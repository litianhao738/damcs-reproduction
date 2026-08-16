"""仿真实验的持久化 checkpoint 与 LLM 响应缓存。

快照使用带版本号的 JSON 文档，并显式编码二进制数组。数据库同时保存
运行时元数据，使不兼容的快照明确报错，而不是在错误状态下静默恢复。
"""

from __future__ import annotations

import hashlib
import json
import os
import platform
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from runtime_snapshot import (
    RUNTIME_SNAPSHOT_VERSION,
    create_runtime_snapshot,
    decode_value,
    encode_value,
    restore_runtime_snapshot,
)


# SQLite 表结构版本；快照载荷本身另有 RUNTIME_SNAPSHOT_VERSION。
CHECKPOINT_SCHEMA_VERSION = 2


@dataclass
class RestoredCheckpoint:
    """从数据库恢复后交给仿真控制器的完整运行状态。"""

    # LangGraph 父图在上一个完整环境步结束时的状态。
    state: dict[str, Any]
    # 重建后的 Crafter 环境。
    env: Any
    # 重建后的全部智能体及其记忆。
    agents: list[Any]
    # 已原子提交的环境步数。
    total_step: int
    # 实验是否已经到达终止条件或最大步数。
    completed: bool
    # 本次实验最初创建的产物根目录。
    output_dir: str | None


class SimulationCheckpointStore:
    """使用 SQLite 为每个 run_id 原子保存一份最新完整快照。"""

    def __init__(self, database_path: str | os.PathLike[str]):
        """规范化数据库路径并初始化所需表结构。"""
        self.database_path = Path(database_path).expanduser().resolve()
        self.database_path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize_database()

    def _connect(self):
        """建立启用 WAL 和完整同步策略的 SQLite 连接。"""
        connection = sqlite3.connect(str(self.database_path), timeout=30)
        connection.execute("PRAGMA journal_mode=WAL")
        connection.execute("PRAGMA synchronous=FULL")
        return connection

    def _initialize_database(self):
        """创建 checkpoint/缓存表，并兼容补充旧数据库缺少的列。"""
        with self._connect() as connection:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS simulation_checkpoints (
                    run_id TEXT PRIMARY KEY,
                    schema_version INTEGER NOT NULL,
                    updated_at TEXT NOT NULL,
                    total_step INTEGER NOT NULL,
                    max_steps INTEGER NOT NULL,
                    n_players INTEGER NOT NULL,
                    completed INTEGER NOT NULL,
                    metadata_json TEXT NOT NULL,
                    snapshot BLOB NOT NULL,
                    output_dir TEXT
                )
                """
            )
            # 旧版数据库没有 output_dir，使用轻量迁移保留已有数据。
            columns = {
                row[1]
                for row in connection.execute(
                    "PRAGMA table_info(simulation_checkpoints)"
                ).fetchall()
            }
            if "output_dir" not in columns:
                connection.execute(
                    "ALTER TABLE simulation_checkpoints ADD COLUMN output_dir TEXT"
                )
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS llm_response_cache (
                    run_id TEXT NOT NULL,
                    step INTEGER NOT NULL,
                    agent_id INTEGER NOT NULL,
                    request_hash TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    response BLOB NOT NULL,
                    PRIMARY KEY (run_id, step, agent_id, request_hash)
                )
                """
            )

    def exists(self, run_id: str) -> bool:
        """判断指定 run_id 是否已经有已提交的 checkpoint。"""
        with self._connect() as connection:
            row = connection.execute(
                "SELECT 1 FROM simulation_checkpoints WHERE run_id = ?",
                (run_id,),
            ).fetchone()
        return row is not None

    def delete(self, run_id: str) -> None:
        """删除指定运行的快照和 LLM 缓存；仅供明确清理操作使用。"""
        with self._connect() as connection:
            connection.execute(
                "DELETE FROM llm_response_cache WHERE run_id = ?", (run_id,)
            )
            connection.execute(
                "DELETE FROM simulation_checkpoints WHERE run_id = ?", (run_id,)
            )

    def get_output_dir(self, run_id: str) -> str | None:
        """取得恢复时必须复用的原实验产物目录。"""
        with self._connect() as connection:
            row = connection.execute(
                "SELECT output_dir FROM simulation_checkpoints WHERE run_id = ?",
                (run_id,),
            ).fetchone()
        if row is None:
            raise ValueError(f"No checkpoint exists for run_id={run_id!r}.")
        return row[0]

    @staticmethod
    def llm_request_hash(context: Any, info: Any) -> str:
        """规范化提示上下文和环境信息，生成稳定的 LLM 请求摘要。"""

        def normalize(value):
            """递归消除容器顺序及 NumPy 类型对哈希结果的影响。"""
            if isinstance(value, dict):
                return {
                    str(key): normalize(item)
                    for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
                }
            if isinstance(value, (list, tuple)):
                return [normalize(item) for item in value]
            if isinstance(value, set):
                return sorted((normalize(item) for item in value), key=repr)
            if isinstance(value, np.ndarray):
                return {
                    "__ndarray__": hashlib.sha256(value.tobytes()).hexdigest(),
                    "dtype": str(value.dtype),
                    "shape": list(value.shape),
                }
            if isinstance(value, np.generic):
                return value.item()
            if hasattr(value, "model_dump"):
                return normalize(value.model_dump(mode="json"))
            if value is None or isinstance(value, (str, int, float, bool)):
                return value
            return repr(value)

        # 使用确定性的紧凑 JSON，确保同一请求在恢复后命中缓存。
        request = json.dumps(
            normalize((context, info)),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
        ).encode("utf-8")
        return hashlib.sha256(request).hexdigest()

    def get_llm_response(
        self, *, run_id: str, step: int, agent_id: int, request_hash: str
    ) -> Any | None:
        """读取某一步、某个 Agent 的已成功 LLM 响应。"""
        with self._connect() as connection:
            row = connection.execute(
                """
                SELECT response FROM llm_response_cache
                WHERE run_id = ? AND step = ? AND agent_id = ?
                  AND request_hash = ?
                """,
                (run_id, step, agent_id, request_hash),
            ).fetchone()
        if row is None:
            return None
        return decode_value(json.loads(row[0].decode("utf-8")))

    def save_llm_response(
        self,
        *,
        run_id: str,
        step: int,
        agent_id: int,
        request_hash: str,
        response: Any,
    ) -> None:
        """把可结构化编码的 LLM 响应写入幂等缓存。"""
        response_blob = json.dumps(
            encode_value(response), separators=(",", ":"), ensure_ascii=False
        ).encode("utf-8")
        with self._connect() as connection:
            connection.execute(
                """
                INSERT OR REPLACE INTO llm_response_cache (
                    run_id, step, agent_id, request_hash, created_at, response
                ) VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    run_id,
                    step,
                    agent_id,
                    request_hash,
                    datetime.now(timezone.utc).isoformat(),
                    sqlite3.Binary(response_blob),
                ),
            )

    def save(
        self,
        *,
        run_id: str,
        state: dict[str, Any],
        env: Any,
        agents: list[Any],
        max_steps: int,
        n_players: int,
        output_dir: str | os.PathLike[str] | None = None,
    ) -> None:
        """在一个立即事务中提交最新完整环境步的运行快照。"""
        total_step = int(state["total_step"])
        completed = bool(state["done"] or total_step >= max_steps)
        # 快照同时包含图状态、环境、Agent 记忆和随机数状态。
        snapshot = create_runtime_snapshot(state, env, agents)
        metadata = {
            "python": platform.python_version(),
            "implementation": platform.python_implementation(),
            "schema_version": CHECKPOINT_SCHEMA_VERSION,
            "runtime_snapshot_version": RUNTIME_SNAPSHOT_VERSION,
        }
        updated_at = datetime.now(timezone.utc).isoformat()

        with self._connect() as connection:
            # 提前获取写锁，保证读者只能看到旧快照或完整的新快照。
            connection.execute("BEGIN IMMEDIATE")
            connection.execute(
                """
                INSERT INTO simulation_checkpoints (
                    run_id, schema_version, updated_at, total_step, max_steps,
                    n_players, completed, metadata_json, snapshot, output_dir
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(run_id) DO UPDATE SET
                    schema_version = excluded.schema_version,
                    updated_at = excluded.updated_at,
                    total_step = excluded.total_step,
                    max_steps = excluded.max_steps,
                    n_players = excluded.n_players,
                    completed = excluded.completed,
                    metadata_json = excluded.metadata_json,
                    snapshot = excluded.snapshot,
                    output_dir = excluded.output_dir
                """,
                (
                    run_id,
                    CHECKPOINT_SCHEMA_VERSION,
                    updated_at,
                    total_step,
                    max_steps,
                    n_players,
                    int(completed),
                    json.dumps(metadata, sort_keys=True),
                    sqlite3.Binary(snapshot),
                    str(Path(output_dir).expanduser().resolve()) if output_dir else None,
                ),
            )

    def load(self, *, run_id: str, max_steps: int, n_players: int):
        """校验运行参数和版本后，重建环境、Agent 与图状态。"""
        with self._connect() as connection:
            row = connection.execute(
                """
                SELECT schema_version, total_step, max_steps, n_players,
                       completed, metadata_json, snapshot, output_dir
                FROM simulation_checkpoints
                WHERE run_id = ?
                """,
                (run_id,),
            ).fetchone()

        if row is None:
            raise ValueError(f"No checkpoint exists for run_id={run_id!r}.")

        schema_version, total_step, saved_max_steps, saved_players, completed, metadata_json, snapshot, output_dir = row
        # 不允许用不同规模的实验参数继续同一个环境状态。
        if schema_version != CHECKPOINT_SCHEMA_VERSION:
            raise ValueError(
                f"Checkpoint schema {schema_version} is incompatible with "
                f"schema {CHECKPOINT_SCHEMA_VERSION}."
            )
        if saved_players != n_players:
            raise ValueError(
                f"Checkpoint has {saved_players} players, but --agent_num is {n_players}."
            )
        if saved_max_steps != max_steps:
            raise ValueError(
                f"Checkpoint max_steps is {saved_max_steps}, but --step_num is {max_steps}."
            )

        metadata = json.loads(metadata_json)
        if metadata.get("python") != platform.python_version():
            raise ValueError(
                "Checkpoint Python version does not match the current interpreter: "
                f"{metadata.get('python')} != {platform.python_version()}."
            )

        # 只有全部兼容性校验通过后才执行对象重建。
        state, env, agents = restore_runtime_snapshot(snapshot)
        return RestoredCheckpoint(
            state=state,
            env=env,
            agents=agents,
            total_step=int(total_step),
            completed=bool(completed),
            output_dir=output_dir,
        )
