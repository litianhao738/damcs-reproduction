"""实验产物目录的创建、恢复与清单生命周期管理。"""

from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


# Windows 文件名禁止使用的字符和控制字符。
_UNSAFE_PATH_CHARS = re.compile(r'[<>:"/\\|?*\x00-\x1f]')
# Windows 设备保留名不能直接作为目录名，即使带有扩展名也不允许。
_WINDOWS_RESERVED_NAMES = {
    "CON", "PRN", "AUX", "NUL",
    *(f"COM{index}" for index in range(1, 10)),
    *(f"LPT{index}" for index in range(1, 10)),
}


def _utc_now() -> datetime:
    """返回带时区信息的当前 UTC 时间。"""
    return datetime.now(timezone.utc)


def sanitize_run_id(run_id: str) -> str:
    """把 run_id 转换为可读且可在 Windows 使用的单级目录名。"""
    # 限制长度，避免实验根目录较深时超过 Windows 路径长度限制。
    safe = _UNSAFE_PATH_CHARS.sub("_", run_id).strip(" .")[:80]
    if not safe:
        return "run"
    if safe.split(".", 1)[0].upper() in _WINDOWS_RESERVED_NAMES:
        safe = f"run_{safe}"
    return safe


def _write_manifest(experiment_dir: Path, manifest: dict[str, Any]) -> None:
    """通过临时文件替换的方式原子更新 manifest.json。"""
    path = experiment_dir / "manifest.json"
    temporary_path = experiment_dir / "manifest.json.tmp"
    temporary_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    temporary_path.replace(path)


def create_experiment_directory(
    results_root: str | Path,
    run_id: str,
    config: dict[str, Any],
) -> Path:
    """创建带 UTC 时间和 run_id 的独立实验目录及初始清单。"""
    now = _utc_now()
    timestamp = now.strftime("%Y%m%dT%H%M%S%fZ")
    experiment_dir = (
        Path(results_root).expanduser().resolve()
        / f"{timestamp}_{sanitize_run_id(run_id)}"
    )
    experiment_dir.mkdir(parents=True, exist_ok=False)
    # 清单保存复现实验所需的参数和后续恢复状态。
    manifest = {
        "run_id": run_id,
        "started_at_utc": now.isoformat().replace("+00:00", "Z"),
        "updated_at_utc": now.isoformat().replace("+00:00", "Z"),
        "status": "running",
        "resume_count": 0,
        "config": config,
        "rounds": [],
    }
    _write_manifest(experiment_dir, manifest)
    return experiment_dir


def read_manifest(experiment_dir: str | Path) -> dict[str, Any]:
    """读取指定实验目录中的 manifest.json。"""
    path = Path(experiment_dir).expanduser().resolve() / "manifest.json"
    if not path.is_file():
        raise ValueError(f"Experiment manifest does not exist: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def update_manifest(experiment_dir: str | Path, **updates: Any) -> dict[str, Any]:
    """合并清单字段并刷新最后更新时间。"""
    experiment_dir = Path(experiment_dir).expanduser().resolve()
    manifest = read_manifest(experiment_dir)
    manifest.update(updates)
    manifest["updated_at_utc"] = _utc_now().isoformat().replace("+00:00", "Z")
    _write_manifest(experiment_dir, manifest)
    return manifest


def reopen_experiment_directory(experiment_dir: str | Path, run_id: str) -> Path:
    """恢复已有实验目录，并增加恢复次数而不创建新目录。"""
    experiment_dir = Path(experiment_dir).expanduser().resolve()
    manifest = read_manifest(experiment_dir)
    if manifest.get("run_id") != run_id:
        raise ValueError(
            f"Manifest run_id={manifest.get('run_id')!r} does not match {run_id!r}."
        )
    update_manifest(
        experiment_dir,
        status="running",
        resume_count=int(manifest.get("resume_count", 0)) + 1,
        error=None,
    )
    return experiment_dir
