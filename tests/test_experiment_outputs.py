"""实验产物隔离、清单更新和恢复目录复用测试。"""

import json
import tempfile
import unittest
from pathlib import Path

from experiment_outputs import (
    create_experiment_directory,
    read_manifest,
    reopen_experiment_directory,
    sanitize_run_id,
    update_manifest,
)


class ExperimentOutputTests(unittest.TestCase):
    """验证实验目录命名和 manifest 生命周期。"""

    def test_creates_utc_isolated_directory_and_manifest(self):
        """新实验应创建唯一 UTC 目录并写入完整初始清单。"""
        with tempfile.TemporaryDirectory() as directory:
            experiment_dir = create_experiment_directory(
                directory,
                "damcs:01/test",
                {"agent_num": 6},
            )

            self.assertEqual(experiment_dir.parent, Path(directory).resolve())
            self.assertRegex(
                experiment_dir.name,
                r"^\d{8}T\d{12}Z_damcs_01_test$",
            )
            manifest = read_manifest(experiment_dir)
            self.assertEqual(manifest["run_id"], "damcs:01/test")
            self.assertEqual(manifest["status"], "running")
            self.assertEqual(manifest["config"]["agent_num"], 6)

    def test_resume_reuses_directory_and_increments_counter(self):
        """恢复必须复用原目录，且不能额外创建第二个实验目录。"""
        with tempfile.TemporaryDirectory() as directory:
            experiment_dir = create_experiment_directory(directory, "run-1", {})
            update_manifest(experiment_dir, status="failed")

            reopened = reopen_experiment_directory(experiment_dir, "run-1")

            self.assertEqual(reopened, experiment_dir)
            manifest = json.loads(
                (experiment_dir / "manifest.json").read_text(encoding="utf-8")
            )
            self.assertEqual(manifest["status"], "running")
            self.assertEqual(manifest["resume_count"], 1)
            self.assertEqual(len(list(Path(directory).iterdir())), 1)

    def test_sanitize_run_id_never_returns_empty_component(self):
        """非法字符、空名称和 Windows 保留名均应转换为安全名称。"""
        self.assertEqual(sanitize_run_id("<>:/\\|?*"), "________")
        self.assertEqual(sanitize_run_id("..."), "run")
        self.assertEqual(sanitize_run_id("CON"), "run_CON")


if __name__ == "__main__":
    unittest.main()
