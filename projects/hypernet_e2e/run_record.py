"""e2e の単段 fit に必要な不変 run artifact を記録する。"""

from __future__ import annotations

import hashlib
import json
import math
import re
import secrets
import subprocess
import sys
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
from omegaconf import DictConfig, OmegaConf

from projects.hypernet_e2e.run_logging import inject_logger_outputs
from projects.hypernet_e2e.runtime_paths import normalize_runtime_paths, resolve_repository_path

_MISSING = object()


class RunRecorder:
    """新規 e2e fit の directory 予約から状態確定までを所有する。

    ``prepare_fit`` は解決済み Hydra config を受け、既存 run を再利用せずに directory、
    config、train/val の data manifest、golden preflight を作成する。preflight が失敗した
    場合も run は ``failed`` として残る。呼び出し側は fit が終わったら ``record_checkpoints`` と
    ``succeed``、例外時は ``fail`` を一度だけ呼ぶ。run directory は
    ``projects/hypernet_e2e/runs`` 配下に限る。

    ``model.backbone_checkpoint_path`` を持つ run は 2 段学習の 2 段目なので、``prepare_fit`` が
    読み込む checkpoint の来歴を ``parent_run`` に確定させる。
    """

    schema_version = 2

    def __init__(self, run_dir: Path, run_record: dict[str, Any]) -> None:
        self.run_dir = run_dir
        self._run_record = run_record

    @classmethod
    def prepare_fit(cls, config: DictConfig, *, project_dir: str | Path | None = None) -> RunRecorder:
        """fit 前の artifact を作成し、golden preflight を通した recorder を返す。"""
        normalize_runtime_paths(config)
        resolved = OmegaConf.to_container(config, resolve=True)
        if not isinstance(resolved, dict):
            raise TypeError("run config は mapping である必要がある")

        root = cls._project_dir(resolved, project_dir)
        run_dir, run_id = cls._reserve_run_dir(root, resolved)
        cls._inject_run_paths(config, run_dir, run_id)
        recorder = cls(
            run_dir,
            {
                "schema_version": cls.schema_version,
                "run_id": run_id,
                "kind": "fit",
                "status": "running",
                "started_at": cls._timestamp(),
                "finished_at": None,
                "git_commit": cls._git_commit(),
                "seed": resolved.get("seed"),
                "loggers": [],
                "parent_run": None,
                "checkpoints": [],
                "result_summary": None,
            },
        )
        try:
            for name in ("logs", "metrics", "checkpoints", "artifacts/cohorts"):
                (run_dir / name).mkdir(parents=True, exist_ok=False)
            recorder._run_record["parent_run"] = cls._parent_run(resolved)
            OmegaConf.save(config=config, f=run_dir / "config.yaml", resolve=True)
            recorder._write_json("run.json", recorder._run_record)
            recorder._write_data_manifest(resolved)
            recorder._write_preflight()
        except BaseException as error:
            recorder.fail(error)
            raise
        return recorder

    def record_loggers(self, references: Sequence[Mapping[str, Any]]) -> None:
        """experiment logger の run 参照を記録する。

        fit の前に呼ぶことで、失敗した run からも wandb dashboard を辿れる。
        """
        self._run_record["loggers"] = [dict(reference) for reference in references]
        self._write_json("run.json", self._run_record)

    def record_checkpoints(self, callbacks: Sequence[Any]) -> None:
        """fit が出力した checkpoint を、選択基準と SHA-256 付きの参照として記録する。

        2 段学習の 2 段目は、この記録から backbone checkpoint の path を選ぶ。どの基準で選ばれた
        checkpoint なのかが分からないと段の意味が決まらないので、``monitor`` と ``mode`` も残す。
        checkpoint を出力していない callback は記録しない。
        """
        self._run_record["checkpoints"] = [reference for callback in callbacks if (reference := self._checkpoint_callback_reference(callback)) is not None]
        self._write_json("run.json", self._run_record)

    def succeed(self, result_summary: Mapping[str, Any] | None = None) -> None:
        """成功した fit の終了時刻と JSON 化可能な結果要約を確定する。"""
        self._finish("succeeded", result_summary=result_summary)

    def fail(self, error: BaseException) -> None:
        """失敗した fit の例外種別とメッセージを残して状態を確定する。"""
        self._finish("failed", result_summary={"error_type": type(error).__name__, "error_message": str(error)})

    @classmethod
    def _parent_run(cls, config: Mapping[str, Any]) -> dict[str, Any] | None:
        """2 段目 run が読み込む backbone checkpoint の来歴を確定する。

        checkpoint の path は人が CLI で渡すので、どの run の出力かは path からしか辿れない。
        ``<run-dir>/checkpoints/<name>.ckpt`` という配置を前提に親 run の ``run.json`` を読み、
        run ID と、その run が記録した SHA-256 との一致を確かめる。記録と食い違う checkpoint で
        学習を始めると、来歴だけが残って中身は別物という run になるため、ここで止める。
        """
        model = config.get("model")
        if not isinstance(model, Mapping) or model.get("backbone_checkpoint_path") is None:
            return None

        path = Path(str(model["backbone_checkpoint_path"]))
        if not path.is_file():
            raise ValueError(f"backbone checkpoint が見つからない: {path}")
        digest = cls._sha256_file(path)
        record: dict[str, Any] = {"run_id": None, "backbone_checkpoint": {"path": str(path.resolve()), "sha256": digest}}

        parent_record = path.parent.parent / "run.json"
        if path.parent.name != "checkpoints" or not parent_record.is_file():
            return record
        parent = json.loads(parent_record.read_text())
        record["run_id"] = parent.get("run_id")
        recorded = cls._recorded_digest(parent, path)
        if recorded is not None and recorded != digest:
            raise RuntimeError(f"backbone checkpoint の SHA-256 が {parent_record} の記録と一致しない: {path}（記録 {recorded} / 実体 {digest}）")
        return record

    @staticmethod
    def _recorded_digest(parent: Mapping[str, Any], path: Path) -> str | None:
        """親 run の ``run.json`` が同じ checkpoint に対して残した SHA-256 を探す。"""
        resolved = str(path.resolve())
        for entry in parent.get("checkpoints") or []:
            for reference in (entry.get("best"), entry.get("last")):
                if reference and str(Path(str(reference["path"])).resolve()) == resolved:
                    return str(reference["sha256"])
        return None

    @classmethod
    def _checkpoint_callback_reference(cls, callback: Any) -> dict[str, Any] | None:
        best_path = getattr(callback, "best_model_path", None)
        if not best_path:
            return None
        last_path = getattr(callback, "last_model_path", None)
        return {
            "monitor": getattr(callback, "monitor", None),
            "mode": getattr(callback, "mode", None),
            "best": cls._checkpoint_reference(Path(best_path), cls._score(getattr(callback, "best_model_score", None))),
            "last": cls._checkpoint_reference(Path(last_path), None) if last_path else None,
        }

    @classmethod
    def _checkpoint_reference(cls, path: Path, score: float | None) -> dict[str, Any]:
        return {"path": str(path), "score": score, "sha256": cls._sha256_file(path)}

    @staticmethod
    def _score(value: Any) -> float | None:
        """checkpoint の選択スコアを JSON に書ける float へ落とす。"""
        if value is None:
            return None
        numeric = float(value)
        return numeric if math.isfinite(numeric) else None

    @staticmethod
    def _project_dir(config: Mapping[str, Any], project_dir: str | Path | None) -> Path:
        if project_dir is not None:
            return resolve_repository_path(project_dir)
        paths = config.get("paths")
        if not isinstance(paths, Mapping) or "project_dir" not in paths:
            raise ValueError("run config に paths.project_dir が必要")
        return resolve_repository_path(str(paths["project_dir"]))

    @staticmethod
    def _inject_run_paths(config: DictConfig, run_dir: Path, run_id: str) -> None:
        """予約済みの出力先を実行時 config に反映する。"""
        OmegaConf.update(config, "run_dir", str(run_dir), merge=False)
        checkpoint_path = "callbacks.model_checkpoint.dirpath"
        if OmegaConf.select(config, checkpoint_path, default=_MISSING) is not _MISSING:
            OmegaConf.update(config, checkpoint_path, str(run_dir / "checkpoints"), merge=False)
        inject_logger_outputs(config, run_dir, run_id)

    @classmethod
    def _reserve_run_dir(cls, project_dir: Path, config: Mapping[str, Any]) -> tuple[Path, str]:
        experiment = cls._run_id_part(str(config.get("experiment_name", "experiment")))
        seed = config.get("seed")
        seed_part = "snone" if seed is None else f"s{seed}"
        runs_dir = project_dir / "runs"
        runs_dir.mkdir(parents=True, exist_ok=True)
        for _ in range(100):
            run_id = f"{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}-{experiment}-{seed_part}-{secrets.token_hex(2)}"
            run_dir = runs_dir / run_id
            try:
                run_dir.mkdir()
            except FileExistsError:
                continue
            return run_dir, run_id
        raise RuntimeError("一意な run directory を予約できなかった")

    def _write_data_manifest(self, config: Mapping[str, Any]) -> None:
        data = config.get("data")
        if not isinstance(data, Mapping):
            raise ValueError("run config に data 設定が必要")
        splits_dir = Path(str(data["cv_splits_dir"]))
        manifest = {
            "schema_version": self.schema_version,
            "image_root": str(Path(str(data["data_dir"])).resolve()),
            "splits": {split: self._split_manifest(splits_dir / f"{split}.csv") for split in ("train", "val")},
        }
        self._write_json("data_manifest.json", manifest)

    @classmethod
    def _split_manifest(cls, path: Path) -> dict[str, Any]:
        frame = pd.read_csv(path)
        if "image" not in frame or "target" not in frame:
            raise ValueError(f"split manifest には image と target 列が必要: {path}")
        target_counts = frame["target"].value_counts(dropna=False).sort_index()
        return {
            "path": str(path.resolve()),
            "sha256": cls._sha256_file(path),
            "num_rows": len(frame),
            "image_set_sha256": cls._sha256_lines(frame["image"].astype(str).sort_values().tolist()),
            "target_distribution": {str(key): int(value) for key, value in target_counts.items()},
        }

    def _write_preflight(self) -> None:
        # runtime_paths.repository_root() と同じ解決（symlink 経由の起動でも同じ root を指す）。
        repository_root = Path(__file__).resolve().parents[2]
        golden_dir = repository_root / "tests" / "golden"
        golden_files = sorted(golden_dir.glob("*.v*.json"))
        command = [sys.executable, "-m", "pytest", "projects/hypernet_e2e/tests", "-m", "preflight", "-q"]
        completed = subprocess.run(command, cwd=repository_root, capture_output=True, text=True, check=False)
        record = {
            "schema_version": self.schema_version,
            "git_commit": self._run_record["git_commit"],
            "command": command,
            "goldens": [{"name": path.name, "sha256": self._sha256_file(path)} for path in golden_files],
            "exit_code": completed.returncode,
            "stdout": completed.stdout,
            "stderr": completed.stderr,
        }
        self._write_json("preflight.json", record)
        if completed.returncode != 0:
            raise RuntimeError("golden preflight に失敗した")

    def _finish(self, status: str, *, result_summary: Mapping[str, Any] | None) -> None:
        if self._run_record["status"] != "running":
            raise RuntimeError(f"run はすでに {self._run_record['status']!r} として確定済み")
        self._run_record["status"] = status
        self._run_record["finished_at"] = self._timestamp()
        self._run_record["result_summary"] = dict(result_summary) if result_summary is not None else {}
        self._write_json("run.json", self._run_record)

    def _write_json(self, filename: str, value: Mapping[str, Any]) -> None:
        (self.run_dir / filename).write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n")

    @staticmethod
    def _run_id_part(value: str) -> str:
        normalized = re.sub(r"[^a-z0-9]+", "-", value.lower()).strip("-")
        return normalized or "experiment"

    @staticmethod
    def _timestamp() -> str:
        return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")

    @staticmethod
    def _git_commit() -> str | None:
        completed = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=False)
        return completed.stdout.strip() if completed.returncode == 0 else None

    @staticmethod
    def _sha256_file(path: Path) -> str:
        digest = hashlib.sha256()
        with path.open("rb") as file:
            for chunk in iter(lambda: file.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()

    @staticmethod
    def _sha256_lines(lines: list[str]) -> str:
        return hashlib.sha256("\n".join(lines).encode()).hexdigest()
