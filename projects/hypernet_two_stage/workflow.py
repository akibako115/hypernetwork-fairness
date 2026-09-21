"""Stage 1 の最良 checkpoint を凍結した Stage 2 へ渡す固定二段 workflow。"""

from __future__ import annotations

import hashlib
import json
import math
import secrets
import subprocess
import sys
from collections import Counter
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import lightning as L
import pandas as pd
import torch
from hydra.utils import instantiate
from lightning import LightningDataModule, LightningModule, Trainer
from omegaconf import DictConfig, OmegaConf

from projects.hypernet_two_stage.run_logging import (
    as_trainer_loggers,
    experiment_loggers,
    inject_logger_outputs,
    logger_references,
    text_log,
)
from projects.hypernet_two_stage.runtime_paths import normalize_runtime_paths, repository_root


def run_two_stage(config: DictConfig) -> Path:
    """新規 parent run で Stage 1 と Stage 2 を順に実行し、Stage 2 最良 checkpoint を選択する。

    実行ログは親 run の ``logs/train.log`` に両 stage 分をまとめる。experiment logger は stage ごと
    に別 run として作り、その参照を各 stage の結果に残す。
    """
    normalize_runtime_paths(config)
    _resolve_inverse_class_weights(config)
    run_dir = _reserve_run_dir(config)
    stages = run_dir / "stages"
    stages.mkdir()
    record: dict[str, Any] = {"schema_version": 1, "status": "running", "started_at": _timestamp(), "stages": {}, "selected_checkpoint": None}
    try:
        with text_log(run_dir / "logs"):
            OmegaConf.save(config, run_dir / "config.yaml", resolve=True)
            _write_json(run_dir / "run.json", record)
            _write_data_manifest(run_dir, config)
            _write_preflight(run_dir)
            first = _run_stage("stage1", config, stages / "stage1", run_dir.name)
            record["stages"]["stage1"] = first
            stage2_config = _stage2_config(config, first["checkpoints"]["best_val_auroc"]["path"])
            second = _run_stage("stage2", stage2_config, stages / "stage2", run_dir.name)
            record["stages"]["stage2"] = second
            record["selected_checkpoint"] = second["checkpoints"]["best_val_auroc"]
            record["status"] = "succeeded"
    except BaseException as error:
        record["status"] = "failed"
        record["error_type"] = type(error).__name__
        record["error_message"] = str(error)
        raise
    finally:
        record["finished_at"] = _timestamp()
        _write_json(run_dir / "run.json", record)
    return run_dir


def _stage2_config(config: DictConfig, checkpoint_path: str) -> DictConfig:
    """Stage 1 checkpoint を共有重みの入力にした Spatial LoRA stage 設定を返す。"""
    result = OmegaConf.create(OmegaConf.to_container(config, resolve=True))
    model_config = OmegaConf.load(Path(__file__).with_name("configs") / "model" / "spatial_lora_resnet50.yaml")
    class_weight = OmegaConf.select(result, "model.loss_fn.class_weight")
    OmegaConf.update(result, "model", model_config, merge=False)
    OmegaConf.update(result, "model.loss_fn.class_weight", class_weight, merge=False)
    OmegaConf.update(result, "model.backbone_checkpoint_path", checkpoint_path, merge=False)
    OmegaConf.update(result, "model.freeze_backbone", True, merge=False)
    OmegaConf.update(result, "experiment_name", "two_stage_spatial_lora", merge=False)
    return result


def _run_stage(name: str, config: DictConfig, stage_dir: Path, run_id: str) -> dict[str, Any]:
    """予約済み stage directory で一回の Lightning fit を実行し、checkpoint と scalar metrics を返す。"""
    stage_dir.mkdir(parents=True)
    for directory in ("metrics", "checkpoints"):
        (stage_dir / directory).mkdir()
    OmegaConf.update(config, "callbacks.model_checkpoint.dirpath", str(stage_dir / "checkpoints"), merge=False)
    inject_logger_outputs(config, stage_dir, f"{run_id}-{name}")
    OmegaConf.save(config, stage_dir / "config.yaml", resolve=True)
    if config.get("seed") is not None:
        L.seed_everything(config.seed, workers=True)
    datamodule: LightningDataModule = instantiate(config.data)
    model: LightningModule = instantiate(config.model)
    callbacks = [instantiate(value) for value in config.callbacks.values()] if config.get("callbacks") else []
    with experiment_loggers(config) as loggers:
        references = logger_references(loggers)
        trainer: Trainer = instantiate(config.trainer, callbacks=callbacks, logger=as_trainer_loggers(loggers), default_root_dir=str(stage_dir))
        trainer.fit(model=model, datamodule=datamodule)
        metrics = _scalar_metrics(trainer.callback_metrics)
    _write_json(stage_dir / "metrics" / "fit.json", metrics)
    checkpoint = _best_auroc_checkpoint(callbacks, name)
    result = {
        "metrics": metrics,
        "loggers": references,
        "checkpoints": {
            "best_val_auroc": _checkpoint_reference(Path(checkpoint.best_model_path)),
            "last": _checkpoint_reference(Path(checkpoint.last_model_path)) if checkpoint.last_model_path else None,
        },
    }
    _write_json(stage_dir / "result.json", result)
    return result


def _best_auroc_checkpoint(callbacks: Sequence[Any], name: str) -> Any:
    """`val/auroc` を monitor する checkpoint callback をちょうど1つ選ぶ。

    `best_model_path` を持つ先頭の callback を選ぶと、`callbacks.model_checkpoint.monitor` を
    差し替えた run でも別基準の checkpoint が `best_val_auroc` として記録されてしまう。stage の
    選択基準は workflow の契約なので、monitor まで照合してから選ぶ。
    """
    selected = [callback for callback in callbacks if getattr(callback, "monitor", None) == "val/auroc" and hasattr(callback, "best_model_path")]
    if len(selected) != 1:
        raise RuntimeError(f"{name} には val/auroc を monitor する checkpoint callback がちょうど1つ必要だが、{len(selected)} 個見つかった")
    if not selected[0].best_model_path:
        raise RuntimeError(f"{name} は best val/auroc checkpoint を出力する必要がある")
    return selected[0]


def _resolve_inverse_class_weights(config: DictConfig) -> None:
    if config.get("weighting", "none") != "inverse":
        return
    labels = [int(value) for value in pd.read_csv(Path(str(config.data.cv_splits_dir)) / "train.csv")["target"]]
    counts = Counter(labels)
    classes = list(range(int(config.data.num_classes)))
    if sorted(counts) != classes:
        raise ValueError("train.csv の target は全 class を含む必要がある")
    raw = [len(labels) / (len(classes) * counts[index]) for index in classes]
    mean = sum(raw) / len(raw)
    OmegaConf.update(config, "model.loss_fn.class_weight", [round(value / mean, 6) for value in raw], merge=False)


def _reserve_run_dir(config: DictConfig) -> Path:
    project_dir = Path(str(config.paths.project_dir))
    for _ in range(100):
        run_id = f"{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}-two-stage-s{config.seed}-{secrets.token_hex(2)}"
        candidate = project_dir / "runs" / run_id
        try:
            candidate.mkdir(parents=True)
            return candidate
        except FileExistsError:
            continue
    raise RuntimeError("一意な run directory を予約できない")


def _write_data_manifest(run_dir: Path, config: DictConfig) -> None:
    """親 run が用いる train / val split の内容を hash とともに保存する。"""
    splits_dir = Path(str(config.data.cv_splits_dir))
    splits = {}
    for name in ("train", "val"):
        path = splits_dir / f"{name}.csv"
        frame = pd.read_csv(path)
        splits[name] = {
            "path": str(path.resolve()),
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "num_rows": len(frame),
            "target_distribution": {str(key): int(value) for key, value in frame["target"].value_counts().sort_index().items()},
        }
    _write_json(
        run_dir / "data_manifest.json",
        {"schema_version": 1, "image_root": str(Path(str(config.data.data_dir)).resolve()), "splits": splits},
    )


def _write_preflight(run_dir: Path) -> None:
    """学習前に project 自身の golden preflight を実行して記録する。"""
    root = repository_root()
    command = [sys.executable, "-m", "pytest", "projects/hypernet_two_stage/tests", "-m", "preflight", "-q"]
    completed = subprocess.run(command, cwd=root, capture_output=True, text=True, check=False)
    _write_json(run_dir / "preflight.json", {"command": command, "exit_code": completed.returncode, "stdout": completed.stdout, "stderr": completed.stderr})
    if completed.returncode:
        raise RuntimeError("golden preflight に失敗した")


def _checkpoint_reference(path: Path) -> dict[str, Any]:
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    return {"path": str(path), "sha256": digest}


def _scalar_metrics(metrics: Mapping[str, Any]) -> dict[str, float | None]:
    result: dict[str, float | None] = {}
    for key, value in metrics.items():
        numeric = float(value.detach().cpu().item()) if isinstance(value, torch.Tensor) else float(value)
        result[key] = numeric if math.isfinite(numeric) else None
    return result


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n")


def _timestamp() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
