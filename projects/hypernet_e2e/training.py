"""e2e の一回限りの Lightning fit を実行・記録する。"""

from __future__ import annotations

import json
import math
from collections import Counter
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import lightning as L
import pandas as pd
import torch
from hydra.utils import instantiate
from lightning import Callback, LightningDataModule, LightningModule, Trainer
from omegaconf import DictConfig, OmegaConf

from projects.hypernet_e2e.run_logging import as_trainer_loggers, experiment_loggers, logger_references, text_log
from projects.hypernet_e2e.run_record import RunRecorder
from projects.hypernet_e2e.runtime_paths import normalize_runtime_paths


def run_fit(config: DictConfig) -> Path:
    """解決済み設定から新規 run を記録し、一回だけ Lightning ``fit`` を実行する。

    ``weighting=inverse`` の場合、train split の target 頻度から class weight を計算して config に
    反映する。実行ログは ``logs/train.log``、epoch ごとの metric は ``logger`` group が指す
    experiment logger が持ち、fit に成功すれば最終の scalar callback metrics を
    ``metrics/fit.json`` と ``run.json`` に保存する。fit が出力した checkpoint は、2 段学習の
    2 段目へ渡せるよう選択基準と SHA-256 付きで ``run.json`` に記録する。既存 run の再開・
    test 実行・stage 制御はこの入口の責務に含めない。
    """
    normalize_runtime_paths(config)
    _resolve_inverse_class_weights(config)
    recorder = RunRecorder.prepare_fit(config)
    try:
        with text_log(recorder.run_dir / "logs"), experiment_loggers(config) as loggers:
            recorder.record_loggers(logger_references(loggers))
            if config.get("seed") is not None:
                L.seed_everything(config.seed, workers=True)

            datamodule: LightningDataModule = instantiate(config.data)
            model: LightningModule = instantiate(config.model)
            callbacks = _instantiate_callbacks(config.get("callbacks"))
            trainer: Trainer = instantiate(
                config.trainer,
                callbacks=callbacks,
                logger=as_trainer_loggers(loggers),
                default_root_dir=str(recorder.run_dir),
            )
            trainer.fit(model=model, datamodule=datamodule)

            metrics = _scalar_metrics(trainer.callback_metrics)
            _write_metrics(recorder.run_dir / "metrics" / "fit.json", metrics)
            recorder.record_checkpoints(callbacks)
        recorder.succeed(metrics)
    except BaseException as error:
        recorder.fail(error)
        raise
    return recorder.run_dir


def _resolve_inverse_class_weights(config: DictConfig) -> None:
    """inverse weighting 時に train split から class weight を設定する。"""
    if config.get("weighting", "none") != "inverse":
        return
    frame = pd.read_csv(Path(str(config.data.cv_splits_dir)) / "train.csv")
    labels = [int(value) for value in frame["target"]]
    num_classes = int(config.data.num_classes)
    weights = _inverse_frequency_weights(labels, num_classes)
    OmegaConf.update(config, "model.loss_fn.class_weight", weights, merge=False)


def _inverse_frequency_weights(labels: list[int], num_classes: int) -> list[float]:
    """平均が 1 になる逆頻度 class weight を返す。"""
    if not labels:
        raise ValueError("train.csv の target は空にできない")
    expected = list(range(num_classes))
    observed = sorted(set(labels))
    if observed != expected:
        raise ValueError(f"train.csv の target は {expected} である必要があるが、{observed} が指定された")
    counts = Counter(labels)
    sample_count = len(labels)
    raw_weights = [sample_count / (num_classes * counts[class_index]) for class_index in expected]
    mean_weight = sum(raw_weights) / len(raw_weights)
    return [round(weight / mean_weight, 6) for weight in raw_weights]


def _instantiate_callbacks(config: Mapping[str, Any] | None) -> list[Callback]:
    """callbacks config の各 entry を Lightning callback として生成する。"""
    if config is None:
        return []
    return [instantiate(callback_config) for callback_config in config.values()]


def _scalar_metrics(metrics: Mapping[str, Any]) -> dict[str, float | None]:
    """Lightning callback metrics を JSON scalar に変換し、非有限値を ``null`` にする。"""
    result: dict[str, float | None] = {}
    for name, value in metrics.items():
        if isinstance(value, torch.Tensor):
            if value.numel() != 1:
                raise ValueError(f"metric {name!r} は scalar である必要がある")
            numeric_value = float(value.detach().cpu().item())
        elif isinstance(value, (float, int)):
            numeric_value = float(value)
        else:
            raise TypeError(f"metric {name!r} は float または Tensor である必要がある")
        result[name] = numeric_value if math.isfinite(numeric_value) else None
    return result


def _write_metrics(path: Path, metrics: Mapping[str, float | None]) -> None:
    """比較用の fit metric を安定した JSON で保存する。"""
    path.write_text(json.dumps(metrics, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n")
