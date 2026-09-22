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

from projects.hypernet_e2e.run_logging import as_trainer_loggers, experiment_loggers, log_run_config, logger_references, text_log
from projects.hypernet_e2e.run_record import RunRecorder
from projects.hypernet_e2e.runtime_paths import normalize_runtime_paths, repository_root


def run_fit(config: DictConfig) -> Path:
    """解決済み設定から新規 run を記録し、一回だけ Lightning ``fit`` を実行する。

    `study` が指す `analysis/<study>/` の実在を確かめてから始める。
    ``weighting=inverse`` の場合、train split の target 頻度から class weight を計算して config に
    反映する。実行ログは ``logs/train.log``、epoch ごとの metric は ``logger`` group が指す
    experiment logger が持ち、fit に成功すれば最終の scalar callback metrics を
    ``metrics/fit.json`` と ``run.json`` に保存する。fit が出力した checkpoint は、2 段学習の
    2 段目へ渡せるよう選択基準と SHA-256 付きで ``run.json`` に記録する。既存 run の再開・
    test 実行・stage 制御はこの入口の責務に含めない。

    Args:
        config: Hydra が合成した解決済み設定

    Returns:
        Path: 記録した run directory

    Raises:
        BaseException: fit 中の例外はそのまま送出する。送出前に run を失敗として確定する。
    """
    normalize_runtime_paths(config)
    _validate_study(config)
    _validate_attribute_invariance(config)
    _resolve_inverse_class_weights(config)
    recorder = RunRecorder.prepare_fit(config)
    try:
        with text_log(recorder.run_dir / "logs"), experiment_loggers(config) as loggers:
            recorder.record_loggers(logger_references(loggers))
            log_run_config(loggers, config)
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


def _validate_study(config: DictConfig) -> None:
    """run が属する study が、分析側の package として実在することを確かめる。

    study は「この run がどの仮説のためのものか」を指し、`analysis/<study>/` が正本になる。
    打ち間違いを許すと run と分析の対応が静かに切れるので、fit を始める前に照合する。

    Args:
        config: `study` を持つ実行設定。

    Returns:
        None

    Raises:
        ValueError: `study` が未指定か、対応する `analysis/<study>/` が無い場合。
    """
    # `???` のままでも key ごと無くても、同じ「指定されていない」として扱う。
    study = OmegaConf.select(config, "study", default=None)
    if not study:
        raise ValueError("study を指定する（値は analysis/<study>/ の directory 名。探りの run は study=scratch）")
    if not (repository_root() / "analysis" / str(study)).is_dir():
        raise ValueError(f"study に対応する分析 package が無い: analysis/{study}（探りの run は study=scratch）")


def _validate_attribute_invariance(config: DictConfig) -> None:
    """属性不変化の第1段を ERM 以外の学習 strategy と併用させない。

    Group DRO の Hydra override は model.loss_fn を直接差し替える。そのまま許すと属性不変
    wrapper が静かに消えるため、fit 前に意図しない組み合わせを明示的に拒否する。

    Args:
        config: Hydra が合成した実行設定。

    Returns:
        None

    Raises:
        ValueError: 属性不変化を ERM 以外の strategy と組み合わせた場合。
    """
    invariance = config.get("attribute_invariance")
    if not invariance or not invariance.get("enabled", False):
        return
    if config.training_strategy.name != "erm":
        raise ValueError("attribute_invariance は training_strategy=erm でのみ利用できる")


def _resolve_inverse_class_weights(config: DictConfig) -> None:
    """inverse weighting 時に train split から class weight を設定する。"""
    if config.get("weighting", "none") != "inverse":
        return
    frame = pd.read_csv(Path(str(config.data.cv_splits_dir)) / "train.csv")
    labels = [int(value) for value in frame["target"]]
    num_classes = int(config.data.num_classes)
    weights = _inverse_frequency_weights(labels, num_classes)
    loss_target = str(OmegaConf.select(config, "model.loss_fn._target_", default=""))
    path = "model.loss_fn.task_loss.class_weight" if loss_target.endswith("AttributeInvariantTaskLoss") else "model.loss_fn.class_weight"
    OmegaConf.update(config, path, weights, merge=False)


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
