"""親 workflow から独立 process で呼ばれる単一 training stage runner。"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import lightning as L
import torch
from hydra.utils import instantiate
from omegaconf import OmegaConf

from .validation import validate_training_config


def run(config_path: Path, result_path: Path) -> None:
    """保存済み config で一回だけ fit し、checkpoint と scalar metrics を返す。"""
    config = OmegaConf.load(config_path)
    stage_dir = config_path.parent
    # 主 checkpoint だけでなく、BAcc / hidden-cohort 用の補助 checkpoint も stage
    # directory に閉じ込める。callbacks の順番に出力先を依存させない。
    for name, callback in config.callbacks.items():
        if callback.get("_target_") == "lightning.pytorch.callbacks.ModelCheckpoint":
            OmegaConf.update(config, f"callbacks.{name}.dirpath", str(stage_dir / "checkpoints"), merge=False)
    # stage config は parent が生成するため、各 child でも必ず独立に検証する。
    validate_training_config(config)
    if config.seed is not None:
        L.seed_everything(config.seed, workers=True)
    data, model = instantiate(config.data), instantiate(config.model)
    callbacks = [instantiate(value) for value in config.callbacks.values()]
    trainer = instantiate(config.trainer, callbacks=callbacks, logger=False, default_root_dir=str(stage_dir))
    trainer.fit(model=model, datamodule=data)
    metrics = {k: (_scalar(v)) for k, v in trainer.callback_metrics.items()}
    checkpoint = next(callback for callback in callbacks if hasattr(callback, "best_model_path"))
    if not checkpoint.best_model_path or not checkpoint.last_model_path:
        raise RuntimeError("stage did not write best and last checkpoints")
    result_path.write_text(
        json.dumps(
            {"metrics": metrics, "checkpoints": {"val/auroc": {"path": checkpoint.best_model_path, "score": metrics.get("val/auroc")}, "last": {"path": checkpoint.last_model_path}}},
            ensure_ascii=False,
            indent=2,
            allow_nan=False,
        )
        + "\n"
    )


def _scalar(value: Any) -> float | None:
    value = value.detach().cpu().item() if isinstance(value, torch.Tensor) else value
    value = float(value)
    return value if math.isfinite(value) else None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--result", type=Path, required=True)
    args = parser.parse_args()
    run(args.config, args.result)


if __name__ == "__main__":
    main()
