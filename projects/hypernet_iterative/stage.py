"""親 workflow から独立 process で呼ばれる単一 training stage runner。"""

from __future__ import annotations

import argparse
import json
import math
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import lightning as L
import torch
from hydra.utils import instantiate
from lightning.pytorch.loggers import CSVLogger
from omegaconf import OmegaConf

from .validation import validate_training_config

# 次 stage の warm-start と cohort 生成が参照する checkpoint の選択基準。
_SELECTION_MONITOR = "val/auroc"

# stage directory に出力先を閉じ込める checkpoint callback。補助 checkpoint は Lightning 標準のまま置く。
_CHECKPOINT_TARGETS = frozenset({"lightning.pytorch.callbacks.ModelCheckpoint", "projects.hypernet_iterative.callbacks.checkpoint.LastEpochModelCheckpoint"})

# epoch ごとの metric の置き場所。run artifact 契約が stage ごとに予約している。
_METRICS_DIRNAME = "metrics"


def run(config_path: Path, result_path: Path) -> None:
    """保存済み config で一回だけ fit し、checkpoint と scalar metrics を返す。

    parent が組み立てた config でも、子 process 側で必ず `validate_training_config` を通す。

    Args:
        config_path: parent が保存した stage config。その親 directory を stage の出力先に使う
        result_path: metrics と checkpoint path を書き出す先

    epoch ごとの metric は `metrics/metrics.csv` に残す。親はこれを読んで自分の W&B run へ
    集約するので、子は W&B run を作らない。

    Returns:
        None

    Raises:
        RuntimeError: `val/auroc` を monitor する checkpoint callback がちょうど1つでない場合、
            または best / last checkpoint が書かれなかった場合。
    """
    config = OmegaConf.load(config_path)
    stage_dir = config_path.parent
    # 主 checkpoint だけでなく、BAcc / hidden-cohort 用の補助 checkpoint も stage
    # directory に閉じ込める。callbacks の順番に出力先を依存させない。
    for name, callback in config.callbacks.items():
        if callback.get("_target_") in _CHECKPOINT_TARGETS:
            OmegaConf.update(config, f"callbacks.{name}.dirpath", str(stage_dir / "checkpoints"), merge=False)
    # stage config は parent が生成するため、各 child でも必ず独立に検証する。
    validate_training_config(config)
    if config.seed is not None:
        L.seed_everything(config.seed, workers=True)
    data, model = instantiate(config.data), instantiate(config.model)
    callbacks = [instantiate(value) for value in config.callbacks.values()]
    # CSVLogger はローカルの CSV にしか書かないので、W&B run は親の1本のままである。
    # 子が W&B logger を持つと stage ごとに別 run ができるため、それは引き続き禁じる。
    # `trainer.callback_metrics` は fit 完了後の1点しか残らず、epoch ごとの推移は
    # ここでしか残らない。親はこの CSV を読んで自分の W&B run へ集約する。
    logger = CSVLogger(save_dir=str(stage_dir), name=_METRICS_DIRNAME, version="")
    trainer = instantiate(config.trainer, callbacks=callbacks, logger=logger, default_root_dir=str(stage_dir))
    trainer.fit(model=model, datamodule=data)
    metrics = {k: (_scalar(v)) for k, v in trainer.callback_metrics.items()}
    checkpoint = _selected_checkpoint(callbacks)
    if not checkpoint.best_model_path or not checkpoint.last_model_path:
        raise RuntimeError("stage did not write best and last checkpoints")
    result_path.write_text(
        json.dumps(
            {
                "metrics": metrics,
                "metrics_csv": str(Path(logger.log_dir) / "metrics.csv"),
                "checkpoints": {"val/auroc": {"path": checkpoint.best_model_path, "score": metrics.get("val/auroc")}, "last": {"path": checkpoint.last_model_path}},
            },
            ensure_ascii=False,
            indent=2,
            allow_nan=False,
        )
        + "\n"
    )


def _selected_checkpoint(callbacks: Sequence[Any]) -> Any:
    """`val/auroc` を monitor する checkpoint callback をちょうど1つ選ぶ。

    ここで選ばれた checkpoint が次 stage の warm-start と cohort 生成の参照になる。
    `best_model_path` を持つ先頭の callback を取ると、`callbacks.model_checkpoint.monitor` を
    差し替えた run や callback の並びを変えた run で、別基準の checkpoint が黙って
    `val/auroc` として記録されてしまう。stage の選択基準は workflow の契約なので、
    monitor まで照合してから選ぶ。
    """
    selected = [callback for callback in callbacks if getattr(callback, "monitor", None) == _SELECTION_MONITOR and hasattr(callback, "best_model_path")]
    if len(selected) != 1:
        raise RuntimeError(f"stage には {_SELECTION_MONITOR} を monitor する checkpoint callback がちょうど1つ必要だが、{len(selected)} 個見つかった")
    return selected[0]


def _scalar(value: Any) -> float | None:
    value = value.detach().cpu().item() if isinstance(value, torch.Tensor) else value
    value = float(value)
    return value if math.isfinite(value) else None


def main() -> None:
    """`--config` と `--result` を受け取り、単一 stage を実行する CLI 入口。

    Args:
        なし

    Returns:
        None
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--result", type=Path, required=True)
    args = parser.parse_args()
    run(args.config, args.result)


if __name__ == "__main__":
    main()
