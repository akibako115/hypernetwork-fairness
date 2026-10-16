"""群ラベルが欠損する split を、群を使う目的関数の run で推論して予測 cache を書く。

`GroupImageDataModule` は群の属性が欠けた行を拒む。HAM10000 の test には sex が欠けた行（external 施設）があり、
性別 × 年齢の GroupDRO の run は `analysis/common/predictions.py` で test を推論できない。群は学習の損失にしか
使わず、network は画像だけを取るので、datamodule を群なしの `ImageDataModule` に差し替えて推論する。
セル均等抽出の `GroupClassImageDataModule` も同じく差し替える（評価の loader は抽出に関係しない）。
それ以外（checkpoint、transform、cache の形式と metadata）は `predictions.write_cache` と同じ。
`--worst` を付けると、val の性別 × 年齢の最悪群 AUROC で選んだ checkpoint（`best_val_worst_sexage_*.ckpt`）を推論する。
`--swad` を付けると、SWAD で平均した重み（`swad.ckpt`）を推論する。

入力: `runs/<run-id>`（取り込み済みの run）
出力: `cache/<run-id>_<split>.npz`

    uv run python -m analysis.archive.latent_bias_hypernet.predict_ungrouped --split test \
      --run-dir analysis/archive/latent_bias_hypernet/runs/<run-id>
"""

from __future__ import annotations

import argparse
from pathlib import Path
from unittest import mock

import torch
from omegaconf import OmegaConf

from analysis.archive.latent_bias_hypernet.data import PACKAGE
from analysis.common import predictions

LOAD = OmegaConf.load
GROUP_KEYS = (
    "group_attribute_names",
    "group_cardinalities",
    "num_groups",
    "group_key",
    "group_missing_values",
    "sampling_seed",
)


def ungrouped_config(path: Path):
    config = LOAD(path)
    target = config.data.get("_target_", "") if "data" in config else ""
    if target.endswith(("GroupImageDataModule", "GroupClassImageDataModule")):
        config.data._target_ = "projects.hypernet_e2e.data.datamodule.ImageDataModule"
        config.data.train_sampling = "uniform"
        for key in GROUP_KEYS:
            config.data.pop(key, None)
    return config


def worst_checkpoint(run_dir: Path) -> Path:
    (path,) = sorted((run_dir / "checkpoints").glob("best_val_worst_sexage_*.ckpt"))
    return path.resolve()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", action="append", dest="run_dirs", required=True, type=Path)
    parser.add_argument("--split", default="test", choices=("val", "test"))
    parser.add_argument("--worst", action="store_true", help="最悪群 AUROC で選んだ checkpoint を使う")
    parser.add_argument("--swad", action="store_true", help="SWAD で平均した重みを使う")
    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    with mock.patch.object(predictions.OmegaConf, "load", side_effect=ungrouped_config):
        for run_dir in args.run_dirs:
            checkpoint = worst_checkpoint(run_dir) if args.worst else None
            checkpoint = (run_dir / "checkpoints" / "swad.ckpt").resolve() if args.swad else checkpoint
            print(predictions.write_cache(run_dir, args.split, PACKAGE / "cache", device, checkpoint=checkpoint))


if __name__ == "__main__":
    main()
