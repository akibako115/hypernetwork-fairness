"""凍結した ResNet ERM（M01）の表現と split 属性を、head 実験用の配列にまとめる。

入力は `cache/<run-id>_<split>_features.npz`（`analysis/common/predictions.py --features` が書く）と
`data/chexpert/splits_race_known/<split>.csv`。行対応は `load_features` が image 列で検算する。

群は年齢 `<65 / ≥65` × race_group 4 区分の 8 群で、学習側の DRO 群と同じ並び。
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import rootutils

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from analysis.common.paths import REPOSITORY_ROOT  # noqa: E402
from analysis.common.predictions import cache_path, features_cache_path, load_cache, load_features  # noqa: E402

PACKAGE = Path(__file__).resolve().parent
CACHE = PACKAGE / "cache"
SPLITS = REPOSITORY_ROOT / "data" / "chexpert" / "splits_race_known"
RACE_GROUPS = ("White", "Asian", "Black", "Others")
GROUP_NAMES = tuple(f"{age}×{race}" for age in ("<65", "≥65") for race in RACE_GROUPS)
M01_RUNS = {
    42: "20261002T124957Z-resnet-chexpert-race-known-s42-d6d3",
    43: "20261002T142200Z-resnet-chexpert-race-known-s43-bbfc",
    44: "20261002T155222Z-resnet-chexpert-race-known-s44-3d73",
}


@dataclass
class Split:
    """1 split 分の表現・ラベル・属性。"""

    features: np.ndarray  # [n, 2048] float32
    target: np.ndarray  # [n] int64
    group: np.ndarray  # [n] int64、GROUP_NAMES の index
    frame: pd.DataFrame  # split CSV そのもの
    logits: np.ndarray  # [n, 2] 元 classifier の logits


def group_ids(frame: pd.DataFrame) -> np.ndarray:
    """年齢 2 区分 × race_group 4 区分の群 ID を返す。"""
    return ((frame["age"].to_numpy() >= 65).astype(np.int64) * 4 + frame["race_group"].to_numpy()).astype(np.int64)


def load_split(seed: int, split: str) -> Split:
    """M01 の指定 seed・split の表現と属性を読む。"""
    run_id = M01_RUNS[seed]
    frame = pd.read_csv(SPLITS / f"{split}.csv")
    images = frame["image"].to_numpy(dtype=str)
    features = load_features(features_cache_path(CACHE, run_id, split), images)["features"]
    predictions = load_cache(cache_path(CACHE, run_id, split), images)
    target = frame["target"].to_numpy().astype(np.int64)
    if not (predictions["target"] == target).all():
        raise ValueError(f"{run_id} {split} の target が split CSV と一致しない")
    return Split(features, target, group_ids(frame), frame, predictions["logits"])
