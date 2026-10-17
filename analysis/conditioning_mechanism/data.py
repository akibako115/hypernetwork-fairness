"""3 データセット（HAM10000・FairVLMed・CheXpert）の予測 cache と split を、同じ形で読む。

予測は `cache/` の予測 cache（`analysis/common/predictions.py` の形式）を読み、ここでは推論しない。
2026-10-17 に、それまで各 package（`ham10000_architecture_erm`・`fairvlmed_architecture_erm`・archive の
`latent_bias_hypernet`・`race_known_architecture_objective_comparison`）に分かれていた cache を、この package の
`cache/` へ hardlink で集めた。同じ run の cache が 2 か所にあったものは、以前の読み順で後に読んでいた側を残した。
cache の i 行目は split CSV の i 行目に対応し、`image` 列で検算する。

評価する行:
- HAM10000：test は施設が既知で sex・age が揃う行（格子の HAM10000 の主比較と同じ方針）。
  external は年齢欠損の符号化で条件付きモデルが崩れるので、ここで問う「属性が効くか」と混ざる
- FairVLMed・CheXpert：全行
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder

from analysis.common.paths import ANALYSIS_ROOT, REPOSITORY_ROOT

PACKAGE = ANALYSIS_ROOT / "conditioning_mechanism"
SEEDS = (42, 43, 44)
RUN_FILE = re.compile(r"^\d{8}T\d{6}Z-(?P<exp>.+)-s(?P<seed>\d+)-[0-9a-f]{4}_(?P<split>[a-z]+)(?P<suffix>.*)\.npz$")


@dataclass(frozen=True)
class Dataset:
    """1 データセットの読み方。`models` は分析上の名前から experiment 名への対応。"""

    name: str
    splits: str
    models: dict[str, str]
    categorical: tuple[str, ...]
    continuous: tuple[str, ...]


DATASETS = {
    "ham10000": Dataset(
        "ham10000",
        "data/ham10000/splits_latent",
        {
            "resnet": "resnet-ham10000",
            "meta": "spatial-lora-ham10000",
            "hyperadapt": "hyperadapt-ham10000",
            "const": "spatial-lora-ham10000-const",
            "random8": "spatial-lora-ham10000-random8",
            "fc": "spatial-lora-ham10000-fc",
        },
        ("sex", "localization"),
        ("age",),
    ),
    "fairvlmed": Dataset(
        "fairvlmed",
        "data/fairvlmed/splits",
        {
            "resnet": "resnet-fairvlmed",
            "meta": "spatial-lora-fairvlmed",
            "hyperadapt": "hyperadapt-fairvlmed",
            "const": "spatial-lora-fairvlmed-const",
            "random8": "spatial-lora-fairvlmed-random8",
        },
        ("gender", "race", "ethnicity", "language", "maritalstatus"),
        ("age",),
    ),
    "chexpert": Dataset(
        "chexpert",
        "data/chexpert/splits_race_known",
        {
            "resnet": "resnet-chexpert-race-known",
            "meta": "spatial-lora-chexpert-race-known",
            "hyperadapt": "hyperadapt-chexpert-race-known",
        },
        ("sex", "race", "ethnicity", "frontal_lateral", "ap_pa"),
        ("age",),
    ),
}


def read_split(dataset: Dataset, split: str) -> pd.DataFrame:
    return pd.read_csv(REPOSITORY_ROOT / dataset.splits / f"{split}.csv")


def evaluation_rows(dataset: Dataset, frame: pd.DataFrame) -> np.ndarray:
    """評価に使う行の mask。HAM は施設が既知で sex・age の揃う行（理由は module docstring）。"""
    if dataset.name != "ham10000":
        return np.ones(len(frame), dtype=bool)
    return ((frame.site_missing == 0) & (frame.sex_missing == 0) & (frame.age_missing == 0)).to_numpy()


def patients(dataset: Dataset, frame: pd.DataFrame) -> np.ndarray:
    """bootstrap と cross-fit の単位。HAM は病変、CheXpert は患者、FairVLMed は 1 患者 1 枚なので画像。"""
    if dataset.name == "ham10000":
        return frame.lesion_id.to_numpy()
    if dataset.name == "chexpert":
        return frame.image.str.split("/").str[1].to_numpy()
    return frame.image.to_numpy()


def cache_files(dataset: Dataset, split: str, suffix: str = "") -> dict[str, dict[int, Path]]:
    """model 名 → seed → cache file。`suffix` は `_features` や `_cf-constant_typical`。"""
    found: dict[str, dict[int, Path]] = {}
    experiments = {experiment: model for model, experiment in dataset.models.items()}
    for path in sorted((PACKAGE / "cache").glob(f"*_{split}{suffix}.npz")):
        match = RUN_FILE.match(path.name)
        if match is None or match["suffix"] != suffix or match["exp"] not in experiments:
            continue
        if int(match["seed"]) in SEEDS:
            found.setdefault(experiments[match["exp"]], {})[int(match["seed"])] = path
    return found


def load_logits(dataset: Dataset, split: str, images: np.ndarray, suffix: str = "") -> dict[str, list[np.ndarray]]:
    """3 seed が揃う model だけを、ロジット差（陽性 − 陰性）の seed 順の list で返す。

    Raises:
        ValueError: cache の行が split CSV と対応しない場合。
    """
    out = {}
    for model, by_seed in cache_files(dataset, split, suffix).items():
        if len(by_seed) != len(SEEDS):
            continue
        scores = []
        for seed in SEEDS:
            cache = np.load(by_seed[seed], allow_pickle=True)
            if not np.array_equal(cache["image"], images):
                raise ValueError(f"{by_seed[seed].name}: split CSV と行が対応しない")
            scores.append(cache["logits"][:, 1] - cache["logits"][:, 0])
        out[model] = scores
    return out


def load_evaluation(dataset: Dataset, split: str) -> tuple[pd.DataFrame, dict[str, list[np.ndarray]]]:
    """評価行に絞った split と、その行のロジット。"""
    frame = read_split(dataset, split)
    keep = evaluation_rows(dataset, frame)
    scores = load_logits(dataset, split, frame.image.to_numpy())
    kept = {model: [s[keep] for s in seeds] for model, seeds in scores.items()}
    return frame[keep].reset_index(drop=True), kept


class AttributeDesign:
    """条件として model に入る属性を、ロジスティック回帰の説明変数にする。

    カテゴリは one-hot（欠損の値も 1 つのカテゴリ）、連続値は train の平均・SD で標準化し、欠損フラグを足す。
    model の入力と同じ列を使うので、「model が受け取った情報」を線形に使ったときの上限の目安になる。
    """

    def __init__(self, dataset: Dataset, train: pd.DataFrame) -> None:
        self.dataset = dataset
        self.encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
        self.encoder.fit(train[list(dataset.categorical)].astype(int).astype(str))
        self.stats = {k: (train[k].mean(), train[k].std()) for k in dataset.continuous}

    def __call__(self, frame: pd.DataFrame) -> np.ndarray:
        parts = [self.encoder.transform(frame[list(self.dataset.categorical)].astype(int).astype(str))]
        for key, (mean, sd) in self.stats.items():
            parts.append(((frame[key] - mean) / sd).to_numpy()[:, None])
            parts.append(frame[[f"{key}_missing"]].to_numpy())
        return np.hstack(parts)
