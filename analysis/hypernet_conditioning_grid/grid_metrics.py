"""柱 A の格子の評価指標を、予測 cache と split CSV から計算する。

評価の定義は `docs/research-premises.md` の「公平性のスコープ：動作点」に従う。

- 群は年齢（65 歳以上）× race_group（White / Asian / Black / Others）の 8 群。番号は `年齢 × 4 + race`
  で、学習側の `group_labels` と同じ並び。
- 動作点は固定しきい値 0.5（2 クラス softmax の陽性確率）。CheXpert の陽性は No Finding なので、
  害の大きい側は FPR（所見のある人を正常と判定する見落とし）で、最悪群は FPR が最大の群。
- 率を出す最小セルは 20 件。TPR は陽性の数、FPR は陰性の数で別々に判定する。AUROC は両方を満たす群だけ。
- 区間は患者単位の bootstrap。同じ resample を全条件に当てるので、条件間の差の区間になる。
  学習の seed の揺れは含まない（seed は平均してから差を取り、seed 間の SD は別に出す）。

使い方は `grid.ipynb` を参照。
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import rootutils
from sklearn.metrics import roc_auc_score

ROOT = rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from analysis.common.predictions import load_cache  # noqa: E402

AGE_BOUNDARY = 65.0
RACE_LABELS = ("White", "Asian", "Black", "Others")
GROUP_LABELS = tuple(f"{age} × {race}" for age in ("<65", "≥65") for race in RACE_LABELS)
MIN_CELL = 20
THRESHOLD = 0.5


def load_split(split: str) -> pd.DataFrame:
    """race-known の split CSV を読み、群番号と患者 ID を足す。

    dual-holdout の split は train だけを分けたもので、val・test は race-known と同じ行・同じ順序。

    Args:
        split: `val` または `test`

    Returns:
        pd.DataFrame: CSV の列に `group`（0〜7）と `patient` を足したもの
    """
    frame = pd.read_csv(ROOT / "data" / "chexpert" / "splits_race_known" / f"{split}.csv")
    # 学習側（evaluation_attributes）と同じく、65 歳ちょうどは ≥65 に入る。
    age_group = (frame["age"].to_numpy(dtype=float) >= AGE_BOUNDARY).astype(int)
    frame["group"] = age_group * len(RACE_LABELS) + frame["race_group"].to_numpy(dtype=int)
    frame["patient"] = frame["image"].str.split("/").str[1]
    return frame


def positive_probability(cache_file: Path, frame: pd.DataFrame) -> np.ndarray:
    """cache の陽性確率を、split CSV との行対応を確かめてから返す。

    Args:
        cache_file: `analysis/common/predictions.py` が書いた `.npz`
        frame: `load_split` が返した同じ split の frame

    Returns:
        np.ndarray: 各行の陽性確率

    Raises:
        ValueError: cache の target が split CSV と一致しない場合
    """
    cached = load_cache(cache_file, frame["image"].to_numpy(dtype=str))
    if not (cached["target"] == frame["target"].to_numpy()).all():
        raise ValueError(f"{cache_file} の target が split CSV と一致しない")
    return cached["probabilities"][:, 1]


@dataclass(frozen=True)
class GroupRates:
    """8 群それぞれの率と、率を出せたかどうか。"""

    tpr: np.ndarray
    fpr: np.ndarray
    auroc: np.ndarray
    n_pos: np.ndarray
    n_neg: np.ndarray


def group_rates(
    probability: np.ndarray, target: np.ndarray, group: np.ndarray, weight: np.ndarray | None = None
) -> GroupRates:
    """群ごとの TPR・FPR（しきい値 0.5）と AUROC を求める。下限に届かない群は NaN にする。

    Args:
        probability: 陽性確率
        target: 0/1 のラベル
        group: 群番号 0〜7
        weight: bootstrap の重み（各行が resample で何回選ばれたか）。None は全行 1 回

    Returns:
        GroupRates: 8 群分の配列
    """
    weight = np.ones_like(probability) if weight is None else weight
    predicted = probability >= THRESHOLD
    shape = len(GROUP_LABELS)
    tpr, fpr, auroc = (np.full(shape, np.nan) for _ in range(3))
    n_pos, n_neg = np.zeros(shape), np.zeros(shape)
    for g in range(shape):
        in_group = (group == g) & (weight > 0)
        positive = in_group & (target == 1)
        negative = in_group & (target == 0)
        n_pos[g] = weight[positive].sum()
        n_neg[g] = weight[negative].sum()
        if n_pos[g] >= MIN_CELL:
            tpr[g] = weight[positive & predicted].sum() / n_pos[g]
        if n_neg[g] >= MIN_CELL:
            fpr[g] = weight[negative & predicted].sum() / n_neg[g]
        if n_pos[g] >= MIN_CELL and n_neg[g] >= MIN_CELL:
            auroc[g] = roc_auc_score(target[in_group], probability[in_group], sample_weight=weight[in_group])
    return GroupRates(tpr, fpr, auroc, n_pos, n_neg)


def summary(
    probability: np.ndarray, target: np.ndarray, group: np.ndarray, weight: np.ndarray | None = None
) -> dict[str, float]:
    """評価の定義に沿った指標の一式を返す。

    Eopp1 は群の TPR の最大 − 最小、Eopp0 は FPR の最大 − 最小、Eodds はその平均。worst_fpr は FPR が
    最大の群の FPR（CheXpert で害の大きい側）。評価できた群の数も返す（外れた群があると差は小さく見える）。

    Args:
        probability: 陽性確率
        target: 0/1 のラベル
        group: 群番号 0〜7
        weight: bootstrap の重み。None は全行 1 回

    Returns:
        dict[str, float]: 指標名 → 値
    """
    weight = np.ones_like(probability) if weight is None else weight
    rates = group_rates(probability, target, group, weight)
    predicted = probability >= THRESHOLD
    positive, negative = target == 1, target == 0
    eopp1 = np.nanmax(rates.tpr) - np.nanmin(rates.tpr)
    eopp0 = np.nanmax(rates.fpr) - np.nanmin(rates.fpr)
    return {
        "auroc": roc_auc_score(target, probability, sample_weight=weight),
        "worst_group_auroc": np.nanmin(rates.auroc),
        "auroc_gap": np.nanmax(rates.auroc) - np.nanmin(rates.auroc),
        "tpr": weight[positive & predicted].sum() / weight[positive].sum(),
        "fpr": weight[negative & predicted].sum() / weight[negative].sum(),
        "eopp1": eopp1,
        "eopp0": eopp0,
        "eodds": (eopp1 + eopp0) / 2,
        "worst_fpr": np.nanmax(rates.fpr),
        "min_tpr": np.nanmin(rates.tpr),
        "groups_tpr": float(np.isfinite(rates.tpr).sum()),
        "groups_fpr": float(np.isfinite(rates.fpr).sum()),
    }


def patient_bootstrap_weights(patient: pd.Series, n_resamples: int, seed: int = 0) -> np.ndarray:
    """患者単位で resample したときの、各行の重み（選ばれた回数）を返す。

    同じ患者の画像はまとめて選ぶ。全条件に同じ重みを当てると、条件間の差の区間になる。

    Args:
        patient: 各行の患者 ID
        n_resamples: resample の回数
        seed: 乱数の seed

    Returns:
        np.ndarray: `[n_resamples, n_rows]` の重み
    """
    codes, uniques = pd.factorize(patient)
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, len(uniques), size=(n_resamples, len(uniques)))
    counts = np.stack([np.bincount(row, minlength=len(uniques)) for row in draws])
    return counts[:, codes].astype(float)
