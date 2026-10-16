"""test 上の 2 つのスコアの差を、患者単位の paired bootstrap で評価する。"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score


def patient_index(images: pd.Series) -> np.ndarray:
    return pd.factorize(images.str.split("/").str[1])[0]


def paired_bootstrap(
    y: np.ndarray, a: np.ndarray, b: np.ndarray, patients: np.ndarray, metric, n_boot: int = 300, seed: int = 0
) -> dict[str, float]:
    """metric(y, b) − metric(y, a) の点推定・SE・95% 区間。患者を復元抽出する。"""
    rng = np.random.default_rng(seed)
    n_pat = patients.max() + 1
    rows_of = pd.Series(np.arange(len(y))).groupby(patients).apply(np.asarray).to_numpy()
    deltas = []
    for _ in range(n_boot):
        pick = rng.integers(0, n_pat, n_pat)
        idx = np.concatenate(rows_of[pick])
        try:
            deltas.append(metric(y[idx], b[idx], idx) - metric(y[idx], a[idx], idx))
        except ValueError:
            continue
    deltas = np.asarray(deltas)
    point = metric(y, b, np.arange(len(y))) - metric(y, a, np.arange(len(y)))
    return {"delta": point, "se": deltas.std(), "lo": np.quantile(deltas, 0.025), "hi": np.quantile(deltas, 0.975)}


def auroc(y, s, _idx):
    return roc_auc_score(y, s)


def worst_group_auroc(group: np.ndarray):
    def metric(y, s, idx):
        g = group[idx]
        return min(roc_auc_score(y[g == k], s[g == k]) for k in np.unique(group))

    return metric


def subset_auroc(mask: np.ndarray):
    """mask の行だけで測る AUROC（陰性すべて＋陽性の一部、など）。"""

    def metric(y, s, idx):
        m = mask[idx]
        return roc_auc_score(y[m], s[m])

    return metric
