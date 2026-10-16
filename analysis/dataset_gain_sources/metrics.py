"""上がり幅の出どころを測る指標（データセットに依らない純粋な関数）。

- `cv_auc`：特徴を患者（病変）単位の cross-fit でロジスティック回帰にかけた AUROC。後付け（late fusion）の効果に使う
- `within_between`：AUROC を、属性の事前確率が近い組（セル内）と遠い組（セル間）の一致率に分ける
- `boot_delta`：seed 平均の ΔAUROC を、患者（病変）単位の対応のある bootstrap で区間にする
"""

from __future__ import annotations

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold


def standardize(score: np.ndarray) -> np.ndarray:
    """ロジットを平均 0・SD 1 にそろえる。seed や model でロジットの尺度が違うため。"""
    return (score - score.mean()) / score.std()


def cv_auc(x: np.ndarray, y: np.ndarray, groups: np.ndarray, folds: int = 5) -> float:
    """患者単位の cross-fit で学習した線形判別の AUROC。

    同じ患者の画像を学習側と評価側に分けないため、GroupKFold を使う。test の上で学習と評価を両方行うので、
    test の事前確率に合わせた楽観的な値になる。val で学習する版と並べて読む。

    Args:
        x: 説明変数（行 × 列）。
        y: 二値ラベル。
        groups: 患者（病変）の ID。
        folds: 分割数。

    Returns:
        float: out-of-fold の判別値の AUROC。
    """
    pred = np.zeros(len(y))
    for train, test in GroupKFold(folds).split(x, y, groups):
        model = LogisticRegression(max_iter=3000).fit(x[train], y[train])
        pred[test] = model.decision_function(x[test])
    return float(roc_auc_score(y, pred))


def within_between(score: np.ndarray, y: np.ndarray, cell: np.ndarray) -> tuple[float, float, float]:
    """AUROC を、同じセルの陽性・陰性の組と、違うセルの組の一致率に分ける。

    AUROC は陽性・陰性の全組の一致率なので、組をセル内とセル間に分けると、両者の組数で重み付けた平均が
    元の AUROC に戻る。属性の事前確率を足すだけ（後付け）の改善は、主にセル間の組を並べ替える。
    セル内の改善は、事前確率では説明できない（画像側の判定か、画像と属性の相互作用）。

    Args:
        score: 判別値。
        y: 二値ラベル。
        cell: 行ごとのセル番号。

    Returns:
        tuple: セル内の一致率、セル間の一致率、全組に占めるセル内の組の割合。
    """
    pos, neg = y == 1, y == 0
    all_pairs = pos.sum() * neg.sum()
    concordant = roc_auc_score(y, score) * all_pairs
    within_pairs = within_concordant = 0.0
    for c in np.unique(cell):
        member = cell == c
        n_pos, n_neg = (member & pos).sum(), (member & neg).sum()
        if n_pos and n_neg:
            within_pairs += n_pos * n_neg
            within_concordant += roc_auc_score(y[member], score[member]) * n_pos * n_neg
    between = (concordant - within_concordant) / (all_pairs - within_pairs)
    return within_concordant / within_pairs, between, within_pairs / all_pairs


def group_members(groups: np.ndarray) -> list[np.ndarray]:
    """患者（病変）ごとの行番号。bootstrap は患者単位で引き直す。"""
    _, inverse = np.unique(groups, return_inverse=True)
    order = np.argsort(inverse, kind="stable")
    bounds = np.cumsum(np.bincount(inverse))[:-1]
    return np.split(order, bounds)


def boot_delta(
    a_scores: list[np.ndarray], b_scores: list[np.ndarray], y: np.ndarray, groups: np.ndarray, n_boot: int = 500
) -> dict[str, float]:
    """seed 平均の AUROC の差（a − b）と、患者単位の対応のある bootstrap の 95% 区間。

    両 model を同じ引き直しで評価するので、test 標本の難しさの揺れは差から打ち消される。seed の揺れは
    seed 平均で薄めるだけで、区間には入らない。

    Args:
        a_scores: model a の seed ごとの判別値。
        b_scores: model b の seed ごとの判別値。
        y: 二値ラベル。
        groups: 患者（病変）の ID。
        n_boot: 引き直しの回数。

    Returns:
        dict: `delta`・`lo`・`hi`・`se`。
    """

    def mean_delta(index: np.ndarray) -> float:
        a = np.mean([roc_auc_score(y[index], s[index]) for s in a_scores])
        b = np.mean([roc_auc_score(y[index], s[index]) for s in b_scores])
        return a - b

    members = group_members(groups)
    rng = np.random.default_rng(0)
    draws = []
    for _ in range(n_boot):
        index = np.concatenate([members[i] for i in rng.integers(len(members), size=len(members))])
        if y[index].min() != y[index].max():
            draws.append(mean_delta(index))
    lo, hi = np.percentile(draws, [2.5, 97.5])
    return {"delta": mean_delta(np.arange(len(y))), "lo": lo, "hi": hi, "se": float(np.std(draws))}
