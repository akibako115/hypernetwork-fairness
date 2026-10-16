"""年齢公平性の条件付き指標・EO mixture・患者単位比較を検算する。"""

from collections.abc import Callable, Mapping, Sequence
from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.optimize import linprog
from sklearn.metrics import roc_auc_score


def age_metrics(
    scores: ArrayLike,
    target: ArrayLike,
    groups: ArrayLike,
    positive_probability: ArrayLike | None = None,
    *,
    include_auc: bool = True,
) -> dict[str, Any]:
    """年齢2群の AUC と固定閾値の条件付き率を計算する。

    Args:
        scores: z1-z0 の順位スコア。ensemble は平均スコアを明示して渡す。
        target: No Finding=1 の二値ラベル。
        groups: raw age<65=0、65以上=1。
        positive_probability: ランダム化 classifier の陽性予測確率。未指定なら s>0。
        include_auc: False はrateだけを求め、AUCの計算を省く。

    Returns:
        dict: 全体・4ペア AUC と2群の TPR/TNR。
    """
    scores, y, g = np.asarray(scores), np.asarray(target), np.asarray(groups)
    probability = (scores > 0).astype(float) if positive_probability is None else np.asarray(positive_probability)
    if scores.ndim != 1 or not (scores.shape == y.shape == g.shape == probability.shape):
        raise ValueError("同形状の1次元入力が必要")
    if (
        not np.isfinite(scores).all()
        or not np.isfinite(probability).all()
        or (probability < 0).any()
        or (probability > 1).any()
    ):
        raise ValueError("有限なスコアと [0,1] の予測確率が必要")
    if not np.isin(y, [0, 1]).all() or not np.isin(g, [0, 1]).all():
        raise ValueError("ラベル・年齢群は二値")
    if any(not ((g == group) & (y == c)).any() for group in range(2) for c in range(2)):
        raise ValueError("全4セルが必要")
    tpr = [float(probability[(g == group) & (y == 1)].mean()) for group in range(2)]
    tnr = [float(1 - probability[(g == group) & (y == 0)].mean()) for group in range(2)]
    result: dict[str, Any] = dict(tpr=tpr, tnr=tnr, tpr_gap=max(tpr) - min(tpr), fpr_gap=max(tnr) - min(tnr))
    if include_auc:
        pair_auc = []
        for positive_group in range(2):
            for negative_group in range(2):
                mask = ((g == positive_group) & (y == 1)) | ((g == negative_group) & (y == 0))
                pair_auc.append(float(roc_auc_score(y[mask], scores[mask])))
        result.update(
            auroc=float(roc_auc_score(y, scores)),
            pair_auroc=pair_auc,
            worst_age_auroc=min(pair_auc[0], pair_auc[3]),
            min_pair_auroc=min(pair_auc),
        )
    return result


def eo_mixture(
    candidates: Sequence[Mapping[str, Any]], parent: Mapping[str, Any], epsilon: float = 0.1
) -> dict[str, Any]:
    """validation のみで epoch 分布を LP 選択する。

    Args:
        candidates: validation 指標、weighted_ce、checkpoint、epoch を持つ辞書列。
        parent: 親の validation 指標。
        epsilon: 両クラス条件付き率の群差の上限。

    Returns:
        dict: status、checkpoint ごとの確率。成立しない場合は infeasible。
    """
    if not 0 <= epsilon <= 1:
        raise ValueError("epsilon は [0,1]")
    finite = [
        row
        for row in candidates
        if np.isfinite([row["auroc"], row["worst_age_auroc"], row["weighted_ce"], *row["tpr"], *row["tnr"]]).all()
    ]
    rows = [
        row
        for row in finite
        if row["auroc"] >= parent["auroc"] - 0.005 and row["worst_age_auroc"] >= parent["worst_age_auroc"] - 0.005
    ]
    if not rows:
        return dict(status="infeasible", components=[])
    rates = np.array([row["tpr"] + row["tnr"] for row in rows]).T
    differences = np.stack([rates[0] - rates[1], rates[1] - rates[0], rates[2] - rates[3], rates[3] - rates[2]])
    matrix = np.vstack([differences, -rates])
    rhs = np.concatenate([np.full(4, epsilon), -np.array(parent["tpr"] + parent["tnr"]) + 0.02])
    result = linprog(
        [row["weighted_ce"] for row in rows],
        A_ub=matrix,
        b_ub=rhs,
        A_eq=np.ones((1, len(rows))),
        b_eq=[1.0],
        bounds=(0, None),
        method="highs",
        options={"primal_feasibility_tolerance": 1e-9, "dual_feasibility_tolerance": 1e-9},
    )
    if not result.success:
        return dict(status="infeasible", components=[], solver_message=result.message)
    residual = float(np.max(matrix @ result.x - rhs))
    if residual > 1e-8 or abs(result.x.sum() - 1) > 1e-8:
        raise RuntimeError("LP solution が制約を満たさない")
    return dict(
        status="selected",
        components=[
            dict(checkpoint=row["checkpoint"], epoch=row["epoch"], probability=float(p))
            for row, p in zip(rows, result.x, strict=True)
            if p > 1e-12
        ],
        expected_rates=(rates @ result.x).tolist(),
        weighted_ce=float(result.fun),
        maximum_violation=residual,
        solver_feasibility_tolerance=1e-9,
    )


def randomized_positive_probability(scores_by_checkpoint: ArrayLike, weights: ArrayLike) -> NDArray:
    """checkpoint 抽選 classifier の陽性予測確率を返す。

    Args:
        scores_by_checkpoint: [K,N] の順位スコア。
        weights: validation で確定した simplex 分布 [K]。

    Returns:
        ndarray: [N] の陽性予測確率。平均 logits の判定とは異なる。
    """
    scores, weights = np.asarray(scores_by_checkpoint), np.asarray(weights)
    if (
        scores.ndim != 2
        or weights.shape != (len(scores),)
        or not np.isfinite(scores).all()
        or not np.isfinite(weights).all()
        or (weights < 0).any()
        or not np.isclose(weights.sum(), 1)
    ):
        raise ValueError("有限な [K,N] scores と simplex [K] weights が必要")
    return weights @ (scores > 0)


def paired_patient_bootstrap(
    patient_ids: ArrayLike, metric_difference: Callable[[NDArray], float], repetitions: int = 2000, seed: int = 42
) -> dict[str, Any]:
    """同じ患者の全画像をまとめて再抽出し、差の信頼区間を求める。

    Args:
        patient_ids: 画像行に対応する患者 ID。
        metric_difference: 抽出 index を受け取り、同一患者の比較差を返す関数。
        repetitions: 復元抽出回数。
        seed: 再現性のための乱数 seed。

    Returns:
        dict: 点推定、95%区間、有限な bootstrap の回数。
    """
    patient_ids = np.asarray(patient_ids)
    indices = _patient_index_groups(patient_ids)
    if repetitions < 1:
        raise ValueError("正の repetitions が必要")
    rng = np.random.default_rng(seed)
    samples = []
    for _ in range(repetitions):
        chosen = rng.integers(len(indices), size=len(indices))
        value = metric_difference(np.concatenate([indices[i] for i in chosen]))
        if np.isfinite(value):
            samples.append(value)
    if len(samples) < repetitions * 0.95:
        raise ValueError("bootstrap で評価不能な標本が5%を超えた")
    return dict(
        estimate=float(metric_difference(np.arange(len(patient_ids)))),
        confidence_interval=np.quantile(samples, [0.025, 0.975]).tolist(),
        valid_repetitions=len(samples),
    )


def _patient_index_groups(patient_ids: ArrayLike) -> list[NDArray]:
    ids = np.asarray(patient_ids)
    if ids.ndim != 1 or not len(ids):
        raise ValueError("非空の1次元患者IDが必要")
    _, inverse, counts = np.unique(ids, return_inverse=True, return_counts=True)
    order = np.argsort(inverse, kind="stable")
    return list(np.split(order, np.cumsum(counts)[:-1]))


def _seed_metric_difference(
    candidate: Mapping[str, Any],
    reference: Mapping[str, Any],
    labels: Mapping[str, ArrayLike],
    metrics: Sequence[str],
) -> Callable[[NDArray], NDArray]:
    if candidate["seeds"] != reference["seeds"] or not len(candidate["seeds"]):
        raise ValueError("同じ順の非空seed集合が必要")
    allowed = {"auroc", "worst_age_auroc", "min_pair_auroc", "tpr_gap", "fpr_gap"}
    if not metrics or len(set(metrics)) != len(metrics) or not set(metrics) <= allowed:
        raise ValueError("重複のないscalar指標名が必要")
    target, groups = np.asarray(labels["target"]), np.asarray(labels["groups"])
    expected = (len(candidate["seeds"]), len(target))
    prepared = []
    for condition in (candidate, reference):
        scores = np.asarray(condition["scores"])
        probability = condition.get("positive_probability")
        probability = None if probability is None else np.asarray(probability)
        if scores.shape != expected or (probability is not None and probability.shape != expected):
            raise ValueError("scores/probability は同じ [S,N] である必要がある")
        prepared.append((scores, probability))
    include_auc = bool(set(metrics) - {"tpr_gap", "fpr_gap"})

    def difference(index: NDArray) -> NDArray:
        totals = np.zeros(len(metrics))
        for i in range(expected[0]):
            values = []
            for scores, probability in prepared:
                row = age_metrics(
                    scores[i, index],
                    target[index],
                    groups[index],
                    None if probability is None else probability[i, index],
                    include_auc=include_auc,
                )
                values.append(np.array([row[metric] for metric in metrics]))
            totals += values[0] - values[1]
        return totals / expected[0]

    return difference


def bootstrap_seed_differences(
    candidate: Mapping[str, Any],
    reference: Mapping[str, Any],
    labels: Mapping[str, ArrayLike],
    metrics: Sequence[str],
    repetitions: int = 2000,
    seed: int = 42,
) -> dict[str, dict[str, Any]]:
    """同じ患者再抽出から全指標のseed平均差をまとめて求める。

    Args:
        candidate: seeds、[S,N] scores、任意のpositive_probabilityを持つ辞書。
        reference: 同じseed・画像順の比較対象。
        labels: target、groups、patient_idsの辞書。
        metrics: scalar指標名。rateだけならAUCを計算しない。
        repetitions: 患者の復元抽出回数。
        seed: 全指標で共有するbootstrap seed。

    Returns:
        dict: 指標名ごとの点推定、患者単位95%区間、有効反復数。
    """
    if repetitions < 1:
        raise ValueError("正の repetitions が必要")
    difference = _seed_metric_difference(candidate, reference, labels, metrics)
    patients = np.asarray(labels["patient_ids"])
    if len(patients) != len(np.asarray(labels["target"])):
        raise ValueError("patient_ids と target の行数が一致しない")
    groups = _patient_index_groups(patients)
    rng = np.random.default_rng(seed)
    samples = []
    # 元データの不正をbootstrapの欠損標本として隠さない。
    estimate = difference(np.arange(len(patients)))
    for _ in range(repetitions):
        chosen = rng.integers(len(groups), size=len(groups))
        index = np.concatenate([groups[i] for i in chosen])
        try:
            value = difference(index)
        except ValueError:
            continue
        if np.isfinite(value).all():
            samples.append(value)
    if len(samples) < repetitions * 0.95:
        raise ValueError("bootstrap で評価不能な標本が5%を超えた")
    intervals = np.quantile(np.asarray(samples), [0.025, 0.975], axis=0)
    return {
        metric: dict(
            estimate=float(estimate[i]), confidence_interval=intervals[:, i].tolist(), valid_repetitions=len(samples)
        )
        for i, metric in enumerate(metrics)
    }


def bootstrap_seed_difference(
    candidate: Mapping[str, Any],
    reference: Mapping[str, Any],
    labels: Mapping[str, ArrayLike],
    metric: str,
    repetitions: int = 2000,
    seed: int = 42,
) -> dict[str, Any]:
    """単一指標の呼び出し互換を保ち、共通のbootstrap処理へ委譲する。

    Args:
        candidate: seeds、[S,N] scores、任意のpositive_probabilityを持つ辞書。
        reference: 同じseed・画像順の比較対象。
        labels: target、groups、patient_idsの辞書。
        metric: age_metricsのscalar指標名。
        repetitions: 患者の復元抽出回数。
        seed: bootstrap seed。

    Returns:
        dict: seed平均差の点推定と患者単位95%区間。
    """
    return bootstrap_seed_differences(candidate, reference, labels, [metric], repetitions, seed)[metric]
