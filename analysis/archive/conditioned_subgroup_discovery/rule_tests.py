"""`delta_probe` の cache から、ΔW のクラスタ・その安定性・悪性率での説明・差し替えの効果を計算する。

notebook から呼ぶ集計の関数だけを置く。group の切り方の判断（K、stage、差し替え先の下限）は notebook が持つ。

- クラスタ：ΔW の距離（Gram から）、または差し替え先ごとの出力の残差の相関距離に Ward 法をかける。
- 説明の割合：ΔW を多変量の応答とみなし、説明変数の列空間への射影が占める分散の割合
  R² = tr(H G_c) / tr(G_c)（G_c は中心化した Gram、H は説明変数の射影）。
- セル内の一致率：属性がまったく同じ行の組（陽性 × 陰性）だけで数えた AUROC。
  セルの全員が同じ condition を受けるので、切片の違いではこの値は変わらない。
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import squareform

PACKAGE = Path(__file__).resolve().parent
CELL_KEYS = ["sex", "localization", "raw_age"]


def load_cells(run_id: str) -> tuple[pd.DataFrame, dict[str, np.ndarray]]:
    """セル表と stage ごとの Gram を読む。"""
    with np.load(PACKAGE / "cache" / f"{run_id}_cells.npz") as cached:
        table = pd.DataFrame(
            {key.removeprefix("cell_"): cached[key] for key in cached.files if key.startswith("cell_")}
        )
        grams = {key.removeprefix("gram_"): cached[key] for key in cached.files if key.startswith("gram_")}
    return table, grams


def load_swap(run_id: str, split: str) -> dict[str, np.ndarray]:
    """差し替え推論の cache を読む。"""
    with np.load(PACKAGE / "cache" / f"{run_id}_swap_{split}.npz") as cached:
        out = {key: cached[key] for key in cached.files if key != "metadata"}
        out["metadata"] = json.loads(str(cached["metadata"]))
    return out


def distance(gram: np.ndarray) -> np.ndarray:
    """Gram から ΔW の距離行列を作る（数値誤差の負値は 0 に丸める）。"""
    diagonal = np.diag(gram)
    squared = diagonal[:, None] + diagonal[None, :] - 2 * gram
    return np.sqrt(np.clip(squared, 0, None))


def cluster_ward(dist: np.ndarray, k: int) -> np.ndarray:
    """Ward 法の階層クラスタリングで k 個に切ったラベル（0 始まり）。

    average linkage は件数の少ない外れたセルを 1 つずつ切り出し、大きな 1 群が残るので使わない。
    ΔW の距離はユークリッド距離なので Ward 法の前提を満たす。
    """
    tree = linkage(squareform(dist, checks=False), method="ward")
    return fcluster(tree, t=k, criterion="maxclust") - 1


def centered(gram: np.ndarray, weights: np.ndarray | None = None) -> np.ndarray:
    """重み付き平均を引いた Gram（ΔW − 平均 ΔW の内積）を返す。"""
    n = len(gram)
    w = np.full(n, 1 / n) if weights is None else weights / weights.sum()
    projector = np.eye(n) - np.outer(np.ones(n), w)
    return projector @ gram @ projector.T


def explained_share(gram: np.ndarray, predictors: np.ndarray) -> float:
    """ΔW の（セル等重みの）分散のうち、説明変数の線形結合で表せる割合。

    Args:
        gram: `[n, n]` の Gram。
        predictors: `[n, p]` の説明変数（定数列は不要。中で中心化する）。

    Returns:
        float: tr(H G_c) / tr(G_c)。
    """
    g = centered(gram)
    x = predictors - predictors.mean(0)
    hat = x @ np.linalg.pinv(x)
    return float(np.trace(hat @ g) / np.trace(g))


def shrunk_logit_rate(table: pd.DataFrame, pseudo: float = 20.0) -> np.ndarray:
    """セルの悪性率を、性別・部位・年齢の加法モデルへ pseudo 件分だけ縮めた logit。

    件数の少ないセルの素の率は 0 か 1 に張り付くので、そのまま「セルの事前確率」には使えない。
    縮める先は train 全体の率ではなく加法モデルにして、属性の主効果は保つ。
    """
    from sklearn.linear_model import LogisticRegression

    design = attribute_design(table)
    repeated = np.repeat(np.arange(len(table)), 2)
    labels = np.tile([1, 0], len(table))
    weights = np.stack([table["positives"], table["count"] - table["positives"]], 1).ravel()
    keep = weights > 0
    model = LogisticRegression(C=1.0, max_iter=2000)
    model.fit(design[repeated[keep]], labels[keep], sample_weight=weights[keep])
    smooth = model.predict_proba(design)[:, 1]
    rate = (table["positives"] + pseudo * smooth) / (table["count"] + pseudo)
    return np.log(rate / (1 - rate)).to_numpy()


def attribute_design(table: pd.DataFrame, interactions: bool = False) -> np.ndarray:
    """性別・部位（one-hot）・年齢（3 次まで）の加法の説明変数。interactions で性別 × 年齢を足す。"""
    age = (table["raw_age"].to_numpy() - 50) / 20
    columns = [table["sex"].to_numpy()[:, None], age[:, None], age[:, None] ** 2, age[:, None] ** 3]
    columns.append(pd.get_dummies(table["localization"]).to_numpy(dtype=float)[:, 1:])
    if interactions:
        columns.append((table["sex"].to_numpy() * age)[:, None])
    return np.concatenate(columns, 1).astype(float)


def cell_index(swap: dict[str, np.ndarray]) -> np.ndarray:
    """評価行ごとに、属性が同じ行をまとめたセル番号を返す。"""
    keys = pd.DataFrame({key: swap[key] for key in CELL_KEYS})
    return keys.groupby(CELL_KEYS, sort=False).ngroup().to_numpy()


def within_cell_counts(margins: np.ndarray, cells: np.ndarray, target: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """セルごとの一致数と組数。

    Args:
        margins: `[T, n]` の logit 差（T 通りの condition）。
        cells: `[n]` のセル番号。
        target: `[n]` の 0/1。

    Returns:
        tuple: 一致数 `[n_cells, T]`（同点は 0.5）、組数 `[n_cells]`。
    """
    n_cells = cells.max() + 1
    concordant = np.zeros((n_cells, len(margins)))
    pairs = np.zeros(n_cells)
    for cell in range(n_cells):
        rows = cells == cell
        positive = margins[:, rows & (target == 1)]
        negative = margins[:, rows & (target == 0)]
        if positive.shape[1] == 0 or negative.shape[1] == 0:
            continue
        difference = positive[:, :, None] - negative[:, None, :]
        concordant[cell] = (difference > 0).sum((1, 2)) + 0.5 * (difference == 0).sum((1, 2))
        pairs[cell] = positive.shape[1] * negative.shape[1]
    return concordant, pairs


def cell_keys_of(swap: dict[str, np.ndarray], cells: np.ndarray) -> pd.DataFrame:
    """評価のセル番号 → (sex, localization, raw_age) の表。"""
    keys = pd.DataFrame({key: swap[key] for key in CELL_KEYS}).assign(cell=cells)
    return keys.drop_duplicates("cell").set_index("cell").sort_index()


def block_auroc(concordant, pairs, source_labels, target_labels, k) -> np.ndarray:
    """源のクラスタ × 差し替え先のクラスタごとの、セル内 AUROC（差し替え先のセルは等重みで平均）。

    Args:
        concordant: `[n_cells, T]`。
        pairs: `[n_cells]`。
        source_labels: 評価セルのクラスタ（`[n_cells]`、−1 は除外）。
        target_labels: 差し替え先セルのクラスタ（`[T]`）。
        k: クラスタ数。

    Returns:
        ndarray: `[k, k]`。組の無いブロックは NaN。
    """
    out = np.full((k, k), np.nan)
    for a in range(k):
        rows = source_labels == a
        total = pairs[rows].sum()
        if total == 0:
            continue
        auroc_per_target = concordant[rows].sum(0) / total
        for b in range(k):
            if (target_labels == b).any():
                out[a, b] = auroc_per_target[target_labels == b].mean()
    return out


def interaction(block: np.ndarray) -> float:
    """ブロック表の二重中心化の対角の平均。

    行（源）の主効果は「そのクラスタのセルの解きやすさ」、列（差し替え先）の主効果は
    「その ΔW の全般的な良さ」。対角の交互作用だけが「自分のクラスタの ΔW が自分にだけ合う」＝規則の違いを表す。
    """
    double = block - np.nanmean(block, 1, keepdims=True) - np.nanmean(block, 0, keepdims=True) + np.nanmean(block)
    return float(np.nanmean(np.diag(double)))


def two_way_centered(values: np.ndarray, row_weights: np.ndarray) -> np.ndarray:
    """`[S, T]` の表から、行（源のセル）と列（差し替え先のセル）の主効果を除く。

    行の重みは組数、列は等重み。完全な表なので、重み付きの二重中心化が二元固定効果の射影と一致する。
    """
    w = row_weights / row_weights.sum()
    row_mean = values.mean(1, keepdims=True)
    column_mean = (w[:, None] * values).sum(0, keepdims=True)
    grand = (w[:, None] * values).sum() / values.shape[1]
    return values - row_mean - column_mean + grand


def distance_slope(auroc: np.ndarray, pairs: np.ndarray, dist: np.ndarray) -> float:
    """主効果を除いたセル内 AUROC を、源と差し替え先の ΔW 距離（SD 単位）に回帰した傾き。

    負なら「ΔW が遠いセルのものほど、そのセル自身の判別を損なう」＝セルごとに規則が違う。
    差し替え先の ΔW の全般的な良し悪し（列の主効果）と、セルの解きやすさ（行の主効果）は除いてある。

    Args:
        auroc: `[S, T]` のセル内 AUROC（源のセル × 差し替え先）。
        pairs: `[S]` の組数（行の重み）。
        dist: `[S, T]` の ΔW 距離。
    """
    y = two_way_centered(auroc, pairs)
    x = two_way_centered(dist / dist.std(), pairs)
    w = pairs[:, None] * np.ones_like(x)
    return float((w * x * y).sum() / (w * x * x).sum())


def response_residual(swapped: np.ndarray) -> np.ndarray:
    """差し替え先の condition ごとの出力 `[T, n]` から、条件の切片と画像の主効果を除いた残差。

    残るのは「その条件が、どの画像を他の条件より高く・低く採点するか」で、条件ごとの規則の違いにあたる。
    ラベルは使わない。
    """
    response = swapped.astype(float)
    response = response - response.mean(1, keepdims=True)
    return response - response.mean(0, keepdims=True)


def response_clusters(swapped: np.ndarray, k: int) -> np.ndarray:
    """条件（差し替え先のセル）を、出力の残差の相関距離で Ward 法により k 個に分ける。"""
    correlation = np.corrcoef(response_residual(swapped))
    dist = np.sqrt(np.clip(2 - 2 * correlation, 0, None))
    np.fill_diagonal(dist, 0)
    return cluster_ward(dist, k)


def source_labels(
    table: pd.DataFrame, swap: dict[str, np.ndarray], cells: np.ndarray, labels: np.ndarray
) -> np.ndarray:
    """評価のセル番号ごとに、セル表の上で付けたラベルを引く。セル表に無いセルは −1。

    Args:
        table: `load_cells` のセル表。
        swap: `load_swap` の出力。
        cells: `cell_index(swap)`。
        labels: セル表の行ごとのラベル（`[len(table)]`、付けないセルは −1）。
    """
    index = {tuple(row): i for i, row in enumerate(table[CELL_KEYS].to_numpy())}
    keys = cell_keys_of(swap, cells)
    rows = np.array([index.get(tuple(row), -1) for row in keys.to_numpy()])
    return np.where(rows >= 0, labels[np.maximum(rows, 0)], -1)


def interaction_test(swap, table, labels, k, n_boot=300, seed=0) -> dict[str, float]:
    """セル表の上のラベルで、差し替えの交互作用（自分の群の ΔW が自分にだけ合う度合い）と区間を返す。

    区間は評価のセル単位の bootstrap。差し替え先（列）は固定する。

    Args:
        swap: `load_swap` の出力。
        table: `load_cells` のセル表。
        labels: セル表の行ごとのラベル（0..k−1、付けないセルは −1）。
        k: 群の数。
    """
    cells = cell_index(swap)
    concordant, pairs = within_cell_counts(swap["swapped"], cells, swap["target"])
    source = source_labels(table, swap, cells, labels)
    keep = np.flatnonzero((source >= 0) & (pairs > 0))
    target = labels[swap["target_cells"]]
    point = interaction(block_auroc(concordant[keep], pairs[keep], source[keep], target, k))
    rng = np.random.default_rng(seed)
    boots = []
    for _ in range(n_boot):
        draw = rng.choice(keep, len(keep))
        boots.append(interaction(block_auroc(concordant[draw], pairs[draw], source[draw], target, k)))
    low, high = np.nanpercentile(boots, [2.5, 97.5])
    return {"interaction": point, "low": low, "high": high, "source_cells": len(keep)}


def planted_minority(table: pd.DataFrame, variant: str) -> np.ndarray:
    """セル表の各セルが、植え込みの少数側 B に入るか（`attribute_rule_planting.minority` と同じ規則）。"""
    from analysis.archive.conditioned_subgroup_discovery.attribute_rule_planting import minority

    frame = table.rename(columns={"raw_age": "age"}).assign(sex_missing=0, localization_missing=0, age_missing=0)
    return minority(frame, variant)


def marker_effect(swap: dict[str, np.ndarray], variant: str) -> np.ndarray:
    """差し替え先の condition ごとの、マーカーの効果。

    付いた画像 − 付かない画像の logit 差の平均を、クラス内で取って平均する。

    植え込みの答えを使う診断で、発見の手順には入れない。モデルが B の規則をどこまで学んだかを見る。
    """
    from analysis.archive.conditioned_subgroup_discovery.attribute_rule_planting import marker_mask

    frame = pd.DataFrame(
        {
            "image": swap["image"],
            "target": swap["target"],
            "sex": swap["sex"],
            "localization": swap["localization"],
            "age": swap["raw_age"],
        }
    ).assign(sex_missing=0, localization_missing=0, age_missing=0)
    marked = marker_mask(frame, variant)
    effects = []
    for label in (0, 1):
        rows = swap["target"] == label
        effects.append(swap["swapped"][:, rows & marked].mean(1) - swap["swapped"][:, rows & ~marked].mean(1))
    return np.mean(effects, 0)
