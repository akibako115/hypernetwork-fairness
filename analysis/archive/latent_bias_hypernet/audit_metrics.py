"""Q1 の監査：属性セルと潜在スライスとで、どの公平性基準の違反が見えるかを測る（CheXpert・ISIC）。

同じ 1 段目の test 予測を、属性セル・潜在スライス・対照（ランダム、スコア分位）で切り、族ごとに次を求める。
指標は医用画像の公平性研究で標準の順位・しきい値の指標を主にし、較正は別の基準（sufficiency）として並べる。
群ごとの較正と TPR・FPR の均等は、群の有病率が違えば一般に両立しない（Kleinberg et al. 2016）。

- 順位：最悪群 AUROC と、群間の AUROC 差（最大 − 最小）
- しきい値（val で決めた Youden しきい値）：Eopp1（TPR 差）、Eopp0（TNR 差）、Eodds（(TPR 差 + FPR 差) / 2）。
  定義は projects/hypernet_iterative/utils/metrics.py の 2 値の場合と同じ。
- 較正：log O/E の SD（較正が合っている帰無仮説の分を差し引いたもの。audit_run.py・isic_discovery.py と同じ）

潜在の切り方は、ラベルを使わない GMM（LHCF 型）とラベルの残差を使う監査器を、同じ特徴空間の組でそろえる。
1 段目自身の表現 h(x)、医用の基盤モデル（CheXpert は RAD-DINO）、汎用の基盤モデル（DINOv2）、h と基盤モデルの連結。
どの表現に偏りが乗っているかを、2 つの切り方で別々に読むためである。

最悪群や差は群が小さいほど標本誤差で悪く出るため、どれも「群の大きさを保ったまま群を入れ替えた」帰無分布と比べる。
入れ替えは患者（CheXpert）・病変（ISIC）単位で行い、同じ単位の画像は同じ群に置く。帰無分布は 2 種類作る。

- 無層別（`*_null`・`*_p`）：単位の間で自由に入れ替える。
- スコア層別（`*_snull`・`*_sp`）：単位の平均スコアの 10 分位の中だけで入れ替える。各群のスコアの分布を保つので、
  スコアの近い画像が集まった群で AUROC が機械的に下がり、TPR が 0 か 1 に張り付く分を差し引ける。
  1 段目の表現 h で切った群はスコアで分かれやすく、無層別の帰無分布では順位・しきい値の偏りが過大に出る。

入力: 1 段目のスコア・特徴のキャッシュ（`data.load`）
出力: `results/audit_metrics.csv`（族ごとに 1 行。`*_null`・`*_snull` は帰無分布の平均、`*_p`・`*_sp` は帰無分布で同じか
より悪い割合）

    uv run python -m analysis.archive.latent_bias_hypernet.audit_metrics
"""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

from analysis.archive.latent_bias_hypernet import audit, isic_discovery
from analysis.archive.latent_bias_hypernet.audit_corrected import logit
from analysis.archive.latent_bias_hypernet.audit_run import slice_families
from analysis.archive.latent_bias_hypernet.data import PACKAGE, load
from analysis.archive.latent_bias_hypernet.planting import hidden_subgroup

DRAWS = 200
SCORE_STRATA = 10


def stats(table: pd.DataFrame) -> dict[str, float]:
    t = table.dropna(subset=["auroc", "tpr"])
    tpr_gap = float(t.tpr.max() - t.tpr.min())
    fpr_gap = float(t.fpr.max() - t.fpr.min())
    return {
        "worst_auroc": float(t.auroc.min()),
        "auroc_gap": float(t.auroc.max() - t.auroc.min()),
        "eopp1": tpr_gap,
        "eopp0": fpr_gap,  # TNR 差は FPR 差と同じ値になる
        "eodds": (tpr_gap + fpr_gap) / 2,
    }


def unit_strata(p: np.ndarray, cluster: np.ndarray) -> np.ndarray:
    """単位（患者・病変）ごとの平均スコアの分位番号。単位は np.unique(cluster) の順。"""
    _, inv = np.unique(cluster, return_inverse=True)
    mean = np.bincount(inv, weights=p) / np.bincount(inv)
    edges = np.quantile(mean, np.linspace(0, 1, SCORE_STRATA + 1)[1:-1])
    return np.digitize(mean, edges)


def block_shuffle(
    group: np.ndarray, cluster: np.ndarray, rng: np.random.Generator, strata: np.ndarray | None = None
) -> np.ndarray:
    """単位ごとに最初の画像の群を持たせ、単位の間で群を入れ替える。strata を渡すと同じ層の中だけで入れ替える。"""
    _, first, inv = np.unique(cluster, return_index=True, return_inverse=True)
    unit_group = group[first].copy()
    layers = np.zeros(len(first), dtype=int) if strata is None else strata
    for layer in np.unique(layers):
        idx = np.flatnonzero(layers == layer)
        unit_group[idx] = unit_group[rng.permutation(idx)]
    return unit_group[inv]


def null_stats(g, y, p, threshold, cluster, limits, strata) -> pd.DataFrame:
    """群を入れ替えた帰無分布の指標（DRAWS 回）。"""
    rng = np.random.default_rng(0)
    draws = []
    for _ in range(DRAWS):
        shuffled = block_shuffle(g, cluster, rng, strata)
        draws.append(stats(audit.group_metrics(shuffled, y, p, threshold, *limits)))
    return pd.DataFrame(draws)


def measure(name, g, y, p, threshold, cluster, min_n, min_exp) -> dict:
    keep = g >= 0
    g, y, p, cluster = g[keep], y[keep], p[keep], cluster[keep]
    table = audit.group_metrics(g, y, p, threshold, min_n, min_exp)
    d = audit.dispersion(table, g, y, p, cluster)
    row = {"family": name, "groups": len(table), "logoe_excess_sd": d["excess_sd"], "logoe_above_null": d["above_null"]}
    obs = stats(table)
    null = null_stats(g, y, p, threshold, cluster, (min_n, min_exp), None)
    snull = null_stats(g, y, p, threshold, cluster, (min_n, min_exp), unit_strata(p, cluster))
    for k, v in obs.items():
        row[k] = v
        for tag, dist in (("", null), ("s", snull)):
            worse = dist[k] <= v if k == "worst_auroc" else dist[k] >= v
            row |= {f"{k}_{tag}null": float(dist[k].mean()), f"{k}_{tag}p": float(worse.mean())}
    return row


def block_pcs(fit_blocks: list[np.ndarray], blocks: list[list[np.ndarray]], dims: int) -> list[np.ndarray]:
    """表現ごとに主成分 dims 次元へ落とし、表現ごとの全体 SD で割って連結する。

    h（2048 次元）と基盤モデルの埋め込みをそのまま連結して主成分を取ると、次元と分散の大きい側だけで群が決まる。
    表現ごとに同じ次元数・同じ尺度にしてから並べ、どちらの表現も群の決め方に効くようにする。

    Args:
        fit_blocks: 主成分と尺度を決める split の、表現ごとの行列。
        blocks: 変換する split ごとに、fit_blocks と同じ順の表現の行列。
        dims: 表現 1 つあたりの主成分の次元数。

    Returns:
        blocks の split ごとの、連結した行列。
    """
    parts: list[list[np.ndarray]] = [[] for _ in blocks]
    for i, ref in enumerate(fit_blocks):
        pca = PCA(dims, random_state=0).fit(ref)
        scale = pca.transform(ref).std()
        for j, split_blocks in enumerate(blocks):
            parts[j].append(pca.transform(split_blocks[i]) / scale)
    return [np.concatenate(p, axis=1) for p in parts]


def latent_families(views: dict[str, tuple], K: int, gmm_dims: int, auditor_args: tuple) -> dict:
    """特徴空間ごとに、GMM（ラベル不使用）と監査器（val の残差、3 seed）の群番号を test に振る。

    Args:
        views: 特徴空間の名前 → (GMM を当てる split, 監査器を学習する val, test) の行列。
        K: 群の数。監査器は K 分位で切る。
        gmm_dims: GMM の前の主成分の次元数。
        auditor_args: (val の較正後 logit, val のラベル, val の患者・病変 id)。

    Returns:
        族の名前 → test の群番号。
    """
    offset_v, y_v, units_v = auditor_args
    fam = {}
    for name, (x_fit, x_v, x_t) in views.items():
        fam[f"gmm{K}_{name}"] = audit.GMMSlicer(x_fit, K, dims=min(gmm_dims, x_fit.shape[1]))(x_t)
        for seed in (0, 1, 2):
            r = audit.ResidualAuditor(x_v, offset_v, y_v, units_v, seed)(x_t)
            fam[f"auditor{K}_{name}_s{seed}"] = audit.quantile_slicer(r, K)(r)
        print(f"latent {name}", flush=True)
    return fam


def chexpert_views(stage1: str) -> dict[str, tuple]:
    """CheXpert の特徴空間。GMM は train_b（1 段目が見ていない split）に当てる。"""
    S = {split: load(stage1, split, "raddino") for split in ("train_b", "val", "test")}
    dino = {split: load(stage1, split, "dinov2").route for split in S}
    h = {split: S[split].features for split in S}
    rad = {split: S[split].route for split in S}
    order = ("train_b", "val", "test")
    joint = block_pcs([h["train_b"], rad["train_b"]], [[h[k], rad[k]] for k in order], dims=32)
    pcs = {}
    for name, x in (("h", h), ("raddino", rad), ("dinov2", dino)):
        pca = PCA(64, random_state=0).fit(x["train_b"])
        pcs[name] = tuple(pca.transform(x[k]) for k in order)
    return pcs | {"h+raddino": tuple(joint)}


def confounder_families(context: pd.DataFrame) -> dict[str, np.ndarray]:
    """交絡の候補：撮影文脈（1 族）と、画像に写る部分集団（要因ごとに有無の 2 群）。"""
    fam = {"imaging_context": audit.imaging_context_cells(context)}
    for rule in ("pacer", "picc", "central_line", "sternotomy", "surgical_clip"):
        fam[f"sub_{rule}"] = hidden_subgroup(context, rule).astype(int)
    return fam


def chexpert(stage1: str) -> list[dict]:
    D = load(stage1, "train_b", "raddino")
    T = load(stage1, "test", "raddino")
    V = load(stage1, "val", "raddino")
    cal = audit.Calibrator(V.base, V.target)
    p_d, p_t, p_v = cal(D.base), cal(T.base), cal(V.base)
    fam = slice_families(argparse.Namespace(K=[32], field=[]), D, T, p_d, D.target - p_d)
    # 属性セルは年齢を 65 歳で 2 区分にした 2 本に絞る（細かいセルは評価の下限で少数群が落ちる）。
    # GMM と監査器は latent_families で特徴空間をそろえて作るので、slice_families の版は使わない。
    fam = {k: fam[k] for k in ("demo8", "random32", "score32", "tree32")}
    fam["demo16"] = audit.demo_cells(T.context, "demo16")
    fam |= confounder_families(T.context)
    patients_v = V.context.image.str.split("/").str[1].to_numpy()
    fam |= latent_families(chexpert_views(stage1), 32, 64, (logit(p_v), V.target, patients_v))
    patients_t = T.context.image.str.split("/").str[1].to_numpy()
    rows = []
    for name, g in fam.items():
        rows.append({"dataset": "chexpert", **measure(name, g, T.target, p_t, cal.threshold, patients_t, 150, 15.0)})
        print(rows[-1]["family"], flush=True)
    return rows


def isic_views() -> dict[str, tuple]:
    """ISIC の特徴空間。GMM は監査セットの母集団（`isic_audit`）に当てる。DINOv2 の GMM は isic_discovery にある。"""
    S = {split: load(isic_discovery.STAGE1, split, "dinov2") for split in ("isic_audit", "isic_audit_v", "isic_test")}
    order = tuple(S)
    h_pca = PCA(32, random_state=0).fit(S["isic_audit"].features)
    joint = block_pcs(
        [S["isic_audit"].features, S["isic_audit"].route], [[S[k].features, S[k].route] for k in order], dims=16
    )
    dino_pca = PCA(32, random_state=0).fit(S["isic_audit"].route)
    return {
        "h": tuple(h_pca.transform(S[k].features) for k in order),
        "dinov2": tuple(dino_pca.transform(S[k].route) for k in order),
        "h+dinov2": tuple(joint),
    }


def isic() -> list[dict]:
    fam, T, p_t, threshold, lesions = isic_discovery.families()
    # CheXpert と同じく年齢は 2 区分だけにし、GMM・監査器は latent_families の版にそろえる
    keep = ("性別", "年齢（65 歳で 2 区分）", "性別 × 年齢（4）", "部位", "ランダム（16）", "スコア分位（16、対照）")
    fam = {k: fam[k] for k in keep}
    V = load(isic_discovery.STAGE1, "isic_audit_v", "dinov2")
    cal = audit.Calibrator(V.base, V.target)
    lesions_v = V.context.lesion_id.fillna(V.context.image).to_numpy()
    fam |= latent_families(isic_views(), 16, 32, (logit(cal(V.base)), V.target, lesions_v))
    rows = []
    for name, g in fam.items():
        rows.append({"dataset": "isic", **measure(name, g, T.target, p_t, threshold, lesions, 100, 10.0)})
        print(rows[-1]["family"], flush=True)
    return rows


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage1", default="a_s42")
    args = parser.parse_args(argv)
    out = pd.DataFrame(chexpert(args.stage1) + isic())
    out.to_csv(PACKAGE / "results" / "audit_metrics.csv", index=False)
    with pd.option_context("display.width", 250, "display.max_columns", 30):
        print(out.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
