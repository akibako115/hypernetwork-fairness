"""年齢群の AUROC の差が、構成（撮影条件・潜在部分集団）の違いでどこまで説明できるかを調べる。

若年群（65 歳未満）の各サンプルに、層の構成を高齢群に合わせる重み w(s, y) = P_old(s | y) / P_young(s | y) を付け、
重み付き AUROC を求める。ラベル別に合わせるので、陽性率の違いは混ざらない。差の縮み
（若年 − 重み付き若年）/（若年 − 高齢）を「構成で説明できる割合」とする。逆向き（高齢を若年の構成に合わせる）も求める。

層は test 以外で決める（GMM と k-means は train_b に当てる）。不確実性は患者単位の bootstrap。

注意：ラベル別に構成を合わせるので、ラベルを当てられる層ほど機械的に差を「説明」する。1 段目のスコアの分位を
層にすると差の 91〜99% が説明され、h(x) を含む `kmeans32([h;e])`（94〜97%）はこれと区別できない。
解釈に使えるのは、ラベルをあまり当てない層（撮影条件、粗い GMM）の値だけ。対照として `score32` を出す。

出力: `results/diag_age/<stage1>.csv`

    uv run python -m analysis.archive.latent_bias_hypernet.diag_age --stage1 a_s42 --runs base im_a_s42_router_he_s0
"""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd
from sklearn.cluster import MiniBatchKMeans
from sklearn.decomposition import PCA
from sklearn.metrics import roc_auc_score

from analysis.archive.latent_bias_hypernet.audit import GMMSlicer, quantile_slicer
from analysis.archive.latent_bias_hypernet.audit_corrected import scores
from analysis.archive.latent_bias_hypernet.data import PACKAGE, load
from analysis.archive.latent_bias_hypernet.stats import patient_index


def context_strata(c: pd.DataFrame, with_report: bool) -> np.ndarray:
    s = (
        c.frontal_lateral.to_numpy().astype(int) * 8
        + c.ap_pa.to_numpy().astype(int) * 4
        + c.device.to_numpy().astype(int) * 2
        + c.followup.to_numpy().astype(int)
    )
    if with_report:
        rt = pd.Categorical(c.report_type, categories=["affirmed_normal", "focused_followup", "other"]).codes
        s = s * 3 + rt
    return s


def he_kmeans(D, T, K: int) -> np.ndarray:
    ph = PCA(64, random_state=0).fit(D.features)
    pe = PCA(64, random_state=0).fit(D.route)

    def z(s):
        a, b = ph.transform(s.features), pe.transform(s.route)
        return np.concatenate([a / a.std(), b / b.std()], axis=1)

    km = MiniBatchKMeans(K, random_state=0, n_init=5, batch_size=4096).fit(z(D))
    return km.predict(z(T))


def reweight(src: np.ndarray, dst: np.ndarray, strata: np.ndarray, y: np.ndarray) -> np.ndarray:
    """src の行に、層 × ラベルの構成を dst に合わせる重みを返す（dst に無い層は重み 0）。"""
    w = np.zeros(len(y))
    for label in (0, 1):
        ms, md = src & (y == label), dst & (y == label)
        ps = pd.Series(strata[ms]).value_counts(normalize=True)
        pd_ = pd.Series(strata[md]).value_counts(normalize=True)
        ratio = (pd_ / ps).reindex(ps.index).fillna(0.0)
        w[ms] = pd.Series(strata[ms]).map(ratio).to_numpy()
    return w


def gap_stats(y, s, old, strata, idx=None) -> dict[str, float]:
    if idx is not None:
        y, s, old, strata = y[idx], s[idx], old[idx], strata[idx]
    young = ~old
    a_y = roc_auc_score(y[young], s[young])
    a_o = roc_auc_score(y[old], s[old])
    w = reweight(young, old, strata, y)
    a_yw = roc_auc_score(y[young], s[young], sample_weight=w[young])
    w2 = reweight(old, young, strata, y)
    a_ow = roc_auc_score(y[old], s[old], sample_weight=w2[old])
    gap = a_y - a_o
    return {
        "auroc_young": a_y,
        "auroc_old": a_o,
        "gap": gap,
        "young_as_old": a_yw,
        "old_as_young": a_ow,
        "explained_y": (a_y - a_yw) / gap,
        "explained_o": (a_ow - a_o) / gap,
    }


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage1", default="a_s42")
    parser.add_argument("--runs", nargs="+", default=["base"])
    parser.add_argument("--boot", type=int, default=200)
    args = parser.parse_args(argv)

    D = load(args.stage1, "train_b", "raddino")
    V, T = load(args.stage1, "val", "raddino"), load(args.stage1, "test", "raddino")
    strata = {
        "none": np.zeros(len(T.target), dtype=int),
        "context": context_strata(T.context, False),
        "context+report": context_strata(T.context, True),
        "gmm8(RAD-DINO)": GMMSlicer(D.route, 8)(T.route),
        "gmm32(RAD-DINO)": GMMSlicer(D.route, 32)(T.route),
        "kmeans32([h;e])": he_kmeans(D, T, 32),
    }
    strata["context×gmm8"] = strata["context"] * 8 + strata["gmm8(RAD-DINO)"]
    strata["score32(対照)"] = quantile_slicer(D.base, 32)(T.base)
    old = T.context.age.to_numpy() >= 65
    y = T.target
    patients = patient_index(T.context.image)
    rows_of = pd.Series(np.arange(len(y))).groupby(patients).apply(np.asarray).to_numpy()
    rng = np.random.default_rng(0)
    boots = [np.concatenate(rows_of[rng.integers(0, len(rows_of), len(rows_of))]) for _ in range(args.boot)]
    rows = []
    for run in args.runs:
        _, s = scores(run, V, T)
        for name, st in strata.items():
            row = {"run": run, "strata": name, "n_strata": len(np.unique(st)), **gap_stats(y, s, old, st)}
            b = pd.DataFrame([gap_stats(y, s, old, st, idx) for idx in boots])
            for key in ("gap", "explained_y", "explained_o"):
                row[f"{key}_lo"], row[f"{key}_hi"] = np.quantile(b[key], [0.025, 0.975])
            rows.append(row)
            print(run, name, {k: round(v, 3) for k, v in row.items() if isinstance(v, float)}, flush=True)
    out = PACKAGE / "results" / "diag_age"
    out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out / f"{args.stage1}.csv", index=False)


if __name__ == "__main__":
    main()
