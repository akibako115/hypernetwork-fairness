"""Q1 の GEORGE 型の群の数を振り、原典どおり silhouette で選ぶと K がいくつになるかと、見える差を並べる。

GEORGE（Sohoni et al. 2020）はクラスごとに K を silhouette で選ぶ（上限 10）。Q1 では属性にそろえて
クラスごとに 4（計 8）に固定した。ここではクラスごとの K を 2〜10 で振り、次を求める（CheXpert、1 段目 a_s42）。

- silhouette（train_b の各クラスから 5000 枚）と、患者で半分に分けた train_b どうしの test の割り当ての ARI（3 回の平均）
- Eopp1（陽性の群の TPR の幅）・Eopp0（陰性の群の FPR の幅）と、クラスの中で群ラベルを並べ替えた帰無分布
  （200 回）の平均を引いた超過分
- 陽性と陰性で K をそろえた行に加え、silhouette でクラスごとに選んだ K の組み合わせの行（`selected`）

原典の UMAP は Q1 と同じく主成分 64 次元に替えた。

出力: `results/q1_george_k.csv`

    OMP_NUM_THREADS=8 uv run python -m analysis.archive.latent_bias_hypernet.q1_george_k
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score, silhouette_score
from sklearn.mixture import GaussianMixture

from analysis.archive.latent_bias_hypernet.data import PACKAGE, load
from analysis.archive.latent_bias_hypernet.q1_definitions import STAGE1, group_rows, summarize, youden

KS = range(2, 11)
NULL = 200
SPLITS = 3


def gmm(z: np.ndarray, K: int, seed: int) -> GaussianMixture:
    return GaussianMixture(K, covariance_type="diag", random_state=seed, max_iter=200).fit(z)


def class_sweep(D, T, c: int, rng) -> tuple[pd.DataFrame, dict[int, np.ndarray]]:
    """クラス c の中で K を振り、K ごとの silhouette・ARI と test の割り当てを返す。"""
    fit_rows, test_rows = D.target == c, T.target == c
    pca = PCA(64, random_state=0).fit(D.features[fit_rows])
    z_fit, z_test = pca.transform(D.features[fit_rows]), pca.transform(T.features[test_rows])
    patients = D.context.image[fit_rows].str.split("/").str[1].to_numpy()
    uniq = np.unique(patients)
    sil_idx = rng.choice(len(z_fit), min(5000, len(z_fit)), replace=False)
    rows, assign = [], {}
    for K in KS:
        model = gmm(z_fit, K, 0)
        assign[K] = model.predict(z_test)
        aris = []
        for r in range(SPLITS):
            half = np.isin(patients, rng.permutation(uniq)[: len(uniq) // 2])
            aris.append(adjusted_rand_score(gmm(z_fit[half], K, r).predict(z_test), gmm(z_fit[~half], K, r).predict(z_test)))
        sizes = np.bincount(assign[K], minlength=K)
        rows.append({
            "cls": c, "K": K, "silhouette": silhouette_score(z_fit[sil_idx], model.predict(z_fit[sil_idx])),
            "split_ari": np.mean(aris), "split_ari_min": np.min(aris), "min_size": int(sizes.min()),
        })
        print("class", c, "K", K, flush=True)
    return pd.DataFrame(rows), assign


def gaps(g, y, score, thr, rng) -> dict:
    obs = summarize(group_rows(g, y, score, thr, True))
    null = []
    for _ in range(NULL):
        perm = g.copy()
        for c in (0, 1):
            perm[y == c] = rng.permutation(g[y == c])
        null.append(summarize(group_rows(perm, y, score, thr, True)))
    null = pd.DataFrame(null)
    out = {}
    for k in ("eopp1", "eopp0"):
        out |= {k: obs[k], f"{k}_null": null[k].mean(), f"{k}_excess": obs[k] - null[k].mean(), f"{k}_p": (null[k] >= obs[k]).mean()}
    return out


def main() -> None:
    rng = np.random.default_rng(0)
    D, V, T = (load(STAGE1, s) for s in ("train_b", "val", "test"))
    y, score, thr = T.target, T.base, youden(V.base, V.target)
    tables, assigns = {}, {}
    for c in (0, 1):
        tables[c], assigns[c] = class_sweep(D, T, c, rng)

    def combined(kn: int, kp: int) -> np.ndarray:
        g = np.empty(len(y), dtype=int)
        g[y == 0] = assigns[0][kn]
        g[y == 1] = assigns[1][kp] + kn
        return g

    picked = {c: int(tables[c].loc[tables[c].silhouette.idxmax(), "K"]) for c in (0, 1)}
    combos = [(K, K, "same") for K in KS] + [(picked[0], picked[1], "selected")]
    rows = []
    for kn, kp, kind in combos:
        neg, pos = tables[0].set_index("K").loc[kn], tables[1].set_index("K").loc[kp]
        row = {"kind": kind, "K_neg": kn, "K_pos": kp, "sil_neg": neg.silhouette, "sil_pos": pos.silhouette}
        row |= {"ari_neg": neg.split_ari, "ari_pos": pos.split_ari, "min_size_neg": neg.min_size, "min_size_pos": pos.min_size}
        rows.append(row | gaps(combined(kn, kp), y, score, thr, rng))
    out = pd.DataFrame(rows)
    out.to_csv(PACKAGE / "results" / "q1_george_k.csv", index=False)
    with pd.option_context("display.width", 300, "display.max_columns", 40):
        print(out.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
