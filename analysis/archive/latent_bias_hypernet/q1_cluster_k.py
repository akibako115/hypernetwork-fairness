"""Q1 の特徴空間クラスタの群の数 K を振り、クラスタの安定性・妥当性と、見える性能差を並べる。

Q1 では K を属性（8 群）にそろえた。この K が妥当かを、次の 3 つで見る（CheXpert、1 段目 a_s42）。

- 安定性：train_b を患者で半分に分けた 2 つにそれぞれクラスタを当て、test の割り当ての ARI を求める（3 回の平均）。
  初期値だけを変えた ARI も添える。
- 妥当性：silhouette（test から 5000 枚）、GMM の BIC（train_b）、撮影方向・属性との相互情報量。
- 見える差：Q1 の 4 指標。群の数が増えると偏りが無くても差は大きく出るので、test の群ラベルを並べ替えた
  帰無分布（群の大きさは同じ、200 回）の平均を引いた超過分も求める。属性の粒度（2〜16 群）にも同じ処理をする。

出力: `results/q1_cluster_k.csv`

    OMP_NUM_THREADS=8 uv run python -m analysis.archive.latent_bias_hypernet.q1_cluster_k
"""

from __future__ import annotations

import itertools

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score, mutual_info_score, silhouette_score
from sklearn.mixture import GaussianMixture

from analysis.archive.latent_bias_hypernet import audit
from analysis.archive.latent_bias_hypernet.data import PACKAGE, load
from analysis.archive.latent_bias_hypernet.q1_definitions import METRICS, STAGE1, group_rows, summarize, youden

KS = (2, 3, 4, 6, 8, 12, 16, 24, 32)
NULL = 200
SPLITS = 3


def fit(z: np.ndarray, method: str, K: int, seed: int):
    if method == "kmeans":
        return KMeans(K, n_init=4, random_state=seed).fit(z)
    return GaussianMixture(K, covariance_type="diag", random_state=seed, max_iter=200).fit(z)


def entropy(a: np.ndarray) -> float:
    p = np.bincount(a) / len(a)
    p = p[p > 0]
    return float(-(p * np.log(p)).sum())


def gaps(g, y, score, thr, rng) -> dict:
    obs = summarize(group_rows(g, y, score, thr, False))
    null = pd.DataFrame([summarize(group_rows(rng.permutation(g), y, score, thr, False)) for _ in range(NULL)])
    out = {}
    for k in METRICS:
        out[k] = obs[k]
        out[f"{k}_null"] = null[k].mean()
        out[f"{k}_excess"] = obs[k] - null[k].mean()
        # worst_auroc は小さいほど悪いので、帰無より悪い側の割合を p とする
        out[f"{k}_p"] = (null[k] <= obs[k]).mean() if k == "worst_auroc" else (null[k] >= obs[k]).mean()
    t = group_rows(g, y, score, thr, False)
    out["groups"] = int(t.auroc.notna().sum())
    out["min_n"] = int(t.n.min())
    return out


def main() -> None:
    rng = np.random.default_rng(0)
    V = load(STAGE1, "val")
    T0 = load(STAGE1, "test")
    y, score, thr = T0.target, T0.base, youden(V.base, V.target)
    ctx = T0.context
    view = ctx.frontal_lateral.to_numpy().astype(int) * 2 + ctx.ap_pa.to_numpy().astype(int)
    factors = {"view": view, "demo8": audit.demo_cells(ctx, "demo8"), "device": ctx.device.to_numpy().astype(int)}
    rows = []

    attrs = {
        "sex(2)": ctx.sex.to_numpy().astype(int),
        "age(2)": (ctx.age.to_numpy() >= 65).astype(int),
        "race(4)": ctx.race_group.to_numpy().astype(int),
        "age×race(8)": audit.demo_cells(ctx, "demo8"),
        "sex×age×race(16)": audit.demo_cells(ctx, "demo16"),
    }
    for name, g in attrs.items():
        rows.append({"space": "属性", "method": name, "K": len(np.unique(g))} | gaps(g, y, score, thr, rng))
        print(rows[-1]["method"], flush=True)

    for space in ("raddino", "dinov2"):
        D, T = load(STAGE1, "train_b", space), load(STAGE1, "test", space)
        pca = PCA(64, random_state=0).fit(D.route)
        z_fit, z_test = pca.transform(D.route), pca.transform(T.route)
        patients = D.context.image.str.split("/").str[1].to_numpy()
        uniq = np.unique(patients)
        sil_idx = rng.choice(len(z_test), 5000, replace=False)
        for method, K in itertools.product(("kmeans", "gmm"), KS):
            model = fit(z_fit, method, K, 0)
            g = model.predict(z_test)
            split_ari = []
            for r in range(SPLITS):
                half = np.isin(patients, rng.permutation(uniq)[: len(uniq) // 2])
                a = fit(z_fit[half], method, K, r).predict(z_test)
                b = fit(z_fit[~half], method, K, r).predict(z_test)
                split_ari.append(adjusted_rand_score(a, b))
            seed_ari = adjusted_rand_score(g, fit(z_fit, method, K, 1).predict(z_test))
            row = {"space": space, "method": method, "K": K, "split_ari": np.mean(split_ari)}
            row |= {"split_ari_min": np.min(split_ari), "seed_ari": seed_ari}
            row["silhouette"] = silhouette_score(z_test[sil_idx], g[sil_idx]) if len(np.unique(g[sil_idx])) > 1 else np.nan
            row["bic"] = model.bic(z_fit) if method == "gmm" else np.nan
            row["test_loglik"] = model.score(z_test) if method == "gmm" else np.nan
            for f, v in factors.items():
                row[f"mi_{f}"] = mutual_info_score(v, g) / entropy(v)  # 要因のエントロピーのうちクラスタが説明する割合
            row["prev_min"], row["prev_max"] = (pd.Series(y).groupby(g).mean().agg(["min", "max"]))
            rows.append(row | gaps(g, y, score, thr, rng))
            print(space, method, K, flush=True)

    out = pd.DataFrame(rows)
    out.to_csv(PACKAGE / "results" / "q1_cluster_k.csv", index=False)
    cols = ["space", "method", "K", "groups", "split_ari", "seed_ari", "silhouette", "bic"]
    cols += [f"{k}{s}" for k in METRICS for s in ("", "_excess")]
    with pd.option_context("display.width", 300, "display.max_columns", 40):
        print(out[cols].round(3).to_string(index=False))


if __name__ == "__main__":
    main()
