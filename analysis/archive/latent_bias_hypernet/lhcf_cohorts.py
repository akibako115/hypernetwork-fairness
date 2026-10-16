"""LHCF（Masroor et al., MICCAI 2026）の隠れコホートを HAM10000 で作り、学習用の split に書き出す。

LHCF の作り方に合わせる：ERM で学習した encoder の train 埋め込みに GMM を当て、K は BIC で選ぶ。
LHCF は ResNet18（512 次元）だが、ここでは比較の基準と同じ ResNet-50 ERM（`resnet-ham10000-fg-erm` の seed 42）の
2048 次元を、標準化して PCA 32 に落としてから全共分散の GMM を当てる（本文に次元の扱いの記載がない）。
val・test は同じ変換と GMM で割り当てるだけ。ラベルも属性も使わない。

入力: `cache/<fg-erm s42 run-id>_{train,val,test}_features.npz`、`data/ham10000/splits_latent/`
出力:
- `data/ham10000/splits_lhcf/{train,val,test}.csv`（`splits_latent` に `lhcf`・`lhcf_missing` を足したもの）。
  対照の群の列も足す：`lhcf_random`（split ごとに `lhcf` を並べ替えた、大きさだけ同じランダムなコホート）と
  `label_group`（クラスそのもの。コホートが悪性度で切れることの対照）
- `results/ham10000_e2e/lhcf_bic.csv`（K ごとの BIC）、`lhcf_cohorts.csv`（コホートごとの件数・悪性率・属性の構成）

    uv run python -m analysis.archive.latent_bias_hypernet.lhcf_cohorts
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score as ari
from sklearn.metrics import normalized_mutual_info_score as nmi
from sklearn.mixture import GaussianMixture
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from analysis.archive.latent_bias_hypernet.data import PACKAGE, ROOT
from analysis.archive.latent_bias_hypernet.e2e_eval import CACHE, sex_age
from analysis.common.predictions import features_cache_path, load_features

SPLITS = ("train", "val", "test")
KS = range(2, 13)
SEED = 42
RUN = "resnet-ham10000-fg-erm-s42"


def fit_gmm(x: np.ndarray) -> tuple[GaussianMixture, pd.DataFrame]:
    """BIC 最小の K の GMM と、K ごとの BIC。"""
    models = {k: GaussianMixture(k, covariance_type="full", n_init=5, random_state=SEED).fit(x) for k in KS}
    bic = pd.DataFrame({"k": list(models), "bic": [m.bic(x) for m in models.values()]})
    return models[int(bic.loc[bic.bic.idxmin(), "k"])], bic


def describe(frame: pd.DataFrame, split: str) -> pd.DataFrame:
    """コホートごとの件数・悪性率・性別 × 年齢の構成（LHCF の Table 1 に相当）。"""
    groups = sex_age(frame, "sex")
    rows = []
    for k, part in frame.groupby("lhcf"):
        g = groups[frame.lhcf.to_numpy() == k]
        row = {
            "split": split,
            "cohort": k,
            "n": len(part),
            "n_pos": int(part.target.sum()),
            "pos_rate": part.target.mean(),
        }
        row.update({f"share_sexage{v}": float(np.mean(g == v)) for v in range(4)})
        row["share_female"] = float((part.sex == 1).mean())
        rows.append(row)
    return pd.DataFrame(rows)


def main() -> None:
    source, dest = ROOT / "data" / "ham10000" / "splits_latent", ROOT / "data" / "ham10000" / "splits_lhcf"
    frames = {s: pd.read_csv(source / f"{s}.csv") for s in SPLITS}
    (run_id,) = [p.name.removesuffix("_train_features.npz") for p in CACHE.glob(f"*{RUN}-*_train_features.npz")]
    h = {
        s: load_features(features_cache_path(CACHE, run_id, s), frames[s].image.to_numpy(str))["features"]
        for s in SPLITS
    }
    reduce = make_pipeline(StandardScaler(), PCA(32, random_state=SEED)).fit(h["train"])
    z = {s: reduce.transform(h[s]) for s in SPLITS}
    gmm, bic = fit_gmm(z["train"])
    print(bic.to_string(index=False))
    tables = []
    for s in SPLITS:
        frames[s]["lhcf"] = gmm.predict(z[s])
        frames[s]["lhcf_missing"] = 0
        frames[s]["lhcf_random"] = np.random.default_rng(SEED).permutation(frames[s].lhcf.to_numpy())
        frames[s]["label_group"] = frames[s].target
        frames[s]["lhcf_random_missing"] = frames[s]["label_group_missing"] = 0
        tables.append(describe(frames[s], s))
    train = frames["train"]
    print(
        f"K = {gmm.n_components}; train の NMI：ラベル {nmi(train.target, train.lhcf):.3f}、"
        f"性別 × 年齢 {nmi(sex_age(train, 'sex'), train.lhcf):.3f}、施設 {nmi(train.dataset, train.lhcf):.3f}、"
        f"DINOv2 GMM 8 との ARI {ari(train.gmm8, train.lhcf):.3f}"
    )
    table = pd.concat(tables)
    print(table.round(3).to_string(index=False))
    out = PACKAGE / "results" / "ham10000_e2e"
    bic.to_csv(out / "lhcf_bic.csv", index=False)
    table.to_csv(out / "lhcf_cohorts.csv", index=False)
    dest.mkdir(exist_ok=True)
    for s in SPLITS:
        frames[s].to_csv(dest / f"{s}.csv", index=False)


if __name__ == "__main__":
    main()
