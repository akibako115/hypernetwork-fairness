"""発見側の切り方（[h; e] の k-means）を、学習時の公平性手法の群として split に書き出す（HAM10000）。

`discovery_to_remedy.py` の `he_km8` と同じ作り方を、学習に使えるよう train・test にも当てはめる。
seed ごとに、同じ seed の ERM（`resnet-ham10000-fg-erm`）の特徴 h を使う。

- val の h を標準化 → PCA 32、DINOv2 の PCA 16（`e0`〜`e15`）と並べて標準化 → k-means 8
- 変換と k-means は val だけで決め、train・test は当てはめるだけ（ラベルは使わない）

入力: `cache/<fg-erm run-id>_{train,val,test}_features.npz`、`data/ham10000/splits_latent/`
出力: `data/ham10000/splits_discovery/{train,val,test}.csv`（`splits_latent` に `he8_s<seed>` と欠損列を足したもの）

    uv run python -m analysis.archive.latent_bias_hypernet.discovery_groups
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from analysis.archive.latent_bias_hypernet.data import ROOT
from analysis.archive.latent_bias_hypernet.discovery_to_remedy import DINO, K
from analysis.archive.latent_bias_hypernet.e2e_eval import CACHE, conditions, runs
from analysis.common.predictions import features_cache_path, load_features

SPLITS = ("train", "val", "test")


def assign(frames: dict[str, pd.DataFrame], run_dir, seed: int) -> dict[str, np.ndarray]:
    """val で決めた [h; e] の k-means を、各 split に当てはめた群。"""
    h = {
        s: load_features(features_cache_path(CACHE, run_dir.name, s), frames[s].image.to_numpy(str))["features"]
        for s in SPLITS
    }
    reduce = make_pipeline(StandardScaler(), PCA(32, random_state=seed)).fit(h["val"])
    joined = {s: np.column_stack([reduce.transform(h[s]), frames[s][DINO].to_numpy()]) for s in SPLITS}
    scale = StandardScaler().fit(joined["val"])
    km = KMeans(K, n_init=10, random_state=seed).fit(scale.transform(joined["val"]))
    return {s: km.predict(scale.transform(joined[s])) for s in SPLITS}


def main() -> None:
    names = conditions("ham10000")
    names["resnet-ham10000-fg-erm"] = "ResNet-50 ERM（公平性比較の基準）"
    table = runs(names)
    table = table[table.exp == "resnet-ham10000-fg-erm"]
    source, dest = ROOT / "data" / "ham10000" / "splits_latent", ROOT / "data" / "ham10000" / "splits_discovery"
    frames = {s: pd.read_csv(source / f"{s}.csv") for s in SPLITS}
    for r in table.itertuples():
        groups = assign(frames, r.run_dir, r.seed)
        for s in SPLITS:
            frames[s][f"he8_s{r.seed}"] = groups[s]
            frames[s][f"he8_s{r.seed}_missing"] = 0
        counts = pd.crosstab(frames["train"][f"he8_s{r.seed}"], frames["train"].target)
        print(f"seed {r.seed} train [群, クラス] 件数\n{counts.T.to_string()}")
    dest.mkdir(exist_ok=True)
    for s in SPLITS:
        frames[s].to_csv(dest / f"{s}.csv", index=False)


if __name__ == "__main__":
    main()
