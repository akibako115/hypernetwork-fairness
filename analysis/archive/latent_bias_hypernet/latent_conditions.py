"""HAM10000・FairVLMed の E2E 条件付きモデル用に、条件の列を足した split を作る（潜在の振り分け、段階 1〜3）。

`data/<dataset>/splits` の各 CSV に次の列（とそれぞれの `<name>_missing`）を足し、
`data/<dataset>/splits_latent/` に書く。hypernet_e2e の MetadataEncoder は列挙された列をそのまま読むので、
config の `attribute_names` を差し替えるだけで条件を変えられる。

- `site`（HAM10000 のみ）：撮影施設（`dataset` の 1〜4 を 0〜3 に詰める）。学習にない external（test のみ）は欠損。
  潜在の振り分けが施設を回収できたときの上限として使う
- `gmm8`：DINOv2（CLS＋patch 平均）を train で PCA 32 → GMM 8。ラベル不使用（LHCF と同じ作り方）
- `gmm8_lo`：同じ埋め込みから、train のラベルを予測できる方向を反復的に除いて（INLP、5 回）から GMM 8。
  振り分けが「難しさ・ラベル」で切れるのを防いだ版。射影は train で決め、適用時にラベルは使わない
- `random8`：病変単位（FairVLMed は画像単位）の一様ランダムな 8 区分。容量が増えたことの効果を除く対照
- `const`：全行 0。条件を持たない同じ構成（adapter を足したこと自体の効果）の対照
- `e0`〜`e15`：DINOv2 の PCA 16 成分（段階 2 の連続入力）
- `elo0`〜`elo15`：ラベル直交化した埋め込み（`gmm8_lo` の入力）の PCA 16 成分（段階 2 の連続入力）

HAM10000 の `dx_type` はラベルの漏れ（陽性は histo のみ）なので条件に入れない。
埋め込みは probes/embed_splits.py が先に作っておく。

    uv run python -m analysis.archive.latent_bias_hypernet.latent_conditions [--dataset fairvlmed]
"""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import adjusted_rand_score
from sklearn.mixture import GaussianMixture

from analysis.archive.latent_bias_hypernet.data import CACHE, ROOT

PREFIX = {"ham10000": "ham", "fairvlmed": "fvl"}
SPLITS = ("train", "val", "test")
K = 8
INLP_ROUNDS = 5


def inlp(x: np.ndarray, y: np.ndarray, rounds: int) -> np.ndarray:
    """ラベルを線形に予測できる方向を `rounds` 回取り除く射影行列を返す。"""
    proj = np.eye(x.shape[1])
    for _ in range(rounds):
        w = LogisticRegression(max_iter=2000, C=1.0).fit(x @ proj, y).coef_[0]
        w = proj @ w
        w /= np.linalg.norm(w)
        proj = proj - np.outer(w, w)
    return proj


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="ham10000", choices=tuple(PREFIX))
    args = parser.parse_args()
    ham, prefix = args.dataset == "ham10000", PREFIX[args.dataset]
    SOURCE = ROOT / "data" / args.dataset / "splits"
    DEST = ROOT / "data" / args.dataset / "splits_latent"
    frames = {s: pd.read_csv(SOURCE / f"{s}.csv") for s in SPLITS}
    emb = {}
    for s in SPLITS:
        npz = np.load(CACHE / "embed" / "dinov2" / f"{prefix}_{s}.npz")
        if not np.array_equal(npz["image"], frames[s].image.to_numpy(dtype=str)):
            raise ValueError(f"{prefix}_{s} の埋め込みの行順が split CSV と一致しない")
        emb[s] = npz["embedding"].astype(np.float32)
    mu, sd = emb["train"].mean(0), emb["train"].std(0) + 1e-6
    pca = PCA(32, random_state=0).fit((emb["train"] - mu) / sd)
    z = {s: pca.transform((emb[s] - mu) / sd) for s in SPLITS}
    gmm = GaussianMixture(K, covariance_type="diag", random_state=0, n_init=3).fit(z["train"])
    zs = z["train"].std(0)
    proj = inlp(z["train"] / zs, frames["train"].target.to_numpy(), INLP_ROUNDS)
    zlo = {s: (z[s] / zs) @ proj for s in SPLITS}
    gmm_lo = GaussianMixture(K, covariance_type="diag", random_state=0, n_init=3).fit(zlo["train"])
    pca_lo = PCA(16, random_state=0).fit(zlo["train"])
    DEST.mkdir(parents=True, exist_ok=True)
    for s in SPLITS:
        f = frames[s].copy()
        if ham:
            external = f.dataset.to_numpy() == 0
            f["site"] = np.where(external, 0, f.dataset - 1)
            f["site_missing"] = external.astype(int)
        f["gmm8"] = gmm.predict(z[s])
        f["gmm8_lo"] = gmm_lo.predict(zlo[s])
        lesions = f.lesion_id.fillna(f.image) if ham else f.image
        uniq = np.sort(lesions.unique())
        rand = dict(
            zip(
                uniq, np.random.default_rng({"train": 0, "val": 1, "test": 2}[s]).integers(0, K, len(uniq)), strict=True
            )
        )
        f["random8"] = lesions.map(rand).to_numpy()
        f["const"] = 0
        for name in ("gmm8", "gmm8_lo", "random8", "const"):
            f[f"{name}_missing"] = 0
        for j in range(16):
            f[f"e{j}"] = z[s][:, j]
            f[f"e{j}_missing"] = 0
        elo = pca_lo.transform(zlo[s])
        for j in range(16):
            f[f"elo{j}"] = elo[:, j]
            f[f"elo{j}_missing"] = 0
        # 学習中の run が読む CSV なので、書き終えてから置き換える
        f.to_csv(DEST / f".{s}.csv.tmp", index=False)
        (DEST / f".{s}.csv.tmp").replace(DEST / f"{s}.csv")
        if ham:
            known = ~external
            ari = {
                n: round(adjusted_rand_score(f.dataset[known], f[n][known]), 3) for n in ("gmm8", "gmm8_lo", "random8")
            }
            print(s, len(f), "ARI(site)", ari)
        else:
            print(s, len(f))
        for name in ("gmm8", "gmm8_lo"):
            tab = f.groupby(name).target.agg(["size", "mean"]).round(3)
            print(f"  {name} size/pos-rate:", tab.to_dict("index"))


if __name__ == "__main__":
    main()
