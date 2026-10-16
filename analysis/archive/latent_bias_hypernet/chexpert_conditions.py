"""CheXpert の E2E 条件付けと公平性損失に使う、潜在の群の列を足した split を作る（組み直し計画の Q2）。

`data/chexpert/splits_race_known` の各 CSV に次の列（条件に使う列は `<name>_missing` も）を足す。

- `gmm8`：RAD-DINO（CLS＋patch 平均）を train で標準化 → PCA 32 → 対角共分散の GMM 8。ラベル不使用。
  Q1 で胸部 X 線の偏りが見えた埋め込みを使う
- `random8`：患者単位の一様ランダムな 8 区分。容量が増えたことの効果を除く対照
- `const`：全行 0。adapter を足したこと自体の効果を除く対照
- `lhcf`：LHCF（Masroor et al., MICCAI 2026）の隠れコホート。ResNet-50 ERM（race_known の seed 42）の train 埋め込みを
  標準化 → PCA 32 → 全共分散の GMM、K は BIC（2〜12）。HAM10000 の `lhcf_cohorts.py` と同じ作り方。
  train 15.8 万枚では全共分散の GMM の探索が重いので、K の選択と当てはめは train の無作為 4 万枚で行い、全行に割り当てる
- `label_group`：クラスそのもの（LHCF の利得がコホートによるかの対照）

変換と GMM はすべて train だけで決め、val・test は当てはめるだけ。cross-fit の 2 段学習用に、train を
`splits_crossfit` の A・B に分けた版も書く（val・test は同じ行）。

入力:
- `data/chexpert/splits_race_known/{train,val,test}.csv`、`data/chexpert/splits_crossfit/{train_a,train_b}.csv`
- `analysis/archive/latent_bias_hypernet/cache/embed/raddino/{train_a,train_b,val,test}.npz`
- `analysis/archive/race_known_architecture_objective_comparison/cache/<ERM s42 run-id>_{train,val,test}_features.npz`
出力:
- `data/chexpert/splits_latent/{train,val,test}.csv`
- `data/chexpert/splits_latent_cf/{stage1,stage2}/{train,val,test}.csv`（train はそれぞれ A・B）
- `analysis/archive/latent_bias_hypernet/results/chexpert_latent/{lhcf_bic,groups}.csv`

    uv run python -m analysis.archive.latent_bias_hypernet.chexpert_conditions
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from analysis.archive.latent_bias_hypernet.data import CACHE, PACKAGE, ROOT
from analysis.common.predictions import features_cache_path, load_features

DATA = ROOT / "data" / "chexpert"
ERM_CACHE = ROOT / "analysis" / "archive" / "race_known_architecture_objective_comparison" / "cache"
ERM_RUN = "20261002T124957Z-resnet-chexpert-race-known-s42-d6d3"
SPLITS = ("train", "val", "test")
K, KS, SEED, LHCF_FIT = 8, range(2, 13), 42, 40_000


def patient_of(image: pd.Series) -> pd.Series:
    return image.str.split("/").str[1]


def raddino(frames: dict[str, pd.DataFrame]) -> dict[str, np.ndarray]:
    """split CSV の行順に並べた RAD-DINO 埋め込み。train は A・B の cache を image で突き合わせる。"""
    parts = {s: np.load(CACHE / "embed" / "raddino" / f"{s}.npz") for s in ("train_a", "train_b", "val", "test")}
    out = {}
    for s, frame in frames.items():
        sources = ("train_a", "train_b") if s == "train" else (s,)
        images = np.concatenate([parts[p]["image"] for p in sources])
        emb = np.concatenate([parts[p]["embedding"] for p in sources]).astype(np.float32)
        index = pd.Series(np.arange(len(images)), index=images)
        rows = index.reindex(frame.image.to_numpy(str))
        if rows.isna().any():
            raise ValueError(f"{s}: RAD-DINO の埋め込みが無い画像が {int(rows.isna().sum())} 枚ある")
        out[s] = emb[rows.to_numpy(int)]
    return out


def gmm8(emb: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    reduce = make_pipeline(StandardScaler(), PCA(32, random_state=0)).fit(emb["train"])
    z = {s: reduce.transform(e) for s, e in emb.items()}
    gmm = GaussianMixture(K, covariance_type="diag", random_state=0, n_init=3).fit(z["train"])
    return {s: gmm.predict(v) for s, v in z.items()}


def lhcf(frames: dict[str, pd.DataFrame]) -> tuple[dict[str, np.ndarray], pd.DataFrame]:
    h = {
        s: load_features(features_cache_path(ERM_CACHE, ERM_RUN, s), f.image.to_numpy(str))["features"]
        for s, f in frames.items()
    }
    reduce = make_pipeline(StandardScaler(), PCA(32, random_state=SEED)).fit(h["train"])
    z = {s: reduce.transform(v) for s, v in h.items()}
    rng = np.random.default_rng(SEED)
    fit = z["train"][rng.choice(len(z["train"]), LHCF_FIT, replace=False)]
    models = {k: GaussianMixture(k, covariance_type="full", n_init=3, random_state=SEED).fit(fit) for k in KS}
    bic = pd.DataFrame({"k": list(models), "bic": [m.bic(fit) for m in models.values()]})
    best = models[int(bic.loc[bic.bic.idxmin(), "k"])]
    return {s: best.predict(v) for s, v in z.items()}, bic


def main() -> None:
    frames = {s: pd.read_csv(DATA / "splits_race_known" / f"{s}.csv") for s in SPLITS}
    g8 = gmm8(raddino(frames))
    cohorts, bic = lhcf(frames)
    for s, f in frames.items():
        f["gmm8"] = g8[s]
        patients = patient_of(f.image)
        uniq = np.sort(patients.unique())
        draw = np.random.default_rng({"train": 0, "val": 1, "test": 2}[s]).integers(0, K, len(uniq))
        f["random8"] = patients.map(dict(zip(uniq, draw, strict=True))).to_numpy()
        f["const"] = 0
        f["lhcf"] = cohorts[s]
        f["label_group"] = f.target.astype(int)
        for name in ("gmm8", "random8", "const", "lhcf"):
            f[f"{name}_missing"] = 0

    def write(dest, name, frame):
        dest.mkdir(parents=True, exist_ok=True)
        # 学習中の run が読む CSV なので、書き終えてから置き換える
        frame.to_csv(dest / f".{name}.csv.tmp", index=False)
        (dest / f".{name}.csv.tmp").replace(dest / f"{name}.csv")

    for s, f in frames.items():
        write(DATA / "splits_latent", s, f)
    for stage, half in (("stage1", "train_a"), ("stage2", "train_b")):
        keep = set(pd.read_csv(DATA / "splits_crossfit" / f"{half}.csv").image)
        train = frames["train"][frames["train"].image.isin(keep)]
        if len(train) != len(keep):
            raise ValueError(f"{half} の画像が race_known の train に揃わない")
        dest = DATA / "splits_latent_cf" / stage
        write(dest, "train", train)
        for s in ("val", "test"):
            write(dest, s, frames[s])

    out = PACKAGE / "results" / "chexpert_latent"
    out.mkdir(parents=True, exist_ok=True)
    bic.to_csv(out / "lhcf_bic.csv", index=False)
    rows = []
    for s, f in frames.items():
        for name in ("gmm8", "lhcf"):
            for k, part in f.groupby(name):
                rows.append({"split": s, "partition": name, "group": k, "n": len(part), "pos_rate": part.target.mean()})
    pd.DataFrame(rows).to_csv(out / "groups.csv", index=False)
    print(bic.to_string(index=False))
    print(pd.DataFrame(rows).query("split == 'train'").round(3).to_string(index=False))


if __name__ == "__main__":
    main()
