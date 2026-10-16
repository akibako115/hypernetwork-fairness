"""2 段学習（cross-fit）の split と、2 段目の条件・群に使う区分を作る（HAM10000）。

2 段学習では、1 段目（ResNet-50 ERM）を train の半分（stage1）で学習して凍結し、2 段目（Spatial LoRA だけ）を
1 段目が見ていない残りの半分（stage2）で学習する。val・test は `splits_latent` と同じ行。
場は 2 つ：植え込みのない HAM10000（`real`）と、規則の食い違いを強めて植え込んだ版（`rule`、`rule_planting.py
--dose strong`）。

区分は 2 種類。どちらもラベルと属性を条件の計算に使わないので、test でもそのまま割り当てられる。
- `lhcf`：1 段目の stage2 埋め込み（標準化して PCA 32）の GMM。K は BIC（2〜12）。LHCF のコホートの作り方
- `iter<N>`：誤りの向きで作り直す区分。前の周のモデル（`iter1` は 1 段目、それ以降は `iter<N-1>` の 2 段目）の
  stage2 の残差 p − y で、1 段目の埋め込みを重み付けたもの（最終層の損失勾配の向き）に GMM 8 を当てる。
  この区分はラベルを使うので、埋め込みからの多項ロジスティック回帰（クラス均衡）で区分を当てる振り分け係を学習し、
  train・val・test のすべてに、その予測を割り当てる
  - `iter1` の残差は 1 段目にとって held-out。`iter2` 以降は 2 段目の学習データ上の残差になる（2 段目は
    凍結した backbone の上の低ランクの補正なので、残差は消えきらない。消えたかは出力の `resid_abs` で読む）

区分を 1 つ作るたびに、区分の性質を `results/ham10000_e2e/crossfit_partitions.csv` に追記する。
`rule` では植え込んだ少数側 B（`rule_planting.minority`）をどれだけ取り戻したか（ARI、B の多い区分が B を覆う割合）、
`real` ではラベル・性別 × 年齢・施設との NMI。

入力: `data/ham10000/splits_latent/`、1 段目と前の周の run（`runs/`）
出力:
- `split`：`data/ham10000/splits_crossfit/{stage1,stage2}/{train,val,test}.csv`
- `partition`：`data/ham10000/splits_crossfit/<field>_s<seed>_<name>/{train,val,test}.csv`（stage2 に
  `part`・`part_missing` を足したもの）と `k.txt`（区分の数）
- `cache/<run-id>_cf_<split>.npz`（stage2 の split で推論した logit 差と埋め込み）
- `checkpoint`：1 段目の best checkpoint の path（標準出力）

    uv run python -m analysis.archive.latent_bias_hypernet.crossfit split
    uv run python -m analysis.archive.latent_bias_hypernet.crossfit partition --field rule --seed 42 --name iter1
    uv run python -m analysis.archive.latent_bias_hypernet.crossfit checkpoint --field rule --seed 42
"""

from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd
import torch
from hydra.utils import instantiate
from omegaconf import OmegaConf
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import adjusted_rand_score as ari
from sklearn.metrics import normalized_mutual_info_score as nmi
from sklearn.mixture import GaussianMixture
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from analysis.archive.latent_bias_hypernet.data import PACKAGE, ROOT
from analysis.archive.latent_bias_hypernet.e2e_eval import CACHE, RUN_ID, sex_age
from analysis.archive.latent_bias_hypernet.rule_planting import minority
from analysis.common.predictions import _forward_split, load_network, local_data_path

SPLITS = ("train", "val", "test")
SOURCE = ROOT / "data" / "ham10000" / "splits_latent"
DEST = ROOT / "data" / "ham10000" / "splits_crossfit"
STAGE1 = {"real": "resnet-ham10000-cf-stage1", "rule": "resnet-ham10000-rule-strong-cf-stage1"}
STAGE2 = {"real": "spatial-lora-ham10000-cf-{name}", "rule": "spatial-lora-ham10000-rule-strong-cf-{name}"}
KS = range(2, 13)
K_ITER = 8
SEED = 42


def halve_by_lesion(frame: pd.DataFrame, seed: int = 0) -> tuple[pd.DataFrame, pd.DataFrame]:
    """病変を、悪性・良性それぞれで無作為に半分ずつ stage1 / stage2 へ割り当てる。同じ病変の画像は同じ側に入る。"""
    rng = np.random.default_rng(seed)
    label = frame.groupby("lesion_id").target.first()
    first = set()
    for c in (0, 1):
        lesions = np.sort(label.index[label == c])
        first |= set(rng.permutation(lesions)[: len(lesions) // 2])
    mask = frame.lesion_id.isin(first).to_numpy()
    return frame[mask], frame[~mask]


def split() -> None:
    train = pd.read_csv(SOURCE / "train.csv")
    halves = dict(zip(("stage1", "stage2"), halve_by_lesion(train), strict=True))
    for name, half in halves.items():
        (DEST / name).mkdir(parents=True, exist_ok=True)
        half.to_csv(DEST / name / "train.csv", index=False)
        for s in ("val", "test"):
            pd.read_csv(SOURCE / f"{s}.csv").to_csv(DEST / name / f"{s}.csv", index=False)
        print(f"{name}: {len(half)} 枚、{half.lesion_id.nunique()} 病変、陽性率 {half.target.mean():.3f}")


def run_dir(exp: str, seed: int):
    """取り込み済みで成功した run のうち最新のもの。"""
    found = []
    for d in sorted((PACKAGE / "runs").iterdir()):
        m = RUN_ID.match(d.name)
        if m and m["exp"] == exp and int(m["seed"]) == seed:
            if json.loads((d / "run.json").read_text())["status"] == "succeeded":
                found.append(d)
    if not found:
        raise FileNotFoundError(f"{exp} s{seed} の成功した run がない")
    return found[-1]


def infer(run, split_name: str) -> dict[str, np.ndarray]:
    """run を stage2 の split で推論した logit 差と埋め込み（ResNet だけ）。結果は cache に置く。"""
    path = CACHE / f"{run.name}_cf_{split_name}.npz"
    if path.exists():
        return dict(np.load(path))
    config = OmegaConf.load(run / "config.yaml")
    resnet = config.model.net._target_.rsplit(".", 1)[-1] == "ResNet"
    if resnet:
        # 1 段目は stage1 で学習した run。stage2 の行で推論する
        config.data.cv_splits_dir = str(DEST / "stage2")
    else:
        config.data.cv_splits_dir = str(local_data_path(config.data.cv_splits_dir))
    config.data.data_dir = str(local_data_path(config.data.data_dir))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    network = load_network(run, device)
    loader = instantiate(config.data).evaluation_dataloader(split_name, persistent_workers=False)
    logits, target, features = _forward_split(network, loader, device, bool(config.model.use_attributes), resnet)
    out = {"score": logits[:, 1] - logits[:, 0], "target": target}
    if resnet:
        out["features"] = features.astype(np.float32)
    np.savez(path, **out)
    return out


def embed(h: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """stage2 の train で当てた標準化 → PCA 32。"""
    h = {s: x.astype(np.float64) for s, x in h.items()}
    reduce = make_pipeline(StandardScaler(), PCA(32, random_state=SEED)).fit(h["train"])
    return {s: reduce.transform(x) for s, x in h.items()}


def lhcf_partition(z: dict[str, np.ndarray]) -> tuple[dict[str, np.ndarray], dict]:
    models = {k: GaussianMixture(k, covariance_type="full", n_init=5, random_state=SEED).fit(z["train"]) for k in KS}
    bic = {k: m.bic(z["train"]) for k, m in models.items()}
    best = min(bic, key=bic.get)
    return {s: models[best].predict(x) for s, x in z.items()}, {"k": best}


def error_partition(z: dict[str, np.ndarray], residual: np.ndarray) -> tuple[dict[str, np.ndarray], dict]:
    """残差で重み付けた埋め込みの GMM 8 を、埋め込みだけから当てる振り分け係で全 split に割り当てる。"""
    g = residual[:, None] * z["train"]
    raw = GaussianMixture(K_ITER, covariance_type="diag", n_init=5, random_state=SEED).fit_predict(g)
    router = make_pipeline(StandardScaler(), LogisticRegression(max_iter=5000, class_weight="balanced"))
    router.fit(z["train"], raw)
    out = {s: router.predict(x) for s, x in z.items()}
    info = {
        "k": K_ITER,
        "router_agree": float(np.mean(out["train"] == raw)),
        "resid_abs": float(np.abs(residual).mean()),
    }
    return out, info


def describe(frame: pd.DataFrame, part: np.ndarray, field: str) -> dict:
    """区分の性質。rule では少数側 B をどれだけ取り戻したか、real ではラベル・属性・施設との関係。"""
    out = {"n_parts": int(len(np.unique(part))), "largest_share": float(np.bincount(part).max() / len(part))}
    out["nmi_label"] = nmi(frame.target, part)
    if field == "rule":
        b = minority(frame)
        share = pd.Series(b).groupby(part).mean()
        b_parts = share.index[share > 0.5]
        covered = np.isin(part, b_parts)
        out.update(
            {
                "ari_minority": ari(b, part),
                "nmi_minority": nmi(b, part),
                "b_coverage": float(covered[b].mean()),
                "b_purity": float(b[covered].mean()) if covered.any() else np.nan,
            }
        )
    else:
        known = frame.sex.ge(0).to_numpy()
        out["nmi_sexage"] = nmi(sex_age(frame, "sex")[known], part[known])
        out["nmi_site"] = nmi(frame.dataset, part)
    return out


def partition(field: str, seed: int, name: str) -> None:
    frames = {s: pd.read_csv(DEST / "stage2" / f"{s}.csv") for s in SPLITS}
    stage1 = run_dir(STAGE1[field], seed)
    first = {s: infer(stage1, s) for s in SPLITS}
    for s in SPLITS:
        np.testing.assert_array_equal(first[s]["target"], frames[s].target.to_numpy())
    z = embed({s: first[s]["features"] for s in SPLITS})
    if name == "lhcf":
        parts, info = lhcf_partition(z)
    else:
        n = int(name.removeprefix("iter"))
        model = stage1 if n == 1 else run_dir(STAGE2[field].format(name=f"iter{n - 1}"), seed)
        score = first["train"]["score"] if n == 1 else infer(model, "train")["score"]
        residual = 1 / (1 + np.exp(-score)) - frames["train"].target.to_numpy()
        parts, info = error_partition(z, residual)
        info["from_run"] = model.name
    dest = DEST / f"{field}_s{seed}_{name}"
    dest.mkdir(exist_ok=True)
    for s in SPLITS:
        frames[s]["part"] = parts[s]
        frames[s]["part_missing"] = 0
        frames[s].to_csv(dest / f"{s}.csv", index=False)
    (dest / "k.txt").write_text(f"{info['k']}\n")
    rows = [{"field": field, "seed": seed, "name": name, "split": s, **info} for s in SPLITS]
    for row in rows:
        row.update(describe(frames[row["split"]], parts[row["split"]], field))
    table = pd.DataFrame(rows)
    print(table.round(3).to_string(index=False))
    out = PACKAGE / "results" / "ham10000_e2e" / "crossfit_partitions.csv"
    if out.exists():
        old = pd.read_csv(out)
        keep = ~((old.field == field) & (old.seed == seed) & (old.name == name))
        table = pd.concat([old[keep], table], ignore_index=True)
    table.to_csv(out, index=False)


def main() -> None:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("split")
    p = sub.add_parser("partition")
    p.add_argument("--field", choices=list(STAGE1), required=True)
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--name", required=True, help="lhcf または iter<N>")
    c = sub.add_parser("checkpoint", help="1 段目の best checkpoint の path を表示する（2 段目の起動用）")
    c.add_argument("--field", choices=list(STAGE1), required=True)
    c.add_argument("--seed", type=int, required=True)
    args = parser.parse_args()
    if args.command == "split":
        split()
    elif args.command == "checkpoint":
        record = json.loads((run_dir(STAGE1[args.field], args.seed) / "run.json").read_text())
        print(record["checkpoints"][0]["best"]["path"])
    else:
        partition(args.field, args.seed, args.name)


if __name__ == "__main__":
    main()
