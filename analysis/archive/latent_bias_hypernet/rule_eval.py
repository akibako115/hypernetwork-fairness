"""規則の食い違いを植え込んだ場（`rule_planting.py`）で、クラスタで振り分ける条件付きモデルが少数側の規則を守れたかを見る。

比べる条件（各 3 seed）：ResNet-50 ERM、Spatial LoRA の条件を定数・GMM 8（規則が反転する境界を含む）・ランダム 8 区分。
`--dose strong` は規則を強めた版（`rule_planting.py --dose strong`）で、Spatial LoRA の 3 条件だけを比べる。
1. test の AUROC：全体、多数側 A、少数側 B（マーカーの意味が反転するクラスタ 2・4）
2. マーカーの反実仮想の効果：同じ test 画像にマーカーを付けた版と付けない版を推論し、logit 差（悪性 − 良性）の差を
   A・B で平均する。規則を学んでいれば A で正、B で負になる
3. 到達できる上限の目安：植え込み前の画像で学習した ERM（`resnet-ham10000-fg-erm`）の test 得点に、
   マーカーとクラスタの情報を val のロジスティック回帰で足したときの AUROC
   - 規則を知る：[得点, マーカー × A, マーカー × B]
   - 規則を知らない：[得点, マーカー]

条件間の差は、同じ seed 平均どうしの差を、病変単位の bootstrap（`e2e_eval.bootstrap_index`）で区間にする。

入力: `runs/*-ham10000-rule*`、`cache/<run-id>_test.npz`、`data/ham10000/splits_latent/`、`data/ham10000/images`
出力:
- `data/ham10000/images_rule_{on,off}/`（test の全画像にマーカーを付けた版・付けない版。品質は同じ）
- `results/ham10000_e2e/rule_summary.csv`（run ごと）、`rule_contrast.csv`（条件間の差）
- `results/ham10000_e2e/rule_ceiling.csv`（上限の目安）
- `--dose strong` では各 CSV の名前に `_strong` が付く

    uv run python -m analysis.archive.latent_bias_hypernet.rule_eval [--dose strong]
"""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd
import torch
from hydra.utils import instantiate
from joblib import Parallel, delayed
from omegaconf import OmegaConf
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

from analysis.archive.latent_bias_hypernet.data import PACKAGE, ROOT
from analysis.archive.latent_bias_hypernet.e2e_eval import CACHE, bootstrap_index, runs
from analysis.archive.latent_bias_hypernet.planting import stamp
from analysis.archive.latent_bias_hypernet.rule_planting import QUALITY, SOURCE, marker_mask, minority
from analysis.common.predictions import _forward_split, load_cache, load_network, local_data_path

CONDITIONS = {
    "resnet-ham10000-rule": "ResNet-50 ERM",
    "spatial-lora-ham10000-rule-const": "Spatial LoRA（条件：定数）",
    "spatial-lora-ham10000-rule-gmm8": "Spatial LoRA（条件：GMM 8、反転の境界を含む）",
    "spatial-lora-ham10000-rule-random8": "Spatial LoRA（条件：ランダム 8 区分）",
}
REFERENCES = ("resnet-ham10000-rule", "spatial-lora-ham10000-rule-const")
# 規則を強めた版（Spatial LoRA の 3 条件だけ）
STRONG = {k.replace("-rule-", "-rule-strong-"): v for k, v in CONDITIONS.items() if k.startswith("spatial")}
STRONG_REFERENCES = ("spatial-lora-ham10000-rule-strong-const",)
KEYS = ["auroc_all", "auroc_A", "auroc_B"]
SPLITS = ROOT / "data" / "ham10000" / "splits_latent"


def write_counterfactual(test: pd.DataFrame) -> None:
    """test の全画像を、マーカー付き（on）と無し（off）の 2 版で書く。"""
    from PIL import Image

    def one(image: str) -> None:
        with Image.open(SOURCE / image) as img:
            out = img.convert("RGB")
        for tag, version in (("on", stamp(out)), ("off", out)):
            version.save(ROOT / "data" / "ham10000" / f"images_rule_{tag}" / image, quality=QUALITY)

    for tag in ("on", "off"):
        (ROOT / "data" / "ham10000" / f"images_rule_{tag}").mkdir(exist_ok=True)
    if all((ROOT / "data" / "ham10000" / f"images_rule_{t}" / test.image.iloc[-1]).exists() for t in ("on", "off")):
        return
    Parallel(n_jobs=16)(delayed(one)(image) for image in test.image)


def counterfactual_logits(run_dir, device: torch.device) -> dict[str, np.ndarray]:
    """test をマーカー on・off の画像で推論した logit 差。"""
    config = OmegaConf.load(run_dir / "config.yaml")
    config.data.cv_splits_dir = str(local_data_path(config.data.cv_splits_dir))
    network = load_network(run_dir, device)
    out = {}
    for tag in ("on", "off"):
        config.data.data_dir = str(ROOT / "data" / "ham10000" / f"images_rule_{tag}")
        loader = instantiate(config.data).evaluation_dataloader("test", persistent_workers=False)
        logits, _, _ = _forward_split(network, loader, device, bool(config.model.get("use_attributes", False)), False)
        out[tag] = logits[:, 1] - logits[:, 0]
    return out


def metrics(test: pd.DataFrame, s: np.ndarray) -> dict:
    y, b = test.target.to_numpy(), minority(test)
    return {
        "auroc_all": roc_auc_score(y, s),
        "auroc_A": roc_auc_score(y[~b], s[~b]),
        "auroc_B": roc_auc_score(y[b], s[b]),
    }


def ceiling(val: pd.DataFrame, test: pd.DataFrame, dose: str) -> pd.DataFrame:
    """植え込み前の ERM の得点に、マーカー（とクラスタ）の情報を足したときの test の AUROC。"""
    rows = []
    for r in runs({"resnet-ham10000-fg-erm": "ERM（植え込み前）"}).itertuples():
        score = {}
        for name, frame in (("val", val), ("test", test)):
            logits = load_cache(CACHE / f"{r.run_dir.name}_{name}.npz", frame.image.to_numpy(str))["logits"]
            score[name] = logits[:, 1] - logits[:, 0]
        for rule, cols in (
            (
                "規則を知る",
                lambda f, s: np.column_stack(
                    [s, marker_mask(f, dose) & ~minority(f), marker_mask(f, dose) & minority(f)]
                ),
            ),
            ("規則を知らない", lambda f, s: np.column_stack([s, marker_mask(f, dose)])),
        ):
            model = LogisticRegression(max_iter=2000).fit(cols(val, score["val"]), val.target)
            s = model.decision_function(cols(test, score["test"]))
            rows.append({"seed": r.seed, "reference": rule, **metrics(test, s)})
        rows.append(
            {"seed": r.seed, "reference": "マーカーなし（植え込み前の ERM そのもの）", **metrics(test, score["test"])}
        )
    return pd.DataFrame(rows)


def seed_delta(test, scores, seeds, exp: str, ref: str, idx: np.ndarray) -> pd.Series:
    """行 idx の上で、exp と ref の指標の seed 平均の差。"""
    f = test.iloc[idx]

    def mean(e: str) -> pd.Series:
        return pd.DataFrame([metrics(f, scores[(e, s)][idx]) for s in seeds[e]]).mean()

    return mean(exp) - mean(ref)


def contrast(test: pd.DataFrame, scores: dict, seeds: dict, conditions: dict, references) -> pd.DataFrame:
    """条件どうしの seed 平均の差と、病変単位の bootstrap 区間。"""
    index = bootstrap_index(test, "ham10000")
    rows = []
    for ref in references:
        for exp in conditions:
            if exp == ref:
                continue
            point = seed_delta(test, scores, seeds, exp, ref, np.arange(len(test)))
            boot = pd.DataFrame([seed_delta(test, scores, seeds, exp, ref, i) for i in index])
            for k in KEYS:
                lo, hi = np.percentile(boot[k], [2.5, 97.5])
                rows.append(
                    {
                        "condition": conditions[exp],
                        "reference": conditions[ref],
                        "metric": k,
                        "delta": point[k],
                        "lo": lo,
                        "hi": hi,
                    }
                )
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dose", choices=["base", "strong"], default="base")
    dose = parser.parse_args().dose
    conditions, references = (CONDITIONS, REFERENCES) if dose == "base" else (STRONG, STRONG_REFERENCES)
    suffix = "" if dose == "base" else "_strong"
    val, test = pd.read_csv(SPLITS / "val.csv"), pd.read_csv(SPLITS / "test.csv")
    write_counterfactual(test)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    table = runs(conditions)
    b, y = minority(test), test.target.to_numpy()
    rows, scores = [], {}
    for r in table.itertuples():
        logits = load_cache(CACHE / f"{r.run_dir.name}_test.npz", test.image.to_numpy(str))["logits"]
        s = logits[:, 1] - logits[:, 0]
        scores[(r.exp, r.seed)] = s
        cf = counterfactual_logits(r.run_dir, device)
        effect = cf["on"] - cf["off"]
        row = {"exp": r.exp, "condition": conditions[r.exp], "seed": r.seed, **metrics(test, s)}
        row.update({"marker_effect_A": effect[~b].mean(), "marker_effect_B": effect[b].mean()})
        row.update(
            {
                f"marker_effect_{side}_y{c}": effect[(b == (side == "B")) & (y == c)].mean()
                for side in "AB"
                for c in (0, 1)
            }
        )
        rows.append(row)
    summary = pd.DataFrame(rows)
    dest = PACKAGE / "results" / "ham10000_e2e"
    summary.to_csv(dest / f"rule_summary{suffix}.csv", index=False)
    cols = KEYS + ["marker_effect_A", "marker_effect_B"]
    print(summary.groupby("condition")[cols].agg(["mean", "std"]).round(3).to_string())
    seeds = summary.groupby("exp").seed.apply(list).to_dict()
    frame = contrast(test, scores, seeds, conditions, references)
    frame.to_csv(dest / f"rule_contrast{suffix}.csv", index=False)
    print(frame.round(3).to_string(index=False))
    ref = ceiling(val, test, dose)
    ref.to_csv(dest / f"rule_ceiling{suffix}.csv", index=False)
    print(ref.groupby("reference")[KEYS].mean().round(3).to_string())


if __name__ == "__main__":
    main()
