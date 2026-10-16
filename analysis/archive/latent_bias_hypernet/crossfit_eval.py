"""2 段学習（cross-fit）の 2 段目を、条件・損失・区分の作り直しの周ごとに test で比べる（HAM10000）。

場は `crossfit.py` と同じ 2 つ。
- `real`（植え込みなし）：1 段目そのもの、2 段目の定数・LHCF のコホートで条件付け（それぞれ ERM と ℓ_wst）、
  誤りの向きで作り直した区分 iter1〜3。指標は `e2e_eval.metrics`（全体・external・性別 × 年齢・GMM 8・LHCF の
  コホートの最悪 AUROC など）。しきい値は val の Youden
- `rule`（規則の食い違いを強めた版）：1 段目、2 段目の定数・GMM 8（反転の境界を含む）・ランダム 8・iter1〜3。
  E2E の 3 条件（`rule_eval.py --dose strong`）も同じ表に並べる。指標は `rule_eval` と同じ AUROC（全体・A・B）と
  マーカーの反実仮想の効果
区間は、基準（2 段目の定数、`rule` では E2E の定数も）との seed 平均の差を、病変単位の対応のある bootstrap で求める。
あわせて、区分の性質（`crossfit.py` が書いた `crossfit_partitions.csv`）を周ごとに seed 平均して表示する。

入力: `runs/` の cross-fit run、`cache/<run-id>_{val,test}.npz`（`analysis/common/predictions.py`）、
`data/ham10000/splits_lhcf/`、`results/ham10000_e2e/crossfit_partitions.csv`
出力: `results/ham10000_e2e/crossfit_{real,rule}_{summary,contrast}.csv`

足りない予測 cache は、作るコマンドを表示して止まる。

    uv run python -m analysis.archive.latent_bias_hypernet.crossfit_eval --field real
    uv run python -m analysis.archive.latent_bias_hypernet.crossfit_eval --field rule
"""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd
import torch

from analysis.archive.latent_bias_hypernet import rule_eval
from analysis.archive.latent_bias_hypernet.data import PACKAGE, ROOT
from analysis.archive.latent_bias_hypernet.e2e_eval import contrasts, metrics, runs, scores, thresholds
from analysis.archive.latent_bias_hypernet.rule_planting import minority

ITER = {f"iter{n}": f"誤りの向きで作り直した区分（{n} 周目）" for n in (1, 2, 3)}
REAL = {
    "resnet-ham10000-cf-stage1": "1 段目（ResNet-50 ERM、train の半分）",
    "spatial-lora-ham10000-cf-const": "2 段目：定数",
    "spatial-lora-ham10000-cf-const-wst": "2 段目：定数 ＋ 最悪コホート損失",
    "spatial-lora-ham10000-cf-lhcf": "2 段目：LHCF のコホートで条件付け",
    "spatial-lora-ham10000-cf-lhcf-wst": "2 段目：LHCF のコホートで条件付け ＋ 最悪コホート損失",
} | {f"spatial-lora-ham10000-cf-{k}": f"2 段目：{v}" for k, v in ITER.items()}
REAL_REFERENCES = ("spatial-lora-ham10000-cf-const", "resnet-ham10000-cf-stage1")
RULE = (
    {
        "resnet-ham10000-rule-strong-cf-stage1": "1 段目（ResNet-50 ERM、train の半分）",
        "spatial-lora-ham10000-rule-strong-cf-const": "2 段目：定数",
        "spatial-lora-ham10000-rule-strong-cf-gmm8": "2 段目：GMM 8（反転の境界を含む）",
        "spatial-lora-ham10000-rule-strong-cf-random8": "2 段目：ランダム 8 区分",
    }
    | {f"spatial-lora-ham10000-rule-strong-cf-{k}": f"2 段目：{v}" for k, v in ITER.items()}
    | {k: f"E2E：{v.removeprefix('Spatial LoRA')}" for k, v in rule_eval.STRONG.items()}
)
RULE_REFERENCES = ("spatial-lora-ham10000-rule-strong-cf-const", "spatial-lora-ham10000-rule-strong-const")
REAL_KEYS = ["auroc_all", "auroc_external", "sexage_min", "sexage_gap", "gmm8_min", "lhcf_min"]
DEST = PACKAGE / "results" / "ham10000_e2e"


def partitions(field: str) -> None:
    path = DEST / "crossfit_partitions.csv"
    if not path.exists():
        return
    table = pd.read_csv(path)
    table = table[(table.field == field) & (table.split == "test")]
    print("区分の性質（test、seed 平均）")
    print(table.groupby("name").mean(numeric_only=True).drop(columns="seed").round(3).to_string())


def real() -> None:
    table = runs(REAL)
    splits = ROOT / "data" / "ham10000" / "splits_lhcf"
    val, test = pd.read_csv(splits / "val.csv"), pd.read_csv(splits / "test.csv")
    sv, st = scores(val, "val", table), scores(test, "test", table)
    thr = thresholds(val, sv)
    rows = [
        {"exp": exp, "condition": REAL[exp], "seed": seed, **metrics("ham10000", test, s, thr[(exp, seed)])}
        for (exp, seed), s in st.items()
    ]
    summary = pd.DataFrame(rows)
    summary.to_csv(DEST / "crossfit_real_summary.csv", index=False)
    print(summary.groupby("condition", sort=False)[REAL_KEYS].agg(["mean", "std"]).round(3).to_string())
    seeds_of = summary.groupby("exp").seed.apply(list).to_dict()
    frame = pd.concat(
        [
            contrasts("ham10000", test, st, thr, seeds_of, REAL, ref, REAL_KEYS).assign(reference=REAL[ref])
            for ref in REAL_REFERENCES
        ],
        ignore_index=True,
    )
    frame.to_csv(DEST / "crossfit_real_contrast.csv", index=False)
    ci = frame.assign(ci=frame.apply(lambda r: f"{r.delta:+.3f} [{r.lo:+.3f}, {r.hi:+.3f}]", axis=1))
    print(ci.pivot(index=["reference", "condition"], columns="metric", values="ci").to_string())
    partitions("real")


def rule() -> None:
    test = pd.read_csv(rule_eval.SPLITS / "test.csv")
    rule_eval.write_counterfactual(test)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    table = runs(RULE)
    raw = scores(test, "test", table)
    b, y = minority(test), test.target.to_numpy()
    rows, logit = [], {}
    for r in table.itertuples():
        p = np.clip(raw[(r.exp, r.seed)], 1e-7, 1 - 1e-7)
        logit[(r.exp, r.seed)] = np.log(p / (1 - p))
        cf = rule_eval.counterfactual_logits(r.run_dir, device)
        effect = cf["on"] - cf["off"]
        row = {
            "exp": r.exp,
            "condition": RULE[r.exp],
            "seed": r.seed,
            **rule_eval.metrics(test, logit[(r.exp, r.seed)]),
        }
        row.update({"marker_effect_A": effect[~b].mean(), "marker_effect_B": effect[b].mean()})
        row.update({f"marker_effect_B_y{c}": effect[b & (y == c)].mean() for c in (0, 1)})
        rows.append(row)
    summary = pd.DataFrame(rows)
    summary.to_csv(DEST / "crossfit_rule_summary.csv", index=False)
    cols = rule_eval.KEYS + ["marker_effect_A", "marker_effect_B"]
    print(summary.groupby("condition", sort=False)[cols].agg(["mean", "std"]).round(3).to_string())
    seeds = summary.groupby("exp").seed.apply(list).to_dict()
    references = [ref for ref in RULE_REFERENCES if ref in seeds]
    frame = rule_eval.contrast(test, logit, seeds, {k: v for k, v in RULE.items() if k in seeds}, references)
    frame.to_csv(DEST / "crossfit_rule_contrast.csv", index=False)
    print(frame.round(3).to_string(index=False))
    partitions("rule")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--field", choices=["real", "rule"], required=True)
    {"real": real, "rule": rule}[parser.parse_args().field]()


if __name__ == "__main__":
    main()
