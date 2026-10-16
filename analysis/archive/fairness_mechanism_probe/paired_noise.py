"""既存の主比較ペアで、群別 AUROC 差の患者単位 paired bootstrap 標準誤差を求める。

問い：この test 集合（19,617 件、≥65×Others の陽性 68 件）で、条件間の worst-group AUROC 差は
どの大きさから標本雑音と区別できるか。3 seed の標準偏差は同じ test 集合での学習のばらつきで、
test 標本の雑音を含まない。

入力は `race_known_architecture_objective_comparison/cache/` の予測（3 seed の logit 平均）。
出力: `results/paired_noise/paired_bootstrap.csv`

    uv run python analysis/archive/fairness_mechanism_probe/paired_noise.py
"""

from __future__ import annotations

import glob

import numpy as np
import pandas as pd
from features import GROUP_NAMES, PACKAGE, REPOSITORY_ROOT, SPLITS, group_ids
from sklearn.metrics import roc_auc_score

COMPARISON_CACHE = REPOSITORY_ROOT / "analysis/archive/race_known_architecture_objective_comparison/cache"
CONDITIONS = {
    "M01 ResNet ERM": "resnet-chexpert-race-known-s4",
    "M02 ResNet DRO": "resnet-chexpert-age-race-group-dro-race-known-s4",
    "M03 Spatial ERM": "spatial-lora-chexpert-race-known-s4",
    "M04 Spatial DRO": "spatial-lora-chexpert-age-race-group-dro-race-known-s4",
    "M06 Spatial 2-stage DRO": "spatial-lora-chexpert-from-resnet-age-race-group-dro-race-known-s4",
    "M07 HyperAdapt ERM": "hyperadapt-chexpert-race-known-s4",
}
PAIRS = [
    ("M01 ResNet ERM", "M02 ResNet DRO"),
    ("M01 ResNet ERM", "M03 Spatial ERM"),
    ("M04 Spatial DRO", "M06 Spatial 2-stage DRO"),
    ("M01 ResNet ERM", "M07 HyperAdapt ERM"),
]


def mean_score(stem: str) -> np.ndarray:
    files = sorted(f for f in glob.glob(str(COMPARISON_CACHE / f"*-{stem}?-*_test.npz")) if "probe" not in f)
    if len(files) != 3:
        raise ValueError(f"{stem}: {len(files)} files")
    return np.mean([np.load(f)["logits"][:, 1] - np.load(f)["logits"][:, 0] for f in files], axis=0)


def main(n_boot: int = 300) -> None:
    frame = pd.read_csv(SPLITS / "test.csv")
    y, g = frame.target.to_numpy(), group_ids(frame)
    scores = {name: mean_score(stem) for name, stem in CONDITIONS.items()}
    patient = frame.image.str.split("/").str[1].to_numpy()
    _, inverse = np.unique(patient, return_inverse=True)
    members = pd.Series(np.arange(len(frame))).groupby(inverse).apply(np.array).to_numpy()
    rng = np.random.default_rng(0)
    rows = []
    for a, b in PAIRS:
        diffs = np.zeros((n_boot, 9))
        for i in range(n_boot):
            idx = np.concatenate(members[rng.integers(0, len(members), len(members))])
            per = []
            for k in range(8):
                m = g[idx] == k
                per.append(roc_auc_score(y[idx][m], scores[b][idx][m]) - roc_auc_score(y[idx][m], scores[a][idx][m]))
            worst_b = min(roc_auc_score(y[idx][g[idx] == k], scores[b][idx][g[idx] == k]) for k in range(8))
            worst_a = min(roc_auc_score(y[idx][g[idx] == k], scores[a][idx][g[idx] == k]) for k in range(8))
            diffs[i] = per + [worst_b - worst_a]
        point = [
            roc_auc_score(y[g == k], scores[b][g == k]) - roc_auc_score(y[g == k], scores[a][g == k]) for k in range(8)
        ]
        for k, name in enumerate(list(GROUP_NAMES) + ["worst"]):
            rows.append(
                dict(pair=f"{a} → {b}", cell=name, delta=point[k] if k < 8 else np.nan, paired_se=diffs[:, k].std())
            )
    table = pd.DataFrame(rows)
    out = PACKAGE / "results" / "paired_noise"
    out.mkdir(parents=True, exist_ok=True)
    table.to_csv(out / "paired_bootstrap.csv", index=False)
    print(table.pivot(index="cell", columns="pair", values="paired_se").round(4).to_string())


if __name__ == "__main__":
    main()
