"""属性の群の粒度ごとに、最小 AUROC がどれだけ信頼できるかを調べる。

粒度ごとに次を求める（植え込みなしの 1 段目、test）。
- 評価できる群の数（n ≥ 100、陽性・陰性とも 20 以上）と、最小 AUROC の群の陽性数
- 最小 AUROC の患者単位 bootstrap SD
- 勝者の呪い：test の患者を半分に分け、片方で最悪の群を選び、もう片方でその群の AUROC を測る。
  選んだ側の最小値と、もう片方での値の差（200 回の平均）が、選ぶことによる下振れ

出力: `results/granularity/<stage1>.csv`

    uv run python -m analysis.archive.latent_bias_hypernet.granularity --stage1 a_s42
"""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from analysis.archive.latent_bias_hypernet.data import PACKAGE, load
from analysis.archive.latent_bias_hypernet.eval_groups import MIN_CLASS, MIN_N, groupings
from analysis.archive.latent_bias_hypernet.stats import patient_index


def group_aurocs(y, s, g, rows=None) -> dict:
    if rows is not None:
        y, s, g = y[rows], s[rows], g[rows]
    out = {}
    for k in np.unique(g):
        m = g == k
        pos = int(y[m].sum())
        if m.sum() >= MIN_N and pos >= MIN_CLASS and m.sum() - pos >= MIN_CLASS:
            out[k] = (roc_auc_score(y[m], s[m]), pos)
    return out


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage1", default="a_s42")
    parser.add_argument("--reps", type=int, default=200)
    args = parser.parse_args(argv)
    T = load(args.stage1, "test")
    y, s = T.target, T.base
    patients = patient_index(T.context.image)
    rows_of = pd.Series(np.arange(len(y))).groupby(patients).apply(np.asarray).to_numpy()
    n_pat = len(rows_of)
    rng = np.random.default_rng(0)
    out = []
    for name, g in groupings(T.context).items():
        full = group_aurocs(y, s, g)
        worst = min(full, key=lambda k: full[k][0])
        boot = []
        for _ in range(args.reps):
            idx = np.concatenate(rows_of[rng.integers(0, n_pat, n_pat)])
            a = group_aurocs(y, s, g, idx)
            boot.append(min(v[0] for v in a.values()))
        curse = []
        for _ in range(args.reps):
            perm = rng.permutation(n_pat)
            half_a = np.concatenate(rows_of[perm[: n_pat // 2]])
            half_b = np.concatenate(rows_of[perm[n_pat // 2 :]])
            a, b = group_aurocs(y, s, g, half_a), group_aurocs(y, s, g, half_b)
            common = [k for k in a if k in b]
            k = min(common, key=lambda k: a[k][0])
            curse.append(b[k][0] - a[k][0])
        out.append(
            {
                "grouping": name,
                "groups": len(full),
                "min_pos_over_groups": min(v[1] for v in full.values()),
                "min_auroc": full[worst][0],
                "worst_group_pos": full[worst][1],
                "boot_sd_min_auroc": float(np.std(boot)),
                "winners_curse": float(np.mean(curse)),
            }
        )
    table = pd.DataFrame(out)
    dest = PACKAGE / "results" / "granularity"
    dest.mkdir(parents=True, exist_ok=True)
    table.to_csv(dest / f"{args.stage1}.csv", index=False)
    print(table.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
