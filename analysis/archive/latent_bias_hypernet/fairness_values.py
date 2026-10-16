"""公平性手法の比較（`fairness_eval.py` と同じ run）の、各条件の値の 95% 区間を求める（HAM10000）。

`fairness_eval.py` は ERM との差の区間だけを書く。図で値そのものを並べるために、seed 平均の値に
病変単位の bootstrap（500 回、差と同じ標本）で区間をつける。選び方は val の全体 AUROC だけ
（SWAD は平均した重み 1 つ）。

入力: `fairness_eval.py` と同じ予測 cache
出力: `results/ham10000_e2e/fairness_values.csv`（condition ごとに KEYS の値と `_lo`・`_hi`）

    uv run python -m analysis.archive.latent_bias_hypernet.fairness_values
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from analysis.archive.latent_bias_hypernet.data import PACKAGE, ROOT
from analysis.archive.latent_bias_hypernet.e2e_eval import bootstrap_index, metrics, runs, thresholds
from analysis.archive.latent_bias_hypernet.fairness_eval import KEYS, load_scores, names


def main() -> None:
    labels = names()
    table = runs(labels)
    splits = ROOT / "data" / "ham10000" / "splits_lhcf"
    val, test = pd.read_csv(splits / "val.csv"), pd.read_csv(splits / "test.csv")
    sv, st = load_scores(table, val, test)
    thr = thresholds(val, sv)
    seeds_of = table.groupby("exp").seed.apply(list).to_dict()
    boots = bootstrap_index(test, "ham10000")

    def seed_mean(exp, idx):
        part = test.iloc[idx].reset_index(drop=True)
        key = f"{exp}|overall"
        vals = [metrics("ham10000", part, st[(key, s)][idx], thr[(key, s)]) for s in seeds_of[exp]]
        return {k: np.mean([v[k] for v in vals]) for k in KEYS}

    rows = []
    for exp in seeds_of:
        row = {"exp": exp, "condition": labels[exp], **seed_mean(exp, np.arange(len(test)))}
        draws = {k: [] for k in KEYS}
        for b in boots:
            try:
                cur = seed_mean(exp, b)
            except ValueError:
                continue
            for k in KEYS:
                draws[k].append(cur[k])
        for k in KEYS:
            row[f"{k}_lo"], row[f"{k}_hi"] = np.percentile(draws[k], [2.5, 97.5])
        rows.append(row)
        print(exp, flush=True)
    out = pd.DataFrame(rows)
    out.to_csv(PACKAGE / "results" / "ham10000_e2e" / "fairness_values.csv", index=False)
    print(out.set_index("condition")[KEYS].round(3).to_string())


if __name__ == "__main__":
    main()
