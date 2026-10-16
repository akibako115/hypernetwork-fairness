"""`results/latent_codes/*/summary.json` を 1 表にまとめる。

出力: `results/latent_codes_summary.csv`

    uv run python -m analysis.archive.latent_bias_hypernet.summarize [name の部分文字列 ...]
"""

from __future__ import annotations

import json
import sys

import pandas as pd

from analysis.archive.latent_bias_hypernet.data import PACKAGE

ARGS = ("stage1", "fit", "assign", "route", "K", "rank", "lam", "beta", "seed", "select")
PERF = ("loss", "auroc", "worst8", "mean8", "auroc_old", "auroc_pos_focused_followup", "auroc_marked", "fpr0_marked")


def collect() -> pd.DataFrame:
    rows = []
    for path in sorted((PACKAGE / "results" / "latent_codes").glob("*/summary.json")):
        s = json.loads(path.read_text())
        row = {"name": path.parent.name, **{k: s["args"].get(k, "val") for k in ARGS}, "epoch": s["best_epoch"]}
        row["codes"] = s["codes_used"]
        for k in PERF:
            if k in s["corrected"]:
                row[f"d_{k}"] = s["corrected"][k] - s["base"][k]
                row[k] = s["corrected"][k]
        for k, v in s["recovery"].items():
            row[k] = v
        rows.append(row)
    return pd.DataFrame(rows)


def main() -> None:
    table = collect()
    table.to_csv(PACKAGE / "results" / "latent_codes_summary.csv", index=False)
    if len(sys.argv) > 1:
        table = table[table.name.str.contains("|".join(sys.argv[1:]))]
    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 40)
    print(table.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
