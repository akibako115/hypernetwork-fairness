"""bgrid.py の結果を、偏りの量 × 大きさ × 手法ごとに seed 平均（± SD）で表示する。

uv run python -m analysis.archive.latent_bias_hypernet.summarize_bgrid pacer
"""

from __future__ import annotations

import sys

import pandas as pd

from analysis.archive.latent_bias_hypernet.data import PACKAGE

METRICS = {
    "pacer": ["auroc", "S_pos_auroc", "S_log_oe", "detect"],
    "marker": ["auroc", "shortcut", "auroc_marked", "detect"],
}
SIZES = ["1000", "5000", "20000", "all"]


def main() -> None:
    plant = sys.argv[1]
    tag = sys.argv[2] if len(sys.argv) > 2 else ""
    t = pd.read_csv(PACKAGE / "results" / "bgrid" / f"{plant}{('_' + tag) if tag else ''}.csv")
    t["size"] = pd.Categorical(t["size"].astype(str), SIZES, ordered=True)
    methods = list(dict.fromkeys(t.method))
    t["method"] = pd.Categorical(t.method, methods, ordered=True)
    for metric in METRICS[plant]:
        mean = t.pivot_table(index=["bias", "method"], columns="size", values=metric, aggfunc="mean", observed=True)
        sd = t.pivot_table(index=["bias", "method"], columns="size", values=metric, aggfunc="std", observed=True)
        print(f"===== {metric}（seed 平均、括弧内は SD）")
        cell = mean.round(3).astype(str) + " (" + sd.round(3).astype(str) + ")"
        with pd.option_context("display.width", 250, "display.max_rows", 200):
            print(cell.to_string())


if __name__ == "__main__":
    main()
