"""audit_corrected の結果を条件ごと（seed 平均）にまとめて表示する。

uv run python -m analysis.archive.latent_bias_hypernet.summarize_corrected im_a_s42
"""

from __future__ import annotations

import sys

import pandas as pd

from analysis.archive.latent_bias_hypernet.data import PACKAGE

COLS = [
    "auroc",
    "demo32_max",
    "demo32_worst_auroc",
    "demo96_sd",
    "context_sd",
    "aud_he32_sd",
    "aud_e32_sd",
    "aud_d32_sd",
    "aud_d32_max",
    "aud_d32_min",
    "S_log_oe",
    "S_tpr",
    "S_auroc",
]


def table(tag: str) -> pd.DataFrame:
    t = pd.read_csv(PACKAGE / "results" / "audit_corrected" / f"{tag}.csv")
    t["condition"] = t.run.str.replace(r"^im_[a-z0-9_]+?_s42_", "", regex=True).str.replace(r"_s\d$", "", regex=True)
    cols = [c for c in COLS if c in t.columns]
    mean = t.groupby("condition", sort=False)[cols].mean()
    sd = t.groupby("condition", sort=False)[cols].std()
    return mean, sd


if __name__ == "__main__":
    mean, sd = table(sys.argv[1])
    with pd.option_context("display.width", 250, "display.max_columns", 30):
        print(mean.round(3).to_string())
        print("--- seed SD")
        print(sd.round(3).to_string())
