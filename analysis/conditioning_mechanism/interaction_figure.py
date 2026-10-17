"""セル内 AUROC が入力属性でどう動くかを 2 枚の図にする。

1. `age_curve.png`：入力年齢 t ごとのセル内 AUROC。セルの年齢群の代表値（45・75 歳）での値を 0 とした差で描く。
   切片だけに効くモデルなら全セルが水平線になる。狙いどおり「年齢で読み方が変わる」なら、
   65 歳以上のセルは右上がり、65 歳未満のセルは右下がりになる。
2. `contrasts.png`：セル内 AUROC の差（正しい属性 − 反対・他の属性）と 95% 区間。

入力: `results/age_curve.csv`・`results/contrasts.csv`（`interaction_summary.py` の出力）
出力: `figures/age_curve.png`・`figures/contrasts.png`

    uv run python -m analysis.conditioning_mechanism.interaction_figure
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rootutils

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from analysis.common.paths import STYLE_SHEET  # noqa: E402
from analysis.conditioning_mechanism.interaction_counterfactual import AGE_OLD, AGE_YOUNG, PACKAGE  # noqa: E402

DATASETS = {"ham10000": "HAM10000 (cells: sex × age 65)", "chexpert": "CheXpert (cells: race × age 65)"}
MODELS = {
    "spatial_lora": "Spatial LoRA (stage3–4 + fc)",
    "spatial_lora_fc": "Spatial LoRA (fc only)",
    "hyperadapt": "HyperAdapt (stage1–4 + fc)",
}
CONTRASTS = {
    "age_matched_minus_flipped": "Input age: own age group − opposite age group",
    "cell_attr_matched_minus_other": "Input sex (HAM) / race (CheXpert): own − others",
}


def plot_age_curve(curve: pd.DataFrame) -> plt.Figure:
    """データセット × モデルの格子に、セルごとの年齢曲線を描く。"""
    curve = curve[curve.variant != "observed"].copy()
    panels = [(d, m) for d in DATASETS for m in MODELS if ((curve.dataset == d) & (curve.model == m)).any()]
    fig, axes = plt.subplots(1, len(panels), figsize=(4.2 * len(panels), 3.8), sharey=True, squeeze=False)
    for ax, (dataset, model) in zip(axes[0], panels, strict=True):
        sub = curve[(curve.dataset == dataset) & (curve.model == model)]
        for cell, rows in sub.groupby("cell"):
            old = cell.endswith("≥65")
            anchor = rows.loc[rows.input_age == (AGE_OLD if old else AGE_YOUNG), "auroc"].iloc[0]
            ax.plot(
                rows.input_age,
                rows.auroc - anchor,
                marker="o",
                ms=3,
                ls="-" if old else "--",
                color="#c0392b" if old else "#2471a3",
                alpha=0.8,
                label=cell,
            )
        ax.axhline(0, color="#888", lw=0.8)
        ax.axvline(65, color="#bbb", lw=0.8, ls=":")
        ax.set_title(f"{DATASETS[dataset].split(' (')[0]}\n{MODELS[model]}", fontsize=9)
        ax.set_xlabel("Age given to the hypernetwork (years)")
        ax.legend(fontsize=6, ncol=2, loc="lower center")
    axes[0][0].set_ylabel("Within-cell AUROC\n(relative to own-group age 45 / 75)")
    fig.suptitle(
        "Within-cell AUROC when every image in the cell gets the same input age\n"
        "Flat = attributes only shift the score; solid = age ≥65 cells, dashed = <65 cells",
        fontsize=10,
    )
    fig.tight_layout()
    return fig


def plot_contrasts(table: pd.DataFrame) -> plt.Figure:
    """対比ごとに、セル × モデルの差と 95% 区間を横棒で描く。"""
    fig, axes = plt.subplots(len(DATASETS), len(CONTRASTS), figsize=(11, 8.5), squeeze=False)
    colors = dict(zip(MODELS, ["#2a6f97", "#e07a5f", "#6a994e"], strict=True))
    for i, dataset in enumerate(DATASETS):
        for j, (contrast, title) in enumerate(CONTRASTS.items()):
            ax = axes[i][j]
            sub = table[(table.dataset == dataset) & (table.contrast == contrast)]
            cells = [c for c in sorted(sub.cell.unique()) if c != "all cells"] + ["all cells"]
            models = [m for m in MODELS if (sub.model == m).any()]
            for k, model in enumerate(models):
                rows = sub[sub.model == model].set_index("cell").loc[cells]
                y = np.arange(len(cells)) + (k - (len(models) - 1) / 2) * 0.25
                ax.errorbar(
                    rows.delta,
                    y,
                    xerr=[rows.delta - rows.low, rows.high - rows.delta],
                    fmt="o",
                    ms=3,
                    color=colors[model],
                    label=MODELS[model],
                    capsize=2,
                )
            ax.axvline(0, color="#888", lw=0.8)
            ax.set_yticks(np.arange(len(cells)), cells, fontsize=8)
            ax.invert_yaxis()
            ax.set_title(f"{DATASETS[dataset]}\n{title}", fontsize=9)
            ax.set_xlabel("Δ within-cell AUROC (3-seed mean, 95% bootstrap over patients/lesions)")
            if i == 0 and j == 0:
                ax.legend(fontsize=7, loc="lower right")
    fig.suptitle("Does giving the correct attribute improve ranking within the same cell?  (> 0 = yes)", fontsize=11)
    fig.tight_layout()
    return fig


def main() -> None:
    """図を figures/ に書く。"""
    plt.style.use(STYLE_SHEET)
    results, figures = PACKAGE / "results", PACKAGE / "figures"
    figures.mkdir(exist_ok=True)
    plot_age_curve(pd.read_csv(results / "age_curve.csv")).savefig(figures / "age_curve.png", dpi=150)
    plot_contrasts(pd.read_csv(results / "contrasts.csv")).savefig(figures / "contrasts.png", dpi=150)


if __name__ == "__main__":
    main()
