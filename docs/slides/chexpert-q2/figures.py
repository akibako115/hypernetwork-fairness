"""CheXpert Q2（公平性損失 × 学習手順）のスライドの図を生成する。

入力:
- analysis/archive/latent_bias_hypernet/results/q2_fairness_loss.csv（最悪群 AUROC・gap・Eopp1・Eopp0・全体 BAcc・BAcc gap の値と ERM との差）
- analysis/archive/latent_bias_hypernet/results/q2_threshold_equivalence.csv（全体 AUROC の値と 95% 区間）
- analysis/archive/latent_bias_hypernet/results/q2_auroc_decomposition.csv（全体 AUROC の ERM との差と 95% 区間）
出力: figures/*.png（図のラベル・凡例・タイトルは英語）
- q2_values.png: 手順 × 損失ごとの値と 95% 区間
- q2_diffs.png: 同じ手順の ERM との差と 95% 区間
実行例: uv run python docs/slides/chexpert-q2/figures.py
"""

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
RES = ROOT / "analysis/archive/latent_bias_hypernet/results"
OUT = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).parent / "figures"
OUT.mkdir(exist_ok=True)

plt.rcParams.update(
    {
        "font.family": "Noto Sans CJK JP",
        "font.size": 9,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.titlesize": 10,
        "axes.titleweight": "bold",
        "legend.frameon": False,
    }
)
THEME, GREY, INK, ACCENT = "#1F3B57", "#A9B4BF", "#222222", "#C0504D"
PROCS = [("onestage", "One-stage"), ("insample", "Two-stage, in-sample"), ("crossfit", "Two-stage, cross-fit")]
LOSSES = [("erm", "ERM", "#8C99A6"), ("eo", "EO rate", THEME), ("minimax_auc", "minimax AUC", ACCENT)]
# (列名, パネルの題, 悪い向き)
PANELS = [
    ("overall_auroc", "Overall AUROC", "← worse"),
    ("overall_bacc", "Overall BAcc", "← worse"),
    ("worst_auroc", "Worst-group AUROC", "← worse"),
    ("auroc_gap", "AUROC gap", "worse →"),
    ("bacc_gap", "BAcc gap", "worse →"),
    ("eopp1", "Eopp1 (TPR gap)", "worse →"),
    ("eopp0", "Eopp0 (FPR gap)", "worse →"),
]
GROUPS_NOTE = "Groups: age (<65 / ≥65) × race, 8 groups. Threshold: val Youden. 95% CI: patient-level bootstrap, 3 seeds."


def save(fig, name: str) -> None:
    fig.savefig(OUT / f"{name}.png", dpi=250, bbox_inches="tight")
    plt.close(fig)


def style_panel(ax, title: str, note: str) -> None:
    ax.set_title(f"{title}\n", loc="center")
    ax.text(0.5, 1.0, note, transform=ax.transAxes, ha="center", va="bottom", fontsize=8, color="#666666")
    ax.grid(axis="x", color="#EEEEEE", lw=0.6)
    ax.set_axisbelow(True)
    ax.spines["left"].set_visible(False)
    ax.tick_params(axis="y", length=0)


def row_positions(per_proc: int) -> list[float]:
    """手順ごとに per_proc 行、手順の間に 0.4 の隙間。"""
    return [i + 0.4 * (i // per_proc) for i in range(per_proc * len(PROCS))]


def separators(axes, per_proc: int) -> None:
    for k in range(1, len(PROCS)):
        for ax in axes:
            ax.axhline(k * per_proc + 0.4 * (k - 1) - 0.3, color="#DDDDDD", lw=0.8)


def q2_values() -> None:
    """手順 × 損失ごとの値と 95% 区間。"""
    metrics = pd.read_csv(RES / "q2_fairness_loss.csv").set_index("condition")
    overall = pd.read_csv(RES / "q2_threshold_equivalence.csv").set_index(["procedure", "rule"])
    ys = row_positions(len(LOSSES))
    fig, axes = plt.subplots(1, len(PANELS), figsize=(14.5, 3.3), sharey=True)
    labels = []
    rows = [(p, pname, loss, lname, color) for p, pname in PROCS for loss, lname, color in LOSSES]
    for (procedure, pname, loss, lname, color), yi in zip(rows, ys, strict=True):
        labels.append(f"{pname}: {lname}" if loss == "erm" else lname)
        for ax, (metric, *_) in zip(axes, PANELS, strict=True):
            r = overall.loc[(procedure, loss)] if metric == "overall_auroc" else metrics.loc[f"{procedure}:{loss}:auroc"]
            ax.plot([r[f"{metric}_lo"], r[f"{metric}_hi"]], [yi, yi], color=color, lw=2, solid_capstyle="round")
            ax.plot(r[metric], yi, "o", color=color, ms=6)
    separators(axes, len(LOSSES))
    for ax, (_, title, note) in zip(axes, PANELS, strict=True):
        style_panel(ax, title, note)
    axes[0].set_yticks(ys, labels)
    axes[0].invert_yaxis()
    fig.text(0.5, -0.01, GROUPS_NOTE, ha="center", fontsize=7.5, color="#666666")
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    save(fig, "q2_values")


def q2_diffs() -> None:
    """公平性損失の、同じ手順の ERM との差と 95% 区間。区間が 0 を含まないものを濃くする。"""
    metrics = pd.read_csv(RES / "q2_fairness_loss.csv").set_index("condition")
    decomposition = pd.read_csv(RES / "q2_auroc_decomposition.csv").set_index(["procedure", "loss"])
    losses = [(loss, lname) for loss, lname, _ in LOSSES if loss != "erm"]
    ys = row_positions(len(losses))
    rows = [(p, pname, loss, lname) for p, pname in PROCS for loss, lname in losses]
    fig, axes = plt.subplots(1, len(PANELS), figsize=(14.5, 2.9), sharey=True)
    for ax, (metric, *_) in zip(axes, PANELS, strict=True):
        ax.axvline(0, color=INK, lw=0.9, ls="--", zorder=1)
        for (procedure, _, loss, _), yi in zip(rows, ys, strict=True):
            if metric == "overall_auroc":
                r = decomposition.loc[(procedure, loss)]
                d, lo, hi = r["overall_diff"], r["overall_diff_lo"], r["overall_diff_hi"]
            else:
                r = metrics.loc[f"{procedure}:{loss}:auroc"]
                d, lo, hi = r[f"{metric}_diff"], r[f"{metric}_diff_lo"], r[f"{metric}_diff_hi"]
            color = THEME if lo > 0 or hi < 0 else GREY
            ax.plot([lo, hi], [yi, yi], color=color, lw=2, solid_capstyle="round", zorder=2)
            ax.plot(d, yi, "o", color=color, ms=6, zorder=3)
    separators(axes, len(losses))
    for ax, (_, title, note) in zip(axes, PANELS, strict=True):
        style_panel(ax, f"Δ {title}", note)
    axes[0].set_yticks(ys, [f"{pname}: {lname}" for _, pname, _, lname in rows])
    axes[0].invert_yaxis()
    handles = [
        plt.Line2D([], [], color=INK, ls="--", lw=0.9, label="ERM of the same procedure"),
        plt.Line2D([], [], color=THEME, marker="o", lw=2, label="95% CI of difference excludes 0"),
        plt.Line2D([], [], color=GREY, marker="o", lw=2, label="includes 0"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=3, fontsize=8, bbox_to_anchor=(0.5, -0.02))
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    save(fig, "q2_diffs")


q2_values()
q2_diffs()
