"""HAM10000 で、属性で条件付けた hypernet（Spatial LoRA）が全体性能を底上げし、公平性指標は変えないことを示す図を生成する。

入力: analysis/archive/latent_bias_hypernet/results/ham10000_e2e/architecture_groups.csv（architecture_groups.py）
出力: figures/*.png（ResNet-50・Spatial LoRA の定数・属性を並べる。図のラベル・凡例・タイトルは英語）
- hypernet_overall.png: 全体の AUROC と BAcc
- hypernet_fairness.png: 性別 × 年齢 4 群の AUROC gap・Eopp1・Eopp0
- hypernet_groups.png: 性別 × 年齢 4 群ごとの AUROC・TPR・FPR の棒グラフ（AUROC の棒は偶然の 0.5 から）
点の横の数字は ResNet-50 との差で、95% 区間が 0 を含まないものを太字にする。
実行例: uv run python docs/slides/ham10000-fairness-methods/hypernet_gain.py
"""

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
RES = ROOT / "analysis/archive/latent_bias_hypernet/results/ham10000_e2e"
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
THEME, ACCENT, BASE = "#1F3B57", "#C0504D", "#8C99A6"
MODELS = [
    ("resnet-ham10000", "ResNet-50", BASE),
    ("spatial-lora-ham10000-const", "Spatial LoRA, constant condition", THEME),
    ("spatial-lora-ham10000", "Spatial LoRA, conditioned on attributes", ACCENT),
]
# (列名, パネルの題, 悪い向き)
OVERALL = [("overall_auroc", "Overall AUROC", "← worse"), ("overall_bacc", "Overall BAcc", "← worse")]
FAIRNESS = [
    ("auroc_gap", "AUROC gap", "worse →"),
    ("eopp1", "Eopp1 (TPR gap)", "worse →"),
    ("eopp0", "Eopp0 (FPR gap)", "worse →"),
]
NOTE = {
    "overall": "HAM10000 (ISIC 2018 test), 1,217 images with sex and age recorded.",
    "fairness": "HAM10000 (ISIC 2018 test). Groups: sex × age (<65 / ≥65), 4 groups.",
}
COMMON = (
    " Threshold: val Youden. Numbers: difference from ResNet-50 (bold: 95% CI excludes 0)."
    " 95% CI: lesion-level bootstrap, 3 seeds."
)


def figure(panels, name: str, note: str, width: float) -> None:
    table = pd.read_csv(RES / "architecture_groups.csv").set_index(["exp", "metric"])
    fig, axes = plt.subplots(1, len(panels), figsize=(width, 2.3), sharey=True)
    ys = range(len(MODELS))
    for ax, (metric, title, direction) in zip(axes, panels, strict=True):
        ax.axvline(table.loc[(MODELS[0][0], metric), "value"], color=BASE, lw=0.9, ls=":", zorder=1)
        for (exp, _, color), yi in zip(MODELS, ys, strict=True):
            r = table.loc[(exp, metric)]
            ax.plot([r.lo, r.hi], [yi, yi], color=color, lw=2, solid_capstyle="round", zorder=2)
            ax.plot(r.value, yi, "o", color=color, ms=6, zorder=3)
            if exp != MODELS[0][0]:
                significant = r.diff_lo > 0 or r.diff_hi < 0
                ax.annotate(
                    f"{r['diff']:+.3f}",
                    (r.hi, yi),
                    xytext=(5, 0),
                    textcoords="offset points",
                    va="center",
                    fontsize=8,
                    color=color,
                    fontweight="bold" if significant else "normal",
                )
        ax.set_title(f"{title}\n", loc="center")
        ax.text(0.5, 1.0, direction, transform=ax.transAxes, ha="center", va="bottom", fontsize=8, color="#666666")
        ax.grid(axis="x", color="#EEEEEE", lw=0.6)
        ax.set_axisbelow(True)
        ax.spines["left"].set_visible(False)
        ax.tick_params(axis="y", length=0)
        ax.margins(x=0.18)
    axes[0].set_yticks(list(ys), [label for _, label, _ in MODELS])
    axes[0].invert_yaxis()
    h = fig.get_figheight()
    fig.tight_layout(rect=(0, 0.4 / h, 1, 1))
    fig.text(0.5, 0.12 / h, note + COMMON, ha="center", va="center", fontsize=7.5, color="#666666")
    fig.savefig(OUT / f"{name}.png", dpi=250, bbox_inches="tight")
    plt.close(fig)


# (列名の接頭辞, パネルの題, 悪い向き, 棒の基点)
GROUP_PANELS = [
    ("auroc", "AUROC", "higher is better", 0.5),
    ("tpr", "TPR (sensitivity)", "higher is better", 0.0),
    ("fpr", "FPR (1 − specificity)", "lower is better", 0.0),
]
GROUPS = [
    ("female_lt65", "Female\n<65"),
    ("female_ge65", "Female\n≥65"),
    ("male_lt65", "Male\n<65"),
    ("male_ge65", "Male\n≥65"),
]


def group_bars() -> None:
    """群ごとの値の棒と 95% 区間。ResNet-50 との差の区間が 0 を含まない棒に * を付ける。"""
    table = pd.read_csv(RES / "architecture_groups.csv").set_index(["exp", "metric"])
    fig, axes = plt.subplots(1, len(GROUP_PANELS), figsize=(13, 3.4))
    width = 0.26
    for ax, (prefix, title, direction, base) in zip(axes, GROUP_PANELS, strict=True):
        for j, (exp, label, color) in enumerate(MODELS):
            for i, (group, _) in enumerate(GROUPS):
                r = table.loc[(exp, f"{prefix}_{group}")]
                x = i + (j - 1) * width
                ax.bar(x, r.value - base, width * 0.92, bottom=base, color=color, label=label if i == 0 else None)
                ax.plot([x, x], [r.lo, r.hi], color="#444444", lw=1, zorder=3)
                if exp != MODELS[0][0] and (r.diff_lo > 0 or r.diff_hi < 0):
                    ax.text(x, r.hi + 0.01, "*", ha="center", va="bottom", fontsize=11, color="#222222")
        ax.set_xticks(range(len(GROUPS)), [name for _, name in GROUPS])
        ax.set_ylim(base, 1.0 if prefix != "fpr" else 0.5)
        ax.set_title(f"{title}\n", loc="center")
        ax.text(0.5, 1.0, direction, transform=ax.transAxes, ha="center", va="bottom", fontsize=8, color="#666666")
        ax.grid(axis="y", color="#EEEEEE", lw=0.6)
        ax.set_axisbelow(True)
        ax.tick_params(axis="x", length=0)
    h = fig.get_figheight()
    fig.tight_layout(rect=(0, 0.6 / h, 1, 1))
    fig.legend(*axes[0].get_legend_handles_labels(), loc="center", ncol=3, fontsize=8, bbox_to_anchor=(0.5, 0.36 / h))
    note = (
        "HAM10000 (ISIC 2018 test), sex × age groups. Threshold: val Youden. Error bars: 95% CI, lesion-level bootstrap,"
        " 3 seeds. *: 95% CI of difference from ResNet-50 excludes 0. AUROC bars start at chance (0.5)."
    )
    fig.text(0.5, 0.1 / h, note, ha="center", va="center", fontsize=7.5, color="#666666")
    fig.savefig(OUT / "hypernet_groups.png", dpi=250, bbox_inches="tight")
    plt.close(fig)


figure(OVERALL, "hypernet_overall", NOTE["overall"], 9.5)
figure(FAIRNESS, "hypernet_fairness", NOTE["fairness"], 12.5)
group_bars()
