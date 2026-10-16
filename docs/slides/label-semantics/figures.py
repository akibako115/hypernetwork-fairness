"""slide-label-semantics.pptx の図を生成する。

入力: analysis/archive/direction_poc/results/d5/{metrics,decomposition,bootstrap}.csv（d5_label_semantics.py の出力）
出力: figures/*.png（図のラベル・凡例・タイトルは英語）
実行例: uv run python docs/slides/label-semantics/figures.py
"""

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
RES = ROOT / "analysis/archive/direction_poc/results/d5"
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
THEME, ACCENT, GREY, INK = "#1F3B57", "#C0504D", "#A9B4BF", "#222222"
BASE = "M01 ResNet ERM"
TYPES = [
    ("affirmed_normal", "Report affirms normal"),
    ("focused_followup", "Answers a follow-up question only"),
    ("other", "Other"),
]
# (metrics.csv のモデル名, 図の行名)
CONDITIONS = [
    ("M01 ResNet ERM", "ResNet-50, ERM"),
    ("M02 ResNet DRO", "ResNet-50, GroupDRO"),
    ("M03 Spatial ERM", "Spatial LoRA, ERM"),
    ("M04 Spatial DRO", "Spatial LoRA, GroupDRO"),
    ("M06 Spatial 2-stage DRO", "Spatial LoRA, 2-stage GroupDRO"),
    ("M07 HyperAdapt ERM", "HyperAdapt, ERM"),
]


def save(fig, name: str) -> None:
    fig.savefig(OUT / f"{name}.png", dpi=250, bbox_inches="tight")
    plt.close(fig)


def composition() -> None:
    """年齢 2 区分ごとの、No Finding 陽性のレポート種別の構成比。"""
    row = pd.read_csv(RES / "decomposition.csv").query("model == @BASE and band == '<65 vs ≥65'").iloc[0]
    colors = [THEME, ACCENT, GREY]
    fig, ax = plt.subplots(figsize=(4.6, 2.0))
    for y, (age, key) in enumerate([("Age <65", "young"), ("Age ≥65", "old")]):
        left = 0.0
        for (t, label), c in zip(TYPES, colors):
            w = row[f"w {key} {t}"]
            ax.barh(y, w, left=left, color=c, height=0.6, label=label if y == 0 else None)
            ax.text(left + w / 2, y, f"{w:.0%}", ha="center", va="center", color="white", fontsize=9, fontweight="bold")
            left += w
    ax.set_yticks([0, 1], ["Age <65", "Age ≥65"])
    ax.invert_yaxis()
    ax.set_xlim(0, 1)
    ax.set_xticks([0, 0.5, 1], ["0%", "50%", "100%"])
    ax.set_xlabel("Share of No Finding positives")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.38), ncol=2, fontsize=8)
    save(fig, "composition")


def auroc_by_age() -> None:
    """陽性の絞り方ごとの、年齢 2 区分の AUROC（陰性は全件）。"""
    m = pd.read_csv(RES / "metrics.csv").query("model == @BASE").set_index("positives")
    boot = pd.read_csv(RES / "bootstrap.csv").query("model == @BASE")
    sets = [("all", "All positives"), ("affirmed_normal", "Affirms normal"), ("focused_followup", "Follow-up only")]
    fig, ax = plt.subplots(figsize=(4.6, 3.0))
    width = 0.36
    for i, (key, _) in enumerate(sets):
        young, old = m.loc[key, "auc <65"], m.loc[key, "auc ≥65"]
        ax.bar(i - width / 2, young, width, color=GREY, label="Age <65" if i == 0 else None)
        ax.bar(i + width / 2, old, width, color=THEME, label="Age ≥65" if i == 0 else None)
        text = f"gap {young - old:.3f}"
        col = {"all": "all age_gap", "affirmed_normal": "aff age_gap"}.get(key)
        if col:
            lo, hi = boot[col].quantile([0.025, 0.975])
            text += f"\n[{lo:.3f}, {hi:.3f}]"
        ax.text(i, max(young, old) + 0.008, text, ha="center", va="bottom", fontsize=8, color=INK)
    ax.set_xticks(range(len(sets)), [s[1] for s in sets])
    ax.set_ylim(0.7, 1.0)
    ax.set_ylabel("AUROC (all negatives kept)")
    ax.legend(loc="upper right", ncol=2, fontsize=8, bbox_to_anchor=(1.0, 1.12))
    save(fig, "auroc_by_age")


def six_conditions() -> None:
    """既存 6 条件の年齢差：陽性全体と、正常を明言した陽性だけ。"""
    m = pd.read_csv(RES / "metrics.csv").set_index(["model", "positives"])
    fig, ax = plt.subplots(figsize=(4.6, 2.9))
    for y, (key, label) in enumerate(CONDITIONS):
        a, b = m.loc[(key, "all"), "age_gap"], m.loc[(key, "affirmed_normal"), "age_gap"]
        ax.annotate("", xy=(b, y), xytext=(a, y), arrowprops=dict(arrowstyle="->", color=GREY, lw=1))
        ax.plot(a, y, "o", color=GREY, ms=6, label="All positives" if y == 0 else None)
        ax.plot(b, y, "o", color=THEME, ms=6, label="Positives affirming normal" if y == 0 else None)
    ax.set_yticks(range(len(CONDITIONS)), [c[1] for c in CONDITIONS])
    ax.invert_yaxis()
    ax.axvline(0, color=INK, lw=0.5)
    ax.set_xlim(-0.005, 0.08)
    ax.set_xlabel("Age gap in AUROC (<65 minus ≥65)")
    ax.legend(loc="upper center", bbox_to_anchor=(0.4, -0.2), ncol=2, fontsize=8)
    save(fig, "six_conditions")


composition()
auroc_by_age()
six_conditions()
