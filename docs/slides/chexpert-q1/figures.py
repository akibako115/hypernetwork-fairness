"""slide-chexpert-q1.pptx の図を生成する。

入力:
- analysis/archive/latent_bias_hypernet/results/q1_definitions.csv（q1_definitions.py の出力）
- analysis/archive/latent_bias_hypernet/results/q2_fairness_loss.csv・q2_threshold_equivalence.csv・q2_auroc_decomposition.csv
- analysis/dataset_gain_sources/results/gains.csv・attribute_information.csv
出力: figures/*.png（図のラベル・凡例・タイトルは英語）
実行例: uv run python docs/slides/chexpert-q1/figures.py
"""

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
RES = ROOT / "analysis/archive/latent_bias_hypernet/results"
GAIN = ROOT / "analysis/dataset_gain_sources/results"
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
THEME, GREY, INK = "#1F3B57", "#A9B4BF", "#222222"
# (列名, パネルの題, 悪い向き)。悪い向きを題に書き、どのパネルも「どちらが悪いか」を読めるようにする
METRICS = [
    ("worst_auroc", "Worst-group AUROC", "← worse"),
    ("auroc_gap", "AUROC gap", "worse →"),
    ("eopp1", "Eopp1 (TPR gap)", "worse →"),
    ("eopp0", "Eopp0 (FPR gap)", "worse →"),
]
# (q1_definitions.csv の定義名, 図の行名)。None は区切り線
ROWS = {
    "chexpert": [
        ("年齢 × race", "Attribute: age × race"),
        None,
        ("k-means（raddino）", "k-means, RAD-DINO"),
        ("GMM（raddino）", "GMM, RAD-DINO"),
        ("k-means（dinov2）", "k-means, DINOv2"),
        ("GMM（dinov2）", "GMM, DINOv2"),
        ("GEORGE 型（h）", "GEORGE, ERM features"),
        None,
        ("撮影文脈", "Imaging context"),
    ],
    "isic": [
        ("性別 × 年齢", "Attribute: sex × age"),
        None,
        ("k-means（dinov2）", "k-means, DINOv2"),
        ("GMM（dinov2）", "GMM, DINOv2"),
        ("GEORGE 型（h）", "GEORGE, ERM features"),
        None,
        ("部位", "Body site"),
    ],
}


def save(fig, name: str) -> None:
    fig.savefig(OUT / f"{name}.png", dpi=250, bbox_inches="tight")
    plt.close(fig)


def value_forest(dataset: str, height: float) -> None:
    """定義ごとの値と 95% 区間。属性の値に縦線を引き、属性との差が有意な行を濃い色にする。"""
    table = pd.read_csv(RES / "q1_definitions.csv").query("dataset == @dataset").set_index("definition")
    rows = ROWS[dataset]
    base = rows[0][0]
    y_of, labels, seps = {}, [], []
    y = 0.0
    for row in rows:
        if row is None:
            seps.append(y - 0.3)
            y += 0.4
            continue
        y_of[row[0]] = y
        labels.append((y, row[1]))
        y += 1.0
    fig, axes = plt.subplots(1, 4, figsize=(9, height), sharey=True)
    for ax, (metric, title, worse) in zip(axes, METRICS, strict=True):
        ax.axvline(table.loc[base, metric], color=INK, lw=0.9, ls="--", zorder=1)
        for key, yi in y_of.items():
            r = table.loc[key]
            if pd.isna(r[metric]):
                # GEORGE は群が片方のクラスだけなので、群の AUROC を定義できない
                ax.text(0.5, yi, "n/a (single-class groups)", transform=ax.get_yaxis_transform(), ha="center",
                        va="center", fontsize=7.5, color="#888888")
                continue
            differs = r[f"{metric}_diff_lo"] > 0 or r[f"{metric}_diff_hi"] < 0
            color = INK if key == base else THEME if differs else GREY
            ax.plot([r[f"{metric}_lo"], r[f"{metric}_hi"]], [yi, yi], color=color, lw=2, solid_capstyle="round", zorder=2)
            ax.plot(r[metric], yi, "o", color=color, ms=6, zorder=3)
        for sep in seps:
            ax.axhline(sep, color="#DDDDDD", lw=0.8)
        ax.set_title(f"{title}\n", loc="center")
        ax.text(0.5, 1.0, worse, transform=ax.transAxes, ha="center", va="bottom", fontsize=8, color="#666666")
        ax.grid(axis="x", color="#EEEEEE", lw=0.6)
        ax.set_axisbelow(True)
        ax.spines["left"].set_visible(False)
        ax.tick_params(axis="y", length=0)
    axes[0].set_yticks([yy for yy, _ in labels], [t for _, t in labels])
    axes[0].invert_yaxis()
    handles = [
        plt.Line2D([], [], color=INK, ls="--", lw=0.9, label="attribute value"),
        plt.Line2D([], [], color=THEME, marker="o", lw=2, label="differs from attribute (95% CI of difference excludes 0)"),
        plt.Line2D([], [], color=GREY, marker="o", lw=2, label="not distinguishable from attribute"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=3, fontsize=8, bbox_to_anchor=(0.5, -0.02))
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    save(fig, f"q1_values_{dataset}")


value_forest("chexpert", 3.3)
value_forest("isic", 2.5)


# ---- Q2・Q3 ----
PROCS = [("onestage", "One-stage"), ("insample", "Two-stage, in-sample"), ("crossfit", "Two-stage, cross-fit")]
LOSS_NAMES = {"eo": "EO", "minimax_auc": "minimax AUC"}


def interval(ax, lo, hi, x, yi, differs) -> None:
    color = THEME if differs else GREY
    ax.plot([lo, hi], [yi, yi], color=color, lw=2, solid_capstyle="round", zorder=2)
    ax.plot(x, yi, "o", color=color, ms=6, zorder=3)


def style_panel(ax, title: str, note: str) -> None:
    ax.set_title(f"{title}\n", loc="center")
    ax.text(0.5, 1.0, note, transform=ax.transAxes, ha="center", va="bottom", fontsize=8, color="#666666")
    ax.grid(axis="x", color="#EEEEEE", lw=0.6)
    ax.set_axisbelow(True)
    ax.spines["left"].set_visible(False)
    ax.tick_params(axis="y", length=0)


def loss_rows() -> list[tuple[str, str, str]]:
    """(手順, 損失, 行名)。区切りは手順ごと。"""
    return [(p, loss, f"{name}: {LOSS_NAMES[loss]}") for p, name in PROCS for loss in LOSS_NAMES]


def q2_loss_effects() -> None:
    """公平性損失の、同じ手順の ERM との差。全体 AUROC・最悪群 AUROC・gap・Eopp1・Eopp0。"""
    table = pd.read_csv(RES / "q2_fairness_loss.csv").set_index("condition")
    decomposition = pd.read_csv(RES / "q2_auroc_decomposition.csv").set_index(["procedure", "loss"])
    panels = [
        ("overall_auroc", "Overall AUROC", "← worse"),
        ("worst_auroc", "Worst-group AUROC", "← worse"),
        ("auroc_gap", "AUROC gap", "worse →"),
        ("eopp1", "Eopp1 (TPR gap)", "worse →"),
        ("eopp0", "Eopp0 (FPR gap)", "worse →"),
    ]
    rows = loss_rows()
    ys = [i + 0.4 * (i // 2) for i in range(len(rows))]
    fig, axes = plt.subplots(1, len(panels), figsize=(10, 2.9), sharey=True)
    for ax, (metric, title, note) in zip(axes, panels, strict=True):
        ax.axvline(0, color=INK, lw=0.9, ls="--", zorder=1)
        for (procedure, loss, _), yi in zip(rows, ys, strict=True):
            if metric == "overall_auroc":
                r = decomposition.loc[(procedure, loss)]
                d, lo, hi = r["overall_diff"], r["overall_diff_lo"], r["overall_diff_hi"]
            else:
                r = table.loc[f"{procedure}:{loss}:auroc"]
                d, lo, hi = r[f"{metric}_diff"], r[f"{metric}_diff_lo"], r[f"{metric}_diff_hi"]
            interval(ax, lo, hi, d, yi, lo > 0 or hi < 0)
        for sep in (1.7, 4.1):
            ax.axhline(sep, color="#DDDDDD", lw=0.8)
        style_panel(ax, title, note)
    axes[0].set_yticks(ys, [name for *_, name in rows])
    axes[0].invert_yaxis()
    handles = [
        plt.Line2D([], [], color=INK, ls="--", lw=0.9, label="ERM of the same procedure"),
        plt.Line2D([], [], color=THEME, marker="o", lw=2, label="95% CI of difference excludes 0"),
        plt.Line2D([], [], color=GREY, marker="o", lw=2, label="includes 0"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=3, fontsize=8, bbox_to_anchor=(0.5, -0.02))
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    save(fig, "q2_loss_effects")


def q2_threshold() -> None:
    """ERM・ERM + 群別しきい値・損失のモデルの、全体 AUROC・Eopp1・Eopp0（値と 95% 区間）。"""
    table = pd.read_csv(RES / "q2_threshold_equivalence.csv").set_index(["procedure", "rule"])
    rules = [
        ("erm", "ERM (common threshold)", GREY),
        ("erm_group_match", "ERM + group thresholds", ACCENT),
        ("eo", "EO", THEME),
        ("minimax_auc", "minimax AUC", THEME),
    ]
    panels = [("overall_auroc", "Overall AUROC", "← worse"), ("eopp1", "Eopp1 (TPR gap)", "worse →"),
              ("eopp0", "Eopp0 (FPR gap)", "worse →")]
    fig, axes = plt.subplots(1, 3, figsize=(10, 3.1), sharey=True)
    labels, y = [], 0.0
    for p_index, (procedure, name) in enumerate(PROCS):
        for rule, label, color in rules:
            r = table.loc[(procedure, rule)]
            for ax, (metric, *_) in zip(axes, panels, strict=True):
                ax.plot([r[f"{metric}_lo"], r[f"{metric}_hi"]], [y, y], color=color, lw=2, solid_capstyle="round")
                ax.plot(r[metric], y, "o", color=color, ms=6, mfc="white" if rule == "minimax_auc" else color)
            labels.append((y, f"{name}: {label}" if rule == "erm" else label))
            y += 1.0
        if p_index < len(PROCS) - 1:
            for ax in axes:
                ax.axhline(y - 0.3, color="#DDDDDD", lw=0.8)
            y += 0.4
    for ax, (_, title, note) in zip(axes, panels, strict=True):
        style_panel(ax, title, note)
    axes[0].set_yticks([yy for yy, _ in labels], [t for _, t in labels])
    axes[0].invert_yaxis()
    fig.tight_layout()
    save(fig, "q2_threshold")


def q2_decomposition() -> None:
    """全体 AUROC の低下を、群をまたぐ組と群の中の組の寄与に分けた積み上げ棒。"""
    table = pd.read_csv(RES / "q2_auroc_decomposition.csv").query("loss != 'erm'")
    fig, ax = plt.subplots(figsize=(6.2, 2.7))
    names = []
    for i, r in enumerate(table.itertuples()):
        between = (1 - r.within_share) * r.between_diff
        within = r.within_share * r.within_diff
        ax.barh(i, between, color=THEME, height=0.6, label="pairs across groups" if i == 0 else None)
        ax.barh(i, within, left=between, color=ACCENT, height=0.6, label="pairs within a group" if i == 0 else None)
        names.append(f"{dict(PROCS)[r.procedure]}: {LOSS_NAMES[r.loss]}")
    ax.set_yticks(range(len(names)), names)
    ax.invert_yaxis()
    ax.axvline(0, color=INK, lw=0.8)
    ax.set_xlabel("Contribution to the change in overall AUROC (vs ERM)")
    ax.grid(axis="x", color="#EEEEEE", lw=0.6)
    ax.set_axisbelow(True)
    ax.tick_params(axis="y", length=0)
    ax.legend(loc="lower left", bbox_to_anchor=(0, 1.0), ncol=2, fontsize=8)
    fig.tight_layout()
    save(fig, "q2_decomposition")


def q3_datasets() -> None:
    """3 データセットの、条件付けの上がり幅と、属性を後から足したときの上がり幅。"""
    gains = pd.read_csv(GAIN / "gains.csv").query("split == 'test'").set_index(["dataset", "contrast"])
    info = pd.read_csv(GAIN / "attribute_information.csv").set_index(["dataset", "quantity"])
    datasets = [("ham10000", "HAM10000"), ("fairvlmed", "FairVLMed"), ("chexpert", "CheXpert")]
    bars = [
        ("meta − resnet", "Attribute-conditioned Spatial LoRA − ResNet", THEME),
        ("const − resnet", "Constant-condition Spatial LoRA − ResNet", GREY),
        ("ceiling", "ResNet output + attributes, late fusion (val-fit)", ACCENT),
    ]
    fig, ax = plt.subplots(figsize=(7.2, 2.9))
    width = 0.26
    for i, (key, label, color) in enumerate(bars):
        for j, (dataset, _) in enumerate(datasets):
            x = j + (i - 1) * width
            if key == "ceiling":
                ax.bar(x, info.loc[(dataset, "late_fusion_valfit_gain_resnet"), "value"], width * 0.9, color=color,
                       label=label if j == 0 else None)
                continue
            if (dataset, key) not in gains.index:
                ax.text(x, 0.001, "no run", ha="center", va="bottom", fontsize=7, color="#888888", rotation=90)
                continue
            r = gains.loc[(dataset, key)]
            ax.bar(x, r["delta"], width * 0.9, color=color, label=label if j == 0 else None)
            ax.plot([x, x], [r["lo"], r["hi"]], color=INK, lw=1)
    ax.axhline(0, color=INK, lw=0.8)
    ax.set_xticks(range(len(datasets)), [name for _, name in datasets])
    ax.set_ylabel("Δ test AUROC")
    ax.grid(axis="y", color="#EEEEEE", lw=0.6)
    ax.set_axisbelow(True)
    ax.legend(loc="upper right", fontsize=7.5)
    fig.tight_layout()
    save(fig, "q3_datasets")


ACCENT = "#C0504D"
q2_loss_effects()
q2_threshold()
q2_decomposition()
q3_datasets()
