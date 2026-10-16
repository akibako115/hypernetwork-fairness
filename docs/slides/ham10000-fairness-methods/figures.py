"""HAM10000 の公平性手法の比較（潜在の群 × 公平性損失 × hypernet）のスライドの図を生成する。

報告は analysis/archive/latent_bias_hypernet/reports/fairness_methods.md の 1・4・6 節。図の形は docs/slides/chexpert-q2 と揃える。

入力:
- analysis/archive/latent_bias_hypernet/results/ham10000_e2e/fairness_values.csv（各条件の値と 95% 区間、fairness_values.py）
- analysis/archive/latent_bias_hypernet/results/ham10000_e2e/fairness_contrast.csv（ResNet-50 ERM との差と 95% 区間）
- analysis/archive/latent_bias_hypernet/results/ham10000_e2e/fairness_pairs.csv（条件どうしの差と 95% 区間）
出力: figures/*.png（図のラベル・凡例・タイトルは英語）
- methods_{values,diffs}.png: 群（性別 × 年齢・GMM 8・LHCF のコホート）× 公平性損失（ResNet-50）
- swadwst_{values,diffs}.png: SWAD ＋ 最悪コホート損失の群を入れ替えた対照
- spatial_{values,diffs}.png: Spatial LoRA（hypernet）× 条件（定数・LHCF のコホート）× 損失
実行例: uv run python docs/slides/ham10000-fairness-methods/figures.py
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
THEME, GREY, INK, ACCENT, BASE = "#1F3B57", "#A9B4BF", "#222222", "#C0504D", "#8C99A6"
R, SL = "resnet-ham10000-fg-", "spatial-lora-ham10000-fg-"
# (列名, パネルの題, 悪い向き)
PANELS = [
    ("auroc_all", "Overall AUROC", "← worse"),
    ("auroc_external", "External-site AUROC", "← worse"),
    ("sexage_min", "Worst-group AUROC", "← worse"),
    ("sexage_gap", "AUROC gap", "worse →"),
    ("sexage_tpr_gap", "Eopp1 (TPR gap)", "worse →"),
    ("sexage_fpr_gap", "Eopp0 (FPR gap)", "worse →"),
]
NOTE = (
    "HAM10000. Fairness metrics on sex × age (<65 / ≥65), 4 groups. Checkpoint: val overall AUROC. "
    "Threshold: val Youden. 95% CI: lesion-level bootstrap, 3 seeds."
)
# 群の入れ方ごとの block。(block の名前, [(exp, 行の名前)])
METHOD_BLOCKS = [
    (
        "Groups: sex × age",
        [
            (R + "gdro-sex-age", "GroupDRO"),
            (R + "ug-sex-age", "Group reweighting"),
            (R + "resample-sex-age", "Group × class resampling"),
            (R + "fairdro-sex-age", "FairDRO"),
            (R + "minimaxauc-sex-age", "minimax AUC"),
            (R + "popauc-sex-age", "AUC loss, fixed group weights"),
            (R + "eo-sex-age", "EO constraint"),
            (R + "dann-sex-age", "Adversarial removal (DANN)"),
            (R + "wst-sex-age", "Worst-cohort loss ℓ_wst"),
            (R + "fis-sex-age", "FIS"),
            (R + "fami-sex-age", "FaMI"),
        ],
    ),
    (
        "Groups: DINOv2 GMM, 8 clusters",
        [
            (R + "gdro-gmm8", "GroupDRO"),
            (R + "ug-gmm8", "Group reweighting"),
            (R + "resample-gmm8", "Group × class resampling"),
            (R + "fairdro-gmm8", "FairDRO"),
            (R + "minimaxauc-gmm8", "minimax AUC"),
            (R + "popauc-gmm8", "AUC loss, fixed group weights"),
            (R + "eo-gmm8", "EO constraint"),
        ],
    ),
    (
        "Groups: LHCF hidden cohorts",
        [
            (R + "wst-lhcf", "Worst-cohort loss ℓ_wst"),
            (R + "fis-lhcf", "FIS"),
            (R + "fami-lhcf", "FaMI"),
        ],
    ),
]
SWAD_BLOCKS = [
    ("SWAD alone", [(R + "swad", "SWAD (no groups)")]),
    (
        "SWAD + ℓ_wst, groups swapped",
        [
            (R + "swadwst-lhcf", "LHCF hidden cohorts"),
            (R + "swadwst-label", "Class (benign / malignant)"),
            (R + "swadwst-lhcfrand", "Random cohorts, same sizes"),
            (R + "swadwst-sex-age", "Sex × age"),
        ],
    ),
]
SPATIAL_BLOCKS = [
    ("ResNet-50", [(R + "erm", "ERM"), (R + "swadwst-lhcf", "SWAD + ℓ_wst (LHCF cohorts)")]),
    (
        "Spatial LoRA, constant condition",
        [(SL + "erm", "ERM"), (SL + "swadwst-lhcf", "SWAD + ℓ_wst (LHCF cohorts)")],
    ),
    (
        "Spatial LoRA, conditioned on LHCF cohorts",
        [(SL + "lhcfcond-erm", "ERM"), (SL + "lhcfcond-swadwst-lhcf", "SWAD + ℓ_wst (LHCF cohorts)")],
    ),
]
# Spatial LoRA の差は、比べる相手が行ごとに違う。(block, [(対象, 基準, 行の名前)])
SPATIAL_PAIRS = [
    ("Add the hypernet", [(SL + "erm", R + "erm", "Spatial LoRA − ResNet-50 (both ERM)")]),
    ("Add the loss", [(SL + "swadwst-lhcf", SL + "erm", "SWAD + ℓ_wst − ERM (constant condition)")]),
    (
        "Condition on LHCF cohorts",
        [
            (SL + "lhcfcond-erm", SL + "erm", "LHCF cohorts − constant (ERM)"),
            (SL + "lhcfcond-swadwst-lhcf", SL + "swadwst-lhcf", "LHCF cohorts − constant (SWAD + ℓ_wst)"),
        ],
    ),
]


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


def layout(sizes: list[int]) -> tuple[list[float], list[float]]:
    """block ごとに行を並べ、block の間に 0.4 の隙間。行の位置と区切り線の位置を返す。"""
    ys, seps, y = [], [], 0.0
    for k, n in enumerate(sizes):
        if k:
            seps.append(y - 0.7)
        ys += [y + i for i in range(n)]
        y += n + 0.4
    return ys, seps


def row_labels(blocks, name_index: int) -> list[str]:
    return [f"{block}: {row[name_index]}" if i == 0 else row[name_index] for block, rows in blocks for i, row in enumerate(rows)]


def finish(fig, axes, ys, seps, labels, titles_prefix: str, legend=None) -> None:
    for ax in axes:
        for s in seps:
            ax.axhline(s, color="#DDDDDD", lw=0.8)
    for ax, (_, title, note) in zip(axes, PANELS, strict=True):
        style_panel(ax, f"{titles_prefix}{title}", note)
    axes[0].set_yticks(ys, labels)
    axes[0].invert_yaxis()
    # 凡例と注記は図の高さに依らず、軸の下に一定の幅（インチ）で置く
    h = fig.get_figheight()
    fig.tight_layout(rect=(0, 0.75 / h, 1, 1))
    if legend:
        fig.legend(handles=legend, loc="center", ncol=len(legend), fontsize=8, bbox_to_anchor=(0.5, 0.38 / h))
    fig.text(0.5, 0.1 / h, NOTE, ha="center", va="center", fontsize=7.5, color="#666666")


def height(n_rows: int) -> float:
    return 1.9 + 0.3 * n_rows


def values_figure(blocks, name: str, erm_row: bool) -> None:
    """各条件の値と 95% 区間。erm_row なら先頭に ResNet-50 ERM の行を足す。点線は ResNet-50 ERM の値。"""
    values = pd.read_csv(RES / "fairness_values.csv").set_index("exp")
    if erm_row:
        blocks = [("Baseline", [(R + "erm", "ResNet-50 ERM")]), *blocks]
    ys, seps = layout([len(rows) for _, rows in blocks])
    rows = [row for _, rs in blocks for row in rs]
    fig, axes = plt.subplots(1, len(PANELS), figsize=(14.5, height(len(rows))), sharey=True)
    for ax, (metric, *_) in zip(axes, PANELS, strict=True):
        ax.axvline(values.loc[R + "erm", metric], color=BASE, lw=0.9, ls=":", zorder=1)
        for (exp, _), yi in zip(rows, ys, strict=True):
            r = values.loc[exp]
            color = BASE if exp.endswith("-erm") else THEME
            ax.plot([r[f"{metric}_lo"], r[f"{metric}_hi"]], [yi, yi], color=color, lw=2, solid_capstyle="round", zorder=2)
            ax.plot(r[metric], yi, "o", color=color, ms=6, zorder=3)
    legend = [
        plt.Line2D([], [], color=BASE, ls=":", lw=0.9, label="ResNet-50 ERM"),
        plt.Line2D([], [], color=BASE, marker="o", lw=2, label="ERM"),
        plt.Line2D([], [], color=THEME, marker="o", lw=2, label="with group-based loss / SWAD"),
    ]
    finish(fig, axes, ys, seps, row_labels(blocks, 1), "", legend)
    save(fig, name)


def diff_figure(rows_of_blocks, name: str, lookup, reference_label: str) -> None:
    """差と 95% 区間。区間が 0 を含まないものを濃くする。lookup(row, metric) -> (差, 下限, 上限)。"""
    ys, seps = layout([len(rows) for _, rows in rows_of_blocks])
    rows = [row for _, rs in rows_of_blocks for row in rs]
    fig, axes = plt.subplots(1, len(PANELS), figsize=(14.5, height(len(rows))), sharey=True)
    for ax, (metric, *_) in zip(axes, PANELS, strict=True):
        ax.axvline(0, color=INK, lw=0.9, ls="--", zorder=1)
        for row, yi in zip(rows, ys, strict=True):
            d, lo, hi = lookup(row, metric)
            color = THEME if lo > 0 or hi < 0 else GREY
            ax.plot([lo, hi], [yi, yi], color=color, lw=2, solid_capstyle="round", zorder=2)
            ax.plot(d, yi, "o", color=color, ms=6, zorder=3)
    legend = [
        plt.Line2D([], [], color=INK, ls="--", lw=0.9, label=reference_label),
        plt.Line2D([], [], color=THEME, marker="o", lw=2, label="95% CI of difference excludes 0"),
        plt.Line2D([], [], color=GREY, marker="o", lw=2, label="includes 0"),
    ]
    finish(fig, axes, ys, seps, row_labels(rows_of_blocks, -1), "Δ ", legend)
    save(fig, name)


def erm_lookup():
    """ResNet-50 ERM との差。contrast の condition は日本語の名前なので、values の exp で引き直す。"""
    label = pd.read_csv(RES / "fairness_values.csv").set_index("exp").condition
    contrast = pd.read_csv(RES / "fairness_contrast.csv")
    contrast = contrast[contrast.select == "overall"].set_index(["condition", "metric"])

    def lookup(row, metric):
        r = contrast.loc[(label[row[0]], metric)]
        return r.delta, r.lo, r.hi

    return lookup


def pair_lookup():
    label = pd.read_csv(RES / "fairness_values.csv").set_index("exp").condition
    pairs = pd.read_csv(RES / "fairness_pairs.csv")
    pairs = pairs[pairs.select == "overall"].set_index(["condition", "reference", "metric"])

    def lookup(row, metric):
        r = pairs.loc[(label[row[0]], label[row[1]], metric)]
        return r.delta, r.lo, r.hi

    return lookup


values_figure(METHOD_BLOCKS, "methods_values", erm_row=True)
diff_figure(METHOD_BLOCKS, "methods_diffs", erm_lookup(), "ResNet-50 ERM")
values_figure(SWAD_BLOCKS, "swadwst_values", erm_row=True)
diff_figure(SWAD_BLOCKS, "swadwst_diffs", erm_lookup(), "ResNet-50 ERM")
values_figure(SPATIAL_BLOCKS, "spatial_values", erm_row=False)
diff_figure(SPATIAL_PAIRS, "spatial_diffs", pair_lookup(), "Reference of each row")
