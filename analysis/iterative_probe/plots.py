"""`results/` の表だけを入力に、notebook から再利用する Figure を描く helper。

**色の規約**: 線の色は GroupDRO の step size を、図そのものは変調範囲を表す。4 条件を
1 枚に重ねると線が交差して読めないので、変調範囲は figure の単位に置く。baseline は
step size を持たないので黒（点線・四角）に固定し、条件の 1 つではなく参照線として扱う。

run artifact はここでは読まない。図を描き直すたびに run を読み直すのを避けるためで、
入力は `epoch_metrics.py`・`cohort_analysis.py`・`baseline_comparison.py`・`groups.py` が書いた表に限る。

入力: `results/epoch_metrics.csv` / `baseline_epoch_metrics.csv` / `cohort_groups.csv`
      / `group_metrics_<split>.csv`
notebook はこの module の Figure 関数を呼んで inline 表示する。外部レポート用に PNG を
一括生成する CLI も互換性のため残す。
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import matplotlib.cm as cm
import matplotlib.patheffects as patheffects
import matplotlib.pyplot as plt
import matplotlib.style
import numpy as np
import pandas as pd
import rootutils
from matplotlib.axes import Axes
from matplotlib.figure import Figure

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from analysis.common.paths import STYLE_SHEET  # noqa: E402
from analysis.iterative_probe.groups import BASELINE_LABEL, GROUPING_NAMES  # noqa: E402

PACKAGE = Path(__file__).parent
RESULTS, FIGURES = PACKAGE / "results", PACKAGE / "figures"

MUTED, GRID = "#52514e", "#dcdcd8"
STEP_COLOR = {0.001: "#0173B2", 0.01: "#DE8F05"}
BASELINE_STYLE = {"color": "#000000", "marker": "s", "linestyle": ":"}
MODULATIONS = ["fc", "stage4+fc"]

LEARNING_CURVES = [
    ("val/loss", "Validation loss"),
    ("val/auroc", "Validation AUROC"),
    ("val/bacc", "Validation balanced accuracy"),
    ("val/hidden_min_auroc", "Worst-group AUROC"),
    ("val/hidden_min_bacc", "Worst-group balanced accuracy"),
    ("val/hidden_auroc_gap", "Cohort AUROC spread (max - min)"),
]
GLOBAL_PANELS = [
    ("val/loss", "Validation loss"),
    ("val/auroc", "Validation AUROC"),
    ("val/bacc", "Validation balanced accuracy"),
]
GROUP_DRO_PANELS = [
    ("train/group_dro/weight_entropy", "Weight entropy", float(np.log(10)), "log(10)"),
    ("train/group_dro/max_q", "Max q", 0.1, "uniform"),
]
# 群ごとに記録されうる指標。run に列が無いものは panel ごと落とす。bacc と loss は
# hidden_cohort_logger に後から足した記録なので、それ以前の run には存在しない。
COHORT_PANELS = [
    ("train/group_dro/q_{:02d}", "GroupDRO weight q", 0.1, "uniform"),
    ("val/hidden_auroc_{:02d}", "Validation AUROC per cohort", None, ""),
    ("val/hidden_bacc_{:02d}", "Validation bACC per cohort", None, ""),
    ("val/hidden_loss_{:02d}", "Validation loss per cohort", None, ""),
    ("val/hidden_support_{:02d}", "Validation support", None, ""),
]


def file_name(modulation: str, prefix: str, suffix: str = "") -> str:
    """変調範囲を file 名に使える形にして繋ぐ。

    Args:
        modulation: 変調範囲
        prefix: 図の種類
        suffix: 追加の識別子

    Returns:
        str: `<prefix>_<modulation>[_<suffix>].png`
    """
    parts = [prefix, modulation.replace("+", "_"), suffix]
    return "_".join(part for part in parts if part) + ".png"


def panel(
    axis: Axes,
    frame: pd.DataFrame,
    modulation: str,
    column: str,
    title: str,
    reference: float | None = None,
    reference_label: str = "",
) -> None:
    """1 つの panel に、指定した変調範囲の step size 別の推移を描く。

    縦の区切りは stage の境目で、そこで cohort が引き直され GroupDRO の `q` も初期化される。

    Args:
        axis: 描画先
        frame: `epoch_metrics.csv`
        modulation: 描く変調範囲
        column: 描く列
        title: panel の見出し
        reference: 水平の基準線。不要なら None
        reference_label: 基準線に添える名前

    Returns:
        None
    """
    data = frame[frame["modulation"] == modulation]
    if reference is not None:
        axis.axhline(reference, color=MUTED, linewidth=0.8, linestyle="--", zorder=1)
        axis.text(data["run_epoch"].max() + 0.3, reference, reference_label, color=MUTED, fontsize=8, va="center")
    for _, group in data.groupby("stage_index"):
        left = group["run_epoch"].min()
        if left > 0:
            axis.axvline(left - 0.5, color=GRID, linewidth=0.8, zorder=0)
    for step in sorted(frame["step_size"].unique()):
        series = data[data["step_size"] == step]
        values = series[column]
        axis.plot(series["run_epoch"], values, color=STEP_COLOR[step], label=f"step {step:g}", zorder=3)
        valid = values.dropna()
        if valid.empty:
            continue
        end = (series.loc[valid.index[-1], "run_epoch"], valid.iloc[-1])
        axis.annotate(
            f"{end[1]:.3f}", end, textcoords="offset points", xytext=(6, 0), va="center", color=MUTED, fontsize=8
        )
    axis.set_title(title)
    axis.set_xlabel("Epoch")
    axis.set_xlim(-0.5, data["run_epoch"].max() + 2.0)


def epoch_figure(frame: pd.DataFrame, modulation: str, panels: list[tuple], columns: int = 3) -> Figure:
    """変調範囲 1 つ分の figure を、指標ごとの panel を並べて作る。

    Args:
        frame: `epoch_metrics.csv`
        modulation: 描く変調範囲
        panels: `(列名, 見出し)` または `(列名, 見出し, 基準線, 基準線の名前)` の並び
        columns: 1 行あたりの panel 数

    Returns:
        Figure: 描画した figure
    """
    rows = -(-len(panels) // columns)
    figure, axes = plt.subplots(rows, columns, figsize=(4.4 * columns, 3.6 * rows), squeeze=False)
    for axis, spec in zip(axes.ravel(), panels, strict=False):
        panel(axis, frame, modulation, *spec)
    for axis in axes.ravel()[len(panels) :]:
        axis.set_visible(False)
    handles, labels = axes[0][0].get_legend_handles_labels()
    figure.legend(handles, labels, loc="upper center", ncol=len(labels), bbox_to_anchor=(0.5, 1.05))
    figure.suptitle(f"modulation: {modulation}", y=1.10, fontsize=12)
    return figure


def against_baseline(frame: pd.DataFrame, baseline: pd.DataFrame, modulation: str) -> Figure:
    """iterative の推移に baseline の推移を重ねる。

    baseline は iterative より長く回しているので、同じ epoch 予算での位置（左半分）と
    収束先（右端）が 1 枚で読める。

    Args:
        frame: `epoch_metrics.csv`
        baseline: `baseline_epoch_metrics.csv`
        modulation: 描く変調範囲

    Returns:
        Figure: 描画した figure
    """
    figure, axes = plt.subplots(1, len(GLOBAL_PANELS), figsize=(4.4 * len(GLOBAL_PANELS), 3.6), squeeze=False)
    for axis, (column, title) in zip(axes[0], GLOBAL_PANELS, strict=True):
        panel(axis, frame, modulation, column, title)
        color = BASELINE_STYLE["color"]
        axis.plot(
            baseline["run_epoch"],
            baseline[column],
            color=color,
            linewidth=1.2,
            linestyle=":",
            label="ResNet (ERM)",
            zorder=2,
        )
        end = (baseline["run_epoch"].iloc[-1], baseline[column].iloc[-1])
        axis.annotate(
            f"{end[1]:.3f}", end, textcoords="offset points", xytext=(6, 0), va="center", color=color, fontsize=8
        )
        axis.set_xlim(-0.5, baseline["run_epoch"].max() + 4.0)
        # 終端の数値は baseline の線と重なる位置に出ることがあるので、白で縁取って読めるようにする。
        for text in axis.texts:
            text.set_path_effects([patheffects.withStroke(linewidth=3.0, foreground="white")])
            text.set_zorder(5)
    handles, labels = axes[0][0].get_legend_handles_labels()
    figure.legend(handles, labels, loc="upper center", ncol=len(labels), bbox_to_anchor=(0.5, 1.09))
    figure.suptitle(f"modulation: {modulation}   (dotted: plain ResNet, ERM)", y=1.16, fontsize=12)
    return figure


def global_performance(comparison: pd.DataFrame) -> Figure:
    """baseline と iterative の global AUROC / bACC を snapshot 別に比較する。

    `global_comparison.csv` の行順を保ち、baseline の予算時点・best・final と、iterative の
    同一予算時点を横並びにする。学習曲線で推移を見たあと、条件間の到達点を一枚で確認するための図。

    Args:
        comparison: `global_comparison.csv` を index 付きで読んだ表

    Returns:
        Figure: global AUROC と balanced accuracy の比較図
    """
    metrics = [("global AUROC", "Global AUROC"), ("global bACC", "Global balanced accuracy")]
    labels = comparison.index.tolist()
    positions = np.arange(len(labels))
    figure, axes = plt.subplots(1, len(metrics), figsize=(13.0, 5.2), sharey=True)
    for axis, (column, title) in zip(axes, metrics, strict=True):
        for position, label in zip(positions, labels, strict=True):
            if label.startswith("ResNet"):
                color, marker = BASELINE_STYLE["color"], "s"
            elif "fc / 0.001" in label:
                color, marker = STEP_COLOR[0.001], "o"
            elif "fc / 0.01" in label:
                color, marker = STEP_COLOR[0.01], "o"
            else:
                color, marker = MUTED, "o"
            axis.scatter(comparison.loc[label, column], position, color=color, marker=marker, s=42, zorder=3)
        axis.set_title(title)
        axis.set_xlabel(title)
        axis.grid(axis="x", color=GRID, linewidth=0.7)
    axes[0].set_yticks(positions, labels)
    axes[0].invert_yaxis()
    figure.legend(
        [
            plt.Line2D([], [], color="#000000", marker="s", linestyle="none"),
            plt.Line2D([], [], color=STEP_COLOR[0.001], marker="o", linestyle="none"),
            plt.Line2D([], [], color=STEP_COLOR[0.01], marker="o", linestyle="none"),
        ],
        ["ResNet ERM", "GroupDRO step 0.001", "GroupDRO step 0.01"],
        loc="upper center",
        ncol=3,
        bbox_to_anchor=(0.5, 1.04),
    )
    figure.suptitle("Global performance snapshots", y=1.10, fontsize=12)
    figure.tight_layout()
    return figure


def model_styles(groups: pd.DataFrame, modulation: str) -> list[tuple[str, dict[str, Any]]]:
    """baseline と、指定した変調範囲の 2 条件を、描画順に返す。

    Args:
        groups: `group_metrics_<split>.csv`
        modulation: 描く変調範囲

    Returns:
        list[tuple[str, dict[str, Any]]]: `(ラベル, 描画 kwargs)`
    """
    styles: list[tuple[str, dict[str, Any]]] = []
    for row in groups.drop_duplicates("model")[["model", "modulation", "step_size"]].itertuples():
        if row.model == BASELINE_LABEL:
            styles.append((row.model, dict(BASELINE_STYLE)))
        elif row.modulation == modulation:
            styles.append((row.model, {"color": STEP_COLOR[row.step_size], "marker": "o", "linestyle": "-"}))
    return styles


def fairness_ranges(groups: pd.DataFrame, modulation: str, split: str) -> Figure:
    """粒度ごとに worst → best の幅を描く。

    線が短いほど群間が揃っていて、線の位置が全体の水準になる。

    Args:
        groups: `group_metrics_<split>.csv`
        modulation: 描く変調範囲
        split: 評価した split（見出しに出す）

    Returns:
        Figure: 描画した figure
    """
    styles = model_styles(groups, modulation)
    grouping_order = list(dict.fromkeys(groups["grouping"]))
    figure, axes = plt.subplots(1, 2, figsize=(7.0, 4.6), sharey=True)
    offsets = np.linspace(0.26, -0.26, len(styles))
    for axis, (metric, title) in zip(axes, [("auroc", "AUROC"), ("bacc", "balanced accuracy")], strict=True):
        for (label, style), offset in zip(styles, offsets, strict=True):
            for position, grouping in enumerate(grouping_order):
                part = groups[(groups["model"] == label) & (groups["grouping"] == grouping)][metric]
                span, height = [part.min(), part.max()], [position + offset] * 2
                axis.plot(span, height, color=style["color"], linewidth=1.6, solid_capstyle="round", zorder=3)
                axis.scatter(span, height, s=18, color=style["color"], marker=style["marker"], zorder=4)
            axis.plot([], [], label=label, **style)
        axis.set_yticks(range(len(grouping_order)), grouping_order)
        axis.set_title(f"worst - best {title}")
        axis.set_xlabel(title)
    # sharey なので反転は 1 回だけ。2 回呼ぶと元に戻る。
    axes[0].invert_yaxis()
    handles, labels = axes[0].get_legend_handles_labels()
    figure.legend(handles, labels, loc="upper center", ncol=len(labels), bbox_to_anchor=(0.5, 1.06))
    figure.suptitle(f"modulation: {modulation}   ({split} split)", y=1.13, fontsize=12)
    return figure


def intersection_groups(groups: pd.DataFrame, modulation: str, split: str) -> Figure:
    """3 属性の交差群を 1 群 1 行で描く。

    Args:
        groups: `group_metrics_<split>.csv`
        modulation: 描く変調範囲
        split: 評価した split（見出しに出す）

    Returns:
        Figure: 描画した figure
    """
    styles = model_styles(groups, modulation)
    cells = groups[groups["grouping"] == "age x sex x race"]
    reference = cells[cells["model"] == BASELINE_LABEL]
    # 群の並びは baseline の AUROC 順に固定する。model ごとに並べ替えると行の対応が取れない。
    order = reference.sort_values("auroc")["group"].tolist()
    sizes = reference.set_index("group")["n"]
    labels = [f"{name}  (n={int(sizes[name])})" for name in order]
    figure, axes = plt.subplots(1, 2, figsize=(7.6, 4.8), sharey=True)
    for axis, (metric, title) in zip(axes, [("auroc", "AUROC"), ("bacc", "balanced accuracy")], strict=True):
        for label, style in styles:
            part = cells[cells["model"] == label].set_index("group").loc[order, metric]
            axis.scatter(
                part.to_numpy(),
                range(len(order)),
                s=26,
                color=style["color"],
                marker=style["marker"],
                label=label,
                zorder=3,
            )
        axis.set_yticks(range(len(order)), labels)
        axis.set_title(f"{title} per intersectional group")
        axis.set_xlabel(title)
    axes[0].invert_yaxis()
    handles, axis_labels = axes[0].get_legend_handles_labels()
    figure.legend(handles, axis_labels, loc="upper center", ncol=len(axis_labels), bbox_to_anchor=(0.5, 1.06))
    figure.suptitle(f"modulation: {modulation}   ({split} split, age x sex x race)", y=1.12, fontsize=12)
    return figure


def fairness_metrics(summary: pd.DataFrame, modulation: str, split: str, groups: pd.DataFrame | None = None) -> Figure:
    """Eopp0 / Eopp1 / Eodds を粒度ごとに model 間で比較する。

    いずれも群間の gap なので、値が小さいほど予測の機会が揃っている。ただし
    gap の大小だけでは性能水準を表さないため、AUROC や bACC の図と合わせて読む。
    `summary` は `fairness_summary_<split>.csv` を index 付きで読んだ表を想定する。

    Args:
        summary: 粒度・model ごとの公平性集計表
        modulation: 描く変調範囲
        split: 評価した split（見出しに出す）
        groups: model の描画 style を解決するための群別表。省略時は単色で描く。

    Returns:
        Figure: Eopp0 / Eopp1 / Eodds の比較図
    """
    if isinstance(summary.index, pd.MultiIndex):
        data = summary.reset_index()
    else:
        data = summary.copy()
    data = data[data["modulation"].eq(modulation)] if "modulation" in data else data
    metrics = [("Eopp0", "Eopp0 (TNR gap)"), ("Eopp1", "Eopp1 (TPR gap)"), ("Eodds", "Eodds")]
    styles = model_styles(groups, modulation) if groups is not None else []
    if not styles:
        styles = [(label, dict(BASELINE_STYLE) if label == BASELINE_LABEL else {}) for label in data["model"].unique()]
    grouping_order = [grouping for grouping in GROUPING_NAMES if grouping in data["grouping"].unique()]
    figure, axes = plt.subplots(1, len(metrics), figsize=(12.0, 4.4), sharey=True)
    positions = np.arange(len(grouping_order))
    offsets = np.linspace(-0.18, 0.18, len(styles)) if styles else []
    for axis, (metric, title) in zip(axes, metrics, strict=True):
        for offset, (label, style) in zip(offsets, styles, strict=True):
            values = data[data["model"] == label].set_index("grouping")[metric].reindex(grouping_order)
            color = style.get("color", MUTED)
            marker = style.get("marker", "o")
            axis.scatter(positions + offset, values, color=color, marker=marker, s=34, label=label, zorder=3)
        axis.set_title(title)
        axis.set_xticks(positions, grouping_order, rotation=45, ha="right")
        axis.set_xlabel("Grouping")
        axis.grid(axis="y", color=GRID, linewidth=0.7)
    axes[0].set_ylabel("Gap (lower is more equal)")
    handles, labels = axes[0].get_legend_handles_labels()
    figure.legend(handles, labels, loc="upper center", ncol=max(1, len(labels)), bbox_to_anchor=(0.5, 1.05))
    figure.suptitle(f"Equalized opportunity / odds gaps   ({modulation}, {split} split)", y=1.12, fontsize=12)
    figure.tight_layout()
    return figure


def all_cohorts(frame: pd.DataFrame, cohorts: pd.DataFrame, modulation: str, step: float) -> Figure:
    """10 群すべての推移を stage ごとに描く。色は stage 最終 epoch の q の順位。

    top3 / bottom3 のような区分は epoch ごとの順位変動を潰すので、圧縮せずに全群を描く。
    同じ図の中で同じ色は同じ群を指すが、stage をまたぐと cohort が変わるので色の対応も
    stage 内で閉じる。

    Args:
        frame: `epoch_metrics.csv`
        cohorts: `cohort_groups.csv`
        modulation: 描く変調範囲
        step: 描く step size

    Returns:
        Figure: 描画した figure
    """
    stages = sorted(cohorts["stage"].unique())
    panels = [spec for spec in COHORT_PANELS if spec[0].format(0) in frame.columns]
    figure, axes = plt.subplots(len(stages), len(panels), figsize=(4.6 * len(panels), 3.5 * len(stages)), squeeze=False)
    for row, stage in enumerate(stages):
        selected = (cohorts["modulation"] == modulation) & (cohorts["step_size"] == step) & (cohorts["stage"] == stage)
        ranked = cohorts[selected].sort_values("q")
        # q の小さい群を薄く、大きい群を濃くする。順位は 1 つの sequential ramp で表す。
        shade = {
            group: cm.YlGnBu(0.25 + 0.7 * index / (len(ranked) - 1)) for index, group in enumerate(ranked["group"])
        }
        epochs = frame[(frame["modulation"] == modulation) & (frame["step_size"] == step) & (frame["stage"] == stage)]
        for column, (template, title, reference, reference_label) in enumerate(panels):
            axis = axes[row][column]
            if reference is not None:
                axis.axhline(reference, color=MUTED, linewidth=0.8, linestyle="--", zorder=1)
                axis.text(
                    epochs["epoch"].max() + 0.05, reference, reference_label, color=MUTED, fontsize=8, va="center"
                )
            for group in ranked["group"]:
                axis.plot(epochs["epoch"], epochs[template.format(group)], color=shade[group], linewidth=1.4, zorder=3)
            axis.set_title(f"{title} — {stage}")
            axis.set_xlabel("Stage epoch")
    figure.suptitle(f"modulation: {modulation} / step {step:g}   (color: darker = larger q)", y=1.02, fontsize=12)
    return figure


def main() -> None:
    """`results/` の表を読んで `figures/` を作り直す。

    Args:
        なし

    Returns:
        None
    """
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--split", default="test", choices=("val", "test"))
    args = parser.parse_args()

    # 旧 repo から持ってきた図 style。Okabe–Ito の色循環と論文向けの軸設定を持つ。import した
    # だけで global な matplotlib state を書き換えないよう、作図の入口で適用する。
    matplotlib.style.use(STYLE_SHEET)
    frame = pd.read_csv(RESULTS / "epoch_metrics.csv")
    baseline = pd.read_csv(RESULTS / "baseline_epoch_metrics.csv")
    comparison = pd.read_csv(RESULTS / "global_comparison.csv", index_col="run")
    cohorts = pd.read_csv(RESULTS / "cohort_groups.csv")
    groups = pd.read_csv(RESULTS / f"group_metrics_{args.split}.csv")
    FIGURES.mkdir(exist_ok=True)

    global_figure = global_performance(comparison)
    global_figure.savefig(FIGURES / "global_performance.png")
    plt.close(global_figure)
    written = ["global_performance.png"]
    for modulation in MODULATIONS:
        figures = {
            file_name(modulation, "learning_curves"): epoch_figure(frame, modulation, LEARNING_CURVES),
            file_name(modulation, "baseline_global"): against_baseline(frame, baseline, modulation),
            file_name(modulation, "group_dro"): epoch_figure(frame, modulation, GROUP_DRO_PANELS, columns=2),
            file_name(modulation, "fairness_ranges"): fairness_ranges(groups, modulation, args.split),
            file_name(modulation, "fairness_intersection"): intersection_groups(groups, modulation, args.split),
            file_name(modulation, "fairness_metrics"): fairness_metrics(
                pd.read_csv(RESULTS / f"fairness_summary_{args.split}.csv", index_col=["grouping", "model"]),
                modulation,
                args.split,
                groups,
            ),
        }
        for step in sorted(frame["step_size"].unique()):
            figures[file_name(modulation, "all_groups", f"step{step:g}")] = all_cohorts(
                frame, cohorts, modulation, step
            )
        for name, figure in figures.items():
            figure.savefig(FIGURES / name)
            plt.close(figure)
            written.append(name)
    print(f"{len(written)} figures -> {FIGURES}")


if __name__ == "__main__":
    main()
