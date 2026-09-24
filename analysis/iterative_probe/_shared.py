"""iterative_probe の各分析で共有する run artifact の読み取り。"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from analysis.common.run_artifacts import read_config, read_epoch_metrics


def add_hidden_summaries(row: dict[str, Any]) -> dict[str, Any]:
    """cohort ごとの raw 指標から hidden の派生サマリを後計算する。"""
    for metric in ("auroc", "bacc", "loss"):
        prefix = f"val/hidden_{metric}_"
        values = [
            value
            for key, value in row.items()
            if key.startswith(prefix) and key[len(prefix) :].isdigit() and pd.notna(value)
        ]
        if metric == "loss":
            row["val/hidden_max_loss"] = max(values) if values else None
            row["val/hidden_loss_gap"] = max(values) - min(values) if len(values) >= 2 else None
        else:
            row[f"val/hidden_min_{metric}"] = min(values) if values else None
            row[f"val/hidden_{metric}_gap"] = max(values) - min(values) if len(values) >= 2 else None
    auroc_values = [
        value
        for key, value in row.items()
        if key.startswith("val/hidden_auroc_") and key[len("val/hidden_auroc_") :].isdigit() and pd.notna(value)
    ]
    row["val/hidden_valid_auroc_groups"] = len(auroc_values)
    return row


def read_condition(run_dir: Path) -> dict[str, Any]:
    """run の config から変調範囲と GroupDRO step size を読む。"""
    config = read_config(run_dir / "config.yaml")
    modulation = "+".join(config["model"]["net"]["modulation_stages"])
    step_size = float(config["iteration"]["group_dro_step_size"])
    return {"modulation": modulation, "step_size": step_size, "condition": f"{modulation} / {step_size:g}"}


def collect_epoch_rows(run_dir: Path) -> list[dict[str, Any]]:
    """warmup と stage の metrics を通し epoch 番号付きで読む。"""
    stages = json.loads((run_dir / "run.json").read_text(encoding="utf-8"))["stages"]
    condition = read_condition(run_dir)
    names = ["warmup"] + sorted(name for name in stages if name.startswith("stage"))
    rows: list[dict[str, Any]] = []
    run_epoch = 0
    for stage_index, name in enumerate(names):
        metrics_path = run_dir / "stages" / name / "metrics" / "metrics.csv"
        for row in read_epoch_metrics(metrics_path):
            row = add_hidden_summaries(row)
            position = {"stage": name, "stage_index": stage_index, "run_epoch": run_epoch}
            rows.append({"run_id": run_dir.name, **condition, **position, **row})
            run_epoch += 1
    return rows


def collect_cohort_rows(run_dir: Path) -> list[dict[str, Any]]:
    """cohort stage の q・class weight・support・AUROC を読む。"""
    condition = read_condition(run_dir)
    rows: list[dict[str, Any]] = []
    for stage_dir in sorted((run_dir / "stages").glob("stage*")):
        config = read_config(stage_dir / "config.yaml")
        weights = config["model"]["loss_fn"].get("class_weight")
        if weights is None or not isinstance(weights[0], list):
            continue
        last = read_epoch_metrics(stage_dir / "metrics" / "metrics.csv")[-1]
        cohort_name = f"cohort{stage_dir.name.removeprefix('stage')}"
        assignments = pd.read_parquet(run_dir / "artifacts" / "cohorts" / cohort_name / "assignments.parquet")
        train_size = assignments[assignments["split"] == "train"].groupby("group_id").size()
        for group, weight in enumerate(weights):
            rows.append(
                {
                    "run_id": run_dir.name,
                    **condition,
                    "stage": stage_dir.name,
                    "group": group,
                    "q": last.get(f"train/group_dro/q_{group:02d}"),
                    "negative_weight": weight[0],
                    "positive_weight": weight[1],
                    "train_size": int(train_size.get(group, 0)),
                    "val_support": last.get(f"val/hidden_support_{group:02d}"),
                    "val_auroc": last.get(f"val/hidden_auroc_{group:02d}"),
                }
            )
    return rows


def rank_cohorts_by_q(epochs: pd.DataFrame, pick: int = 2, cohort_count: int = 10) -> pd.DataFrame:
    """stage ごとに cohort を `q` の stage 内平均で順位付けし、上位・下位 `pick` 個に印を付ける。

    Args:
        epochs: `collect_epoch_rows` の行を並べた表（`run_id`・`condition`・`stage`・`train/group_dro/q_NN` を持つ）
        pick: 上位・下位それぞれから選ぶ cohort の数
        cohort_count: stage あたりの cohort 数

    Returns:
        pd.DataFrame: run × stage × cohort ごとの `q_mean`・`q_end`・`rank_mean`・`rank_end`・`pick`
        （`high` / `middle` / `low`）。順位は 1 が最大の `q`
    """
    stages = epochs[epochs["stage"].str.startswith("stage")]
    rows = []
    for (run_id, condition, stage), part in stages.groupby(["run_id", "condition", "stage"], sort=False):
        for cohort in range(cohort_count):
            q = part[f"train/group_dro/q_{cohort:02d}"]
            rows.append(
                {
                    "run_id": run_id,
                    "condition": condition,
                    "stage": stage,
                    "cohort": cohort,
                    "q_mean": q.mean(),
                    "q_end": q.iloc[-1],
                }
            )
    ranks = pd.DataFrame(rows)
    by_stage = ranks.groupby(["run_id", "stage"])
    ranks["rank_mean"] = by_stage["q_mean"].rank(ascending=False).astype(int)
    ranks["rank_end"] = by_stage["q_end"].rank(ascending=False).astype(int)
    ranks["pick"] = pd.cut(
        ranks["rank_mean"], bins=[0, pick, cohort_count - pick, cohort_count], labels=["high", "middle", "low"]
    ).astype(str)
    return ranks
