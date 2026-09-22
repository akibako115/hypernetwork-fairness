"""iterative_probe の各分析で共有する run artifact の読み取り。"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from analysis.common.run_artifacts import read_config, read_epoch_metrics

HEADLINE = [
    "val/auroc",
    "val/bacc",
    "val/loss",
    "val/hidden_min_auroc",
    "val/hidden_auroc_gap",
    "val/hidden_min_bacc",
    "train/group_dro/weight_entropy",
    "train/group_dro/max_q",
    "val/sex/worst_group_auroc",
    "val/race/worst_group_auroc",
    "val/ethnicity/worst_group_auroc",
    "val/age_group_65/worst_group_auroc",
]


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
            if metric == "auroc":
                row["val/hidden_auroc_gap"] = max(values) - min(values) if len(values) >= 2 else None
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
