"""iterative run と ResNet baseline の epoch 推移・到達点を比較する。"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import pandas as pd
import rootutils

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from analysis.common.paths import runs_root  # noqa: E402
from analysis.common.run_artifacts import read_config, read_wandb_history, write_csv  # noqa: E402

RESULTS = Path(__file__).parent / "results"
GLOBAL_COLUMNS = {
    "val/auroc": "global AUROC",
    "val/bacc": "global bACC",
    "val/loss": "val loss",
    "val/sex/worst_group_auroc": "sex worst AUROC",
    "val/race/worst_group_auroc": "race worst AUROC",
    "val/ethnicity/worst_group_auroc": "ethnicity worst AUROC",
    "val/age_group_65/worst_group_auroc": "age worst AUROC",
}


def collect_baseline(run_dir: Path) -> list[dict[str, Any]]:
    """baseline の W&B transaction log を epoch 表へ変換する。"""
    condition = read_config(run_dir / "config.yaml")["experiment_name"]
    history = read_wandb_history(run_dir)
    return [{"run_id": run_dir.name, "condition": condition, "run_epoch": int(row["epoch"]), **row} for row in history]


def global_comparison(frame: pd.DataFrame, baseline: pd.DataFrame) -> pd.DataFrame:
    """iterative と baseline の予算時点・best・final を同じ表に並べる。"""
    budget = int(frame["run_epoch"].max())
    best = baseline.loc[baseline["val/auroc"].idxmax()]
    last = baseline.iloc[-1]
    snapshots = [
        (f"ResNet ERM @ epoch {budget}", baseline[baseline["run_epoch"] == budget].iloc[0]),
        (f"ResNet ERM @ best AUROC (epoch {int(best['run_epoch'])})", best),
        (f"ResNet ERM @ epoch {int(last['run_epoch'])} (final)", last),
    ]
    for condition, data in frame.groupby("condition"):
        snapshots.append((f"iterative {condition} @ epoch {budget}", data.sort_values("run_epoch").iloc[-1]))
    rows = [
        {"run": name, **{label: row[column] for column, label in GLOBAL_COLUMNS.items()}} for name, row in snapshots
    ]
    return pd.DataFrame(rows).set_index("run")


def write_results(
    iterative_rows: list[dict[str, Any]], baseline_ids: list[str], baseline_root: Path
) -> list[dict[str, Any]]:
    """baseline の epoch 表と iterative との比較表を書き、baseline 行を返す。"""
    baseline = [row for run_id in baseline_ids for row in collect_baseline(baseline_root / run_id)]
    fixed = ["run_id", "condition", "run_epoch", "epoch"]
    columns = fixed + sorted({key for row in baseline for key in row} - set(fixed))
    RESULTS.mkdir(exist_ok=True)
    write_csv(RESULTS / "baseline_epoch_metrics.csv", baseline, columns)
    comparison = global_comparison(pd.DataFrame(iterative_rows), pd.DataFrame(baseline))
    comparison.to_csv(RESULTS / "global_comparison.csv")
    print(f"{len(baseline_ids)} baseline runs / {len(baseline)} epochs -> {RESULTS}")
    return baseline


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_ids", nargs="+", help="iterative run-id")
    parser.add_argument("--baseline", nargs="+", required=True)
    parser.add_argument("--runs-root", type=Path, default=runs_root("iterative_probe"))
    parser.add_argument("--baseline-runs-root", type=Path, default=runs_root("iterative_probe"))
    args = parser.parse_args()
    from analysis.iterative_probe._shared import collect_epoch_rows

    rows = [row for run_id in args.run_ids for row in collect_epoch_rows(args.runs_root / run_id)]
    write_results(rows, args.baseline, args.baseline_runs_root)


if __name__ == "__main__":
    main()
