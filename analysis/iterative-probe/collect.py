"""iterative run の epoch metric を 1 枚の表へ集約する。

複数の run-id を渡すと、条件列（変調範囲・step size）を付けて 1 つの表にまとめる。
run-id には override した水準が入らないので、条件は各 run の `config.yaml` から読む。

epoch 単位の表に加えて、`(run, stage, cohort)` 単位の表も書く。cohort は stage ごとに
引き直されるので群を stage 間で追えないが、stage の中では `q`・class weight・support・AUROC を
同じ添字で突き合わせられる（`assignments.parquet` が train と val を同じ `group_id` で持つ）。

使い方:
    uv run python analysis/iterative-probe/collect.py <run-id> [<run-id> ...]
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import pandas as pd
import yaml

RUNS_ROOT = Path("projects/hypernet_iterative/runs")

# 反復の効き方を見るために毎回見る列。stage 間で比較する順に並べる。
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


def read_epoch_metrics(csv_path: Path) -> list[dict[str, float]]:
    """`CSVLogger` の出力を epoch ごとの 1 行へまとめる。

    train と val は別行に書かれるので epoch で束ね直す。空セルはその行が記録しなかった指標を
    表すので落とす。`step` は optimizer step であり epoch と混ざるため除く。

    Args:
        csv_path: `stages/<name>/metrics/metrics.csv`

    Returns:
        list[dict[str, float]]: epoch 昇順の metric
    """
    by_epoch: dict[int, dict[str, float]] = {}
    with csv_path.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            epoch = int(float(row["epoch"]))
            merged = by_epoch.setdefault(epoch, {"epoch": float(epoch)})
            for name, value in row.items():
                if name in ("epoch", "step") or value in (None, ""):
                    continue
                merged[name] = float(value)
    return [by_epoch[epoch] for epoch in sorted(by_epoch)]


def read_condition(run_dir: Path) -> dict[str, str]:
    """run の `config.yaml` から、run-id に現れない条件を読む。

    Args:
        run_dir: `projects/hypernet_iterative/runs/<run-id>`

    Returns:
        dict[str, str]: `modulation` / `step_size` / `condition`（表示用の短い名前）
    """
    config = yaml.safe_load((run_dir / "config.yaml").read_text())
    modulation = "+".join(config["model"]["net"]["modulation_stages"])
    step_size = float(config["iteration"]["group_dro_step_size"])
    return {
        "modulation": modulation,
        "step_size": step_size,
        "condition": f"{modulation} / {step_size:g}",
    }


def collect(run_dir: Path) -> list[dict[str, object]]:
    """warmup と各 stage の epoch metric を通し番号付きで 1 本に並べる。

    Args:
        run_dir: `projects/hypernet_iterative/runs/<run-id>`

    Returns:
        list[dict[str, object]]: run-id・条件・stage 名・stage 番号・通し epoch を付けた行
    """
    stages = json.loads((run_dir / "run.json").read_text())["stages"]
    condition = read_condition(run_dir)
    names = ["warmup"] + sorted(n for n in stages if n.startswith("stage"))
    rows: list[dict[str, object]] = []
    step = 0
    for index, name in enumerate(names):
        for row in read_epoch_metrics(run_dir / "stages" / name / "metrics" / "metrics.csv"):
            rows.append({"run_id": run_dir.name, **condition, "stage": name, "stage_index": index, "run_epoch": step, **row})
            step += 1
    return rows


def collect_cohorts(run_dir: Path) -> list[dict[str, object]]:
    """cohort stage ごとに、群単位の `q`・class weight・support・AUROC を並べる。

    stage の最終 epoch の値を取る。`q` は stage 内で累積するので、最終 epoch が
    その stage で DRO が到達した重み配分になる。

    Args:
        run_dir: `projects/hypernet_iterative/runs/<run-id>`

    Returns:
        list[dict[str, object]]: `(run_id, 条件, stage, group)` 1 組につき 1 行
    """
    condition = read_condition(run_dir)
    rows: list[dict[str, object]] = []
    for stage_dir in sorted((run_dir / "stages").glob("stage*")):
        config = yaml.safe_load((stage_dir / "config.yaml").read_text())
        weights = config["model"]["loss_fn"].get("class_weight")
        if weights is None or not isinstance(weights[0], list):
            continue
        last = read_epoch_metrics(stage_dir / "metrics" / "metrics.csv")[-1]
        assignments = pd.read_parquet(run_dir / "artifacts" / "cohorts" / f"cohort{stage_dir.name.removeprefix('stage')}" / "assignments.parquet")
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


def write_csv(path: Path, rows: list[dict[str, object]], columns: list[str]) -> None:
    """指定した列だけを CSV へ書き出す。欠けている列は空欄にする。

    改行は LF に固定する。csv の既定は CRLF で、Git は commit 時に LF へ正規化するため、
    既定のままだと書き出すたびに working tree と index が食い違う。

    Args:
        path: 出力先
        rows: 書き出す行
        columns: 出力する列の順序

    Returns:
        None
    """
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore", lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def main() -> None:
    """run-id を受け取り `results/` へ全列と主要列の表を書く。

    Args:
        なし

    Returns:
        None
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_ids", nargs="+")
    parser.add_argument("--runs-root", type=Path, default=RUNS_ROOT)
    args = parser.parse_args()

    rows = [row for run_id in args.run_ids for row in collect(args.runs_root / run_id)]
    results = Path(__file__).parent / "results"
    fixed = ["run_id", "condition", "modulation", "step_size", "stage", "stage_index", "run_epoch", "epoch"]
    every = fixed + sorted({key for row in rows for key in row} - set(fixed))
    write_csv(results / "epoch_metrics.csv", rows, every)
    write_csv(results / "headline.csv", rows, fixed + HEADLINE)

    cohorts = [row for run_id in args.run_ids for row in collect_cohorts(args.runs_root / run_id)]
    write_csv(results / "cohort_groups.csv", cohorts, list(cohorts[0]) if cohorts else [])
    print(f"{len(args.run_ids)} runs / {len(rows)} epochs / {len(cohorts)} cohort groups -> {results}/epoch_metrics.csv, {results}/headline.csv, {results}/cohort_groups.csv")


if __name__ == "__main__":
    main()
