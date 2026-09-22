"""iterative run の epoch metric を `results/` の表へ集約する。

複数の run-id を渡すと、条件列（変調範囲・step size）を付けて 1 つの表にまとめる。
run-id には override した水準が入らないので、条件は各 run の `config.yaml` から読む。

epoch 単位の表に加えて、`(run, stage, cohort)` 単位の表も書く。cohort は stage ごとに
引き直されるので群を stage 間で追えないが、stage の中では `q`・class weight・support・AUROC を
同じ添字で突き合わせられる（`assignments.parquet` が train と val を同じ `group_id` で持つ）。

`--baseline` に `hypernet_e2e` の run-id を渡すと、global 指標の比較対象として同じ形の表を
別ファイルへ書く。baseline は CSVLogger を付けずに回しているので、epoch 推移は WandbLogger の
transaction log から読む。

入力: `projects/*/runs/<run-id>/`
出力: `results/epoch_metrics.csv` / `results/headline.csv` / `results/cohort_groups.csv`
      / `results/baseline_epoch_metrics.csv` / `results/global_comparison.csv`
      / `results/cohort_correlations.csv`

使い方:
    uv run python analysis/iterative_probe/collect.py <run-id> [<run-id> ...] \
      [--baseline <run-id> ...]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd
import rootutils

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from analysis.common.paths import runs_root  # noqa: E402
from analysis.common.run_artifacts import read_config, read_epoch_metrics, read_wandb_history, write_csv  # noqa: E402

RESULTS = Path(__file__).parent / "results"

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


def read_condition(run_dir: Path) -> dict[str, Any]:
    """run の `config.yaml` から、run-id に現れない条件を読む。

    Args:
        run_dir: `projects/hypernet_iterative/runs/<run-id>`

    Returns:
        dict[str, Any]: `modulation` / `step_size` / `condition`（表示用の短い名前）
    """
    config = read_config(run_dir / "config.yaml")
    modulation = "+".join(config["model"]["net"]["modulation_stages"])
    step_size = float(config["iteration"]["group_dro_step_size"])
    return {"modulation": modulation, "step_size": step_size, "condition": f"{modulation} / {step_size:g}"}


def collect(run_dir: Path) -> list[dict[str, Any]]:
    """warmup と各 stage の epoch metric を通し番号付きで 1 本に並べる。

    Args:
        run_dir: `projects/hypernet_iterative/runs/<run-id>`

    Returns:
        list[dict[str, Any]]: run-id・条件・stage 名・stage 番号・通し epoch を付けた行
    """
    stages = json.loads((run_dir / "run.json").read_text(encoding="utf-8"))["stages"]
    condition = read_condition(run_dir)
    names = ["warmup"] + sorted(name for name in stages if name.startswith("stage"))
    rows: list[dict[str, Any]] = []
    step = 0
    for index, name in enumerate(names):
        for row in read_epoch_metrics(run_dir / "stages" / name / "metrics" / "metrics.csv"):
            position = {"stage": name, "stage_index": index, "run_epoch": step}
            rows.append({"run_id": run_dir.name, **condition, **position, **row})
            step += 1
    return rows


def collect_cohorts(run_dir: Path) -> list[dict[str, Any]]:
    """cohort stage ごとに、群単位の `q`・class weight・support・AUROC を並べる。

    stage の最終 epoch の値を取る。`q` は stage 内で累積するので、最終 epoch が
    その stage で DRO が到達した重み配分になる。

    Args:
        run_dir: `projects/hypernet_iterative/runs/<run-id>`

    Returns:
        list[dict[str, Any]]: `(run_id, 条件, stage, group)` 1 組につき 1 行
    """
    condition = read_condition(run_dir)
    rows: list[dict[str, Any]] = []
    for stage_dir in sorted((run_dir / "stages").glob("stage*")):
        config = read_config(stage_dir / "config.yaml")
        weights = config["model"]["loss_fn"].get("class_weight")
        # 群ごとの class weight を持つ stage だけが cohort stage になる。
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


def collect_baseline(run_dir: Path) -> list[dict[str, Any]]:
    """baseline run の epoch metric を、iterative の表と同じ列名で並べる。

    baseline は stage を持たない 1 本の fit なので `run_epoch` と `epoch` は一致する。
    条件列は `experiment_name` だけを入れる（変調範囲も step size も無い）。

    Args:
        run_dir: `projects/hypernet_e2e/runs/<run-id>`

    Returns:
        list[dict[str, Any]]: run-id・条件・epoch を付けた行
    """
    condition = read_config(run_dir / "config.yaml")["experiment_name"]
    history = read_wandb_history(run_dir)
    return [{"run_id": run_dir.name, "condition": condition, "run_epoch": int(row["epoch"]), **row} for row in history]


# baseline と並べて読める global 指標。hidden cohort 系は baseline に無い（cohort を引かない）。
GLOBAL_COLUMNS = {
    "val/auroc": "global AUROC",
    "val/bacc": "global bACC",
    "val/loss": "val loss",
    "val/sex/worst_group_auroc": "sex worst AUROC",
    "val/race/worst_group_auroc": "race worst AUROC",
    "val/ethnicity/worst_group_auroc": "ethnicity worst AUROC",
    "val/age_group_65/worst_group_auroc": "age worst AUROC",
}

# `q` が何に寄ったのかの対立仮説。q ~ positive weight は「陽性率の低い群に寄る」と同義で、
# q ~ val AUROC が想定どおりの「難しさ由来」、q ~ train size は「少数群は loss の分散が大きい」。
CORRELATION_PAIRS = {
    "q ~ positive weight": "positive_weight",
    "q ~ val AUROC": "val_auroc",
    "q ~ train size": "train_size",
}


def global_comparison(frame: pd.DataFrame, baseline: pd.DataFrame) -> pd.DataFrame:
    """iterative の到達点と baseline を、同じ global 指標の 1 表に並べる。

    baseline は 3 点で見る。iterative と同じ epoch 予算での位置、checkpoint に選ばれる最良、
    最後まで回した先。同じ epoch 数で比べるだけだと、baseline がまだ伸びる途中なのか
    収束済みなのかが読めない。

    Args:
        frame: `collect` が作った epoch 単位の表
        baseline: `collect_baseline` が作った epoch 単位の表

    Returns:
        pd.DataFrame: `run` を index にした比較表
    """
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


def cohort_correlations(cohorts: pd.DataFrame) -> pd.DataFrame:
    """stage の中だけで、`q` が何と相関したのかを見る。

    cohort は stage ごとに引き直されるので「群 k がどうなったか」は追えない。追えるのは
    「DRO が何を難しさと見なしたか」で、これは stage 内で閉じた問いとして立てられる。
    4 run × 2 stage = 8 回の再抽選がそのまま反復サンプルになる。

    Args:
        cohorts: `collect_cohorts` が作った群単位の表

    Returns:
        pd.DataFrame: `(条件, stage)` ごとの Spearman 相関と `q` の広がり
    """
    rows = []
    for (condition, stage), data in cohorts.groupby(["condition", "stage"]):
        correlations = {
            name: data["q"].corr(data[column], method="spearman") for name, column in CORRELATION_PAIRS.items()
        }
        spread = data["q"].max() - data["q"].min()
        rows.append({"condition": condition, "stage": stage, **correlations, "q spread": spread})
    return pd.DataFrame(rows)


def main() -> None:
    """run-id を受け取り `results/` へ表を書く。

    Args:
        なし

    Returns:
        None
    """
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("run_ids", nargs="+")
    parser.add_argument("--runs-root", type=Path, default=runs_root("hypernet_iterative"))
    parser.add_argument("--baseline", nargs="*", default=[], help="global 指標の比較対象にする hypernet_e2e の run-id")
    parser.add_argument("--baseline-runs-root", type=Path, default=runs_root("hypernet_e2e"))
    args = parser.parse_args()

    rows = [row for run_id in args.run_ids for row in collect(args.runs_root / run_id)]
    fixed = ["run_id", "condition", "modulation", "step_size", "stage", "stage_index", "run_epoch", "epoch"]
    metrics = sorted({key for row in rows for key in row} - set(fixed))
    RESULTS.mkdir(exist_ok=True)
    write_csv(RESULTS / "epoch_metrics.csv", rows, fixed + metrics)
    write_csv(RESULTS / "headline.csv", rows, fixed + [column for column in HEADLINE if column in metrics])

    cohorts = [row for run_id in args.run_ids for row in collect_cohorts(args.runs_root / run_id)]
    write_csv(RESULTS / "cohort_groups.csv", cohorts, list(cohorts[0]) if cohorts else [])
    cohort_correlations(pd.DataFrame(cohorts)).to_csv(RESULTS / "cohort_correlations.csv", index=False)
    print(f"{len(args.run_ids)} runs / {len(rows)} epochs / {len(cohorts)} cohort groups -> {RESULTS}")

    if args.baseline:
        baseline = [row for run_id in args.baseline for row in collect_baseline(args.baseline_runs_root / run_id)]
        fixed = ["run_id", "condition", "run_epoch", "epoch"]
        columns = fixed + sorted({key for row in baseline for key in row} - set(fixed))
        write_csv(RESULTS / "baseline_epoch_metrics.csv", baseline, columns)
        comparison = global_comparison(pd.DataFrame(rows), pd.DataFrame(baseline))
        comparison.to_csv(RESULTS / "global_comparison.csv")
        print(f"{len(args.baseline)} baseline runs / {len(baseline)} epochs -> {RESULTS}/baseline_epoch_metrics.csv")


if __name__ == "__main__":
    main()
