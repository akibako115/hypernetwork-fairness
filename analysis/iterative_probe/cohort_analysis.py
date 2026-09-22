"""iterative の cohort と GroupDRO の挙動を `results/` の表へ集約する。"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import rootutils

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from analysis.common.paths import runs_root  # noqa: E402
from analysis.common.run_artifacts import write_csv  # noqa: E402
from analysis.iterative_probe._shared import collect_cohort_rows  # noqa: E402

RESULTS = Path(__file__).parent / "results"
CORRELATION_PAIRS = {
    "q ~ positive weight": "positive_weight",
    "q ~ val AUROC": "val_auroc",
    "q ~ train size": "train_size",
}


def cohort_correlations(cohorts: pd.DataFrame) -> pd.DataFrame:
    """stage 内で q と cohort の性質の Spearman 相関を計算する。"""
    rows = []
    for (condition, stage), data in cohorts.groupby(["condition", "stage"]):
        correlations = {
            name: data["q"].corr(data[column], method="spearman") for name, column in CORRELATION_PAIRS.items()
        }
        rows.append(
            {
                "condition": condition,
                "stage": stage,
                **correlations,
                "q spread": data["q"].max() - data["q"].min(),
            }
        )
    return pd.DataFrame(rows)


def write_results(run_ids: list[str], runs_root_path: Path) -> list[dict[str, object]]:
    """指定 run の cohort 表と相関表を書き、cohort 行を返す。"""
    rows = [row for run_id in run_ids for row in collect_cohort_rows(runs_root_path / run_id)]
    RESULTS.mkdir(exist_ok=True)
    write_csv(RESULTS / "cohort_groups.csv", rows, list(rows[0]) if rows else [])
    cohort_correlations(pd.DataFrame(rows)).to_csv(RESULTS / "cohort_correlations.csv", index=False)
    print(f"{len(run_ids)} runs / {len(rows)} cohort groups -> {RESULTS}")
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_ids", nargs="+")
    parser.add_argument("--runs-root", type=Path, default=runs_root("hypernet_iterative"))
    args = parser.parse_args()
    write_results(args.run_ids, args.runs_root)


if __name__ == "__main__":
    main()
