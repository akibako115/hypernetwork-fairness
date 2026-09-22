"""iterative run の epoch metrics を `results/` の表へ集約する。"""

from __future__ import annotations

import argparse
from pathlib import Path

import rootutils

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from analysis.common.paths import runs_root  # noqa: E402
from analysis.common.run_artifacts import write_csv  # noqa: E402
from analysis.iterative_probe._shared import HEADLINE, collect_epoch_rows  # noqa: E402

RESULTS = Path(__file__).parent / "results"
FIXED_COLUMNS = ["run_id", "condition", "modulation", "step_size", "stage", "stage_index", "run_epoch", "epoch"]


def write_results(run_ids: list[str], runs_root_path: Path) -> list[dict[str, object]]:
    """指定 run の epoch metrics 表を書き、行を返す。"""
    rows = [row for run_id in run_ids for row in collect_epoch_rows(runs_root_path / run_id)]
    metrics = sorted({key for row in rows for key in row} - set(FIXED_COLUMNS))
    RESULTS.mkdir(exist_ok=True)
    write_csv(RESULTS / "epoch_metrics.csv", rows, FIXED_COLUMNS + metrics)
    write_csv(RESULTS / "headline.csv", rows, FIXED_COLUMNS + [column for column in HEADLINE if column in metrics])
    print(f"{len(run_ids)} runs / {len(rows)} epochs -> {RESULTS}")
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_ids", nargs="+")
    parser.add_argument("--runs-root", type=Path, default=runs_root("hypernet_iterative"))
    args = parser.parse_args()
    write_results(args.run_ids, args.runs_root)


if __name__ == "__main__":
    main()
