"""iterative_probe の epoch・cohort・baseline 分析を一括実行する wrapper。"""

from __future__ import annotations

import argparse
from pathlib import Path

import rootutils

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from analysis.common.paths import runs_root  # noqa: E402
from analysis.iterative_probe.baseline_comparison import write_results as write_baseline_results  # noqa: E402
from analysis.iterative_probe.cohort_analysis import write_results as write_cohort_results  # noqa: E402
from analysis.iterative_probe.epoch_metrics import write_results as write_epoch_results  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_ids", nargs="+")
    parser.add_argument("--runs-root", type=Path, default=runs_root("hypernet_iterative"))
    parser.add_argument("--baseline", nargs="*", default=[])
    parser.add_argument("--baseline-runs-root", type=Path, default=runs_root("hypernet_e2e"))
    args = parser.parse_args()

    rows = write_epoch_results(args.run_ids, args.runs_root)
    write_cohort_results(args.run_ids, args.runs_root)
    if args.baseline:
        write_baseline_results(rows, args.baseline, args.baseline_runs_root)


if __name__ == "__main__":
    main()
