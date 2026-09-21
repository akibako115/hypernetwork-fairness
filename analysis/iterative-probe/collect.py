"""iterative run の epoch metric を 1 枚の表へ集約する。

使い方:
    uv run python analysis/iterative-probe/collect.py 20260921T081141Z-iterative-s42-5752
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

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


def collect(run_dir: Path) -> list[dict[str, object]]:
    """warmup と各 stage の epoch metric を通し番号付きで 1 本に並べる。

    Args:
        run_dir: `projects/hypernet_iterative/runs/<run-id>`

    Returns:
        list[dict[str, object]]: stage 名・stage 番号・通し epoch を付けた行
    """
    stages = json.loads((run_dir / "run.json").read_text())["stages"]
    names = ["warmup"] + sorted(n for n in stages if n.startswith("stage"))
    rows: list[dict[str, object]] = []
    step = 0
    for index, name in enumerate(names):
        for row in read_epoch_metrics(run_dir / "stages" / name / "metrics" / "metrics.csv"):
            rows.append({"stage": name, "stage_index": index, "run_epoch": step, **row})
            step += 1
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
    parser.add_argument("run_id")
    parser.add_argument("--runs-root", type=Path, default=RUNS_ROOT)
    args = parser.parse_args()

    rows = collect(args.runs_root / args.run_id)
    results = Path(__file__).parent / "results"
    fixed = ["stage", "stage_index", "run_epoch", "epoch"]
    every = fixed + sorted({key for row in rows for key in row} - set(fixed))
    write_csv(results / "epoch_metrics.csv", rows, every)
    write_csv(results / "headline.csv", rows, fixed + HEADLINE)
    print(f"{len(rows)} epochs -> {results}/epoch_metrics.csv, {results}/headline.csv")


if __name__ == "__main__":
    main()
