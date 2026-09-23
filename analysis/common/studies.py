"""仮説 (study) に属する run を集め、`runs.md` に貼る表を作る。

`study` は学習の起動時に渡す必須 key で、run は `run.json` にその値を残す。ここはその
逆引きだけを持つ: 仮説の名前から、それを検証した run の一覧を出す。run-id・状態・
experiment・W&B URL を手で写すのをやめるためのもので、**なぜその条件なのか**の散文は
これまでどおり `runs.md` が持つ。

`study` は途中から入れた key なので、それ以前の run は持たない。持たない run は
`--all` を付けたときだけ `(study 未記録)` として出す。

走査するのは学習が書いた `projects/*/runs/` で、分析 package へ取り込む前の候補を探すために
使う。`run path` 列は、その study の package に取り込み済みならその path、まだなら空になる。
取り込みは `analysis/common/runs.py import`。

使い方:
    uv run python analysis/common/studies.py iterative_probe
"""

from __future__ import annotations

import argparse
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import rootutils

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from analysis.common.paths import REPOSITORY_ROOT  # noqa: E402
from analysis.common.run_artifacts import read_config, read_run_record  # noqa: E402

COLUMNS = ["run-id", "project", "experiment", "seed", "状態", "開始 (UTC)", "run path", "W&B"]


def iter_run_dirs() -> Iterator[Path]:
    """`projects/*/runs/` にある run directory を名前順に返す。

    Args:
        なし

    Returns:
        Iterator[Path]: `run.json` を持つ directory
    """
    for runs_root in sorted((REPOSITORY_ROOT / "projects").glob("*/runs")):
        for run_dir in sorted(runs_root.iterdir()):
            if (run_dir / "run.json").is_file():
                yield run_dir


def describe(run_dir: Path) -> dict[str, Any]:
    """1 run を表の 1 行にする。

    Args:
        run_dir: `projects/<project>/runs/<run-id>`（学習が書いた正本）

    Returns:
        dict[str, Any]: 表の列と、絞り込みに使う `study`
    """
    record = read_run_record(run_dir)
    config_path = run_dir / "config.yaml"
    config = read_config(config_path) if config_path.is_file() else {}
    # e2e は logger を複数持ちうるので list、iterative は parent run 1 つなので dict になる。
    wandb = record.get("wandb") or next(iter(record.get("loggers") or []), {})
    study = record.get("study")
    imported = _imported_path(study, run_dir)
    return {
        "study": record.get("study"),
        "run-id": run_dir.name,
        "project": run_dir.parents[1].name,
        "experiment": config.get("experiment_name", ""),
        "seed": record.get("seed"),
        "状態": record.get("status"),
        "開始 (UTC)": (record.get("started_at") or "")[:19],
        "run path": f"`{imported}`" if imported else "",
        "W&B": f"[{wandb.get('id', '')}]({wandb['url']})" if wandb.get("url") else "",
    }


def _imported_path(study: str | None, run_dir: Path) -> Path | None:
    """その run が study の package に取り込み済みなら、repo root からの path を返す。"""
    if not study:
        return None
    # `projects/` の走査と同じ root から組み立てる（test が root を差し替えられるように）。
    destination = REPOSITORY_ROOT / "analysis" / study / "runs" / run_dir.name
    if not (destination / "run.json").is_file():
        return None
    return destination.relative_to(REPOSITORY_ROOT)


def runs_for(study: str | None) -> list[dict[str, Any]]:
    """study に属する run を集める。

    Args:
        study: 仮説の名前。`None` なら全 run

    Returns:
        list[dict[str, Any]]: `describe` の行
    """
    rows = [describe(run_dir) for run_dir in iter_run_dirs()]
    if study is None:
        return rows
    return [row for row in rows if row["study"] == study]


def markdown_table(rows: list[dict[str, Any]], columns: list[str] = COLUMNS) -> str:
    """行を markdown の表にする。

    Args:
        rows: `describe` の行
        columns: 出力する列

    Returns:
        str: markdown の表。行が無ければその旨の 1 行
    """
    if not rows:
        return "（この study に属する run はまだ無い）"
    header = f"| {' | '.join(columns)} |"
    rule = f"|{'|'.join(['---'] * len(columns))}|"
    body = [f"| {' | '.join(str(row.get(column, '')) for column in columns)} |" for row in rows]
    return "\n".join([header, rule, *body])


def main() -> None:
    """study を受け取り、その run の表を標準出力へ書く。

    Args:
        なし

    Returns:
        None
    """
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("study", nargs="?", help="analysis/<study> の directory 名")
    parser.add_argument("--all", action="store_true", help="study を問わず全 run を出す")
    args = parser.parse_args()

    if not args.study and not args.all:
        parser.error("study を渡すか --all を付ける")
    rows = runs_for(None if args.all else args.study)
    if args.all:
        for row in rows:
            row["study"] = row["study"] or "(study 未記録)"
        print(markdown_table(rows, ["study", *COLUMNS]))
        return
    print(markdown_table(rows))


if __name__ == "__main__":
    main()
