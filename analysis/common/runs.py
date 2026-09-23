"""学習が書いた run を、分析 package の `runs/` へ hardlink で取り込む。

学習は `projects/<project>/runs/<run-id>/` に書く。分析は **自分の package の `runs/` だけを
読む**ので、使う run を先にここへ取り込む。取り込みは directory を作り直し、file は hardlink
にする（`cp -al` と同じ）。disk は増えず、同じ run を複数の package が引ける。
`projects/` 側の run を整理しても package の中身は消えない。

hardlink は元の file と同じ inode なので、**取り込んだ file を書き換えると正本も変わる**。
run artifact は不変という約束がそのまま効く。実行中の run は取り込まない。取り込んだ後に
学習が作る file（checkpoint など）は package 側に現れないため。

使い方:
    uv run python analysis/common/runs.py import initial_resnet_vs_invariant \
      projects/hypernet_e2e/runs/<run-id> projects/hypernet_e2e/runs/<run-id>
"""

from __future__ import annotations

import argparse
import os
import shutil
from pathlib import Path

import rootutils

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from analysis.common.paths import REPOSITORY_ROOT, run_dir, study_dir  # noqa: E402
from analysis.common.run_artifacts import read_run_record, run_project  # noqa: E402

FINISHED = ("succeeded", "failed")


def import_run(study: str, source: Path) -> tuple[Path, bool]:
    """1 run を `analysis/<study>/runs/<run-id>/` へ hardlink で取り込む。

    Args:
        study: 取り込み先の package 名
        source: 学習が書いた run directory

    Returns:
        tuple[Path, bool]: 取り込み先と、今回新しく作ったかどうか。既に同じ run が
            取り込まれていれば作らずに `False` を返す

    Raises:
        FileNotFoundError: package か `run.json` が無い場合
        ValueError: run が終わっていない場合
        FileExistsError: 取り込み先に別の実体を持つ directory がある場合
    """
    if not study_dir(study).is_dir():
        raise FileNotFoundError(f"analysis/{study}/ が無い。先に package を作る")
    source = source.resolve()
    record_path = source / "run.json"
    if not record_path.is_file():
        raise FileNotFoundError(f"{source} は run directory ではない（run.json が無い）")
    status = read_run_record(source).get("status")
    if status not in FINISHED:
        raise ValueError(f"{source.name} は status={status}。終わった run だけを取り込む")

    destination = run_dir(study, source.name)
    if destination.exists():
        if (destination / "run.json").samefile(record_path):
            return destination, False
        raise FileExistsError(f"{destination} に別の run がある")
    destination.parent.mkdir(parents=True, exist_ok=True)
    # symlink（wandb の latest-run など）は link のまま写す。実体を辿って複製しない。
    shutil.copytree(source, destination, symlinks=True, copy_function=os.link)
    return destination, True


def runs_md_row(destination: Path) -> str:
    """取り込んだ run を `runs.md` の表に貼る 1 行にする。

    Args:
        destination: `import_run` が返した取り込み先

    Returns:
        str: `| run-id | project | run path |`
    """
    relative = destination.relative_to(REPOSITORY_ROOT)
    return f"| `{destination.name}` | `{run_project(destination)}` | `{relative}` |"


def main() -> None:
    """run を package へ取り込み、`runs.md` に貼る行を出す。

    Args:
        なし

    Returns:
        None
    """
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest="command", required=True)
    importer = commands.add_parser("import", help="run を analysis/<study>/runs/ へ hardlink で取り込む")
    importer.add_argument("study", help="取り込み先の analysis/<study>")
    importer.add_argument("run_dirs", nargs="+", type=Path, help="projects/<project>/runs/<run-id>")
    args = parser.parse_args()

    rows = []
    for source in args.run_dirs:
        destination, created = import_run(args.study, REPOSITORY_ROOT / source)
        print(f"{'imported' if created else 'already imported'}: {destination.relative_to(REPOSITORY_ROOT)}")
        rows.append(runs_md_row(destination))
    print("\n| run-id | project | run path |\n|---|---|---|")
    print("\n".join(rows))


if __name__ == "__main__":
    main()
