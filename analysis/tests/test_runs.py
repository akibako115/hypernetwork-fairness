"""run の取り込みが hardlink で、終わった run だけを受け付けることを固定する。"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from analysis.common import paths, runs


def _source(root: Path, status: str = "succeeded") -> Path:
    """学習が書いたのと同じ形の run directory を作る。"""
    source = root / "projects" / "hypernet_e2e" / "runs" / "20260922T063725Z-resnet-chexpert-s43-efcd"
    (source / "checkpoints").mkdir(parents=True)
    (source / "checkpoints" / "best_val_auroc_009.ckpt").write_bytes(b"weights")
    (source / "run.json").write_text(json.dumps({"kind": "fit", "status": status}), encoding="utf-8")
    (source / "wandb").mkdir()
    (source / "wandb" / "latest-run").symlink_to("run-x")
    return source


@pytest.fixture
def repository(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """package の置き場と repo root を差し替える。"""
    (tmp_path / "analysis" / "my_study").mkdir(parents=True)
    monkeypatch.setattr(paths, "ANALYSIS_ROOT", tmp_path / "analysis")
    monkeypatch.setattr(runs, "REPOSITORY_ROOT", tmp_path)
    return tmp_path


def test_an_imported_run_shares_its_files_with_the_source(repository: Path) -> None:
    """copy ではなく hardlink にする。checkpoint が 270MB/本あっても disk は増えない。"""
    source = _source(repository)

    destination, created = runs.import_run("my_study", source)

    assert created
    assert destination == repository / "analysis" / "my_study" / "runs" / source.name
    checkpoint = Path("checkpoints") / "best_val_auroc_009.ckpt"
    assert (destination / checkpoint).samefile(source / checkpoint)
    # symlink は辿らずに link のまま写す。
    assert (destination / "wandb" / "latest-run").is_symlink()


def test_importing_the_same_run_again_changes_nothing(repository: Path) -> None:
    """同じ run を 2 回渡しても失敗しない。`runs.md` の行を出し直すために使える。"""
    source = _source(repository)
    runs.import_run("my_study", source)

    _, created = runs.import_run("my_study", source)

    assert not created


def test_a_running_run_is_not_imported(repository: Path) -> None:
    """取り込んだ後に学習が作る file は package 側に現れない。終わるまで待つ。"""
    source = _source(repository, status="running")

    with pytest.raises(ValueError, match="status=running"):
        runs.import_run("my_study", source)
    assert not (repository / "analysis" / "my_study" / "runs").exists()


def test_a_study_without_a_package_is_refused(repository: Path) -> None:
    """打ち間違えた study に黙って directory を作らない。"""
    with pytest.raises(FileNotFoundError, match="analysis/no_such_study/"):
        runs.import_run("no_such_study", _source(repository))


def test_the_runs_md_row_names_the_project_and_the_package_path(repository: Path) -> None:
    """package の `runs/` には両 project の run が並ぶので、project は `run.json` から引く。"""
    destination, _ = runs.import_run("my_study", _source(repository))

    row = runs.runs_md_row(destination)

    assert row == f"| `{destination.name}` | `hypernet_e2e` | `analysis/my_study/runs/{destination.name}` |"
