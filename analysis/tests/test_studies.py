"""study から run を逆引きする表を固定する。

`runs.md` に貼る表なので、欠けている情報を空欄として出せることが要件になる。study は
途中から入れた key であり、それ以前の run は持たないまま並ぶ。
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from analysis.common import studies


def _run(root: Path, project: str, run_id: str, record: dict, config: dict | None = None) -> Path:
    """`projects/<project>/runs/<run-id>/` を作る。"""
    run_dir = root / "projects" / project / "runs" / run_id
    run_dir.mkdir(parents=True)
    (run_dir / "run.json").write_text(json.dumps(record), encoding="utf-8")
    if config is not None:
        (run_dir / "config.yaml").write_text(json.dumps(config), encoding="utf-8")
    return run_dir


@pytest.fixture
def repository(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """`projects/*/runs/` を走査する起点を差し替える。"""
    monkeypatch.setattr(studies, "REPOSITORY_ROOT", tmp_path)
    return tmp_path


def test_a_directory_without_a_run_record_is_not_a_run(repository: Path) -> None:
    """途中で消えた directory や作業用の directory を run として数えない。"""
    _run(repository, "hypernet_e2e", "20260921T103036Z-resnet-chexpert-s42-5538", {"status": "succeeded"})
    (repository / "projects" / "hypernet_e2e" / "runs" / "scratch").mkdir()

    assert [path.name for path in studies.iter_run_dirs()] == ["20260921T103036Z-resnet-chexpert-s42-5538"]


def test_a_run_row_carries_the_wandb_link_of_either_project_shape(repository: Path) -> None:
    """e2e は logger を複数持ちうるので list、iterative は親 run 1 つなので dict になる。"""
    parent = {"status": "succeeded", "wandb": {"id": "8az23qej", "url": "https://wandb.ai/x/y/runs/8az23qej"}}
    listed = {"status": "succeeded", "loggers": [{"id": "5538ab", "url": "https://wandb.ai/x/y/runs/5538ab"}]}
    iterative = _run(repository, "hypernet_iterative", "20260921T103222Z-iterative-s42-d24d", parent)
    e2e = _run(repository, "hypernet_e2e", "20260921T103036Z-resnet-chexpert-s42-5538", listed)

    assert studies.describe(iterative)["W&B"] == "[8az23qej](https://wandb.ai/x/y/runs/8az23qej)"
    assert studies.describe(e2e)["W&B"] == "[5538ab](https://wandb.ai/x/y/runs/5538ab)"


def test_a_run_without_a_wandb_reference_leaves_the_column_empty(repository: Path) -> None:
    """logger を付けずに回した run も表には並ぶ。"""
    run_id = "20260922T063121Z-spatial-lora-chexpert-fc-s42-825a"
    run_dir = _run(repository, "hypernet_e2e", run_id, {"status": "failed"})

    assert studies.describe(run_dir)["W&B"] == ""
    assert studies.describe(run_dir)["状態"] == "failed"


def test_a_run_recorded_before_study_existed_has_no_study(repository: Path) -> None:
    """study は途中から入れた key。持たない run を「未記録」として扱えることが要件。"""
    run_dir = _run(repository, "hypernet_e2e", "20260921T103036Z-resnet-chexpert-s42-5538", {"status": "succeeded"})

    assert studies.describe(run_dir)["study"] is None
    assert studies.runs_for("iterative_probe") == []


def test_runs_are_filtered_by_the_study_they_recorded(repository: Path) -> None:
    """仮説の名前から、それを検証した run だけを出す。"""
    record = {"status": "succeeded", "study": "iterative_probe", "seed": 42}
    config = {"experiment_name": "resnet-chexpert"}
    _run(repository, "hypernet_e2e", "20260921T103036Z-resnet-chexpert-s42-5538", record, config)
    other = "20260921T125711Z-resnet-chexpert-attribute-invariant-s42-9fd3"
    _run(repository, "hypernet_e2e", other, {"study": "scratch"})

    rows = studies.runs_for("iterative_probe")

    assert [row["run-id"] for row in rows] == ["20260921T103036Z-resnet-chexpert-s42-5538"]
    assert rows[0]["project"] == "hypernet_e2e"
    assert rows[0]["experiment"] == "resnet-chexpert"


def test_the_started_at_column_keeps_only_the_timestamp(repository: Path) -> None:
    """表に貼るので秒までにする。`run.json` の値そのものは run が持っている。"""
    record = {"status": "succeeded", "started_at": "2026-09-21T10:30:36.123456+00:00"}
    run_dir = _run(repository, "hypernet_e2e", "20260921T103036Z-resnet-chexpert-s42-5538", record)

    assert studies.describe(run_dir)["開始 (UTC)"] == "2026-09-21T10:30:36"


def test_an_empty_result_is_a_sentence_not_an_empty_table() -> None:
    """表の枠だけを貼ると「まだ回していない」のか「読み落とした」のか区別できない。"""
    assert studies.markdown_table([]) == "（この study に属する run はまだ無い）"


def test_the_table_has_one_header_one_rule_and_one_row_per_run() -> None:
    """`runs.md` へそのまま貼れる markdown にする。"""
    rows = [{"run-id": "a", "project": "hypernet_e2e", "seed": 42}]

    lines = studies.markdown_table(rows, ["run-id", "project", "seed"]).splitlines()

    assert lines == ["| run-id | project | seed |", "|---|---|---|", "| a | hypernet_e2e | 42 |"]
