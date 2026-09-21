"""子 process の epoch metric を親 W&B run へ集約する経路を検証する。"""

from pathlib import Path

import pytest

from projects.hypernet_iterative.run_logging import log_epoch_metrics, read_epoch_metrics


class _FakeRun:
    """`log` の呼び出しを記録するだけの W&B run の代役。"""

    def __init__(self) -> None:
        self.calls: list[tuple[dict, int]] = []

    def log(self, payload: dict, step: int) -> None:
        self.calls.append((payload, step))


def _write_csv(tmp_path: Path, body: str) -> Path:
    path = tmp_path / "metrics.csv"
    path.write_text(body, encoding="utf-8")
    return path


def test_read_merges_the_train_and_val_rows_of_one_epoch(tmp_path: Path) -> None:
    """CSVLogger は log_metrics 呼び出しごとに1行書くので、同じ epoch が複数行に散る。"""
    path = _write_csv(
        tmp_path,
        "epoch,step,train/group_dro/q_00,val/auroc,val/hidden_min_auroc\n0,0,,0.5,0.3\n0,0,0.1,,\n1,1,,0.6,0.35\n1,1,0.2,,\n",
    )

    rows = read_epoch_metrics(path)

    assert rows == [
        {"epoch": 0.0, "train/group_dro/q_00": 0.1, "val/auroc": 0.5, "val/hidden_min_auroc": 0.3},
        {"epoch": 1.0, "train/group_dro/q_00": 0.2, "val/auroc": 0.6, "val/hidden_min_auroc": 0.35},
    ]


def test_read_drops_the_optimizer_step_column(tmp_path: Path) -> None:
    """`step` は optimizer step であり、親が振る通し epoch と混ざる。"""
    path = _write_csv(tmp_path, "epoch,step,val/auroc\n0,137,0.5\n")

    assert read_epoch_metrics(path) == [{"epoch": 0.0, "val/auroc": 0.5}]


def test_read_returns_epochs_in_order_regardless_of_row_order(tmp_path: Path) -> None:
    path = _write_csv(tmp_path, "epoch,step,val/auroc\n2,2,0.7\n0,0,0.5\n1,1,0.6\n")

    assert [row["epoch"] for row in read_epoch_metrics(path)] == [0.0, 1.0, 2.0]


def test_read_rejects_a_stage_that_left_no_metrics(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="epoch metric"):
        read_epoch_metrics(tmp_path / "missing.csv")


def test_log_uses_a_run_wide_step_and_returns_the_next_offset() -> None:
    """stage をまたいで step が連番になるので、run 全体で1本の曲線になる。"""
    run = _FakeRun()
    rows = [{"epoch": 0.0, "val/auroc": 0.5}, {"epoch": 1.0, "val/auroc": 0.6}]

    offset = log_epoch_metrics(run, 3, rows, 8)

    assert offset == 10
    assert [step for _, step in run.calls] == [8, 9]


def test_log_marks_the_stage_boundary_without_prefixing_the_metric_names() -> None:
    """接頭辞を付けると stage ごとに別系列になる。境目は stage_index で読む。"""
    run = _FakeRun()

    log_epoch_metrics(run, 2, [{"epoch": 1.0, "val/auroc": 0.5}], 5)

    payload, step = run.calls[0]
    assert step == 5
    assert payload == {"val/auroc": 0.5, "stage_index": 2.0, "stage_epoch": 1.0}
