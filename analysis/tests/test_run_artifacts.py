"""run artifact の読み方を固定する。

ここが間違うと、表も図も「もっともらしい別の数値」になって出る。落ちずに間違う経路
（epoch の取り違え、checkpoint の選び違い、列の欠落）を中心に押さえる。
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from analysis.common.run_artifacts import (
    read_config,
    read_epoch_metrics,
    read_run_record,
    selected_checkpoint,
    write_csv,
)


def _metrics_csv(path: Path, rows: list[str]) -> Path:
    """`CSVLogger` が書く形の metrics.csv を作る。"""
    path.write_text("\n".join(["epoch,step,train/loss,val/loss,val/auroc", *rows]) + "\n", encoding="utf-8")
    return path


def test_train_and_val_rows_of_one_epoch_become_one_row(tmp_path: Path) -> None:
    """Lightning は train と val を別行に書く。epoch で束ね直さないと片方しか読めない。"""
    path = _metrics_csv(tmp_path / "metrics.csv", ["0,10,,0.50,0.80", "0,10,0.60,,", "1,20,,0.40,0.85", "1,20,0.55,,"])

    rows = read_epoch_metrics(path)

    assert rows == [
        {"epoch": 0.0, "train/loss": 0.60, "val/loss": 0.50, "val/auroc": 0.80},
        {"epoch": 1.0, "train/loss": 0.55, "val/loss": 0.40, "val/auroc": 0.85},
    ]


def test_blank_cells_are_absent_metrics_not_zeros(tmp_path: Path) -> None:
    """空セルは「その行が記録しなかった」であって 0 ではない。0 にすると平均が動く。"""
    path = _metrics_csv(tmp_path / "metrics.csv", ["0,10,,0.50,"])

    assert read_epoch_metrics(path) == [{"epoch": 0.0, "val/loss": 0.50}]


def test_optimizer_step_is_not_carried_into_the_epoch_table(tmp_path: Path) -> None:
    """`step` は optimizer step の軸で、epoch と混ぜると横軸が二重になる。"""
    path = _metrics_csv(tmp_path / "metrics.csv", ["0,10,0.60,,"])

    assert "step" not in read_epoch_metrics(path)[0]


def test_epochs_come_back_in_ascending_order(tmp_path: Path) -> None:
    """記録順ではなく epoch 順で返す。推移の図がそのまま折れ線になる。"""
    path = _metrics_csv(tmp_path / "metrics.csv", ["2,30,0.4,,", "0,10,0.6,,", "1,20,0.5,,"])

    assert [row["epoch"] for row in read_epoch_metrics(path)] == [0.0, 1.0, 2.0]


def _run(tmp_path: Path, record: dict) -> Path:
    """`run.json` を持つ run directory を作る。"""
    run_dir = tmp_path / "20260921T103222Z-iterative-s42-d24d"
    run_dir.mkdir()
    (run_dir / "run.json").write_text(json.dumps(record), encoding="utf-8")
    return run_dir


def test_recorded_checkpoint_path_is_resolved_relative_to_the_run_directory(tmp_path: Path) -> None:
    """記録された path は実行環境（container）のもの。run directory からの位置だけを使う。"""
    recorded = "/workspaces/x/runs/20260921T103222Z-iterative-s42-d24d/stages/stage02/checkpoints/best.ckpt"
    run_dir = _run(tmp_path, {"selected_checkpoint": {"path": recorded}})

    assert selected_checkpoint(run_dir) == run_dir / "stages/stage02/checkpoints/best.ckpt"


def test_a_single_stage_run_falls_back_to_its_only_val_auroc_checkpoint(tmp_path: Path) -> None:
    """単段 run は stage を持たないので、val AUROC で選んだ checkpoint を直接取る。"""
    run_dir = _run(tmp_path, {"status": "succeeded"})
    (run_dir / "checkpoints").mkdir()
    (run_dir / "checkpoints" / "best_val_auroc_007.ckpt").touch()

    assert selected_checkpoint(run_dir) == run_dir / "checkpoints" / "best_val_auroc_007.ckpt"


@pytest.mark.parametrize("names", [[], ["best_val_auroc_001.ckpt", "best_val_auroc_002.ckpt"]])
def test_an_ambiguous_checkpoint_set_is_an_error_not_a_guess(tmp_path: Path, names: list[str]) -> None:
    """どれを評価したか決まらない run は、黙って 1 つ選ばずに落とす。"""
    run_dir = _run(tmp_path, {"status": "succeeded"})
    (run_dir / "checkpoints").mkdir()
    for name in names:
        (run_dir / "checkpoints" / name).touch()

    with pytest.raises(FileNotFoundError):
        selected_checkpoint(run_dir)


def test_run_record_and_config_are_read_as_written(tmp_path: Path) -> None:
    """run.json と config.yaml は run が残した契約そのものを返す。"""
    run_dir = _run(tmp_path, {"study": "iterative_probe", "seed": 42})
    (run_dir / "config.yaml").write_text("study: iterative_probe\nseed: 42\n", encoding="utf-8")

    assert read_run_record(run_dir) == {"study": "iterative_probe", "seed": 42}
    assert read_config(run_dir / "config.yaml") == {"study": "iterative_probe", "seed": 42}


def test_missing_columns_are_written_as_blanks_and_extra_keys_are_dropped(tmp_path: Path) -> None:
    """行ごとに持つ key が違っても、列は指定した順序で揃える。"""
    path = tmp_path / "out" / "table.csv"

    write_csv(path, [{"a": 1, "b": 2, "extra": 9}, {"a": 3}], ["a", "b"])

    assert path.read_bytes() == b"a,b\n1,2\n3,\n"
