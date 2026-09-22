"""dry-run の表が、実行と同じ解決を通った値を出すことを検証する。"""

from pathlib import Path

import pytest
from omegaconf import OmegaConf

from projects.hypernet_e2e.training import run_plan


def _config(tmp_path: Path):
    """inverse weighting の最小構成を返す。train split は 1:3 で不均衡にする。"""
    split_dir = tmp_path / "splits"
    split_dir.mkdir()
    (split_dir / "train.csv").write_text("target\n0\n1\n1\n1\n")
    return OmegaConf.create(
        {
            "study": "scratch",
            "project": "hypernet_e2e",
            "experiment_name": "unit-test",
            "seed": 42,
            "weighting": "inverse",
            "dataset": "chexpert",
            "data": {"cv_splits_dir": str(split_dir), "num_classes": 2, "batch_size": 8},
            "model": {"loss_fn": {"_target_": "projects.hypernet_e2e.loss.TaskLoss", "class_weight": None}},
            "training_strategy": {"name": "erm"},
            "trainer": {"max_epochs": 3},
            "callbacks": {"model_checkpoint": {}},
        }
    )


def test_plan_shows_the_class_weight_that_only_exists_after_resolution(tmp_path: Path) -> None:
    """`--cfg job` には出ない実値が出ること。ここが出ないと起動前に重みを確認できない。"""
    config = _config(tmp_path)

    table = run_plan(config)

    assert "class_weight" in table
    assert "[1.5, 0.5]" in table
    assert "study" in table and "scratch" in table


def test_plan_leaves_no_run_directory(tmp_path: Path) -> None:
    """dry-run が run を作らないこと。作ると失敗 run が並ぶ。"""
    config = _config(tmp_path)
    config.paths = {"project_dir": str(tmp_path / "project")}

    run_plan(config)

    assert not (tmp_path / "project").exists()


def test_plan_refuses_a_study_without_an_analysis_package(tmp_path: Path) -> None:
    config = _config(tmp_path)
    config.study = "no-such-study"

    with pytest.raises(ValueError, match="analysis/no-such-study"):
        run_plan(config)
