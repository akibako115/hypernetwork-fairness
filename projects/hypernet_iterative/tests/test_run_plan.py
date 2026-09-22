"""dry-run の表が、反復の規模と warmup の class weight を出すことを検証する。"""

from pathlib import Path

import pytest
from omegaconf import OmegaConf

from projects.hypernet_iterative.workflow import run_plan


def _config(tmp_path: Path):
    """inverse weighting の最小構成を返す。train split は 1:3 で不均衡にする。"""
    split_dir = tmp_path / "splits"
    split_dir.mkdir()
    (split_dir / "train.csv").write_text("target\n0\n1\n1\n1\n")
    return OmegaConf.create(
        {
            "study": "scratch",
            "project": "hypernet_iterative",
            "experiment_name": "unit-test",
            "seed": 42,
            "weighting": "inverse",
            "dataset": "chexpert",
            "data": {"cv_splits_dir": str(split_dir), "num_classes": 2, "batch_size": 8},
            "model": {"loss_fn": {"_target_": "projects.hypernet_iterative.loss.TaskLoss", "class_weight": None}},
            "trainer": {"max_epochs": 2},
            "callbacks": {"model_checkpoint": {}},
            "iteration": {"warmup_epochs": 2, "stage_epochs": 3, "stages": 4, "clusters": 10, "n_init": 5, "group_dro_step_size": 0.01},
        }
    )


def test_plan_shows_the_warmup_class_weight_and_the_total_epochs(tmp_path: Path) -> None:
    """起動前に確認したいのは重みの実値と run 全体の長さで、どちらも config には書いていない。"""
    table = run_plan(_config(tmp_path))

    assert "[1.5, 0.5]" in table
    # warmup 2 + stage 3 × 4 回。
    assert "total epochs" in table and "14" in table
    assert "cohort: cohort04" in table and "fit: stage04" in table


def test_plan_refuses_a_study_without_an_analysis_package(tmp_path: Path) -> None:
    config = _config(tmp_path)
    config.study = "no-such-study"

    with pytest.raises(ValueError, match="analysis/no-such-study"):
        run_plan(config)


def test_plan_leaves_no_parent_run_directory(tmp_path: Path) -> None:
    config = _config(tmp_path)
    config.paths = {"project_dir": str(tmp_path / "project")}

    run_plan(config)

    assert not (tmp_path / "project").exists()
