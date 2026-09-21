"""workflow が stage 起動前に使う config validation のテスト。"""

import pytest
from omegaconf import OmegaConf

from projects.hypernet_iterative.validation import validate_training_config


def _config(**overrides: object):
    config = {
        "cohort": {"name": "metadata_kmeans", "num_groups": 3},
        "data": {"group_assignment_path": "assignments.parquet"},
        "trainer": {"devices": 1, "num_nodes": 1},
        "model": {"warm_start_checkpoint_path": None},
        "training_strategy": {"name": "group_dro", "uses_cohort_group_id": True, "supports_warm_start": True},
        "callbacks": {"cohort_validity": {}, "hidden_cohort_logger": {}},
        "checkpoint_selection": {"name": "global_auroc_bacc"},
    }
    config.update(overrides)
    return OmegaConf.create(config)


def test_validation_requires_assignment_before_cohort_datamodule_is_created() -> None:
    config = _config(data={"group_assignment_path": None})

    with pytest.raises(ValueError, match="group_assignment_path"):
        validate_training_config(config)


def test_validation_rejects_warm_start_for_unsupported_strategy(tmp_path) -> None:
    checkpoint = tmp_path / "warm.ckpt"
    checkpoint.touch()
    config = _config(
        model={"warm_start_checkpoint_path": str(checkpoint)},
        training_strategy={"name": "uniform_group", "uses_cohort_group_id": True, "supports_warm_start": False},
    )

    with pytest.raises(ValueError, match="supports_warm_start"):
        validate_training_config(config)


def test_validation_rejects_a_group_class_weight_solved_for_another_cohort() -> None:
    """行数が cohort の group 数と合わない重みは、別 cohort 向けに解いたものである。"""
    config = _config(
        model={"warm_start_checkpoint_path": None, "loss_fn": {"class_weight": [[1.0, 2.0], [1.0, 2.0]]}},
    )

    with pytest.raises(ValueError, match="3 行である必要がある"):
        validate_training_config(config)


def test_validation_rejects_a_group_class_weight_without_a_cohort() -> None:
    config = _config(
        cohort=None,
        data={"group_assignment_path": None},
        training_strategy=None,
        model={"warm_start_checkpoint_path": None, "loss_fn": {"class_weight": [[1.0, 2.0], [1.0, 2.0], [1.0, 2.0]]}},
    )

    with pytest.raises(ValueError, match="cohort を使う stage でのみ"):
        validate_training_config(config)


def test_validation_rejects_a_class_weight_shared_by_every_group_on_a_cohort_stage() -> None:
    """cohort stage で共通の重みを使うと group loss に陽性率依存が残る。"""
    config = _config(model={"warm_start_checkpoint_path": None, "loss_fn": {"class_weight": [0.201358, 1.798642]}})

    with pytest.raises(ValueError, match="group ごとの"):
        validate_training_config(config)


@pytest.mark.parametrize("class_weight", [None, [[1.0, 2.0], [1.0, 2.0], [1.0, 2.0]]])
def test_validation_accepts_every_supported_class_weight_shape(class_weight) -> None:
    config = _config(model={"warm_start_checkpoint_path": None, "loss_fn": {"class_weight": class_weight}})

    validate_training_config(config)


def test_validation_requires_hidden_metric_producers_for_hidden_selection() -> None:
    config = _config(
        callbacks={},
        checkpoint_selection={"name": "hidden_min_auroc", "requires_hidden_cohort": True},
    )

    with pytest.raises(ValueError, match="cohort_validity"):
        validate_training_config(config)
