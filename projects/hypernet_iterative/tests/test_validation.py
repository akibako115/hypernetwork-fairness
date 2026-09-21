"""workflow が stage 起動前に使う config validation のテスト。"""

import pytest
from omegaconf import OmegaConf

from projects.hypernet_iterative.validation import validate_training_config


def _config(**overrides: object):
    config = {
        "cohort": {"name": "metadata_kmeans"},
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

    with pytest.raises(ValueError, match="cohort GroupDRO"):
        validate_training_config(config)


def test_validation_requires_hidden_metric_producers_for_hidden_selection() -> None:
    config = _config(
        callbacks={},
        checkpoint_selection={"name": "hidden_min_auroc", "requires_hidden_cohort": True},
    )

    with pytest.raises(ValueError, match="cohort_validity"):
        validate_training_config(config)
