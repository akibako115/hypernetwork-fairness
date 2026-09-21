"""iterative の Hydra preset が stage ごとに必要な構成を合成できることを検証する。"""

from pathlib import Path

from hydra import compose, initialize_config_dir
from hydra.utils import instantiate
from omegaconf import OmegaConf

from projects.hypernet_iterative import workflow


def _compose(*overrides: str):
    config_dir = Path(__file__).parent.parent / "configs"
    with initialize_config_dir(version_base="1.3", config_dir=str(config_dir)):
        return compose(config_name="train", overrides=list(overrides))


def test_warmup_preset_uses_plain_datamodule_and_task_loss() -> None:
    config = _compose()

    assert config.data._target_ == "projects.hypernet_iterative.data.datamodule.ImageDataModule"
    assert config.model.loss_fn._target_ == "projects.hypernet_iterative.loss.TaskLoss"
    assert config.trainer.max_epochs == config.iteration.warmup_epochs
    assert config.weighting == "inverse"
    assert list(config.tags) == ["chexpert", "spatial_lora", "iterative"]
    assert instantiate(config.model) is not None
    assert config.checkpoint_selection.name == "global_auroc_bacc"
    assert config.callbacks.bacc_checkpoint.monitor == "val/bacc"


def test_cohort_stage_preset_connects_sidecar_group_dro_and_hidden_callbacks() -> None:
    config = _compose("experiment=spatial_lora_iterative_cohort_chexpert")

    assert config.data._target_ == "projects.hypernet_iterative.data.cohort_datamodule.CohortImageDataModule"
    assert config.model.loss_fn._target_ == "projects.hypernet_iterative.loss.GroupDROTaskLoss"
    assert config.data.num_groups == config.cohort.num_groups == config.iteration.clusters
    assert config.callbacks.hidden_cohort_logger.num_groups == config.cohort.num_groups
    assert config.trainer.max_epochs == config.iteration.stage_epochs
    assert config.weighting == "inverse"
    assert list(config.tags) == [
        "chexpert",
        "spatial_lora",
        "iterative_hidden_cohort",
        "strategy_group_dro",
        "cohort_metadata_kmeans",
    ]
    assert config.model.loss_fn.num_groups == config.cohort.num_groups
    assert instantiate(config.model) is not None


def test_hidden_min_auroc_selection_adds_its_checkpoint() -> None:
    config = _compose(
        "experiment=spatial_lora_iterative_cohort_chexpert",
        "checkpoint_selection=hidden_min_auroc",
    )

    assert config.checkpoint_selection.requires_hidden_cohort is True
    assert config.callbacks.hidden_min_auroc_checkpoint.monitor == "val/hidden_min_auroc"


def test_cohort_stage_can_select_each_supported_group_objective() -> None:
    expected = {
        "group_dro": ("GroupDROTaskLoss", True),
        "group_dro_balanced": ("ClassBalancedGroupDROTaskLoss", True),
        "uniform_group": ("UniformGroupTaskLoss", False),
        "uniform_group_iterative": ("UniformGroupTaskLoss", True),
    }

    for strategy, (loss_name, supports_warm_start) in expected.items():
        config = _compose("experiment=spatial_lora_iterative_cohort_chexpert", f"training_strategy={strategy}")
        assert config.model.loss_fn._target_.endswith(loss_name)
        assert config.model.loss_fn.num_groups == config.cohort.num_groups
        assert config.training_strategy.supports_warm_start is supports_warm_start


def test_cohort_stage_config_matches_the_training_strategy_group_it_declares(tmp_path) -> None:
    """parent が組み立てる loss_fn と、宣言側の config group を突き合わせる。

    `cohort_stage_config` は step_size などの数値を自前で持つため、`configs/training_strategy/`
    と二重定義になる。片方だけを変えたら落ちるように、ここで両者を比較する。
    """
    for strategy in ("group_dro", "group_dro_balanced", "uniform_group", "uniform_group_iterative"):
        warmup = _compose(f"iteration.cohort_training_strategy={strategy}")
        stage = workflow.cohort_stage_config(
            warmup,
            assignment_path=tmp_path / "assignments.parquet",
            checkpoint_path=tmp_path / "reference.ckpt",
            reference_id="warmup",
        )
        declared = _compose(
            "experiment=spatial_lora_iterative_cohort_chexpert",
            f"training_strategy={strategy}",
        )

        assert OmegaConf.to_container(stage.model.loss_fn, resolve=True) == OmegaConf.to_container(declared.model.loss_fn, resolve=True)
        assert stage.training_strategy.supports_warm_start == declared.training_strategy.supports_warm_start
        assert stage.training_strategy.uses_cohort_group_id == declared.training_strategy.uses_cohort_group_id
