"""e2e の Hydra preset が独立して合成・インスタンス化できることを検証する。"""

from pathlib import Path

import pytest
from hydra import compose, initialize_config_dir
from hydra.utils import instantiate


@pytest.mark.parametrize("architecture", ["resnet_chexpert", "spatial_lora_chexpert"])
@pytest.mark.parametrize("strategy", ["erm", "inverse_weighted_loss", "inverse_weighted_sampling"])
def test_chexpert_presets_select_only_their_intended_weighting(architecture: str, strategy: str) -> None:
    config_dir = Path(__file__).parent.parent / "configs"
    with initialize_config_dir(version_base="1.3", config_dir=str(config_dir)):
        cfg = compose(config_name="train", overrides=[f"experiment={architecture}_{strategy}"])

    assert cfg.weighting == ("inverse" if strategy == "inverse_weighted_loss" else "none")
    assert cfg.data.train_sampling == ("inverse_frequency" if strategy == "inverse_weighted_sampling" else "uniform")
    assert instantiate(cfg.data) is not None
    assert instantiate(cfg.model) is not None


def test_default_callbacks_include_metrics_and_fairness_without_text_progress() -> None:
    config_dir = Path(__file__).parent.parent / "configs"
    with initialize_config_dir(version_base="1.3", config_dir=str(config_dir)):
        cfg = compose(config_name="train")

    assert set(cfg.callbacks) == {"model_checkpoint", "model_summary", "metrics_logger", "fairness_metrics"}
    assert cfg.callbacks.model_checkpoint.dirpath is None


def test_default_logger_is_wandb_with_its_output_left_to_the_run_record() -> None:
    config_dir = Path(__file__).parent.parent / "configs"
    with initialize_config_dir(version_base="1.3", config_dir=str(config_dir)):
        cfg = compose(config_name="train")

    assert set(cfg.logger) == {"wandb"}
    assert cfg.logger.wandb._target_ == "lightning.pytorch.loggers.wandb.WandbLogger"
    assert cfg.logger.wandb.project == "fairness_hypernet_e2e"
    assert cfg.logger.wandb.log_model is False
    assert cfg.logger.wandb.save_dir is None
    assert cfg.logger.wandb.name is None


def test_hydra_job_logging_keeps_no_shared_log_file_outside_the_run_directory() -> None:
    config_dir = Path(__file__).parent.parent / "configs"
    with initialize_config_dir(version_base="1.3", config_dir=str(config_dir)):
        cfg = compose(config_name="train", return_hydra_config=True)

    assert set(cfg.hydra.job_logging.handlers) == {"console"}
    assert list(cfg.hydra.job_logging.root.handlers) == ["console"]


def test_from_resnet_preset_freezes_the_backbone_and_demands_a_stage1_checkpoint() -> None:
    config_dir = Path(__file__).parent.parent / "configs"
    with initialize_config_dir(version_base="1.3", config_dir=str(config_dir)):
        cfg = compose(config_name="train", overrides=["experiment=spatial_lora_chexpert_from_resnet"])

    assert cfg.model.freeze_backbone is True
    assert cfg.model.backbone_checkpoint_path is None
    with pytest.raises(Exception, match="backbone_checkpoint_path"):
        instantiate(cfg.model)


@pytest.mark.parametrize(
    ("experiment", "loss_target"),
    [
        ("resnet_chexpert_group_dro", "GroupDROTaskLoss"),
        ("spatial_lora_chexpert_group_dro", "GroupDROTaskLoss"),
        ("spatial_lora_chexpert_from_resnet_group_dro", "GroupDROTaskLoss"),
    ],
)
def test_group_presets_pair_the_grouped_datamodule_with_a_group_objective(experiment: str, loss_target: str) -> None:
    config_dir = Path(__file__).parent.parent / "configs"
    with initialize_config_dir(version_base="1.3", config_dir=str(config_dir)):
        cfg = compose(config_name="train", overrides=[f"experiment={experiment}"])

    assert cfg.training_strategy.name == "group_dro"
    assert cfg.data._target_.endswith("GroupImageDataModule")
    assert cfg.model.loss_fn._target_.endswith(loss_target)
    # group 数は data 側の定義が正本であり、目的関数はそれを参照するだけにする。
    assert cfg.model.loss_fn.num_groups == cfg.data.num_groups == 4
    assert cfg.model.loss_fn.group_key == cfg.data.group_key


@pytest.mark.parametrize("strategy", ["erm", "uniform_group", "group_dro", "group_dro_balanced"])
def test_training_strategy_is_selectable_without_changing_the_experiment(strategy: str) -> None:
    config_dir = Path(__file__).parent.parent / "configs"
    with initialize_config_dir(version_base="1.3", config_dir=str(config_dir)):
        cfg = compose(config_name="train", overrides=["experiment=spatial_lora_chexpert_group_dro", f"training_strategy={strategy}"])

    assert cfg.training_strategy.name == strategy
    assert cfg.training_strategy.uses_group_id is (strategy != "erm")
    assert instantiate(cfg.model) is not None


def test_default_experiment_keeps_plain_cross_entropy_and_no_group_id() -> None:
    config_dir = Path(__file__).parent.parent / "configs"
    with initialize_config_dir(version_base="1.3", config_dir=str(config_dir)):
        cfg = compose(config_name="train")

    assert cfg.training_strategy.name == "erm"
    assert cfg.model.loss_fn._target_.endswith("TaskLoss")
    assert cfg.data._target_ == "projects.hypernet_e2e.data.datamodule.ImageDataModule"
    assert "num_groups" not in cfg.data
