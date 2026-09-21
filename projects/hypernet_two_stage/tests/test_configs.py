"""two-stage の Hydra preset が独立して合成・インスタンス化できることを検証する。"""

from pathlib import Path

import pytest
from hydra import compose, initialize_config_dir
from hydra.utils import instantiate

from projects.hypernet_two_stage.workflow import _stage_config


def _compose(**kwargs: object):
    config_dir = Path(__file__).parent.parent / "configs"
    with initialize_config_dir(version_base="1.3", config_dir=str(config_dir)):
        return compose(config_name="train", **kwargs)


def test_default_logger_is_wandb_with_its_output_left_to_the_workflow() -> None:
    cfg = _compose()

    assert set(cfg.logger) == {"wandb"}
    assert cfg.logger.wandb._target_ == "lightning.pytorch.loggers.wandb.WandbLogger"
    assert cfg.logger.wandb.project == "fairness_hypernet_two_stage"
    assert cfg.logger.wandb.log_model is False
    assert cfg.logger.wandb.save_dir is None
    assert cfg.logger.wandb.name is None


def test_hydra_job_logging_keeps_no_shared_log_file_outside_the_run_directory() -> None:
    cfg = _compose(return_hydra_config=True)

    assert set(cfg.hydra.job_logging.handlers) == {"console"}
    assert list(cfg.hydra.job_logging.root.handlers) == ["console"]


@pytest.mark.parametrize("strategy", ["erm", "inverse_weighted_loss", "inverse_weighted_sampling"])
def test_chexpert_presets_select_only_their_intended_weighting(strategy: str) -> None:
    cfg = _compose(overrides=[f"experiment=chexpert_{strategy}"])

    assert cfg.weighting == ("inverse" if strategy == "inverse_weighted_loss" else "none")
    assert cfg.data.train_sampling == ("inverse_frequency" if strategy == "inverse_weighted_sampling" else "uniform")
    assert instantiate(cfg.data) is not None


def test_both_stages_resolve_their_data_interpolations_and_instantiate() -> None:
    """stage の model は `${data.*}` を参照する。stage 設定が config の一部であることを保証する。"""
    cfg = _compose()

    stage1 = _stage_config(cfg, "stage1")
    stage2 = _stage_config(cfg, "stage2")

    assert stage1.model.net._target_.endswith("resnet.network.ResNet")
    assert stage2.model.net._target_.endswith("spatial_hypernet.network.SpatialLoRAResNet")
    assert stage2.model.net.num_classes == cfg.data.num_classes
    assert list(stage2.model.net.metadata_encoder.categorical_cardinalities) == list(cfg.data.attribute_spec.categorical_cardinalities)
    assert instantiate(stage1.model) is not None
    assert instantiate(stage2.model) is not None


def test_stage_one_trains_a_plain_resnet_and_stage_two_freezes_it() -> None:
    """Stage 1 が Spatial LoRA になる preset を残さない。凍結は stage2 設定に見える形で置く。"""
    cfg = _compose()

    assert cfg.stage1.model.use_attributes is False
    assert cfg.stage1.model.freeze_backbone is False
    assert cfg.stage2.model.use_attributes is True
    assert cfg.stage2.model.freeze_backbone is True


def test_stage_knobs_are_overridable_from_the_command_line_without_a_plus() -> None:
    """README が案内する stage2 の変調指定が、struct error なしでその stage にだけ届く。"""
    cfg = _compose(
        overrides=[
            "stage2.model.net.modulation_stages=[stage4,fc]",
            "stage2.model.net.rank=8",
            "stage2.model.optimizer.lr=0.0003",
            "stage2.trainer.max_epochs=10",
        ]
    )

    stage1 = _stage_config(cfg, "stage1")
    stage2 = _stage_config(cfg, "stage2")

    assert list(stage2.model.net.modulation_stages) == ["stage4", "fc"]
    assert stage2.model.net.rank == 8
    assert stage2.model.optimizer.lr == pytest.approx(0.0003)
    assert stage2.trainer.max_epochs == 10
    assert stage1.trainer.max_epochs == cfg.trainer.max_epochs
    assert stage1.model.optimizer.lr == cfg.stage1.model.optimizer.lr


def test_class_weight_is_resolved_in_one_place_for_both_stages() -> None:
    cfg = _compose(overrides=["class_weight=[0.4,1.6]"])

    assert _stage_config(cfg, "stage1").model.loss_fn.class_weight == [0.4, 1.6]
    assert _stage_config(cfg, "stage2").model.loss_fn.class_weight == [0.4, 1.6]
