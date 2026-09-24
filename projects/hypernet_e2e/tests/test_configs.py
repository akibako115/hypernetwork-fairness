"""e2e の Hydra preset が独立して合成・インスタンス化できることを検証する。"""

from pathlib import Path
from typing import Any

import pytest
import yaml
from hydra import compose, initialize_config_dir
from hydra.errors import ConfigCompositionException
from hydra.utils import instantiate
from omegaconf import DictConfig

from projects.hypernet_e2e.training import _validate_attribute_invariance

CONFIG_DIR = Path(__file__).parent.parent / "configs"
EXPERIMENT_DIR = CONFIG_DIR / "experiment"

# CheXpert 系列で「何を振っているか」の正本。既定（inverse class weight・uniform sampling・ERM）から
# 逸脱する preset だけがこの表で違う値を持つ。preset を足したらこの表にも足す。足し忘れは
# test_every_chexpert_preset_is_registered_in_the_table が落とす。
CHEXPERT_PRESETS: dict[str, tuple[str, str, str]] = {
    "resnet_chexpert": ("inverse", "uniform", "erm"),
    "resnet_chexpert_attribute_invariant": ("inverse", "uniform", "erm"),
    "resnet_chexpert_group_dro": ("inverse", "uniform", "group_dro"),
    "resnet_chexpert_inverse_weighted_sampling": ("none", "inverse_frequency", "erm"),
    "spatial_lora_chexpert": ("inverse", "uniform", "erm"),
    "spatial_lora_chexpert_fc": ("inverse", "uniform", "erm"),
    "spatial_lora_chexpert_stage4_fc": ("inverse", "uniform", "erm"),
    "spatial_lora_chexpert_group_dro": ("inverse", "uniform", "group_dro"),
    "spatial_lora_chexpert_inverse_weighted_sampling": ("none", "inverse_frequency", "erm"),
    "spatial_lora_chexpert_from_resnet": ("inverse", "uniform", "erm"),
    "spatial_lora_chexpert_from_attribute_invariant": ("inverse", "uniform", "erm"),
    "spatial_lora_chexpert_from_resnet_group_dro": ("inverse", "uniform", "group_dro"),
}


def _compose(*overrides: str) -> DictConfig:
    """preset を明示した合成結果を返す。

    `experiment` は必須なので、既定に落ちる合成は書けない。

    Args:
        overrides: Hydra の override。`experiment=` を必ず含める

    Returns:
        DictConfig: 合成した設定
    """
    with initialize_config_dir(version_base="1.3", config_dir=str(CONFIG_DIR)):
        return compose(config_name="train", overrides=list(overrides))


def _preset_names() -> list[str]:
    """`configs/experiment/` に置いた、単体で選べる preset 名を返す。"""
    return sorted(path.stem for path in EXPERIMENT_DIR.glob("*.yaml") if not path.name.startswith("_"))


def _preset_body(name: str) -> dict[str, Any]:
    """preset file の、`defaults` を除いた中身を返す。"""
    document = yaml.safe_load((EXPERIMENT_DIR / f"{name}.yaml").read_text()) or {}
    return {key: value for key, value in document.items() if key != "defaults"}


def test_experiment_must_be_named_on_the_command_line() -> None:
    """既定の experiment を持たないこと。持つと、渡し忘れた起動が別条件で完走する。"""
    with pytest.raises(ConfigCompositionException, match="experiment"):
        _compose()


def test_every_chexpert_preset_is_registered_in_the_table() -> None:
    """CheXpert preset の追加・削除を表と一致させる。"""
    on_disk = {name for name in _preset_names() if "chexpert" in name}

    assert on_disk == set(CHEXPERT_PRESETS)


@pytest.mark.parametrize("preset", sorted(CHEXPERT_PRESETS))
def test_chexpert_presets_declare_only_their_intended_deviation(preset: str) -> None:
    """既定（inverse class weight・uniform sampling・ERM）からの逸脱が表どおりであることを見る。"""
    weighting, sampling, strategy = CHEXPERT_PRESETS[preset]
    cfg = _compose(f"experiment={preset}")

    assert (cfg.weighting, cfg.data.train_sampling, cfg.training_strategy.name) == (weighting, sampling, strategy)
    assert instantiate(cfg.data) is not None


@pytest.mark.parametrize("preset", _preset_names())
def test_no_preset_is_a_bare_alias_of_another(preset: str) -> None:
    """`defaults` しか持たない preset を置かない。選択肢だけが増えて条件は増えない。"""
    assert _preset_body(preset), f"{preset}.yaml は defaults だけで、条件を何も変えていない"


@pytest.mark.parametrize(
    ("experiment", "modulation_stages"),
    [
        ("spatial_lora_chexpert_fc", ["fc"]),
        ("spatial_lora_chexpert_stage4_fc", ["stage4", "fc"]),
    ],
)
def test_modulation_range_presets_change_only_the_modulated_stages(experiment: str, modulation_stages: list[str]) -> None:
    cfg = _compose(f"experiment={experiment}")
    default = _compose("experiment=spatial_lora_chexpert")

    assert list(cfg.model.net.modulation_stages) == modulation_stages
    # 変調範囲だけの比較にするため、class weight も目的関数も既定の inverse weighted ERM に揃える。
    assert cfg.weighting == "inverse"
    assert cfg.training_strategy.name == "erm"
    assert {key: value for key, value in cfg.model.net.items() if key != "modulation_stages"} == {key: value for key, value in default.model.net.items() if key != "modulation_stages"}
    assert instantiate(cfg.model) is not None


def test_default_callbacks_include_metrics_and_fairness_without_text_progress() -> None:
    cfg = _compose("experiment=spatial_lora_chexpert")

    assert set(cfg.callbacks) == {"model_checkpoint", "model_summary", "metrics_logger", "fairness_metrics"}
    assert cfg.callbacks.model_checkpoint.dirpath is None
    assert cfg.callbacks.model_checkpoint._target_ == "projects.hypernet_e2e.callbacks.checkpoint.LastEpochModelCheckpoint"


@pytest.mark.parametrize("name", _preset_names())
def test_every_preset_keeps_the_final_epoch_as_last_checkpoint(name: str) -> None:
    """`save_last` を持つ checkpoint は、標準の `ModelCheckpoint` へ戻さない。

    標準の `ModelCheckpoint` は `save_top_k=1` のとき `last.ckpt` を best の epoch で止める。
    preset が callback を差し替えても、`last` が最終 epoch を指し続けることを固定する。
    """
    cfg = _compose(f"experiment={name}", "study=scratch")

    assert cfg.callbacks.model_checkpoint.save_last is True
    for key, value in cfg.callbacks.items():
        if value.get("save_last") is True:
            assert value._target_ == "projects.hypernet_e2e.callbacks.checkpoint.LastEpochModelCheckpoint", key


def test_default_logger_is_wandb_with_its_output_left_to_the_run_record() -> None:
    cfg = _compose("experiment=spatial_lora_chexpert", "study=scratch")

    assert set(cfg.logger) == {"wandb"}
    assert cfg.logger.wandb._target_ == "lightning.pytorch.loggers.wandb.WandbLogger"
    assert cfg.logger.wandb.project == "fairness_hypernet"
    assert cfg.logger.wandb.log_model is False
    assert cfg.logger.wandb.save_dir is None
    assert cfg.logger.wandb.name is None
    # code project は job_type が持つ。W&B project は repo で 1 つに統合している。
    assert cfg.logger.wandb.job_type == "hypernet_e2e"
    # 仮説は config.study で絞る。group は seed 反復などを束ねる用途に空けてある。
    assert cfg.logger.wandb.group is None


def test_hydra_job_logging_keeps_no_shared_log_file_outside_the_run_directory() -> None:
    with initialize_config_dir(version_base="1.3", config_dir=str(CONFIG_DIR)):
        cfg = compose(config_name="train", overrides=["experiment=spatial_lora_chexpert"], return_hydra_config=True)

    assert set(cfg.hydra.job_logging.handlers) == {"console"}
    assert list(cfg.hydra.job_logging.root.handlers) == ["console"]


def test_from_resnet_preset_freezes_the_backbone_and_demands_a_stage1_checkpoint() -> None:
    cfg = _compose("experiment=spatial_lora_chexpert_from_resnet")

    assert cfg.model.freeze_backbone is True
    assert cfg.model.backbone_checkpoint_path is None
    with pytest.raises(Exception, match="backbone_checkpoint_path"):
        instantiate(cfg.model)


def test_attribute_invariant_stage1_and_its_stage2_preset_compose() -> None:
    stage1 = _compose("experiment=resnet_chexpert_attribute_invariant")
    stage2 = _compose("experiment=spatial_lora_chexpert_from_attribute_invariant")

    assert stage1.model.loss_fn._target_.endswith("AttributeInvariantTaskLoss")
    assert stage1.model.loss_fn.feature_dim == 2048
    assert list(stage1.model.loss_fn.adversarial_attribute_names.categorical) == ["sex", "race"]
    assert list(stage1.model.loss_fn.adversarial_attribute_names.continuous) == ["age"]
    assert instantiate(stage1.model).loss_fn.requires_features is True
    assert stage2.model.freeze_backbone is True


def test_attribute_invariant_preset_rejects_group_dro_override() -> None:
    config = _compose("experiment=resnet_chexpert_attribute_invariant", "training_strategy=group_dro")

    with pytest.raises(ValueError, match="training_strategy=erm"):
        _validate_attribute_invariance(config)


@pytest.mark.parametrize(
    ("experiment", "loss_target"),
    [
        ("resnet_chexpert_group_dro", "GroupDROTaskLoss"),
        ("spatial_lora_chexpert_group_dro", "GroupDROTaskLoss"),
        ("spatial_lora_chexpert_from_resnet_group_dro", "GroupDROTaskLoss"),
    ],
)
def test_group_presets_pair_the_grouped_datamodule_with_a_group_objective(experiment: str, loss_target: str) -> None:
    cfg = _compose(f"experiment={experiment}")

    assert cfg.training_strategy.name == "group_dro"
    assert cfg.data._target_.endswith("GroupImageDataModule")
    assert cfg.model.loss_fn._target_.endswith(loss_target)
    # group 数は data 側の定義が正本であり、目的関数はそれを参照するだけにする。
    assert cfg.model.loss_fn.num_groups == cfg.data.num_groups == 4
    assert cfg.model.loss_fn.group_key == cfg.data.group_key


@pytest.mark.parametrize("strategy", ["erm", "uniform_group", "group_dro", "group_dro_balanced"])
def test_training_strategy_is_selectable_without_changing_the_experiment(strategy: str) -> None:
    cfg = _compose("experiment=spatial_lora_chexpert_group_dro", f"training_strategy={strategy}")

    assert cfg.training_strategy.name == strategy
    assert cfg.training_strategy.uses_group_id is (strategy != "erm")
    assert instantiate(cfg.model) is not None


def test_plain_chexpert_preset_keeps_cross_entropy_and_no_group_id() -> None:
    cfg = _compose("experiment=spatial_lora_chexpert")

    assert cfg.training_strategy.name == "erm"
    assert cfg.model.loss_fn._target_.endswith("TaskLoss")
    assert cfg.data._target_ == "projects.hypernet_e2e.data.datamodule.ImageDataModule"
    assert "num_groups" not in cfg.data
