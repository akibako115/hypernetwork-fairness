"""iterative の Hydra preset が stage ごとに必要な構成を合成できることを検証する。"""

from pathlib import Path

import pytest
from hydra import compose, initialize_config_dir
from hydra.errors import ConfigCompositionException
from hydra.utils import instantiate
from omegaconf import OmegaConf

from projects.hypernet_iterative import stage as stage_runner
from projects.hypernet_iterative import workflow

CONFIG_DIR = Path(__file__).parent.parent / "configs"

# warmup 相当の基準 preset。`experiment` は config 側で必須にしてあるので、既定に落ちる合成は
# 書けない。stage を見る test は `experiment=` を自分で渡す。
BASELINE_EXPERIMENT = "spatial_lora_chexpert"


def _compose(*overrides: str):
    named = list(overrides)
    if not any(override.startswith("experiment=") for override in named):
        named.insert(0, f"experiment={BASELINE_EXPERIMENT}")
    # study も必須。`???` のままだと解決する test が MissingMandatoryValue で落ちる。
    if not any(override.startswith("study=") for override in named):
        named.insert(0, "study=scratch")
    with initialize_config_dir(version_base="1.3", config_dir=str(CONFIG_DIR)):
        return compose(config_name="train", overrides=named)


def test_experiment_must_be_named_on_the_command_line() -> None:
    """既定の experiment を持たないこと。持つと、渡し忘れた起動が別の変調範囲で完走する。"""
    with initialize_config_dir(version_base="1.3", config_dir=str(CONFIG_DIR)):
        with pytest.raises(ConfigCompositionException, match="experiment"):
            compose(config_name="train")


def test_warmup_preset_uses_plain_datamodule_and_task_loss() -> None:
    config = _compose()

    assert config.data._target_ == "projects.hypernet_iterative.data.datamodule.ImageDataModule"
    assert config.model.loss_fn._target_ == "projects.hypernet_iterative.loss.TaskLoss"
    assert config.trainer.max_epochs == config.iteration.warmup_epochs
    assert config.weighting == "inverse"
    # 条件は解決済み設定として W&B に載るので、preset は tag を付けない。
    assert list(config.tags) == []
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
    assert list(config.tags) == []
    assert config.model.loss_fn.num_groups == config.cohort.num_groups
    assert config.model.loss_fn.step_size == config.iteration.group_dro_step_size
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
    for strategy in ("group_dro", "uniform_group", "uniform_group_iterative"):
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


def _checkpoint_callbacks(config) -> dict:
    """ModelCheckpoint の callback だけを名前つきで取り出す。"""
    return {name: OmegaConf.to_container(value, resolve=True) for name, value in config.callbacks.items() if value.get("_target_", "").endswith("ModelCheckpoint")}


def test_cohort_stage_config_matches_the_checkpoint_selection_group_it_declares(tmp_path) -> None:
    """parent が置く checkpoint callback と、宣言側の config group を突き合わせる。

    `cohort_stage_config` は warmup の config を写してから差分を当てるので、選択を切り替えても
    前の選択が置いた補助 checkpoint が残りうる。宣言側と比較して、残ったら落ちるようにする。
    """
    for selection in ("global_auroc_bacc", "hidden_min_auroc"):
        warmup = _compose(f"iteration.cohort_checkpoint_selection={selection}")
        stage = workflow.cohort_stage_config(
            warmup,
            assignment_path=tmp_path / "assignments.parquet",
            checkpoint_path=tmp_path / "reference.ckpt",
            reference_id="warmup",
        )
        declared = _compose(
            "experiment=spatial_lora_iterative_cohort_chexpert",
            f"checkpoint_selection={selection}",
        )

        assert _checkpoint_callbacks(stage) == _checkpoint_callbacks(declared)
        assert stage.checkpoint_selection.name == declared.checkpoint_selection.name


def test_every_stage_keeps_the_final_epoch_as_last_and_writes_into_its_stage_directory(tmp_path) -> None:
    """次 stage の warm-start と cohort 生成は `last` を読むので、主 checkpoint は最終 epoch を残す。

    補助 checkpoint も含め、`stage.run` が dirpath を注入する target に入っていないと、
    checkpoint が stage directory の外に出る。
    """
    for selection in ("global_auroc_bacc", "hidden_min_auroc"):
        warmup = _compose(f"iteration.cohort_checkpoint_selection={selection}")
        stage = workflow.cohort_stage_config(
            warmup,
            assignment_path=tmp_path / "assignments.parquet",
            checkpoint_path=tmp_path / "reference.ckpt",
            reference_id="warmup",
        )
        for config in (warmup, stage):
            assert config.callbacks.model_checkpoint._target_ == "projects.hypernet_iterative.callbacks.checkpoint.LastEpochModelCheckpoint"
            assert {value["_target_"] for value in _checkpoint_callbacks(config).values()} <= stage_runner._CHECKPOINT_TARGETS


def test_modulation_presets_change_only_the_modulated_stages() -> None:
    expected = {
        "spatial_lora_chexpert_fc": ["fc"],
        "spatial_lora_chexpert_stage4_fc": ["stage4", "fc"],
    }
    baseline = _compose()

    for preset, stages in expected.items():
        config = _compose(f"experiment={preset}")

        assert list(config.model.net.modulation_stages) == stages
        # 変調範囲以外は既定の preset と同じでなければ、条件の比較にならない。
        assert config.model.net.rank == baseline.model.net.rank
        assert config.model.net.backbone == baseline.model.net.backbone
        assert config.model.backbone_checkpoint_path == baseline.model.backbone_checkpoint_path
        assert config.weighting == baseline.weighting
        assert instantiate(config.model) is not None


def test_modulation_presets_reach_the_cohort_stage_config() -> None:
    warmup = _compose("experiment=spatial_lora_chexpert_fc")

    stage = workflow.cohort_stage_config(
        warmup,
        assignment_path=Path("assignments.parquet"),
        checkpoint_path=Path("last.ckpt"),
        reference_id="warmup",
    )

    assert list(stage.model.net.modulation_stages) == ["fc"]


def test_default_logger_groups_runs_by_config_study_not_by_the_wandb_group() -> None:
    """仮説は `config.study` で絞る。group を埋めると seed 反復を束ねる用途が塞がる。"""
    cfg = _compose()

    assert cfg.logger.wandb.project == "fairness_hypernet"
    assert cfg.logger.wandb.job_type == "hypernet_iterative"
    assert cfg.logger.wandb.group is None
    assert cfg.study == "scratch"
