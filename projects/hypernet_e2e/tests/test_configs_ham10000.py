"""HAM10000 の preset が皮膚画像用の data・transform に繋がることを検証する。"""

from pathlib import Path

import pytest
from hydra import compose, initialize_config_dir
from hydra.utils import instantiate

CONFIG_DIR = Path(__file__).parent.parent / "configs"


@pytest.mark.parametrize("experiment", ["resnet_ham10000", "spatial_lora_ham10000", "hyperadapt_ham10000"])
def test_ham10000_preset_connects_the_dermoscopy_data_and_transforms(experiment: str) -> None:
    """HAM10000 は悪性 2 値で、属性は sex・localization・age、transform は ISIC 2019 と共通になる。"""
    with initialize_config_dir(version_base="1.3", config_dir=str(CONFIG_DIR)):
        config = compose(config_name="train", overrides=[f"experiment={experiment}"])

    assert config.dataset == "ham10000"
    assert config.weighting == "inverse"
    assert config.data.num_classes == config.model.net.num_classes == 2
    assert config.data.cv_splits_dir.endswith("ham10000/splits")
    assert config.data.train_transform._target_.endswith("train_transforms_medfair")
    assert config.data.val_transform._target_.endswith("val_transforms_medfair")
    categorical = list(config.data.attribute_names.categorical)
    assert len(categorical) == len(config.data.attribute_spec.categorical_cardinalities)
    # localization は患者の属性ではないので、モデル入力にだけ使い評価軸には入れない。
    assert "localization" in categorical
    assert "localization" not in config.data.fairness_attribute_names.categorical
    assert config.callbacks.fairness_metrics.intersectional_groups.sex_age_group.attributes == ["sex", "age_group_65"]
    assert instantiate(config.model) is not None


def test_hyperadapt_ham10000_trains_the_backbone_from_imagenet() -> None:
    """HyperAdapt の model 既定は凍結・path 無しなので、E2E の preset で ImageNet 重みと凍結解除を入れる。"""
    with initialize_config_dir(version_base="1.3", config_dir=str(CONFIG_DIR)):
        config = compose(config_name="train", overrides=["experiment=hyperadapt_ham10000"])

    assert config.model.backbone_checkpoint_path.endswith("resnet50_imagenet.pth")
    assert config.model.freeze_backbone is False


LATENT = {
    "spatial_lora_ham10000_const": (["const"], [1], 0),
    "spatial_lora_ham10000_site": (["site"], [4], 0),
    "spatial_lora_ham10000_random8": (["random8"], [8], 0),
    "spatial_lora_ham10000_gmm8": (["gmm8"], [8], 0),
    "spatial_lora_ham10000_gmm8lo": (["gmm8_lo"], [8], 0),
    "spatial_lora_ham10000_emb16": ([], [], 16),
    "spatial_lora_ham10000_emb16lo": ([], [], 16),
}


@pytest.mark.parametrize("experiment", sorted(LATENT))
def test_latent_condition_presets_swap_only_the_condition(experiment: str) -> None:
    """潜在の条件の preset は、条件の列だけを splits_latent の列に替え、評価軸は sex × age のまま残す。"""
    with initialize_config_dir(version_base="1.3", config_dir=str(CONFIG_DIR)):
        config = compose(config_name="train", overrides=[f"experiment={experiment}"])

    categorical, cardinalities, continuous_dim = LATENT[experiment]
    assert config.data.cv_splits_dir.endswith("ham10000/splits_latent")
    assert list(config.data.attribute_names.categorical) == categorical
    assert list(config.data.attribute_spec.categorical_cardinalities) == cardinalities
    assert len(config.data.attribute_names.continuous) == config.data.attribute_spec.continuous_dim == continuous_dim
    assert list(config.model.net.modulation_stages) == ["stage3", "stage4", "fc"]
    assert config.callbacks.fairness_metrics.intersectional_groups.sex_age_group.attributes == ["sex", "age_group_65"]


def test_fc_preset_modulates_only_the_classifier_with_attributes() -> None:
    """分類層だけの対照は、属性の入力を spatial_lora_ham10000 と揃え、変調範囲だけを fc に絞る。"""
    with initialize_config_dir(version_base="1.3", config_dir=str(CONFIG_DIR)):
        base = compose(config_name="train", overrides=["experiment=spatial_lora_ham10000"])
        fc = compose(config_name="train", overrides=["experiment=spatial_lora_ham10000_fc"])

    assert list(fc.model.net.modulation_stages) == ["fc"]
    assert fc.data.attribute_names == base.data.attribute_names
    assert fc.data.cv_splits_dir == base.data.cv_splits_dir


FAIRNESS = {
    "resnet_ham10000_fg_wst_lhcf": ("WorstCohortTaskLoss", "splits_lhcf", 11),
    "resnet_ham10000_fg_wst_sex_age": ("WorstCohortTaskLoss", "splits", 4),
    "resnet_ham10000_fg_fis_lhcf": ("FISTaskLoss", "splits_lhcf", 11),
    "resnet_ham10000_fg_fis_sex_age": ("FISTaskLoss", "splits", 4),
    "resnet_ham10000_fg_fami_lhcf": ("FaMITaskLoss", "splits_lhcf", 11),
    "resnet_ham10000_fg_fami_sex_age": ("FaMITaskLoss", "splits", 4),
    "resnet_ham10000_fg_popauc_sex_age": ("PopulationAUCTaskLoss", "splits", 4),
    "resnet_ham10000_fg_swadwst_lhcf": ("WorstCohortTaskLoss", "splits_lhcf", 11),
    "resnet_ham10000_fg_swadwst_sex_age": ("WorstCohortTaskLoss", "splits", 4),
    "resnet_ham10000_fg_swadwst_lhcfrand": ("WorstCohortTaskLoss", "splits_lhcf", 11),
    "resnet_ham10000_fg_swadwst_label": ("WorstCohortTaskLoss", "splits_lhcf", 2),
}


@pytest.mark.parametrize("experiment", sorted(FAIRNESS))
def test_lhcf_comparison_presets_connect_the_loss_to_the_group_columns(experiment: str) -> None:
    """LHCF の比較の preset は、目的関数の群数を群の列（性別 × 年齢か LHCF のコホート）に合わせる。"""
    with initialize_config_dir(version_base="1.3", config_dir=str(CONFIG_DIR)):
        config = compose(config_name="train", overrides=[f"experiment={experiment}"])

    loss, splits, groups = FAIRNESS[experiment]
    assert config.model.loss_fn._target_.endswith(loss)
    assert config.data.cv_splits_dir.endswith(f"ham10000/{splits}")
    assert config.data.num_groups == config.model.loss_fn.num_groups == groups
    assert config.callbacks.model_checkpoint_worst.monitor == "val/sex_age_group/worst_group_auroc"
    assert ("swad" in config.callbacks) == ("swad" in experiment)
    columns = {"lhcfrand": ["lhcf_random"], "label": ["label_group"], "lhcf": ["lhcf"]}
    expected = next((v for k, v in columns.items() if experiment.endswith(k)), None)
    if expected is not None:
        assert list(config.data.group_attribute_names) == expected


@pytest.mark.parametrize(
    "experiment",
    [
        "resnet_ham10000_rule",
        "spatial_lora_ham10000_rule_gmm8",
        "spatial_lora_ham10000_rule_const",
        "spatial_lora_ham10000_rule_strong_gmm8",
    ],
)
def test_rule_planting_presets_read_the_planted_images(experiment: str) -> None:
    """規則を植え込んだ場の preset は、画像だけを images_rule に替え、split は splits_latent を読む。"""
    with initialize_config_dir(version_base="1.3", config_dir=str(CONFIG_DIR)):
        config = compose(config_name="train", overrides=[f"experiment={experiment}"])

    assert config.data.data_dir.endswith("ham10000/images_rule_strong" if "strong" in experiment else "ham10000/images_rule")
    assert config.data.cv_splits_dir.endswith("ham10000/splits_latent")
    assert "-rule" in config.experiment_name


# Q2 の HAM10000 版。(class weight, train の抽出, 目的関数, held-out の split)
DUAL_HOLDOUT = {
    "resnet_ham10000_dual_holdout_erm": ("inverse", "uniform", "erm", None),
    "resnet_ham10000_sex_age_eo_rate_dual_holdout": ("inverse", "group_class_balanced", "equalized_odds_rate", "holdout"),
    "resnet_ham10000_sex_age_minimax_auc_dual_holdout": ("none", "group_class_balanced", "minimax_auc", "holdout"),
    "spatial_lora_ham10000_dual_holdout_erm": ("inverse", "uniform", "erm", None),
    "spatial_lora_ham10000_sex_age_eo_rate_dual_holdout": ("inverse", "group_class_balanced", "equalized_odds_rate", "holdout"),
    "spatial_lora_ham10000_sex_age_minimax_auc_dual_holdout": ("none", "group_class_balanced", "minimax_auc", "holdout"),
}


@pytest.mark.parametrize("experiment", sorted(DUAL_HOLDOUT))
def test_dual_holdout_presets_differ_only_in_the_objective(experiment: str) -> None:
    """ERM と公平性損失は同じ 80% の train で学習し、損失の run だけが群の重みを held-out で測る。

    モデルの構成と属性の入力は、既存の HAM10000 の主比較（resnet_ham10000・spatial_lora_ham10000）と揃える。
    """
    weighting, sampling, strategy, holdout = DUAL_HOLDOUT[experiment]
    base_name = experiment.split("_ham10000")[0] + "_ham10000"
    with initialize_config_dir(version_base="1.3", config_dir=str(CONFIG_DIR)):
        config = compose(config_name="train", overrides=[f"experiment={experiment}"])
        base = compose(config_name="train", overrides=[f"experiment={base_name}"])

    assert (config.weighting, config.data.train_sampling, config.training_strategy.name) == (weighting, sampling, strategy)
    assert config.data.cv_splits_dir.endswith("ham10000/splits_dual_holdout")
    assert config.data.get("holdout_split") == holdout
    assert config.model.net == base.model.net
    assert config.data.attribute_names == base.data.attribute_names
    assert config.trainer.max_epochs == base.trainer.max_epochs
    assert config.callbacks.model_checkpoint.save_top_k == -1
    if strategy != "erm":
        assert list(config.data.group_attribute_names) == ["sex", "age_group_65"]
        assert config.data.num_groups == config.model.loss_fn.num_groups == 4
    if strategy == "equalized_odds_rate":
        assert config.model.loss_fn.initial_task_weight == 0.9
    assert instantiate(config.data) is not None


# 柱 A の格子の HAM10000 版（analysis/hypernet_conditioning_grid_ham10000）。
# (目的関数, train の抽出, split, 変調段, 2 段か)。1 段の ERM は既存の resnet_ham10000・spatial_lora_ham10000・spatial_lora_ham10000_fc。
S34 = ["stage3", "stage4", "fc"]
CONDITIONING_GRID = {
    "resnet_ham10000_sex_age_group_dro": ("group_dro_global", "uniform", "splits", None, False),
    "resnet_ham10000_sex_age_eo_rate_dual_holdout": ("equalized_odds_rate", "group_class_balanced", "splits_dual_holdout", None, False),
    "spatial_lora_ham10000_sex_age_group_dro": ("group_dro_global", "uniform", "splits", S34, False),
    "spatial_lora_ham10000_sex_age_eo_rate_dual_holdout": ("equalized_odds_rate", "group_class_balanced", "splits_dual_holdout", S34, False),
    "spatial_lora_ham10000_fc_sex_age_group_dro": ("group_dro_global", "uniform", "splits", ["fc"], False),
    "spatial_lora_ham10000_fc_sex_age_eo_rate_dual_holdout": ("equalized_odds_rate", "group_class_balanced", "splits_dual_holdout", ["fc"], False),
    "spatial_lora_ham10000_from_resnet": ("erm", "uniform", "splits", S34, True),
    "spatial_lora_ham10000_from_resnet_sex_age_group_dro": ("group_dro_global", "uniform", "splits", S34, True),
    "spatial_lora_ham10000_from_resnet_sex_age_eo_rate_dual_holdout": ("equalized_odds_rate", "group_class_balanced", "splits_dual_holdout", S34, True),
    "spatial_lora_ham10000_fc_from_resnet": ("erm", "uniform", "splits", ["fc"], True),
    "spatial_lora_ham10000_fc_from_resnet_sex_age_group_dro": ("group_dro_global", "uniform", "splits", ["fc"], True),
    "spatial_lora_ham10000_fc_from_resnet_sex_age_eo_rate_dual_holdout": ("equalized_odds_rate", "group_class_balanced", "splits_dual_holdout", ["fc"], True),
}


@pytest.mark.parametrize("experiment", sorted(CONDITIONING_GRID))
def test_conditioning_grid_presets_vary_only_the_grid_axes(experiment: str) -> None:
    """格子の各条件は、目的関数・変調段・学習手順だけを既存の 1 段 ERM から変える。

    class weight は全条件で inverse、群は sex × age_group_65 の 4 群。2 段は親の checkpoint を起動時に渡す。
    """
    strategy, sampling, splits, modulation_stages, two_stage = CONDITIONING_GRID[experiment]
    base_name = "resnet_ham10000" if experiment.startswith("resnet") else "spatial_lora_ham10000"
    with initialize_config_dir(version_base="1.3", config_dir=str(CONFIG_DIR)):
        config = compose(config_name="train", overrides=[f"experiment={experiment}"])
        base = compose(config_name="train", overrides=[f"experiment={base_name}"])

    assert (config.training_strategy.name, config.data.train_sampling, config.weighting) == (strategy, sampling, "inverse")
    assert config.data.cv_splits_dir.endswith(f"ham10000/{splits}")
    assert config.data.attribute_names == base.data.attribute_names
    assert config.trainer.max_epochs == base.trainer.max_epochs
    assert config.model.optimizer == base.model.optimizer
    if modulation_stages is not None:
        assert list(config.model.net.modulation_stages) == modulation_stages
        assert {k: v for k, v in config.model.net.items() if k != "modulation_stages"} == {k: v for k, v in base.model.net.items() if k != "modulation_stages"}
    if strategy != "erm":
        assert list(config.data.group_attribute_names) == ["sex", "age_group_65"]
        assert config.data.num_groups == config.model.loss_fn.num_groups == 4
    if strategy == "group_dro_global":
        assert len(config.callbacks.group_dro_diagnostics.group_labels) == 4
    if strategy == "equalized_odds_rate":
        assert config.model.loss_fn.initial_task_weight == 0.9
    assert config.model.get("freeze_backbone", False) is two_stage
    if two_stage:
        assert config.model.backbone_checkpoint_path is None
    if two_stage and strategy == "erm":
        # 群の損失は class weight を起動時に解決するので、ここで instantiate できるのは ERM だけ。
        with pytest.raises(Exception, match="backbone_checkpoint_path"):
            instantiate(config.model)
