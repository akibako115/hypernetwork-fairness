"""race-known の主比較・全stage追加条件が共通の比較条件と評価群を使うことを検証する。"""

from pathlib import Path

import pandas as pd
import pytest
import torch
from hydra import compose, initialize_config_dir
from hydra.utils import instantiate

from projects.hypernet_e2e.callbacks.fairness_metrics import FairnessMetricsCallback
from projects.hypernet_e2e.data.group_datamodule import demographic_group_ids
from projects.hypernet_e2e.training import _resolve_inverse_class_weights
from projects.hypernet_e2e.utils.metrics import compute_fairness_metrics

PRESETS = [
    "spatial_lora_chexpert_stage1_4_race_known",
    "spatial_lora_chexpert_stage1_4_age_race_group_dro_race_known",
    "spatial_lora_chexpert_stage1_4_age_race_group_dro_subgroup_race_known",
    "spatial_lora_chexpert_stage1_4_from_resnet_age_race_group_dro_race_known",
    "spatial_lora_chexpert_stage1_4_from_resnet_age_race_group_dro_subgroup_race_known",
    "hyperadapt_chexpert_race_known",
    "hyperadapt_chexpert_age_race_group_dro_race_known",
    "hyperadapt_chexpert_from_resnet_age_race_group_dro_race_known",
    "resnet_chexpert_age_race_group_dro_subgroup_race_known",
    "spatial_lora_chexpert_age_race_group_dro_subgroup_race_known",
    "spatial_lora_chexpert_from_resnet_age_race_group_dro_subgroup_race_known",
    "hyperadapt_chexpert_age_race_group_dro_subgroup_race_known",
    "hyperadapt_chexpert_from_resnet_age_race_group_dro_subgroup_race_known",
    "resnet_chexpert_race_known",
    "resnet_chexpert_age_race_group_dro_race_known",
    "spatial_lora_chexpert_race_known",
    "spatial_lora_chexpert_age_race_group_dro_race_known",
    "spatial_lora_chexpert_from_resnet_age_race_group_dro_race_known",
]


def test_standard_seed_preserves_legacy_default_and_allows_cli_override():
    with initialize_config_dir(version_base="1.3", config_dir=str(Path(__file__).parents[1] / "configs")):
        assert compose(config_name="train", overrides=["experiment=resnet_chexpert"]).seed == 12345
        assert compose(config_name="train", overrides=["experiment=resnet_chexpert_race_known", "seed=43"]).seed == 43


@pytest.mark.parametrize("preset", PRESETS)
def test_standard_config(preset, tmp_path):
    with initialize_config_dir(version_base="1.3", config_dir=str(Path(__file__).parents[1] / "configs")):
        config = compose(config_name="train", overrides=[f"experiment={preset}"])
    assert str(config.data.cv_splits_dir).endswith("/chexpert/splits_race_known")
    assert config.seed == 42
    assert config.data.batch_size == 128
    assert config.trainer.max_epochs == 30
    assert config.model.optimizer.lr == 1e-4
    assert config.model.optimizer.weight_decay == 1e-2
    assert config.model.scheduler is None
    assert config.weighting == "inverse"
    assert config.data.train_sampling == "uniform"
    assert config.logger.wandb.project == "fairness_hypernet"
    definition = config.callbacks.fairness_metrics.intersectional_groups.age_race_group
    assert list(definition.attributes) == ["age_group_65", "race_group"]
    assert list(definition.cardinalities) == [2, 4]
    assert list(config.data.attribute_spec.categorical_cardinalities) == [2, 6, 2, 2, 2]
    assert list(config.data.attribute_names.categorical) == ["sex", "race", "ethnicity", "frontal_lateral", "ap_pa"]
    assert list(config.data.fairness_attribute_names.categorical) == ["sex", "race_group", "ethnicity", "age_group_65"]
    assert isinstance(instantiate(config.callbacks.fairness_metrics), FairnessMetricsCallback)
    assert config.callbacks.model_checkpoint.monitor == "val/auroc"
    if "group_dro" in preset:
        assert config.training_strategy.class_weight_scope == "global"
        assert config.training_strategy.get("q_class_weight_scope") == ("subgroup" if "subgroup" in preset else None)
        assert config.model.loss_fn.step_size == 0.01
        assert config.data.num_groups == 8
        assert list(config.data.group_attribute_names) == ["age_group_65", "race_group"]
        assert list(config.data.group_cardinalities) == [2, 4]
        assert not config.data.group_missing_values
    if "from_resnet" in preset:
        assert config.model.freeze_backbone
        assert config.model.backbone_checkpoint_path is None
    else:
        assert not config.model.freeze_backbone
        assert str(config.model.backbone_checkpoint_path).endswith("resnet50_imagenet.pth")
    pd.DataFrame([{"target": target, "age": age, "age_missing": False, "race_group": race, "race_group_missing": False} for age in [40, 70] for race in range(4) for target in [0, 0, 0, 1]]).to_csv(
        tmp_path / "train.csv", index=False
    )
    config.data.cv_splits_dir = str(tmp_path)
    _resolve_inverse_class_weights(config)
    # モデルの更新はどちらも全群共通の重み。subgroup の strategy だけが q 用の群ごとの重みを持つ。
    assert list(config.model.loss_fn.class_weight) == [0.5, 1.5]
    if "subgroup" in preset:
        assert [list(row) for row in config.model.loss_fn.q_class_weight] == [[0.666667, 2.0]] * 8
    else:
        assert "q_class_weight" not in config.model.loss_fn


class _Module:
    fairness_attribute_names = {"categorical": ["age_group_65", "race_group"]}

    def __init__(self):
        self.logged = {}

    def log(self, name, value):
        self.logged[name] = value


@pytest.mark.parametrize("prefix", ["val", "test"])
def test_intersectional_logging_matches_training_groups_and_direct_metrics(prefix):
    callback = FairnessMetricsCallback({"age_race_group": {"attributes": ["age_group_65", "race_group"], "cardinalities": [2, 4]}})
    module = _Module()
    values = torch.tensor([[age, race] for age in range(2) for race in range(4) for _ in range(2)] + [[99, 0], [0, 99]])
    missing = torch.zeros_like(values, dtype=torch.bool)
    missing[-2, 0] = True
    missing[-1, 1] = True
    targets = torch.tensor([0, 1] * 9)
    logits = torch.tensor([[4.0, 0.0], [0.0, 4.0]] * 8 + [[0.0, 4.0], [4.0, 0.0]])
    frame = pd.DataFrame(values[:-2].numpy(), columns=["age_group_65", "race_group"])
    frame["age_group_65_missing"] = False
    frame["race_group_missing"] = False
    ids = torch.tensor(demographic_group_ids(frame, ["age_group_65", "race_group"], [2, 4], group_key="group_id").to_numpy())
    expected = compute_fairness_metrics(logits[:-2], targets[:-2], {"categorical": ids[:, None]}, {"categorical": ["age_race_group"]})["age_race_group"]
    buffer = [
        {"logits": logits[sl], "target": targets[sl], "attributes": {"evaluation_categorical": values[sl], "evaluation_categorical_missing": missing[sl]}} for sl in [slice(0, 7), slice(7, None)]
    ]
    callback._log_attribute_metrics(module, prefix, buffer)
    for name, value in expected.items():
        assert module.logged[f"{prefix}/age_race_group/{name}"] == pytest.approx(value)


def test_intersectional_logging_rejects_unknown_or_out_of_range_attributes():
    callback = FairnessMetricsCallback({"group": {"attributes": ["race_group"], "cardinalities": [4]}})
    for names, values in [(["other"], [[0]]), (["race_group"], [[4]])]:
        with pytest.raises(ValueError):
            callback._append_intersectional_groups({"categorical": torch.tensor(values)}, {"categorical": names})


@pytest.mark.parametrize(
    "definition",
    [
        {"attributes": [], "cardinalities": []},
        {"attributes": ["race_group"], "cardinalities": [0]},
        {"attributes": ["race_group"], "cardinalities": [4, 2]},
    ],
)
def test_intersectional_logging_rejects_invalid_definition(definition):
    with pytest.raises(ValueError):
        FairnessMetricsCallback({"group": definition})


def test_class_weight_scope_still_puts_subgroup_weights_into_the_model_loss(tmp_path):
    """`class_weight_scope=subgroup` はモデルの更新の重みを群ごとにする knob で、q 用の重みとは別に残す。"""
    overrides = ["experiment=resnet_chexpert_age_race_group_dro_race_known", "training_strategy=uniform_group", "+training_strategy.class_weight_scope=subgroup"]
    with initialize_config_dir(version_base="1.3", config_dir=str(Path(__file__).parents[1] / "configs")):
        config = compose(config_name="train", overrides=overrides)
    rows = [{"target": target, "age": age, "age_missing": False, "race_group": race, "race_group_missing": False} for age in [40, 70] for race in range(4) for target in [0, 0, 0, 1]]
    pd.DataFrame(rows).to_csv(tmp_path / "train.csv", index=False)
    config.data.cv_splits_dir = str(tmp_path)
    _resolve_inverse_class_weights(config)
    assert [list(row) for row in config.model.loss_fn.class_weight] == [[0.666667, 2.0]] * 8
    assert "q_class_weight" not in config.model.loss_fn
    instantiate(config.model.loss_fn)
