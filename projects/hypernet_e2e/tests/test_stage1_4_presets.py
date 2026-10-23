"""全stage版が変調範囲以外の標準条件を変えないことを検証する。"""

from pathlib import Path

import pytest
from hydra import compose, initialize_config_dir
from hydra.utils import instantiate
from omegaconf import OmegaConf

PARENTS = [
    "spatial_lora_chexpert_race_known",
    "spatial_lora_chexpert_age_race_group_dro_race_known",
    "spatial_lora_chexpert_age_race_group_dro_subgroup_race_known",
    "spatial_lora_chexpert_from_resnet_age_race_group_dro_race_known",
    "spatial_lora_chexpert_from_resnet_age_race_group_dro_subgroup_race_known",
]


def _compose(preset):
    with initialize_config_dir(version_base="1.3", config_dir=str(Path(__file__).parents[1] / "configs")):
        return compose(config_name="train", overrides=[f"experiment={preset}", "study=race_known_architecture_objective_comparison"])


@pytest.mark.parametrize("parent", PARENTS)
def test_all_stage_preset_changes_only_modulation_range(parent):
    name = parent.replace("spatial_lora_chexpert", "spatial_lora_chexpert_stage1_4", 1)
    baseline = OmegaConf.to_container(_compose(parent), resolve=True)
    variant = OmegaConf.to_container(_compose(name), resolve=True)
    assert variant.pop("experiment_name") == name.replace("_", "-")
    baseline.pop("experiment_name")
    assert variant["model"]["net"]["modulation_stages"] == ["stage1", "stage2", "stage3", "stage4", "fc"]
    assert baseline["model"]["net"]["modulation_stages"] == ["stage3", "stage4", "fc"]
    variant["model"]["net"].pop("modulation_stages")
    baseline["model"]["net"].pop("modulation_stages")
    assert variant == baseline


def test_model_has_conv2_adapters_in_all_sixteen_bottlenecks_and_keeps_base_frozen():
    config = _compose("spatial_lora_chexpert_stage1_4_race_known")
    net = instantiate(config.model.net)
    adapters = [name for name, module in net.named_modules() if name.endswith("conv2_adapter")]
    assert len(adapters) == 16
    assert not any(name.endswith(("conv1_adapter", "conv3_adapter")) for name, _ in net.named_modules())
    net.freeze_base_model()
    assert all(not parameter.requires_grad for parameter in net.fc.base_linear.parameters())
    assert any(parameter.requires_grad for parameter in net.parameters())
    for name, parameter in net.named_parameters():
        if "adapter" in name or "metadata_encoder" in name:
            continue
        if name.startswith("backbone."):
            assert not parameter.requires_grad, name
