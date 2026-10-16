"""FairVLMed の preset が眼底用の data・transform に繋がることを検証する。"""

from pathlib import Path

import pytest
from hydra import compose, initialize_config_dir
from hydra.utils import instantiate

CONFIG_DIR = Path(__file__).parent.parent / "configs"


@pytest.mark.parametrize("experiment", ["resnet_fairvlmed", "spatial_lora_fairvlmed", "hyperadapt_fairvlmed"])
def test_fairvlmed_preset_connects_the_fundus_data_and_transforms(experiment: str) -> None:
    """FairVLMed は glaucoma の 2 クラスで、属性も transform も CheXpert とは別物になる。"""
    with initialize_config_dir(version_base="1.3", config_dir=str(CONFIG_DIR)):
        config = compose(config_name="train", overrides=[f"experiment={experiment}"])

    assert config.dataset == "fairvlmed"
    assert config.data.num_classes == config.model.net.num_classes == 2
    assert config.data.cv_splits_dir.endswith("fairvlmed/splits")
    assert config.data.train_transform._target_.endswith("train_transforms_fairvlmed")
    assert config.data.val_transform._target_.endswith("val_transforms_fairvlmed")
    categorical = list(config.data.attribute_names.categorical)
    assert len(categorical) == len(config.data.attribute_spec.categorical_cardinalities)
    # maritalstatus は保護属性ではないので、モデル入力にだけ使い評価軸には入れない。
    assert "maritalstatus" in categorical
    assert "maritalstatus" not in config.data.fairness_attribute_names.categorical
    assert instantiate(config.model) is not None


def test_hyperadapt_fairvlmed_trains_the_backbone_from_imagenet() -> None:
    """HyperAdapt の model 既定は凍結・path 無しなので、E2E の preset で ImageNet 重みと凍結解除を入れる。"""
    with initialize_config_dir(version_base="1.3", config_dir=str(CONFIG_DIR)):
        config = compose(config_name="train", overrides=["experiment=hyperadapt_fairvlmed"])

    assert config.model.backbone_checkpoint_path.endswith("resnet50_imagenet.pth")
    assert config.model.freeze_backbone is False
