import pytest
import torch

from projects.hypernet_e2e.models.resnet.network import ResNetBackbone
from projects.hypernet_e2e.models.spatial_hypernet.metadata_encoder import MetadataEncoder
from projects.hypernet_e2e.models.spatial_hypernet.modulation import HyperLinearLayer
from projects.hypernet_e2e.models.spatial_hypernet.network import SpatialLoRAResNet


def _attributes(batch_size: int = 2) -> dict[str, torch.Tensor]:
    return {
        "categorical": torch.tensor([[0], [1]])[:batch_size],
        "categorical_missing": torch.zeros(batch_size, 1, dtype=torch.bool),
        "continuous": torch.empty(batch_size, 0),
        "continuous_missing": torch.empty(batch_size, 0, dtype=torch.bool),
    }


def _metadata_encoder() -> MetadataEncoder:
    return MetadataEncoder(
        categorical_cardinalities=[2],
        num_continuous=0,
        embedding_dim=2,
        hidden_dim=4,
        output_dim=3,
    )


def test_spatial_lora_resnet_assigns_adapters_to_selected_stages() -> None:
    backbone = ResNetBackbone(layers=[1, 1, 2, 2], block="Bottleneck")
    model = SpatialLoRAResNet(
        num_classes=2,
        backbone=backbone,
        metadata_encoder=_metadata_encoder(),
        rank=2,
        modulation_stages=["stage3", "stage4"],
    )

    assert set(model.spatial_adapters) == {"stage3", "stage4"}
    assert len(model.spatial_adapters["stage3"]) == len(backbone.layer3) == 2
    assert len(model.spatial_adapters["stage4"]) == len(backbone.layer4) == 2
    assert model(torch.randn(2, 3, 64, 64), _attributes()).shape == (2, 2)


def test_spatial_lora_resnet_supports_hyperlinear_classifier_and_base_checkpoint() -> None:
    backbone = ResNetBackbone(layers=[1, 1, 1, 1], block="Bottleneck")
    model = SpatialLoRAResNet(
        num_classes=2,
        backbone=backbone,
        metadata_encoder=_metadata_encoder(),
        rank=2,
        modulation_stages=["stage4", "fc"],
        classifier_rank=2,
    )
    checkpoint = {
        "fc.weight": torch.full_like(model.fc.base_linear.weight, 3.0),
        "fc.bias": torch.full_like(model.fc.base_linear.bias, 4.0),
    }

    result = model.load_base_state_dict(checkpoint)

    assert isinstance(model.fc, HyperLinearLayer)
    assert torch.equal(model.fc.base_linear.weight, checkpoint["fc.weight"])
    assert torch.equal(model.fc.base_linear.bias, checkpoint["fc.bias"])
    assert result["loaded_keys"] == ["fc.base_linear.bias", "fc.base_linear.weight"]
    assert model(torch.randn(2, 3, 64, 64), _attributes()).shape == (2, 2)


def test_spatial_lora_resnet_freezes_only_base_model() -> None:
    model = SpatialLoRAResNet(
        num_classes=2,
        backbone=ResNetBackbone(layers=[1, 1, 1, 1], block="Bottleneck"),
        metadata_encoder=_metadata_encoder(),
        rank=2,
        modulation_stages=["stage4", "fc"],
        classifier_rank=2,
    )

    model.freeze_base_model()

    assert not any(parameter.requires_grad for parameter in model.backbone.parameters())
    assert not any(parameter.requires_grad for parameter in model.fc.base_linear.parameters())
    assert all(parameter.requires_grad for parameter in model.spatial_adapters["stage4"].parameters())
    assert all(parameter.requires_grad for parameter in model.fc.a_generator.parameters())
    assert not model.backbone.training


@pytest.mark.parametrize("modulation_stages", [["unknown"], ["stage4", "stage4"]])
def test_spatial_lora_resnet_rejects_invalid_modulation_stages(modulation_stages: list[str]) -> None:
    with pytest.raises(ValueError):
        SpatialLoRAResNet(
            num_classes=2,
            backbone=ResNetBackbone(layers=[1, 1, 1, 1], block="Bottleneck"),
            metadata_encoder=_metadata_encoder(),
            modulation_stages=modulation_stages,
        )


def test_initialize_with_attributes_rescales_b_generators_without_touching_a() -> None:
    """B 生成器だけが Var(c) で初期化され、A のゼロ初期化＝初期出力の一致は保たれる。"""
    torch.manual_seed(0)
    model = SpatialLoRAResNet(
        num_classes=2,
        backbone=ResNetBackbone(layers=[1, 1, 1, 1], block="Bottleneck"),
        metadata_encoder=_metadata_encoder(),
        rank=2,
        modulation_stages=["stage4", "fc"],
    )
    adapter = model.spatial_adapters["stage4"][0].conv2_adapter
    before = adapter.b_generator.weight.detach().clone()
    images = torch.randn(2, 3, 64, 64)
    baseline = model(images, _attributes())

    model.initialize_with_attributes(_attributes(2))

    assert not torch.equal(adapter.b_generator.weight, before)
    assert torch.count_nonzero(adapter.a_generator.weight) == 0
    assert torch.allclose(model(images, _attributes()), baseline)


def test_initialize_with_attributes_is_a_no_op_without_modulation() -> None:
    model = SpatialLoRAResNet(
        num_classes=2,
        backbone=ResNetBackbone(layers=[1, 1, 1, 1], block="Bottleneck"),
        metadata_encoder=_metadata_encoder(),
        modulation_stages=[],
    )

    model.initialize_with_attributes({})
