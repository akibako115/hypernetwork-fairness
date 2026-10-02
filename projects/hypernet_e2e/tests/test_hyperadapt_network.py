import pytest
import torch
import torch.nn.functional as F

from projects.hypernet_e2e.models.hyperadapt import HyperAdaptResNet
from projects.hypernet_e2e.models.resnet.network import ResNetBackbone
from projects.hypernet_e2e.models.spatial_hypernet.metadata_encoder import MetadataEncoder


def _attributes(batch_size: int = 2) -> dict[str, torch.Tensor]:
    return {
        "categorical": torch.tensor([[0], [1]])[:batch_size],
        "categorical_missing": torch.zeros(batch_size, 1, dtype=torch.bool),
        "continuous": torch.empty(batch_size, 0),
        "continuous_missing": torch.empty(batch_size, 0, dtype=torch.bool),
    }


def test_hyperadapt_starts_as_the_loaded_base_resnet() -> None:
    """A をゼロ初期化した直後は、HyperAdapt の出力が共有 ResNet と一致する。"""
    backbone = ResNetBackbone(layers=[1, 1, 1, 1], block="Bottleneck")
    model = HyperAdaptResNet(
        num_classes=2,
        backbone=backbone,
        metadata_encoder=MetadataEncoder([2], 0, embedding_dim=2, hidden_dim=4, output_dim=3),
        rank=2,
        generator_hidden_dim=4,
    )
    image = torch.randn(2, 3, 64, 64)

    expected = model.fc(backbone(image))

    torch.testing.assert_close(model(image, _attributes()), expected)


def test_hyperadapt_conv_scales_each_channel_pair_by_low_rank_modulation() -> None:
    """Conv は論文の Θ[i,j] * (1 + (AB)[i,j]) を計算する。"""
    model = HyperAdaptResNet(
        num_classes=2,
        backbone=ResNetBackbone(layers=[1, 1, 1, 1], block="Bottleneck"),
        metadata_encoder=MetadataEncoder([2], 0, embedding_dim=2, hidden_dim=4, output_dim=3),
        rank=2,
        generator_hidden_dim=4,
    )
    adapter = model.conv_adapters["stage1_block0_conv1"]
    a_head = model.generator.a_generators["conv_64"]
    b_head = model.generator.b_generators["stage1_block0_conv1"]
    a_head.bias.data.fill_(0.25)
    b_head.bias.data.fill_(0.5)
    x = torch.randn(2, adapter._base_conv.in_channels, 8, 8)
    condition = model.metadata_encoder(_attributes())

    a, b = model.generator.factors(condition, "stage1_block0_conv1")
    weight = adapter._base_conv.weight.unsqueeze(0) * (1 + torch.bmm(a, b)[..., None, None])
    expected = F.conv2d(
        x.reshape(1, -1, 8, 8),
        weight.reshape(2 * adapter._base_conv.out_channels, adapter._base_conv.in_channels, 1, 1),
        stride=adapter._base_conv.stride,
        padding=adapter._base_conv.padding,
        dilation=adapter._base_conv.dilation,
        groups=2,
    ).reshape(2, adapter._base_conv.out_channels, 8, 8)

    torch.testing.assert_close(adapter(x, condition), expected)


def test_hyperadapt_freezes_only_base_parameters() -> None:
    model = HyperAdaptResNet(
        num_classes=2,
        backbone=ResNetBackbone(layers=[1, 1, 1, 1], block="Bottleneck"),
        metadata_encoder=MetadataEncoder([2], 0, embedding_dim=2, hidden_dim=4, output_dim=3),
        rank=2,
        generator_hidden_dim=4,
    )

    model.freeze_base_model()

    assert not any(parameter.requires_grad for parameter in model.backbone.parameters())
    assert not any(parameter.requires_grad for parameter in model.fc.parameters())
    assert all(parameter.requires_grad for parameter in model.metadata_encoder.parameters())
    assert all(parameter.requires_grad for parameter in model.generator.parameters())
    assert not model.backbone.training


def test_hyperadapt_b_generators_scale_with_the_target_layer_fan_in() -> None:
    model = HyperAdaptResNet(
        num_classes=2,
        backbone=ResNetBackbone(layers=[1, 1, 1, 1], block="Bottleneck"),
        metadata_encoder=MetadataEncoder([2], 0, embedding_dim=2, hidden_dim=4, output_dim=3),
        rank=2,
        generator_hidden_dim=64,
    )
    b_head = model.generator.b_generators["stage1_block0_conv1"]

    assert b_head.weight.std().item() == pytest.approx((64 * 64) ** -0.5, rel=0.15)
