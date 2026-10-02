import pytest
import torch
import torch.nn as nn

from projects.hypernet_iterative.models.resnet.blocks import Bottleneck
from projects.hypernet_iterative.models.spatial_hypernet.blocks import SpatialLoRABottleneckAdapter, SpatialLoRAConv


def test_spatial_lora_starts_as_base_conv2() -> None:
    conv = nn.Conv2d(4, 5, kernel_size=3, padding=1, bias=False)
    adapter = SpatialLoRAConv(condition_dim=3, conv=conv, rank=2)
    x = torch.randn(2, 4, 7, 7)
    condition = torch.randn(2, 3)

    assert torch.count_nonzero(adapter.a_generator.weight) == 0
    assert torch.count_nonzero(adapter.a_generator.bias) == 0
    assert torch.equal(adapter(x, condition), conv(x))


def test_spatial_lora_path_matches_materialized_weight_delta() -> None:
    conv = nn.Conv2d(4, 5, kernel_size=3, padding=1, bias=False)
    adapter = SpatialLoRAConv(condition_dim=3, conv=conv, rank=2, lora_alpha=2.0)
    adapter.a_generator.bias.data.fill_(1.0)
    x = torch.randn(2, 4, 7, 7)
    condition = torch.randn(2, 3)
    weight_delta = adapter.weight_delta(condition)

    expected = torch.stack(
        [
            conv(x[index : index + 1])
            + torch.nn.functional.conv2d(
                x[index : index + 1],
                weight_delta[index],
                stride=conv.stride,
                padding=conv.padding,
                dilation=conv.dilation,
            )
            for index in range(x.size(0))
        ]
    ).squeeze(1)

    assert adapter.scale == 1.0
    torch.testing.assert_close(adapter(x, condition), expected, rtol=1e-5, atol=1e-5)


def test_lora_alpha_scales_the_effective_weight_delta() -> None:
    conv = nn.Conv2d(4, 5, kernel_size=3, padding=1, bias=False)
    low_alpha = SpatialLoRAConv(condition_dim=3, conv=conv, rank=2, lora_alpha=1.0)
    high_alpha = SpatialLoRAConv(condition_dim=3, conv=conv, rank=2, lora_alpha=2.0)
    high_alpha.load_state_dict(low_alpha.state_dict())
    low_alpha.a_generator.bias.data.fill_(1.0)
    high_alpha.a_generator.bias.data.fill_(1.0)
    condition = torch.randn(2, 3)

    assert low_alpha.scale == 0.5
    assert high_alpha.scale == 1.0
    assert torch.allclose(high_alpha.weight_delta(condition), 2 * low_alpha.weight_delta(condition))


def test_spatial_bottleneck_adapter_starts_as_the_original_block() -> None:
    block = Bottleneck(inplanes=16, planes=4)
    adapter = SpatialLoRABottleneckAdapter(condition_dim=3, block=block, rank=2)
    block.eval()
    adapter.eval()
    x = torch.randn(2, 16, 7, 7)
    condition = torch.randn(2, 3)

    assert torch.equal(adapter(x, condition), block(x))


@pytest.mark.parametrize("stride", [1, 2])
def test_spatial_lora_supports_1x1_convolutions(stride: int) -> None:
    conv = nn.Conv2d(4, 5, kernel_size=1, stride=stride, bias=False)
    adapter = SpatialLoRAConv(condition_dim=3, conv=conv, rank=2)
    x = torch.randn(2, 4, 7, 7)
    condition = torch.randn(2, 3)

    assert torch.equal(adapter(x, condition), conv(x))

    adapter.a_generator.bias.data.fill_(1.0)
    weight_delta = adapter.weight_delta(condition)
    expected = torch.cat([conv(x[index : index + 1]) + torch.nn.functional.conv2d(x[index : index + 1], weight_delta[index], stride=conv.stride) for index in range(x.size(0))])

    assert weight_delta.shape == (2, 5, 4, 1, 1)
    torch.testing.assert_close(adapter(x, condition), expected, rtol=1e-5, atol=1e-5)


def test_spatial_bottleneck_adapter_can_modulate_every_conv() -> None:
    block = Bottleneck(inplanes=16, planes=4)
    adapter = SpatialLoRABottleneckAdapter(condition_dim=3, block=block, rank=2, adapted_convs=["conv1", "conv2", "conv3"])
    block.eval()
    adapter.eval()
    x = torch.randn(2, 16, 7, 7)
    condition = torch.randn(2, 3)

    assert {name.split(".")[0] for name in adapter.state_dict()} == {"conv1_adapter", "conv2_adapter", "conv3_adapter"}
    assert torch.equal(adapter(x, condition), block(x))

    # conv1 の差分だけを有効にすると、出力は元の block から離れる。
    adapter.conv1_adapter.a_generator.bias.data.fill_(1.0)
    assert not torch.allclose(adapter(x, condition), block(x))


def test_spatial_bottleneck_adapter_defaults_to_conv2_only() -> None:
    adapter = SpatialLoRABottleneckAdapter(condition_dim=3, block=Bottleneck(inplanes=16, planes=4), rank=2)

    assert {name.split(".")[0] for name in adapter.state_dict()} == {"conv2_adapter"}


@pytest.mark.parametrize("adapted_convs", [[], ["conv4"], ["downsample"], ["conv1", "conv1"]])
def test_spatial_bottleneck_adapter_rejects_invalid_adapted_convs(adapted_convs: list[str]) -> None:
    with pytest.raises(ValueError):
        SpatialLoRABottleneckAdapter(condition_dim=3, block=Bottleneck(inplanes=16, planes=4), rank=2, adapted_convs=adapted_convs)
