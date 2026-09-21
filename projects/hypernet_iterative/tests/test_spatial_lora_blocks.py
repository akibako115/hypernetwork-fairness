import torch
import torch.nn as nn

from projects.hypernet_iterative.models.resnet.blocks import Bottleneck
from projects.hypernet_iterative.models.spatial_hypernet.blocks import SpatialLoRABottleneckAdapter, SpatialLoRAConv2


def test_spatial_lora_starts_as_base_conv2() -> None:
    conv = nn.Conv2d(4, 5, kernel_size=3, padding=1, bias=False)
    adapter = SpatialLoRAConv2(condition_dim=3, conv=conv, rank=2)
    x = torch.randn(2, 4, 7, 7)
    condition = torch.randn(2, 3)

    assert torch.count_nonzero(adapter.a_generator.weight) == 0
    assert torch.count_nonzero(adapter.a_generator.bias) == 0
    assert torch.equal(adapter(x, condition), conv(x))


def test_spatial_lora_path_matches_materialized_weight_delta() -> None:
    conv = nn.Conv2d(4, 5, kernel_size=3, padding=1, bias=False)
    adapter = SpatialLoRAConv2(condition_dim=3, conv=conv, rank=2, lora_alpha=2.0)
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
    low_alpha = SpatialLoRAConv2(condition_dim=3, conv=conv, rank=2, lora_alpha=1.0)
    high_alpha = SpatialLoRAConv2(condition_dim=3, conv=conv, rank=2, lora_alpha=2.0)
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
