import torch
import torch.nn as nn

_BLOCK_TYPES: dict[str, type[nn.Module]] = {}


def _resolve_block_class(block: type[nn.Module] | str) -> type[nn.Module]:
    """block 名（"BasicBlock" / "Bottleneck"）またはクラス自体からブロッククラスを解決する。"""
    if isinstance(block, str):
        cls = _BLOCK_TYPES.get(block)
        if cls is None:
            raise ValueError(f"Unknown block name: {block!r}")
        return cls
    return block


def conv3x3(in_planes: int, out_planes: int, stride: int = 1, groups: int = 1, dilation: int = 1) -> nn.Conv2d:
    """padding=dilation の 3x3 conv（bias なし）を返す。"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=dilation, groups=groups, bias=False, dilation=dilation)


def conv1x1(in_planes: int, out_planes: int, stride: int = 1) -> nn.Conv2d:
    """downsample・チャンネル数変換に使う 1x1 conv（bias なし）を返す。"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlock(nn.Module):
    """ResNet の BasicBlock（3x3 conv 2層 + 残差接続、expansion=1）。"""

    expansion = 1

    def __init__(
        self, inplanes: int, planes: int, stride: int = 1, downsample: nn.Module | None = None, groups: int = 1, base_width: int = 64, dilation: int = 1, norm_layer: type[nn.Module] | None = None
    ):
        """BasicBlock の畳み込み・正規化・shortcut を構成する。"""
        super().__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError("BasicBlock only supports groups=1 and base_width=64")
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """3x3 conv 2層と残差接続を適用する。出力 shape は downsample の有無に応じ入力と同一またはチャンネル・解像度変化後。"""
        identity = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        if self.downsample is not None:
            identity = self.downsample(x)
        return self.relu(out + identity)


class Bottleneck(nn.Module):
    """ResNet の Bottleneck（1x1-3x3-1x1 conv + 残差接続、expansion=4）。"""

    expansion = 4

    def __init__(
        self, inplanes: int, planes: int, stride: int = 1, downsample: nn.Module | None = None, groups: int = 1, base_width: int = 64, dilation: int = 1, norm_layer: type[nn.Module] | None = None
    ):
        """Bottleneck block の畳み込み・正規化・shortcut を構成する。"""
        super().__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.0)) * groups
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """1x1-3x3-1x1 conv と残差接続を適用する。出力チャンネル数は `planes * expansion`（expansion=4）。"""
        identity = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        if self.downsample is not None:
            identity = self.downsample(x)
        return self.relu(out + identity)


_BLOCK_TYPES.update({"BasicBlock": BasicBlock, "Bottleneck": Bottleneck})
