from collections.abc import Sequence

import torch
import torch.nn as nn

from .blocks import BasicBlock, Bottleneck, _resolve_block_class, conv1x1


class ResNetBackbone(nn.Module):
    """分類ヘッドなしの ResNet 特徴抽出器。出力は (B, feature_dim) の flatten 済み特徴量。

    Args:
        layers: 各 stage（layer1〜layer4）のブロック数。長さ4固定。
        block: ブロッククラス、または `"BasicBlock"` / `"Bottleneck"` の名前。
        zero_init_residual: 残差分岐の最終 BN をゼロ初期化し、恒等写像から学習を始めるか。
        groups: グループ畳み込みのグループ数（ResNeXt 用、既定は通常の conv）。
        width_per_group: グループあたりの幅（ResNeXt / Wide ResNet 用）。
        replace_stride_with_dilation: layer2〜layer4 の stride を dilation に置き換えるか
            （長さ3の bool 列。`None` なら全て置き換えない）。
        norm_layer: 正規化層のクラス（既定は `nn.BatchNorm2d`）。
    """

    def __init__(
        self,
        layers: Sequence[int],
        block: type[nn.Module] | str,
        zero_init_residual: bool = False,
        groups: int = 1,
        width_per_group: int = 64,
        replace_stride_with_dilation: Sequence[bool] | None = None,
        norm_layer: type[nn.Module] | None = None,
    ):
        """指定された stage 構成で ResNet backbone を初期化する。"""
        super().__init__()
        if len(layers) != 4:
            raise ValueError(f"layers must have length 4, got {layers!r}")
        block_cls = _resolve_block_class(block)

        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer
        self.inplanes = 64
        self.dilation = 1

        if replace_stride_with_dilation is None:
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation must be None or 3-element tuple")
        self.groups = groups
        self.base_width = width_per_group

        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block_cls, 64, layers[0])
        self.layer2 = self._make_layer(block_cls, 128, layers[1], stride=2, dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block_cls, 256, layers[2], stride=2, dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(block_cls, 512, layers[3], stride=2, dilate=replace_stride_with_dilation[2])
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.feature_dim = 512 * block_cls.expansion

        # 重み初期化: Conv は Kaiming Normal（ReLU 想定）、BN/GroupNorm は weight=1, bias=0。
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # zero_init_residual: 各ブロックの最終 BN を 0 初期化し、残差分岐を恒等写像から始める。
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block: type[nn.Module], planes: int, num_blocks: int, stride: int = 1, dilate: bool = False) -> nn.Sequential:
        """1つの stage（同一 planes を持つ block の並び）を構築する。

        先頭 block だけ `stride` / チャンネル数変化を引き受け、必要なら downsample 分岐を付与する。
        `dilate=True` の場合は stride の代わりに dilation を蓄積し出力解像度を維持する。

        Args:
            block: 使用するブロッククラス（`BasicBlock` / `Bottleneck`）。
            planes: この stage の基準チャンネル数。
            num_blocks: block の個数。
            stride: 先頭 block の stride。
            dilate: True なら stride の代わりに dilation を使う。

        Returns:
            nn.Sequential: `num_blocks` 個の block を並べた stage。
        """
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )
        layers = [block(self.inplanes, planes, stride, downsample, self.groups, self.base_width, previous_dilation, norm_layer)]
        self.inplanes = planes * block.expansion
        for _ in range(1, num_blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups, base_width=self.base_width, dilation=self.dilation, norm_layer=norm_layer))
        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """stem + 4 stage + global average pooling を適用する。`(B, 3, H, W)` → `(B, feature_dim)`。"""
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        return torch.flatten(x, 1)

    def backbone_parameters(self):
        """freeze_backbone 用に、stem + 4 stage（分類ヘッドを除く）の parameter を列挙する。"""
        for m in (self.conv1, self.bn1, self.layer1, self.layer2, self.layer3, self.layer4):
            yield from m.parameters()

    def backbone_stateful_modules(self):
        """freeze_backbone 時に eval モードへ固定すべき BatchNorm モジュールを列挙する。"""
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                yield m


class ResNet(nn.Module):
    """分類ヘッド付き ResNet。`num_classes` 指定時のみ FC を持ち、None なら特徴量を返す。

    Args:
        layers: `ResNetBackbone` に渡す各 stage のブロック数。
        block: ブロッククラス、または `"BasicBlock"` / `"Bottleneck"` の名前。
        num_classes: 分類クラス数。`None` なら `fc` を持たず特徴量をそのまま返す。
        zero_init_residual: `ResNetBackbone` を参照。
        groups: `ResNetBackbone` を参照（ResNeXt 用）。
        width_per_group: `ResNetBackbone` を参照（ResNeXt / Wide ResNet 用）。
        replace_stride_with_dilation: `ResNetBackbone` を参照。
        norm_layer: 正規化層のクラス（既定は `nn.BatchNorm2d`）。
    """

    def __init__(
        self,
        layers: Sequence[int],
        block: type[nn.Module] | str,
        num_classes: int | None = None,
        zero_init_residual: bool = False,
        groups: int = 1,
        width_per_group: int = 64,
        replace_stride_with_dilation: Sequence[bool] | None = None,
        norm_layer: type[nn.Module] | None = None,
    ):
        """ResNet backbone と分類器を組み合わせたネットワークを初期化する。"""
        super().__init__()
        self.backbone = ResNetBackbone(layers, block, zero_init_residual, groups, width_per_group, replace_stride_with_dilation, norm_layer)
        self.num_classes = num_classes
        self.fc = nn.Linear(self.backbone.feature_dim, num_classes) if num_classes is not None else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """`(B, 3, H, W)` → `num_classes` 指定時は `(B, num_classes)` の logits、未指定時は `(B, feature_dim)` の特徴量。"""
        feat = self.backbone(x)
        return self.fc(feat) if self.fc is not None else feat

    def get_features(self, x: torch.Tensor) -> torch.Tensor:
        """分類ヘッドを経由せず backbone の特徴量 `(B, feature_dim)` を返す。"""
        return self.backbone(x)

    def backbone_parameters(self):
        """freeze_backbone 用に backbone（分類ヘッドを除く）の parameter を列挙する。"""
        return self.backbone.backbone_parameters()

    def backbone_stateful_modules(self):
        """freeze_backbone 時に eval モードへ固定すべき backbone の BatchNorm モジュールを列挙する。"""
        return self.backbone.backbone_stateful_modules()
