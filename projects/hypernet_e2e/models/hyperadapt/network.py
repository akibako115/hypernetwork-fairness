"""HyperAdapt 論文の患者条件付き重み変調 ResNet。

metadata encoder の出力を共有 MLP へ通し、層ごとの低ランク係数を生成する。畳み込みは
channel-pair ごとの乗法変調、分類器は低ランクの加法更新を用いる。
"""

from __future__ import annotations

from collections.abc import Mapping

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..resnet.blocks import Bottleneck
from ..resnet.network import ResNetBackbone
from ..spatial_hypernet.metadata_encoder import MetadataEncoder
from ..utils import load_compatible_state_dict


class HyperAdaptGenerator(nn.Module):
    """condition から共有 A と層固有 B の低ランク因子を生成する。

    Conv と Linear は出力次元が同じでも別の共有 A head を持つ。これにより論文の layer type
    ごとの共有規則を保ちつつ、同一出力次元の層では patient-conditioned な A を共有する。
    `layer_specs` は layer 名から `(kind, Cout, Cin)` への対応であり、`factors` は各層に
    `[B, Cout, rank]` と `[B, rank, Cin]` を返す。
    """

    def __init__(self, condition_dim: int, hidden_dim: int, rank: int, layer_specs: Mapping[str, tuple[str, int, int]]) -> None:
        """共有 trunk と、指定された各層の decoder head を構築する。

        Args:
            condition_dim: metadata embedding の次元。
            hidden_dim: 共有 MLP trunk の隠れ次元。
            rank: 低ランク因子の rank。
            layer_specs: layer 名から `(kind, Cout, Cin)` への対応。

        Returns: None
        """
        super().__init__()
        if condition_dim <= 0 or hidden_dim <= 0 or rank <= 0:
            raise ValueError("condition_dim, hidden_dim, and rank must be positive")
        self.rank = rank
        self.trunk = nn.Sequential(nn.Linear(condition_dim, hidden_dim), nn.ReLU())
        self.a_generators = nn.ModuleDict()
        self.b_generators = nn.ModuleDict()
        self._a_keys: dict[str, str] = {}
        self._layer_specs = dict(layer_specs)
        for name, (kind, out_features, in_features) in self._layer_specs.items():
            a_key = f"{kind}_{out_features}"
            if a_key not in self.a_generators:
                head = nn.Linear(hidden_dim, out_features * rank)
                nn.init.zeros_(head.weight)
                nn.init.zeros_(head.bias)
                self.a_generators[a_key] = head
            self._a_keys[name] = a_key
            head = nn.Linear(hidden_dim, rank * in_features)
            # Var(B) を target layer の input fan-in に合わせる。既定の Linear 初期化は
            # hidden_dim だけを見るため、生成される因子の尺度が層の幅に追従しない。
            nn.init.normal_(head.weight, mean=0.0, std=(hidden_dim * in_features) ** -0.5)
            nn.init.zeros_(head.bias)
            self.b_generators[name] = head

    def factors(self, condition: torch.Tensor, layer_name: str) -> tuple[torch.Tensor, torch.Tensor]:
        """指定層用の A `[B, Cout, k]` と B `[B, k, Cin]` を返す。

        Args:
            condition: `[B, condition_dim]` の metadata embedding。
            layer_name: constructor の `layer_specs` に登録した層名。

        Returns:
            tuple[torch.Tensor, torch.Tensor]: A `[B, Cout, rank]` と B `[B, rank, Cin]`。
        """
        return self._factors_from_hidden(self.trunk(condition), layer_name)

    def _factors_from_hidden(self, hidden: torch.Tensor, layer_name: str) -> tuple[torch.Tensor, torch.Tensor]:
        """共有 trunk 済み condition から指定層の低ランク因子を生成する。"""
        if layer_name not in self._layer_specs:
            raise KeyError(f"Unknown HyperAdapt layer: {layer_name}")
        kind, out_features, in_features = self._layer_specs[layer_name]
        del kind
        batch_size = hidden.size(0)
        a = self.a_generators[self._a_keys[layer_name]](hidden).reshape(batch_size, out_features, self.rank)
        b = self.b_generators[layer_name](hidden).reshape(batch_size, self.rank, in_features)
        return a, b


class HyperAdaptConv(nn.Module):
    """共有 Conv2d に患者ごとの channel-wise 乗法変調を適用する層。

    `groups=1` の Conv2d だけを受け、入力 `[B, Cin, H, W]` と condition から同じ出力 shape
    を返す。base convolution と generator は親 model が所有し、この module は重複登録しない。
    """

    def __init__(self, conv: nn.Conv2d, generator: HyperAdaptGenerator, layer_name: str) -> None:
        """base convolution を保持せず参照し、generator から modulation 行列を得る。

        Args:
            conv: 変調対象の groups=1 Conv2d。
            generator: A/B を生成する親所有の HyperAdaptGenerator。
            layer_name: generator に登録した層名。

        Returns: None
        """
        super().__init__()
        if conv.groups != 1:
            raise ValueError(f"HyperAdaptConv requires groups=1, got {conv.groups}")
        object.__setattr__(self, "_base_conv", conv)
        object.__setattr__(self, "_generator", generator)
        self.layer_name = layer_name

    def forward(self, x: torch.Tensor, condition: torch.Tensor, hidden: torch.Tensor | None = None) -> torch.Tensor:
        """base kernel を `(1 + AB)` で channel-pair ごとにスケールして畳み込む。

        Args:
            x: `[B, Cin, H, W]` の特徴量。
            condition: `[B, condition_dim]` の metadata embedding。
            hidden: 共有 trunk 済みの `[B, hidden_dim]`。指定時は condition から再計算しない。

        Returns:
            torch.Tensor: base convolution と同じ `[B, Cout, H', W']` の出力。
        """
        base_conv = self._base_conv
        generator = self._generator
        a, b = generator._factors_from_hidden(hidden, self.layer_name) if hidden is not None else generator.factors(condition, self.layer_name)
        batch_size = x.size(0)
        if condition.size(0) != batch_size:
            raise ValueError("x and condition batch sizes must match")
        if x.size(1) != base_conv.in_channels:
            raise ValueError(f"x has {x.size(1)} channels, expected {base_conv.in_channels}")
        base = base_conv(x)
        # Θ * (AB) を患者ごとの full kernel として展開すると、stage4 の 3x3 Conv では
        # batch 128 時に GB 級の一時 tensor になる。Σ_r A[:, :, r] Conv(Θ, B[:, r, :] * x)
        # は完全に同値であり、共有 kernel の通常 Conv を rank 回まとめて実行できる。
        rank_inputs = (x.unsqueeze(1) * b[..., None, None]).reshape(batch_size * generator.rank, base_conv.in_channels, x.size(2), x.size(3))
        rank_features = F.conv2d(
            rank_inputs,
            base_conv.weight,
            bias=None,
            stride=base_conv.stride,
            padding=base_conv.padding,
            dilation=base_conv.dilation,
            groups=1,
        )
        rank_features = rank_features.reshape(batch_size, generator.rank, base_conv.out_channels, rank_features.size(2), rank_features.size(3))
        return base + torch.einsum("bor,brohw->bohw", a, rank_features)


class HyperAdaptLinear(nn.Module):
    """共有 Linear に患者ごとの低ランク加法更新を適用する分類層。

    入力 `[B, Cin]` と condition から、共有重みへ `AB` を加えた `[B, Cout]` を返す。base
    classifier と generator は親 model が所有し、この module は重複登録しない。
    """

    def __init__(self, linear: nn.Linear, generator: HyperAdaptGenerator, layer_name: str) -> None:
        """base classifier を保持せず参照し、generator から A/B を得る。

        Args:
            linear: 変調対象の共有分類器。
            generator: A/B を生成する親所有の HyperAdaptGenerator。
            layer_name: generator に登録した層名。

        Returns: None
        """
        super().__init__()
        object.__setattr__(self, "_base_linear", linear)
        object.__setattr__(self, "_generator", generator)
        self.layer_name = layer_name

    def forward(self, x: torch.Tensor, condition: torch.Tensor, hidden: torch.Tensor | None = None) -> torch.Tensor:
        """`base(x) + A(Bx)` を実体化なしで計算する。

        Args:
            x: `[B, in_features]` の特徴量。
            condition: `[B, condition_dim]` の metadata embedding。
            hidden: 共有 trunk 済みの `[B, hidden_dim]`。指定時は condition から再計算しない。

        Returns:
            torch.Tensor: `[B, out_features]` の logits。
        """
        a, b = self._generator._factors_from_hidden(hidden, self.layer_name) if hidden is not None else self._generator.factors(condition, self.layer_name)
        delta = torch.bmm(a, torch.bmm(b, x.unsqueeze(-1))).squeeze(-1)
        return self._base_linear(x) + delta


class HyperAdaptResNet(nn.Module):
    """論文 HyperAdapt の全 ResNet stage + FC を変調する分類モデル。

    Bottleneck ResNet 専用で、stem と projection shortcut は固定し、stage1〜4 主枝の全 Conv と
    FC を変調する。attributes は MetadataEncoder が要求する `categorical` / `continuous` と各
    `*_missing` を持つ。A はゼロ初期化され、初期 logits は共有 ResNet と一致する。
    """

    def __init__(
        self,
        num_classes: int,
        backbone: ResNetBackbone,
        metadata_encoder: MetadataEncoder,
        rank: int = 4,
        generator_hidden_dim: int = 128,
    ) -> None:
        """stem 以外の Bottleneck 主枝 Conv と FC 用の adapter を初期化する。

        Args:
            num_classes: 分類クラス数。
            backbone: Bottleneck で構成した ResNetBackbone。
            metadata_encoder: 属性辞書を condition へ変換する encoder。
            rank: A/B 因子の rank。
            generator_hidden_dim: 共有 generator trunk の隠れ次元。

        Returns: None
        """
        super().__init__()
        if not all(isinstance(block, Bottleneck) for stage in self._stages(backbone) for block in stage):
            raise TypeError("HyperAdaptResNet requires a Bottleneck ResNet backbone")
        self.backbone = backbone
        self.metadata_encoder = metadata_encoder
        self.fc = nn.Linear(backbone.feature_dim, num_classes)
        specs: dict[str, tuple[str, int, int]] = {}
        for stage_index, stage in enumerate(self._stages(backbone), start=1):
            for block_index, block in enumerate(stage):
                for conv_name in ("conv1", "conv2", "conv3"):
                    conv = getattr(block, conv_name)
                    specs[self._conv_key(stage_index, block_index, conv_name)] = ("conv", conv.out_channels, conv.in_channels)
        specs["fc"] = ("linear", num_classes, backbone.feature_dim)
        self.generator = HyperAdaptGenerator(metadata_encoder.output_dim, generator_hidden_dim, rank, specs)
        self.conv_adapters = nn.ModuleDict(
            {
                name: HyperAdaptConv(getattr(block, conv_name), self.generator, self._conv_key(stage_index, block_index, conv_name))
                for stage_index, stage in enumerate(self._stages(backbone), start=1)
                for block_index, block in enumerate(stage)
                for conv_name in ("conv1", "conv2", "conv3")
                for name in [self._conv_key(stage_index, block_index, conv_name)]
            }
        )
        self.fc_adapter = HyperAdaptLinear(self.fc, self.generator, "fc")

    @staticmethod
    def _stages(backbone: ResNetBackbone) -> tuple[nn.Sequential, nn.Sequential, nn.Sequential, nn.Sequential]:
        return backbone.layer1, backbone.layer2, backbone.layer3, backbone.layer4

    @staticmethod
    def _conv_key(stage_index: int, block_index: int, conv_name: str) -> str:
        return f"stage{stage_index}_block{block_index}_{conv_name}"

    def forward(self, x: torch.Tensor, attributes: Mapping[str, torch.Tensor]) -> torch.Tensor:
        """画像と metadata 属性辞書から患者条件付き logits を返す。

        Args:
            x: `[B, 3, H, W]` の画像。
            attributes: MetadataEncoder の属性辞書。

        Returns:
            torch.Tensor: `[B, num_classes]` の logits。
        """
        condition = self.metadata_encoder(attributes)
        hidden = self.generator.trunk(condition)
        return self.fc_adapter(self._forward_backbone(x, condition, hidden), condition, hidden)

    def get_features(self, x: torch.Tensor, attributes: Mapping[str, torch.Tensor]) -> torch.Tensor:
        """患者条件付き Conv 変調後、分類前の特徴量を返す。

        Args:
            x: `[B, 3, H, W]` の画像。
            attributes: MetadataEncoder の属性辞書。

        Returns:
            torch.Tensor: `[B, backbone.feature_dim]` の特徴量。
        """
        condition = self.metadata_encoder(attributes)
        return self._forward_backbone(x, condition, self.generator.trunk(condition))

    def _forward_backbone(self, x: torch.Tensor, condition: torch.Tensor, hidden: torch.Tensor) -> torch.Tensor:
        x = self.backbone.relu(self.backbone.bn1(self.backbone.conv1(x)))
        x = self.backbone.maxpool(x)
        for stage_index, stage in enumerate(self._stages(self.backbone), start=1):
            for block_index, block in enumerate(stage):
                x = self._forward_block(block, stage_index, block_index, x, condition, hidden)
        return torch.flatten(self.backbone.avgpool(x), 1)

    def _forward_block(self, block: Bottleneck, stage_index: int, block_index: int, x: torch.Tensor, condition: torch.Tensor, hidden: torch.Tensor) -> torch.Tensor:
        identity = x
        conv1 = self.conv_adapters[self._conv_key(stage_index, block_index, "conv1")]
        conv2 = self.conv_adapters[self._conv_key(stage_index, block_index, "conv2")]
        conv3 = self.conv_adapters[self._conv_key(stage_index, block_index, "conv3")]
        out = block.relu(block.bn1(conv1(x, condition, hidden)))
        out = block.relu(block.bn2(conv2(out, condition, hidden)))
        out = block.bn3(conv3(out, condition, hidden))
        if block.downsample is not None:
            identity = block.downsample(x)
        return block.relu(out + identity)

    def load_base_state_dict(self, state_dict: dict) -> dict[str, list[str]]:
        """通常 ResNet checkpoint の backbone と FC を HyperAdapt の共有重みへ読む。

        Args:
            state_dict: 素の ResNet または `backbone.` 接頭辞付き checkpoint state dict。

        Returns:
            dict[str, list[str]]: load_compatible_state_dict のロード結果。
        """
        own_keys = set(self.state_dict())
        mapped = {f"backbone.{key}" if f"backbone.{key}" in own_keys else key: value for key, value in state_dict.items()}
        for parameter_name in ("weight", "bias"):
            source = f"fc.{parameter_name}"
            if source in mapped:
                mapped[f"fc.{parameter_name}"] = mapped.pop(source)
        return load_compatible_state_dict(self, mapped)

    def freeze_base_model(self) -> None:
        """論文の adaptation-only 学習用に backbone と共有 FC を凍結する。

        Args: なし

        Returns: None
        """
        for parameter in self.backbone.parameters():
            parameter.requires_grad = False
        for parameter in self.fc.parameters():
            parameter.requires_grad = False
        self.backbone.eval()

    def backbone_parameters(self):
        """凍結対象となる画像 backbone parameter を列挙する。

        Args: なし

        Returns:
            Iterator[nn.Parameter]: stem と stage1〜4 の parameter。
        """
        return self.backbone.backbone_parameters()

    def backbone_stateful_modules(self):
        """凍結時に eval 状態を維持すべき backbone の状態保持 module を列挙する。

        Args: なし

        Returns:
            Iterator[nn.BatchNorm2d]: backbone の状態保持 BatchNorm。
        """
        return self.backbone.backbone_stateful_modules()
