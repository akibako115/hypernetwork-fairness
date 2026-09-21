import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..resnet.blocks import Bottleneck


# ----- blockの実装 -----
# resenetの bottleneck block の conv2 に Spatial LoRA を適用するアダプタ。
class SpatialLoRAConv2(nn.Module):
    """metadata 条件付きの低ランク差分を 3x3 convolution に加える層。

    Args:
        condition_dim: metadata embedding（condition）の次元。
        conv: 差分を加える対象の 3x3 convolution（groups=1 限定）。重みの所有権は保持せず参照のみ。
        rank: 低ランク差分 ΔW=A@B のランク r。
        lora_alpha: スケール係数（`scale = lora_alpha / rank`）。
    """

    def __init__(self, condition_dim: int, conv: nn.Conv2d, rank: int, lora_alpha: float = 1.0):
        """元の畳み込みへ条件付き LoRA adapter を追加する。"""
        super().__init__()
        if condition_dim <= 0:
            raise ValueError(f"condition_dim must be positive, got {condition_dim}")
        if rank <= 0:
            raise ValueError(f"rank must be positive, got {rank}")
        if lora_alpha <= 0:
            raise ValueError(f"lora_alpha must be positive, got {lora_alpha}")
        if conv.groups != 1:
            raise ValueError(f"SpatialLoRAConv2 requires groups=1, got {conv.groups}")
        if conv.kernel_size != (3, 3):
            raise ValueError(f"SpatialLoRAConv2 requires a 3x3 convolution, got {conv.kernel_size}")

        self.condition_dim = condition_dim
        self.in_channels = conv.in_channels
        self.out_channels = conv.out_channels
        self.rank = rank
        self.lora_alpha = lora_alpha
        self.scale = lora_alpha / rank
        self.stride = conv.stride
        self.padding = conv.padding
        self.dilation = conv.dilation
        # 重みの正式な所有者は backbone 側に保ち、state_dict の二重登録を避ける。
        object.__setattr__(self, "_base_conv", conv)
        self.a_generator = nn.Linear(condition_dim, self.out_channels * rank)
        self.b_generator = nn.Linear(condition_dim, rank * self.in_channels * 3 * 3)

        # A 側をゼロ初期化して、初期状態を元の convolution と完全に一致させる。
        nn.init.zeros_(self.a_generator.weight)
        nn.init.zeros_(self.a_generator.bias)
        # B 側の bias は condition に依らない定数カーネルになる。hypernetwork の初期化規約
        # （Principled Weight Initialization for Hypernetworks）に合わせて
        # ゼロから始め、初期の B を condition 経路だけで決める。学習可能なままにしておく。
        nn.init.zeros_(self.b_generator.bias)

    def initialize_from_variance(self, var_input: float) -> None:
        """生成先 3x3 convolution の fan_in に合わせて B 生成器を初期化する。

        既定の `nn.Linear` 初期化は入力次元（condition_dim）だけを見るため、生成される
        3x3 カーネルの大きさが base convolution の Kaiming 初期値と無関係に決まる。
        "Principled Weight Initialization for Hypernetworks" の規則を fan_in=Cin*3*3 で
        適用し、条件 c の各成分の分散が var_input のとき std(B)=1/sqrt(fan_in) となるようにする。
        """
        if var_input <= 0:
            var_input = 1.0

        fan_in = self.in_channels * 3 * 3
        var_w = 2.0 / (2.0 * fan_in * self.condition_dim * var_input)
        bound = math.sqrt(3.0 * var_w)
        nn.init.uniform_(self.b_generator.weight, -bound, bound)
        # bias は condition に依らない定数カーネルなので、初期化後もゼロを保つ。
        nn.init.zeros_(self.b_generator.bias)

    def _factors(self, condition: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """各サンプルの A, B 因子をそれぞれ (B, Cout, r), (B, r, Cin, 3, 3) で返す。"""
        batch_size = condition.size(0)
        a = self.a_generator(condition).reshape(batch_size, self.out_channels, self.rank)
        b = self.b_generator(condition).reshape(batch_size, self.rank, self.in_channels, 3, 3)
        return a, b

    def weight_delta(self, condition: torch.Tensor) -> torch.Tensor:
        """各サンプルの ΔW=(A @ B) を (B, Cout, Cin, 3, 3) で返す。"""
        a, b = self._factors(condition)
        return self.scale * torch.einsum("bor,brihw->boihw", a, b)

    def forward(self, x: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
        """(B, Cin, H, W) に sample-wise の有効 3x3 convolution を適用する。"""
        if x.ndim != 4:
            raise ValueError(f"x must be 4-dimensional, got shape {tuple(x.shape)}")
        if x.size(0) != condition.size(0):
            raise ValueError(f"x and condition batch sizes must match, got {x.size(0)} and {condition.size(0)}")
        if x.size(1) != self.in_channels:
            raise ValueError(f"x has {x.size(1)} channels, expected {self.in_channels}")

        base_conv = self._base_conv
        batch_size = x.size(0)
        a, b = self._factors(condition)

        # ΔW=A@B を full kernel として展開しない。先に B の動的 3x3 convolution を
        # 適用し、続けて A の動的 1x1 projection をかけることで同じ低ランク差分を得る。
        grouped_x = x.reshape(1, batch_size * self.in_channels, x.size(2), x.size(3))
        b_weight = b.reshape(batch_size * self.rank, self.in_channels, 3, 3)
        low_rank_features = F.conv2d(
            grouped_x,
            b_weight,
            bias=None,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
            groups=batch_size,
        )
        a_weight = (self.scale * a).reshape(batch_size * self.out_channels, self.rank, 1, 1)
        delta = F.conv2d(
            low_rank_features,
            a_weight,
            bias=None,
            groups=batch_size,
        ).reshape(batch_size, self.out_channels, low_rank_features.size(2), low_rank_features.size(3))
        return base_conv(x) + delta


# ----- layerの実装 -----
# resenet layer4 の bottleneck に Spatial LoRA を適用するアダプタ。
class SpatialLoRABottleneckAdapter(nn.Module):
    """既存 Bottleneck の conv2 にだけ Spatial LoRA を適用するアダプタ。

    Args:
        condition_dim: metadata embedding（condition）の次元。
        block: 変調対象の `Bottleneck`。所有権は backbone.layer4 に残し、実行時にのみ参照する。
        rank: `SpatialLoRAConv2` の低ランク差分のランク r。
        lora_alpha: `SpatialLoRAConv2` のスケール係数。
    """

    def __init__(self, condition_dim: int, block: Bottleneck, rank: int, lora_alpha: float = 1.0):
        """Bottleneck の conv2 を変調する Spatial LoRA adapter を初期化する。"""
        super().__init__()
        # block の正式な所有者は backbone.layer4。adapter は実行時にだけ参照する。
        object.__setattr__(self, "_block", block)
        self.conv2_adapter = SpatialLoRAConv2(condition_dim, block.conv2, rank, lora_alpha)

    def initialize_from_variance(self, var_input: float) -> None:
        """内部の SpatialLoRAConv2 へ初期化を委譲する。"""
        self.conv2_adapter.initialize_from_variance(var_input)

    def forward(self, x: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
        """Bottleneck residual branch の conv2 だけを metadata 条件付きに置き換える。"""
        block = self._block
        identity = x
        out = block.relu(block.bn1(block.conv1(x)))
        out = block.relu(block.bn2(self.conv2_adapter(out, condition)))
        out = block.bn3(block.conv3(out))
        if block.downsample is not None:
            identity = block.downsample(x)
        return block.relu(out + identity)
