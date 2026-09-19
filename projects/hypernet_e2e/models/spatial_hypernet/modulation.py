import math

import torch
import torch.nn as nn


class HyperLinearLayer(nn.Module):
    """metadata 条件付きの低ランク差分を共有線形分類層に加える層。"""

    def __init__(
        self,
        condition_dim: int,
        in_features: int,
        out_features: int,
        rank: int = 4,
        lora_alpha: float = 1.0,
    ):
        super().__init__()
        if condition_dim <= 0:
            raise ValueError(f"condition_dim must be positive, got {condition_dim}")
        if in_features <= 0:
            raise ValueError(f"in_features must be positive, got {in_features}")
        if out_features <= 0:
            raise ValueError(f"out_features must be positive, got {out_features}")
        if rank <= 0:
            raise ValueError(f"rank must be positive, got {rank}")
        if lora_alpha <= 0:
            raise ValueError(f"lora_alpha must be positive, got {lora_alpha}")

        self.condition_dim = condition_dim
        self.in_features = in_features
        self.out_features = out_features
        self.rank = rank
        self.lora_alpha = lora_alpha
        self.scale = lora_alpha / rank

        self.base_linear = nn.Linear(in_features, out_features)
        self.a_generator = nn.Linear(condition_dim, out_features * rank)
        self.b_generator = nn.Linear(condition_dim, rank * in_features)

        # A 側をゼロ初期化して、学習開始時は共有 Linear classifier と一致させる。
        nn.init.zeros_(self.a_generator.weight)
        nn.init.zeros_(self.a_generator.bias)
        # B 側の bias は condition に依らない定数成分になるため、SpatialLoRAConv2 と同じく
        # ゼロから始める。
        nn.init.zeros_(self.b_generator.bias)

    def initialize_from_variance(self, var_input: float) -> None:
        """生成先 Linear の fan_in に合わせて B 生成器を初期化する。

        SpatialLoRAConv2 と同じ規則を fan_in=in_features で適用する。
        """
        if var_input <= 0:
            var_input = 1.0

        var_w = 2.0 / (2.0 * self.in_features * self.condition_dim * var_input)
        bound = math.sqrt(3.0 * var_w)
        nn.init.uniform_(self.b_generator.weight, -bound, bound)
        # bias は condition に依らない定数成分なので、初期化後もゼロを保つ。
        nn.init.zeros_(self.b_generator.bias)

    def _factors(self, condition: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """各サンプルの A, B 因子を (B, Cout, r), (B, r, Cin) で返す。"""
        batch_size = condition.size(0)
        a = self.a_generator(condition).reshape(batch_size, self.out_features, self.rank)
        b = self.b_generator(condition).reshape(batch_size, self.rank, self.in_features)
        return a, b

    def weight_delta(self, condition: torch.Tensor) -> torch.Tensor:
        """各サンプルの ΔW=(α/r)AB を (B, Cout, Cin) で返す。

        診断・テスト用であり、forward からは呼ばない。分類層では小さいが、低ランク計算を
        一貫して保つため A@B の明示的な実体化を避ける。
        """
        a, b = self._factors(condition)
        return self.scale * torch.bmm(a, b)

    def forward(self, x: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
        """(B, Cin) の特徴量へ共有 Linear と sample-wise な低ランク差分を適用する。"""
        if x.ndim != 2:
            raise ValueError(f"x must be 2-dimensional, got shape {tuple(x.shape)}")
        if condition.ndim != 2:
            raise ValueError(f"condition must be 2-dimensional, got shape {tuple(condition.shape)}")
        if x.size(0) != condition.size(0):
            raise ValueError(f"x and condition batch sizes must match, got {x.size(0)} and {condition.size(0)}")
        if x.size(1) != self.in_features:
            raise ValueError(f"x has {x.size(1)} features, expected {self.in_features}")
        if condition.size(1) != self.condition_dim:
            raise ValueError(f"condition has {condition.size(1)} features, expected {self.condition_dim}")

        a, b = self._factors(condition)
        low_rank_features = torch.bmm(b, x.unsqueeze(-1))
        delta = self.scale * torch.bmm(a, low_rank_features).squeeze(-1)
        return self.base_linear(x) + delta
