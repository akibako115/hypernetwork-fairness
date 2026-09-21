"""backbone 表現から属性情報を除くための勾配反転型属性予測器。"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F


class _GradientReverse(torch.autograd.Function):
    """順伝播を恒等写像とし、逆伝播時だけ勾配の符号を反転する。"""

    @staticmethod
    def forward(ctx: torch.autograd.function.FunctionCtx, features: torch.Tensor, scale: float) -> torch.Tensor:
        """features を値を変えずに返し、逆伝播用の係数を保存する。"""
        ctx.scale = scale
        return features.view_as(features)

    @staticmethod
    def backward(ctx: torch.autograd.function.FunctionCtx, gradient: torch.Tensor) -> tuple[torch.Tensor, None]:
        """backbone に渡す勾配だけを反転・スケールする。"""
        return -ctx.scale * gradient, None


def gradient_reverse(features: torch.Tensor, scale: float) -> torch.Tensor:
    """値を変えずに、逆伝播時の features 勾配を ``-scale`` 倍する。

    Args:
        features: `[B, feature_dim]` の backbone 表現。
        scale: 0 以上の勾配反転係数。

    Returns:
        torch.Tensor: 順伝播では `features` と同値の tensor。

    Raises:
        ValueError: scale が負の場合。
    """
    if scale < 0:
        raise ValueError(f"gradient reversal の scale は 0 以上である必要があるが、{scale} が指定された")
    return _GradientReverse.apply(features, scale)


class AttributeAdversary(nn.Module):
    """GRL 後の backbone 表現から各属性を予測し、その平均損失を返す。

    categorical 属性には列ごとの cross-entropy、continuous 属性には列ごとの MSE を用いる。
    欠損値は各列の損失から除外する。`forward` の戻り値をタスク loss に加えると、属性予測器は
    属性を当てる一方、GRL を挟んだ backbone はその予測を妨げる方向へ更新される。
    `features` は `[B, feature_dim]`、attributes は設定した列数と一致する `categorical` /
    `continuous` および対応する `*_missing` を持つ必要がある。cardinality は各カテゴリで 2 以上、
    `num_continuous` と `gradient_scale` は 0 以上である。戻り値は観測済み列の平均 scalar loss。
    """

    def __init__(
        self,
        feature_dim: int,
        categorical_cardinalities: Sequence[int] = (),
        num_continuous: int = 0,
        hidden_dim: int = 256,
        gradient_scale: float = 1.0,
    ) -> None:
        """属性の型・次元に合わせた予測ヘッドを構築する。

        Args:
            feature_dim: 入力 backbone 表現の次元。1 以上。
            categorical_cardinalities: categorical 属性列ごとのカテゴリ数。各要素は 2 以上。
            num_continuous: 連続属性列数。0 以上。
            hidden_dim: 共有予測 MLP の隠れ次元。1 以上。
            gradient_scale: backbone へ逆向きに渡す勾配の倍率。0 以上。

        Returns:
            None

        Raises:
            ValueError: 次元・カテゴリ数・勾配係数の制約を満たさない場合。
        """
        super().__init__()
        if feature_dim < 1 or hidden_dim < 1:
            raise ValueError("feature_dim と hidden_dim は 1 以上である必要がある")
        if any(cardinality < 2 for cardinality in categorical_cardinalities):
            raise ValueError("categorical_cardinalities は 2 以上の整数だけを指定する必要がある")
        if num_continuous < 0:
            raise ValueError("num_continuous は 0 以上である必要がある")
        if not categorical_cardinalities and num_continuous == 0:
            raise ValueError("少なくとも1つの属性予測対象が必要")
        if gradient_scale < 0:
            raise ValueError("gradient_scale は 0 以上である必要がある")
        self.gradient_scale = gradient_scale
        self.trunk = nn.Sequential(nn.Linear(feature_dim, hidden_dim), nn.ReLU(inplace=True))
        self.categorical_heads = nn.ModuleList([nn.Linear(hidden_dim, cardinality) for cardinality in categorical_cardinalities])
        self.continuous_head = nn.Linear(hidden_dim, num_continuous) if num_continuous else None

    def forward(self, features: torch.Tensor, attributes: Mapping[str, torch.Tensor]) -> torch.Tensor:
        """属性予測の平均損失を計算する。

        Args:
            features: `[B, feature_dim]` の backbone 表現。
            attributes: `categorical` / `continuous` と対応する `*_missing` を持つ属性辞書。

        Returns:
            torch.Tensor: 観測済み属性列だけで平均した scalar loss。

        Raises:
            ValueError: 必要な属性の shape が設定と一致しない場合。
        """
        hidden = self.trunk(gradient_reverse(features, self.gradient_scale))
        losses: list[torch.Tensor] = []
        if self.categorical_heads:
            categorical = attributes["categorical"]
            missing = attributes["categorical_missing"].bool()
            if categorical.ndim != 2 or categorical.shape[1] != len(self.categorical_heads) or missing.shape != categorical.shape:
                raise ValueError("categorical / categorical_missing の shape が属性予測器の設定と一致しない")
            for index, head in enumerate(self.categorical_heads):
                observed = ~missing[:, index]
                if observed.any():
                    losses.append(F.cross_entropy(head(hidden[observed]), categorical[observed, index].long()))
        if self.continuous_head is not None:
            continuous = attributes["continuous"]
            missing = attributes["continuous_missing"].bool()
            if continuous.ndim != 2 or continuous.shape[1] != self.continuous_head.out_features or missing.shape != continuous.shape:
                raise ValueError("continuous / continuous_missing の shape が属性予測器の設定と一致しない")
            predicted = self.continuous_head(hidden)
            for index in range(predicted.shape[1]):
                observed = ~missing[:, index]
                if observed.any():
                    losses.append(F.mse_loss(predicted[observed, index], continuous[observed, index]))
        if not losses:
            # 欠損だけの batch でも optimizer の backward 契約を保つ。
            return features.sum() * 0
        return torch.stack(losses).mean()
