"""学習目的関数が共有する batch 入力を定義する。"""

from collections.abc import Mapping
from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class ObjectiveInput:
    """学習目的関数へ渡す1バッチ分の入力を保持する。

    `logits` は `[B, num_classes]`、`target` は `[B]`、`attributes` は目的関数が必要とする
    属性辞書である。`features` は通常は None とし、`requires_features=True` の目的関数だけが
    `[B, feature_dim]` の backbone 表現を要求する。
    """

    logits: torch.Tensor
    target: torch.Tensor
    attributes: Mapping[str, torch.Tensor]
    features: torch.Tensor | None = None
