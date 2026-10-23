"""学習目的関数が共有する batch 入力を定義する。"""

from collections.abc import Mapping
from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class ObjectiveInput:
    """学習目的関数へ渡す1バッチ分の入力を保持する。

    `logits` は `[B, num_classes]`、`target` は `[B]`、`attributes` は目的関数が必要とする
    属性辞書である。`group_samples` は `group_sample_size` を持つ目的関数
    （FIS）だけが受け取る、群ごとに別に引いたバッチの勾配なしの入力である。
    """

    logits: torch.Tensor
    target: torch.Tensor
    attributes: Mapping[str, torch.Tensor]
    group_samples: "ObjectiveInput | None" = None
