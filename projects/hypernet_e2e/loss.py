"""hypernet_e2e の通常分類学習で使う目的関数を定義する。"""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass(frozen=True)
class ObjectiveInput:
    """学習目的関数へ渡す1バッチ分の入力を保持する。"""

    logits: torch.Tensor
    target: torch.Tensor
    attributes: Mapping[str, torch.Tensor]


def _class_weight_tensor(class_weight: Sequence[float] | None) -> torch.Tensor | None:
    """正の有限値で構成されたクラス重みを Tensor へ変換する。"""
    if class_weight is None:
        return None

    values = list(class_weight)
    if not values:
        raise ValueError(f"class_weight must not be empty, got {values}")

    weight = torch.tensor(values, dtype=torch.float32)
    if not torch.isfinite(weight).all():
        raise ValueError(f"class_weight must contain only finite values, got {values}")
    if (weight <= 0).any():
        raise ValueError(f"class_weight must contain only positive values, got {values}")
    return weight


class TaskLoss(nn.Module):
    """分類タスクの class weight 付き cross-entropy 学習目的を計算する。

    ``ObjectiveInput`` の logits と target を受け、スカラー loss を返す。class weight は
    buffer として保持するため、Lightning module とともに同じ device へ移動する。
    """

    def __init__(self, class_weight: Sequence[float] | None = None):
        """class_weight を buffer 化して保持する。"""
        super().__init__()
        self.register_buffer("_class_weight", _class_weight_tensor(class_weight), persistent=False)

    def forward(self, inputs: ObjectiveInput) -> torch.Tensor:
        """class_weight 付き cross-entropy loss を計算する。"""
        return F.cross_entropy(inputs.logits, inputs.target, weight=self._class_weight)
