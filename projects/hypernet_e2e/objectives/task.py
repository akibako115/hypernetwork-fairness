"""通常の分類 task objective を定義する。"""

from collections.abc import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from .input import ObjectiveInput


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
        """class_weight を buffer 化して保持する。

        Args:
            class_weight: class index 順の正の有限な重み。None なら重み付けしない。

        Returns:
            None

        Raises:
            ValueError: class_weight が空、非有限、または非正の場合。
        """
        super().__init__()
        self.register_buffer("_class_weight", _class_weight_tensor(class_weight), persistent=False)

    def forward(self, inputs: ObjectiveInput) -> torch.Tensor:
        """class_weight 付き cross-entropy loss を計算する。

        Args:
            inputs: `logits` `[B, num_classes]`、`target` `[B]`、`attributes`（未使用）

        Returns:
            torch.Tensor: batch 平均の scalar loss
        """
        return F.cross_entropy(inputs.logits, inputs.target, weight=self._class_weight)
