"""固定 group を使う公平性 objective を定義する。"""

import math
from collections.abc import Callable, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from .input import ObjectiveInput
from .task import _class_weight_tensor


def _subgroup_class_weight_tensor(class_weight: Sequence[Sequence[float]] | None, *, num_groups: int) -> torch.Tensor:
    """subgroup ごとの class weight を検証して Tensor にする。"""
    if class_weight is None:
        raise ValueError("subgroup-wise Group DRO には [num_groups, num_classes] の class_weight が必要")
    rows = list(class_weight)
    if not rows or not isinstance(rows[0], Sequence) or isinstance(rows[0], (str, bytes)):
        raise ValueError("subgroup-wise Group DRO の class_weight は [num_groups, num_classes] である必要がある")
    values = [list(row) for row in rows]
    if len(values) != num_groups:
        raise ValueError(f"group ごとの class_weight は {num_groups} 行である必要があるが、{len(values)} 行が指定された")
    if not values[0] or len({len(row) for row in values}) != 1:
        raise ValueError("group ごとの class_weight は [num_groups, num_classes] である必要がある")
    weight = torch.tensor(values, dtype=torch.float32)
    if not torch.isfinite(weight).all() or (weight <= 0).any():
        raise ValueError(f"class_weight must contain only finite positive values, got {values}")
    return weight


def _global_class_weight_tensor(class_weight: Sequence[float] | None) -> torch.Tensor:
    """global class weight を必須として検証し Tensor にする。"""
    if class_weight is None:
        raise ValueError("global Group DRO には [num_classes] の class_weight が必要")
    values = list(class_weight)
    if values and isinstance(values[0], Sequence) and not isinstance(values[0], (str, bytes)):
        raise ValueError("global Group DRO の class_weight は [num_classes] である必要がある")
    weight = _class_weight_tensor(values)
    if weight is None:
        raise ValueError("global Group DRO には [num_classes] の class_weight が必要")
    return weight


def _optional_group_class_weight_tensor(class_weight: Sequence[float] | Sequence[Sequence[float]] | None, *, num_groups: int) -> torch.Tensor | None:
    """Uniform Group 用に optional な global / subgroup weight を検証する。"""
    if class_weight is None:
        return None
    values = list(class_weight)
    if values and isinstance(values[0], Sequence) and not isinstance(values[0], (str, bytes)):
        return _subgroup_class_weight_tensor(values, num_groups=num_groups)
    return _global_class_weight_tensor(values)


def _per_sample_loss(logits: torch.Tensor, target: torch.Tensor, group_ids: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    """class weight を適用した per-sample cross-entropy を返す。"""
    losses = F.cross_entropy(logits, target, reduction="none")
    return losses * (weight[target.long()] if weight.ndim == 1 else weight[group_ids, target.long()])


def _group_losses(per_sample_loss: torch.Tensor, group_ids: torch.Tensor, *, num_groups: int, group_key: str) -> tuple[torch.Tensor, torch.Tensor]:
    """全 group の平均 loss（欠損 group は0）と観測マスクを返す。"""
    if group_ids.ndim != 1 or group_ids.shape != per_sample_loss.shape:
        raise ValueError(f"attributes[{group_key!r}] must have shape [batch_size]")
    totals = torch.zeros(num_groups, device=per_sample_loss.device, dtype=per_sample_loss.dtype)
    counts = torch.zeros_like(totals)
    totals.scatter_add_(0, group_ids, per_sample_loss)
    counts.scatter_add_(0, group_ids, torch.ones_like(per_sample_loss))
    observed = counts > 0
    group_losses = torch.zeros_like(totals)
    group_losses[observed] = totals[observed] / counts[observed]
    return group_losses, observed


class UniformGroupTaskLoss(nn.Module):
    """観測された group ごとの平均 cross-entropy を等重みで最適化する。"""

    def __init__(self, num_groups: int, class_weight: Sequence[float] | Sequence[Sequence[float]] | None = None, group_key: str = "group_id"):
        super().__init__()
        if num_groups < 2:
            raise ValueError(f"num_groups must be at least 2, got {num_groups}")
        if not group_key:
            raise ValueError("group_key must not be empty")
        self.register_buffer("_class_weight", _optional_group_class_weight_tensor(class_weight, num_groups=num_groups), persistent=False)
        self.num_groups = num_groups
        self.group_key = group_key

    def forward(self, inputs: ObjectiveInput) -> torch.Tensor:
        group_ids = inputs.attributes[self.group_key].long()
        weight = self._class_weight
        per_sample_loss = F.cross_entropy(inputs.logits, inputs.target, reduction="none") if weight is None else _per_sample_loss(inputs.logits, inputs.target, group_ids, weight)
        group_losses, observed = _group_losses(per_sample_loss, group_ids, num_groups=self.num_groups, group_key=self.group_key)
        return group_losses[observed].mean()


class _ClassWeightedGroupDROTaskLoss(nn.Module):
    """class weight の形だけが異なる Group DRO 実装を共有する内部 module。"""

    def __init__(self, num_groups: int, class_weight: object, step_size: float, group_key: str, weight_tensor: Callable[[object], torch.Tensor]):
        super().__init__()
        if num_groups < 2:
            raise ValueError(f"num_groups must be at least 2, got {num_groups}")
        if not math.isfinite(step_size) or step_size <= 0:
            raise ValueError(f"step_size must be finite and positive, got {step_size}")
        if not group_key:
            raise ValueError("group_key must not be empty")
        self.register_buffer("adv_probs", torch.full((num_groups,), 1.0 / num_groups))
        self.register_buffer("_class_weight", weight_tensor(class_weight), persistent=False)
        self.num_groups = num_groups
        self.step_size = float(step_size)
        self.group_key = group_key

    def forward(self, inputs: ObjectiveInput) -> torch.Tensor:
        group_ids = inputs.attributes[self.group_key].long()
        group_losses, _ = _group_losses(_per_sample_loss(inputs.logits, inputs.target, group_ids, self._class_weight), group_ids, num_groups=self.num_groups, group_key=self.group_key)
        with torch.no_grad():
            detached = group_losses.detach()
            updated = self.adv_probs * torch.exp(self.step_size * (detached - detached.max()))
            self.adv_probs.copy_(updated / updated.sum())
        return torch.dot(self.adv_probs, group_losses)


class GlobalClassWeightedGroupDROTaskLoss(_ClassWeightedGroupDROTaskLoss):
    """global `[num_classes]` class weight を使う online Group DRO。"""

    def __init__(self, num_groups: int, class_weight: Sequence[float] | None = None, step_size: float = 0.01, group_key: str = "group_id"):
        """global class weight を必須として Group DRO を初期化する。

        Args:
            num_groups: 固定 group の総数。2 以上。
            class_weight: 全 group 共通の正の有限な `[num_classes]` 重み。必須。
            step_size: adversarial weight の正の指数勾配ステップ幅。
            group_key: ObjectiveInput.attributes 内の group ID キー。

        Returns:
            None
        """
        super().__init__(num_groups, class_weight, step_size, group_key, _global_class_weight_tensor)


class SubgroupClassWeightedGroupDROTaskLoss(_ClassWeightedGroupDROTaskLoss):
    """subgroup ごとの `[num_groups, num_classes]` class weight を使う online Group DRO。"""

    def __init__(self, num_groups: int, class_weight: Sequence[Sequence[float]] | None = None, step_size: float = 0.01, group_key: str = "group_id"):
        """subgroup-wise class weight を必須として Group DRO を初期化する。

        Args:
            num_groups: 固定 group の総数。2 以上。
            class_weight: subgroup ごとの正の有限な `[num_groups, num_classes]` 重み。必須。
            step_size: adversarial weight の正の指数勾配ステップ幅。
            group_key: ObjectiveInput.attributes 内の group ID キー。

        Returns:
            None
        """
        super().__init__(num_groups, class_weight, step_size, group_key, lambda value: _subgroup_class_weight_tensor(value, num_groups=num_groups))
