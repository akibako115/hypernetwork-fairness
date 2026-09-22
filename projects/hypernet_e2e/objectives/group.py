"""固定 group を使う公平性 objective を定義する。"""

import math
from collections.abc import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from .input import ObjectiveInput
from .task import _class_weight_tensor


def _group_losses(per_sample_loss: torch.Tensor, group_ids: torch.Tensor, *, num_groups: int, group_key: str) -> tuple[torch.Tensor, torch.Tensor]:
    """全groupの平均loss（欠損groupは0）と観測マスクを返す。"""
    if group_ids.ndim != 1 or group_ids.shape != per_sample_loss.shape:
        raise ValueError(f"attributes[{group_key!r}] must have shape [batch_size]")
    totals = torch.zeros(num_groups, device=per_sample_loss.device, dtype=per_sample_loss.dtype)
    counts = torch.zeros(num_groups, device=per_sample_loss.device, dtype=per_sample_loss.dtype)
    totals.scatter_add_(0, group_ids, per_sample_loss)
    counts.scatter_add_(0, group_ids, torch.ones_like(per_sample_loss))
    observed = counts > 0
    group_losses = torch.zeros_like(totals)
    group_losses[observed] = totals[observed] / counts[observed]
    return group_losses, observed


def _group_class_losses(
    per_sample_loss: torch.Tensor,
    group_ids: torch.Tensor,
    target: torch.Tensor,
    *,
    num_groups: int,
    num_classes: int,
    group_key: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    """(group, class) セルごとの平均lossと観測マスクを返す。"""
    if group_ids.ndim != 1 or group_ids.shape != per_sample_loss.shape:
        raise ValueError(f"attributes[{group_key!r}] must have shape [batch_size]")
    cell_ids = group_ids * num_classes + target
    totals = torch.zeros(num_groups * num_classes, device=per_sample_loss.device, dtype=per_sample_loss.dtype)
    counts = torch.zeros_like(totals)
    totals = totals.scatter_add(0, cell_ids, per_sample_loss)
    counts.scatter_add_(0, cell_ids, torch.ones_like(per_sample_loss))
    observed = counts > 0
    cell_losses = torch.zeros_like(totals)
    cell_losses[observed] = totals[observed] / counts[observed]
    return cell_losses.view(num_groups, num_classes), observed.view(num_groups, num_classes)


def _balanced_group_losses(cell_losses: torch.Tensor, observed: torch.Tensor) -> torch.Tensor:
    """group内クラス平均を取り、陽性率に依存しないgroup lossを返す。"""
    present_classes = observed.sum(dim=1)
    group_losses = torch.zeros(cell_losses.shape[0], device=cell_losses.device, dtype=cell_losses.dtype)
    has_class = present_classes > 0
    group_losses[has_class] = cell_losses.sum(dim=1)[has_class] / present_classes[has_class]
    return group_losses


class UniformGroupTaskLoss(nn.Module):
    """観測されたgroupごとの平均cross-entropyを等重みで最適化する。"""

    def __init__(self, num_groups: int, class_weight: Sequence[float] | None = None, group_key: str = "group_id"):
        """num_groups・group_key を検証し、class_weight を buffer 化する。

        Args:
            num_groups: 固定 group の総数。2 以上。
            class_weight: class index 順の正の有限な重み。None なら重み付けしない。
            group_key: ObjectiveInput.attributes 内の group ID キー。空文字列は不可。

        Returns:
            None

        Raises:
            ValueError: num_groups、class_weight、または group_key が不正な場合。
        """
        super().__init__()
        if num_groups < 2:
            raise ValueError(f"num_groups must be at least 2, got {num_groups}")
        if not group_key:
            raise ValueError("group_key must not be empty")
        self.register_buffer("_class_weight", _class_weight_tensor(class_weight), persistent=False)
        self.num_groups = num_groups
        self.group_key = group_key

    def forward(self, inputs: ObjectiveInput) -> torch.Tensor:
        """観測されたgroupごとの平均lossを一様平均して返す。

        Args:
            inputs: `logits` `[B, num_classes]`、`target` `[B]`、`attributes[group_key]` `[B]`

        Returns:
            torch.Tensor: batch に現れた group の平均 loss を等重みで平均した scalar

        Raises:
            ValueError: `attributes[group_key]` の shape が `[B]` でない場合。
        """
        per_sample_loss = F.cross_entropy(inputs.logits, inputs.target, weight=self._class_weight, reduction="none")
        group_losses, observed = _group_losses(per_sample_loss, inputs.attributes[self.group_key].long(), num_groups=self.num_groups, group_key=self.group_key)
        return group_losses[observed].mean()


class GroupDROTaskLoss(nn.Module):
    """固定groupに対するonline Group DROの学習目的を計算する。"""

    def __init__(self, num_groups: int, class_weight: Sequence[float] | None = None, step_size: float = 0.01, group_key: str = "group_id"):
        """num_groups・step_size・group_key を検証し、一様初期化した adv_probs buffer を作る。

        Args:
            num_groups: 固定 group の総数。2 以上。
            class_weight: class index 順の正の有限な重み。None なら重み付けしない。
            step_size: adversarial weight の有限な正の指数勾配ステップ幅。
            group_key: ObjectiveInput.attributes 内の group ID キー。空文字列は不可。

        Returns:
            None

        Raises:
            ValueError: num_groups、class_weight、step_size、または group_key が不正な場合。
        """
        super().__init__()
        if num_groups < 2:
            raise ValueError(f"num_groups must be at least 2, got {num_groups}")
        if not math.isfinite(step_size) or step_size <= 0:
            raise ValueError(f"step_size must be finite and positive, got {step_size}")
        if not group_key:
            raise ValueError("group_key must not be empty")
        self.register_buffer("adv_probs", torch.full((num_groups,), 1.0 / num_groups))
        self.register_buffer("_class_weight", _class_weight_tensor(class_weight), persistent=False)
        self.num_groups = num_groups
        self.step_size = float(step_size)
        self.group_key = group_key

    def forward(self, inputs: ObjectiveInput) -> torch.Tensor:
        """group lossでexponentiated gradient更新したadv_probsとの加重和を返す。

        Args:
            inputs: `logits` `[B, num_classes]`、`target` `[B]`、`attributes[group_key]` `[B]`

        Returns:
            torch.Tensor: 更新後の `adv_probs` と group loss の内積となる scalar

        Raises:
            ValueError: `attributes[group_key]` の shape が `[B]` でない場合。
        """
        per_sample_loss = F.cross_entropy(inputs.logits, inputs.target, weight=self._class_weight, reduction="none")
        group_losses, _ = _group_losses(per_sample_loss, inputs.attributes[self.group_key].long(), num_groups=self.num_groups, group_key=self.group_key)
        with torch.no_grad():
            updated = self.adv_probs * torch.exp(self.step_size * group_losses.detach())
            self.adv_probs.copy_(updated / updated.sum())
        return torch.dot(self.adv_probs, group_losses)


class ClassBalancedGroupDROTaskLoss(nn.Module):
    """group内クラス平均lossに対するonline Group DROの学習目的を計算する。

    batch 横断の EMA で (group, class) セル平均を保持し、陽性率ではなくクラスごとの難しさで
    adversarial weight を更新する。backward する loss は、その batch に観測されたセルだけで作る。
    """

    def __init__(
        self,
        num_groups: int,
        num_classes: int = 2,
        class_weight: Sequence[float] | None = None,
        step_size: float = 0.0001,
        loss_ema_momentum: float = 0.01,
        group_key: str = "group_id",
    ):
        """引数を検証し、adv_probs と (group, class) セル用のEMA buffer を初期化する。

        Args:
            num_groups: 固定 group の総数。2 以上。
            num_classes: 分類 class 数。2 以上。
            class_weight: class index 順の正の有限な重み。None なら重み付けしない。
            step_size: adversarial weight の有限な正の指数勾配ステップ幅。
            loss_ema_momentum: (0, 1] のセル loss EMA 係数。
            group_key: ObjectiveInput.attributes 内の group ID キー。空文字列は不可。

        Returns:
            None

        Raises:
            ValueError: num_groups、num_classes、class_weight、step_size、EMA係数、または group_key が不正な場合。
        """
        super().__init__()
        if num_groups < 2:
            raise ValueError(f"num_groups must be at least 2, got {num_groups}")
        if num_classes < 2:
            raise ValueError(f"num_classes must be at least 2, got {num_classes}")
        if not math.isfinite(step_size) or step_size <= 0:
            raise ValueError(f"step_size must be finite and positive, got {step_size}")
        if not 0.0 < loss_ema_momentum <= 1.0:
            raise ValueError(f"loss_ema_momentum must be in (0, 1], got {loss_ema_momentum}")
        if not group_key:
            raise ValueError("group_key must not be empty")
        self.register_buffer("adv_probs", torch.full((num_groups,), 1.0 / num_groups))
        self.register_buffer("cell_loss_ema", torch.zeros(num_groups, num_classes))
        self.register_buffer("cell_ema_weight", torch.zeros(num_groups, num_classes))
        self.register_buffer("_class_weight", _class_weight_tensor(class_weight), persistent=False)
        self.num_groups = num_groups
        self.num_classes = num_classes
        self.step_size = float(step_size)
        self.loss_ema_momentum = float(loss_ema_momentum)
        self.group_key = group_key

    def _update_adversarial_weights(self, cell_losses: torch.Tensor, observed: torch.Tensor) -> None:
        """観測セルのEMAを進め、bias補正済みのgroup lossでadversarial weightを更新する。"""
        momentum = self.loss_ema_momentum * observed.to(self.cell_loss_ema.dtype)
        self.cell_loss_ema.mul_(1.0 - momentum).add_(momentum * cell_losses)
        self.cell_ema_weight.mul_(1.0 - momentum).add_(momentum)
        seen = self.cell_ema_weight > 0
        corrected = torch.zeros_like(self.cell_loss_ema)
        corrected[seen] = self.cell_loss_ema[seen] / self.cell_ema_weight[seen]
        smoothed = _balanced_group_losses(corrected, seen)
        updated = self.adv_probs * torch.exp(self.step_size * (smoothed - smoothed.max()))
        self.adv_probs.copy_(updated / updated.sum())

    def forward(self, inputs: ObjectiveInput) -> torch.Tensor:
        """(group, class) セルのlossでadv_probsを更新し、balanced group lossとの加重和を返す。

        Args:
            inputs: `logits` `[B, num_classes]`、`target` `[B]`、`attributes[group_key]` `[B]`

        Returns:
            torch.Tensor: 更新後の `adv_probs` と balanced group loss の内積となる scalar

        Raises:
            ValueError: `attributes[group_key]` の shape が `[B]` でない場合。
        """
        per_sample_loss = F.cross_entropy(inputs.logits, inputs.target, weight=self._class_weight, reduction="none")
        cell_losses, observed = _group_class_losses(
            per_sample_loss,
            inputs.attributes[self.group_key].long(),
            inputs.target.long(),
            num_groups=self.num_groups,
            num_classes=self.num_classes,
            group_key=self.group_key,
        )
        with torch.no_grad():
            self._update_adversarial_weights(cell_losses.detach(), observed)
        return torch.dot(self.adv_probs, _balanced_group_losses(cell_losses, observed))
