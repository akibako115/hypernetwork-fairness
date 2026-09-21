"""hypernet_e2e の学習で使う目的関数を定義する。

通常の cross-entropy に加え、固定 group への公平性手法（uniform group / Group DRO）を持つ。
group を使う目的関数は ``attributes[group_key]`` に ``[0, num_groups)`` の group ID を要求し、
その供給と範囲検証は ``data.group_datamodule.GroupImageDataModule`` が所有する。
"""

import math
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
        """class_weight 付き cross-entropy loss を計算する。

        Args:
            inputs: `logits` `[B, num_classes]`、`target` `[B]`、`attributes`（未使用）

        Returns:
            torch.Tensor: batch 平均の scalar loss
        """
        return F.cross_entropy(inputs.logits, inputs.target, weight=self._class_weight)


def _group_losses(
    per_sample_loss: torch.Tensor,
    group_ids: torch.Tensor,
    *,
    num_groups: int,
    group_key: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    """全groupの平均loss（欠損groupは0）と観測マスクを返す。

    group ID が [0, num_groups-1] に収まることは `GroupImageDataModule` が Dataset 構築時に
    split 全体へ対して検証済みのため、step ごとには再検証しない（`.any()` を Python の
    if で評価すると毎 step host-device 同期が入る）。
    """
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
    """(group, class) セルごとの平均lossと観測マスクを返す。

    戻り値はどちらも `[num_groups, num_classes]`。未観測セルの平均lossは0で埋める。
    group ID の範囲は `_group_losses` と同じく `GroupImageDataModule` が Dataset 構築時に
    検証済みのため、step ごとには再検証しない。
    """
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
    """group内でクラス平均を取り、陽性率に依存しないgroup lossを返す。

    観測されたクラスだけで平均するため、片方のクラスしか含まない batch では
    そのクラスの平均lossがそのまま group loss になる。
    """
    present_classes = observed.sum(dim=1)
    group_losses = torch.zeros(cell_losses.shape[0], device=cell_losses.device, dtype=cell_losses.dtype)
    has_class = present_classes > 0
    group_losses[has_class] = cell_losses.sum(dim=1)[has_class] / present_classes[has_class]
    return group_losses


class UniformGroupTaskLoss(nn.Module):
    """観測されたgroupごとの平均cross-entropyを等重みで最適化する。"""

    def __init__(
        self,
        num_groups: int,
        class_weight: Sequence[float] | None = None,
        group_key: str = "group_id",
    ):
        """num_groups・group_key を検証し、class_weight を buffer 化する。"""
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
        per_sample_loss = F.cross_entropy(
            inputs.logits,
            inputs.target,
            weight=self._class_weight,
            reduction="none",
        )
        group_losses, observed = _group_losses(
            per_sample_loss,
            inputs.attributes[self.group_key].long(),
            num_groups=self.num_groups,
            group_key=self.group_key,
        )
        return group_losses[observed].mean()


class GroupDROTaskLoss(nn.Module):
    """固定groupに対するonline Group DROの学習目的を計算する。"""

    def __init__(
        self,
        num_groups: int,
        class_weight: Sequence[float] | None = None,
        step_size: float = 0.01,
        group_key: str = "group_id",
    ):
        """num_groups・step_size・group_key を検証し、一様初期化した adv_probs buffer を作る。"""
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

        `adv_probs` は buffer なので、この呼び出しが状態を進める。勾配は更新側へ流さない。

        Args:
            inputs: `logits` `[B, num_classes]`、`target` `[B]`、`attributes[group_key]` `[B]`

        Returns:
            torch.Tensor: 更新後の `adv_probs` と group loss の内積となる scalar

        Raises:
            ValueError: `attributes[group_key]` の shape が `[B]` でない場合。
        """
        per_sample_loss = F.cross_entropy(
            inputs.logits,
            inputs.target,
            weight=self._class_weight,
            reduction="none",
        )
        group_losses, _ = _group_losses(
            per_sample_loss,
            inputs.attributes[self.group_key].long(),
            num_groups=self.num_groups,
            group_key=self.group_key,
        )
        # exponentiated gradientでadv_probsを更新する（勾配はここに流さない）
        with torch.no_grad():
            updated = self.adv_probs * torch.exp(self.step_size * group_losses.detach())
            self.adv_probs.copy_(updated / updated.sum())
        return torch.dot(self.adv_probs, group_losses)


class ClassBalancedGroupDROTaskLoss(nn.Module):
    """group内クラス平均lossに対するonline Group DROの学習目的を計算する。

    `GroupDROTaskLoss` は group ごとの素の平均 cross-entropy を使うため、
    group loss がその group の陽性率とほぼ単調に対応してしまい、adversarial weight が
    「識別が難しい group」ではなく「陽性が多い group」へ集中する。ここでは group loss を
    クラス平均 `mean_c mean_{i in (g,c)} loss_i` に置き換え、陽性率に依存しない量にする。

    batch 単位では (group, class) セルの多くが空になる（陽性率が低い group ほど頻繁に空く）。
    セルが空いた batch をそのまま使うと group loss が陰性側へ引かれ、陽性率依存が
    別経路で復活するため、adversarial weight の更新には batch 横断の EMA を使う。
    backward する目的関数側は、その batch に実在するセルだけで構成する。

    Args:
        num_groups: 固定groupの数
        num_classes: 分類クラス数。(group, class) セルの構成に使う
        class_weight: クラスごとのcross-entropy重み
        step_size: adversarial weightの指数勾配ステップ幅
        loss_ema_momentum: セル平均lossのEMA係数。セルが観測されたstepでのみ更新する
        group_key: ObjectiveInput.attributes 内のグループIDのキー
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
        """引数を検証し、adv_probs と (group, class) セル用のEMA buffer を初期化する。"""
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
        # EMA本体と、その累積重み。後者はbias補正（Adamのstep補正と同じ役割）に使う。
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

        # max を引いてから exp する。正規化後の値は同じで、overflow だけを避ける。
        updated = self.adv_probs * torch.exp(self.step_size * (smoothed - smoothed.max()))
        self.adv_probs.copy_(updated / updated.sum())

    def forward(self, inputs: ObjectiveInput) -> torch.Tensor:
        """(group, class) セルのlossでadv_probsを更新し、balanced group lossとの加重和を返す。

        `adv_probs` と セル平均 loss の EMA buffer は、この呼び出しが状態を進める。

        Args:
            inputs: `logits` `[B, num_classes]`、`target` `[B]`、`attributes[group_key]` `[B]`

        Returns:
            torch.Tensor: 更新後の `adv_probs` と、その batch に実在するセルだけで構成した
                balanced group loss の内積となる scalar

        Raises:
            ValueError: `attributes[group_key]` の shape が `[B]` でない場合。
        """
        per_sample_loss = F.cross_entropy(
            inputs.logits,
            inputs.target,
            weight=self._class_weight,
            reduction="none",
        )
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
