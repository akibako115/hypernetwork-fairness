"""固定 group を使う公平性 objective を定義する。"""

import math
from collections.abc import Sequence

import torch
import torch.nn.functional as F

from .base import Objective
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
        raise ValueError("Group DRO には [num_classes] の class_weight が必要")
    values = list(class_weight)
    if values and isinstance(values[0], Sequence) and not isinstance(values[0], (str, bytes)):
        raise ValueError("Group DRO の class_weight は [num_classes] である必要がある")
    weight = _class_weight_tensor(values)
    if weight is None:
        raise ValueError("Group DRO には [num_classes] の class_weight が必要")
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


def _exponentiated_update(probs: torch.Tensor, group_losses: torch.Tensor, step_size: float) -> torch.Tensor:
    """公式 GroupDRO と同じ乗算・全群正規化で q を更新する。

    欠損群の loss は 0 とし、乗算係数を 1 に保ったまま全群で正規化する。
    自然抽出時も欠損群の確率質量を保存する独自処理は行わない。
    """
    updated = probs * torch.exp(step_size * group_losses.detach())
    return updated / updated.sum()


class UniformGroupTaskLoss(Objective):
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


class _ClassWeightedGroupDROTaskLoss(Objective):
    """q の更新に使う群損失の重みだけが異なる Group DRO 実装を共有する内部 module。

    モデルの更新には、常に全群共通の `[num_classes]` 重みを掛けた群損失を使う。`q_class_weight` が None なら
    q も同じ群損失で更新する（原典の形）。与えれば、q だけをその重みの群損失で更新する。
    """

    def __init__(self, num_groups: int, class_weight: torch.Tensor, q_class_weight: torch.Tensor | None, step_size: float, group_key: str):
        super().__init__()
        if num_groups < 2:
            raise ValueError(f"num_groups must be at least 2, got {num_groups}")
        if not math.isfinite(step_size) or step_size <= 0:
            raise ValueError(f"step_size must be finite and positive, got {step_size}")
        if not group_key:
            raise ValueError("group_key must not be empty")
        self.register_buffer("adv_probs", torch.full((num_groups,), 1.0 / num_groups))
        self.register_buffer("_class_weight", class_weight, persistent=False)
        self.register_buffer("_q_class_weight", q_class_weight, persistent=False)
        self.num_groups = num_groups
        self.step_size = float(step_size)
        self.group_key = group_key

    def _weighted_group_losses(self, inputs: ObjectiveInput, group_ids: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
        per_sample_loss = _per_sample_loss(inputs.logits, inputs.target, group_ids, weight)
        group_losses, _ = _group_losses(per_sample_loss, group_ids, num_groups=self.num_groups, group_key=self.group_key)
        return group_losses

    def forward(self, inputs: ObjectiveInput) -> torch.Tensor:
        group_ids = inputs.attributes[self.group_key].long()
        group_losses = self._weighted_group_losses(inputs, group_ids, self._class_weight)
        q_losses = group_losses if self._q_class_weight is None else self._weighted_group_losses(inputs, group_ids, self._q_class_weight)
        with torch.no_grad():
            self.adv_probs.copy_(_exponentiated_update(self.adv_probs, q_losses, self.step_size))
            counts = torch.bincount(group_ids, minlength=self.num_groups)
            # 最適化で使ったlossをそのまま渡し、loggerで再計算・q再更新をしない。
            self._batch_diagnostics = {
                "q": self.adv_probs.detach().clone(),
                "loss_sum": group_losses.detach().double() * counts,
                "q_loss_sum": q_losses.detach().double() * counts,
                "count": counts.detach(),
            }
        return torch.dot(self.adv_probs, group_losses)

    def batch_diagnostics(self) -> dict[str, torch.Tensor]:
        """直近のtrain forwardで更新したqと、群別loss合計・件数を返す。

        Args:
            なし

        Returns:
            dict[str, Tensor]: qは更新後の値。`loss_sum` はモデルの更新に使った群損失、`q_loss_sum` は q の更新に
                使った群損失の、標本平均用合計（原典の形では両者は同じ）。欠損群のloss合計と件数は0。
                checkpointには記録しない一時的な診断値。
        """
        return self._batch_diagnostics


class GlobalClassWeightedGroupDROTaskLoss(_ClassWeightedGroupDROTaskLoss):
    """global `[num_classes]` class weight を使う online Group DRO（原典の形）。q もモデルもこの重みの群損失で更新する。"""

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
        super().__init__(num_groups, _global_class_weight_tensor(class_weight), None, step_size, group_key)


class SubgroupClassWeightedGroupDROTaskLoss(_ClassWeightedGroupDROTaskLoss):
    """q だけを群ごとの `[num_groups, num_classes]` class weight の群損失で更新する online Group DRO。

    群ごとの重み `n_g / (C · n_{g,c})` を掛けた群損失は「陽性の平均 CE と陰性の平均 CE の平均」になり、群の陽性率に
    よらない。全群共通の重みのままだと q は陽性率の高い群を追う（docs/methods/group-dro.md）ので、q の更新にだけ
    これを使う。モデルの更新には原典の形と同じ全群共通の重みの群損失を使う。群ごとの重みでモデルまで更新すると、
    陽性と陰性の重みの比が群ごとに変わり、群を見分けられるモデルでは最適な logit が群ごとに異なる量
    （log(n_{g,0} / n_{g,1})）だけずれる。しきい値 0.5 の群別 TPR・FPR に、群別しきい値と同じ効果が混ざる。
    """

    def __init__(
        self,
        num_groups: int,
        class_weight: Sequence[float] | None = None,
        q_class_weight: Sequence[Sequence[float]] | None = None,
        step_size: float = 0.01,
        group_key: str = "group_id",
    ):
        """モデル更新用の global class weight と、q 更新用の subgroup class weight を必須として初期化する。

        Args:
            num_groups: 固定 group の総数。2 以上。
            class_weight: モデルの更新に使う、全 group 共通の正の有限な `[num_classes]` 重み。必須。
            q_class_weight: q の更新に使う、subgroup ごとの正の有限な `[num_groups, num_classes]` 重み。必須。
            step_size: adversarial weight の正の指数勾配ステップ幅。
            group_key: ObjectiveInput.attributes 内の group ID キー。

        Returns:
            None
        """
        if q_class_weight is None:
            raise ValueError("subgroup-wise Group DRO には [num_groups, num_classes] の q_class_weight が必要")
        super().__init__(
            num_groups,
            _global_class_weight_tensor(class_weight),
            _subgroup_class_weight_tensor(q_class_weight, num_groups=num_groups),
            step_size,
            group_key,
        )
