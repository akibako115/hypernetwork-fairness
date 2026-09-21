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


def _class_weight_tensor(
    class_weight: Sequence[float] | None,
) -> torch.Tensor | None:
    """正の有限値で構成されたクラス重みをTensorへ変換する。"""
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


def _group_class_weight_tensor(
    class_weight: Sequence[Sequence[float]] | None,
    *,
    num_groups: int,
) -> torch.Tensor | None:
    """group 目的関数の class weight を検証して `[num_groups, num_classes]` の Tensor にする。

    group 目的関数が受け取るのは `None` か group ごとの重みだけで、全 group 共通の
    `[num_classes]` は受け取らない。共通の重みでは group loss がその group の陽性率に
    依存したままになり、adversarial weight が「識別が難しい group」ではなく「陽性が多い
    group」へ寄る。重みを掛けるなら group ごとに掛ける、を型で固定する。

    group ごとに重みを変えるときは、標本平均 `sum_c f_{g,c} * w[g,c]`（`f_{g,c}` は group g
    内のクラス c の比率）が group によらず一定でなければならない。一定でないと group loss の
    尺度が group ごとに変わり、adversarial weight が group の難しさではなく尺度を追う。
    `w[g,c] = 1 / (num_classes * f_{g,c})` はどの group でも標本平均が 1 になり、かつ group
    loss の期待値が群内クラス平均 `mean_c mean_{i in (g,c)} loss_i` に一致する。

    Args:
        class_weight: `None`、または `[num_groups, num_classes]`
        num_groups: 固定cohortのgroup数

    Returns:
        torch.Tensor | None: `[num_groups, num_classes]` の重み。`class_weight` が
            `None` なら `None`

    Raises:
        ValueError: 空の場合、`[num_groups, num_classes]` の入れ子でない場合、行の長さが
            揃っていない場合、行数が `num_groups` と違う場合、または有限の正値でない値を
            含む場合。
    """
    if class_weight is None:
        return None

    rows = list(class_weight)
    if not rows:
        raise ValueError(f"class_weight must not be empty, got {rows}")
    if not isinstance(rows[0], Sequence) or isinstance(rows[0], (str, bytes)):
        raise ValueError(f"group 目的関数の class_weight は [num_groups, num_classes] である必要があるが、{rows} が指定された")

    values = [list(row) for row in rows]
    if len(values) != num_groups:
        raise ValueError(f"group ごとの class_weight は {num_groups} 行である必要があるが、{len(values)} 行が指定された")
    if len({len(row) for row in values}) != 1:
        raise ValueError(f"group ごとの class_weight は全行が同じ長さである必要がある、got {[len(row) for row in values]}")
    if not values[0]:
        raise ValueError(f"class_weight must not be empty, got {values}")

    weight = torch.tensor(values, dtype=torch.float32)
    if not torch.isfinite(weight).all():
        raise ValueError(f"class_weight must contain only finite values, got {values}")
    if (weight <= 0).any():
        raise ValueError(f"class_weight must contain only positive values, got {values}")
    return weight


class TaskLoss(nn.Module):
    """分類タスクの cross-entropy 学習目的を計算する。

    class_weight は register_buffer で保持するため
    LitModule.to(device) 時に自動で GPU に移る。
    """

    def __init__(
        self,
        class_weight: Sequence[float] | None = None,
    ):
        """class_weight を buffer 化して保持する。

        Args:
            class_weight: クラスごとのcross-entropy重み

        Returns:
            None
        """
        super().__init__()

        class_w = _class_weight_tensor(class_weight)
        self.register_buffer("_class_weight", class_w, persistent=False)

        if class_w is None:
            print("[TaskLoss] class_weight=None (unweighted cross-entropy)")
        else:
            print(f"[TaskLoss] class_weight={class_w.tolist()}")

    def forward(
        self,
        inputs: ObjectiveInput,
    ) -> torch.Tensor:
        """class_weight 付き cross-entropy loss を計算する。

        Args:
            inputs: logits・target・属性を持つ1バッチ分の入力

        Returns:
            torch.Tensor: スカラーの loss
        """
        return F.cross_entropy(
            inputs.logits,
            inputs.target,
            weight=self._class_weight,
        )


def _per_sample_loss(
    logits: torch.Tensor,
    target: torch.Tensor,
    group_ids: torch.Tensor,
    weight: torch.Tensor | None,
) -> torch.Tensor:
    """group ごとの class weight を掛けた per-sample cross-entropy を返す。

    `F.cross_entropy(weight=...)` は全 group 共通の重みしか取れないため、重みは
    cross-entropy の外で掛ける。`reduction="none"` の戻り値へ `w[y_i]` を掛けた値は
    `F.cross_entropy(weight=...)` の per-sample 出力と一致するので、全 group 共通の
    重みを渡したときの数値は従来と変わらない。
    """
    losses = F.cross_entropy(logits, target, reduction="none")
    if weight is None:
        return losses
    return losses * weight[group_ids, target.long()]


def _group_losses(
    per_sample_loss: torch.Tensor,
    group_ids: torch.Tensor,
    *,
    num_groups: int,
    group_key: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    """全groupの平均loss（欠損groupは0）と観測マスクを返す。

    平均の分母は group の**サンプル数**であり、重みの総和ではない。`class_weight` に
    `1 / (num_classes * f_{g,c})` を渡したとき、この平均の期待値がちょうど群内クラス平均に
    なるのはそのためである。

    group ID が [0, num_groups-1] に収まることは `CohortImageDataModule` がロード時に
    sidecar 全体へ対して検証済みのため、step ごとには再検証しない（`.any()` を Python の
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


class UniformGroupTaskLoss(nn.Module):
    """観測されたgroupごとの平均cross-entropyを等重みで最適化する。

    `class_weight` の解釈は `GroupDROTaskLoss` と同じで、`None` か group ごとの
    `[num_groups, num_classes]` だけを受け取る。
    """

    def __init__(
        self,
        num_groups: int,
        class_weight: Sequence[Sequence[float]] | None = None,
        group_key: str = "group_id",
    ):
        """num_groups・group_key を検証し、class_weight を buffer 化する。

        Args:
            num_groups: 固定cohortのgroup数
            class_weight: `None`、または group ごとの `[num_groups, num_classes]`
            group_key: ObjectiveInput.attributes 内のグループIDのキー

        Returns:
            None

        Raises:
            ValueError: num_groups が 2 未満、group_key が空、または class_weight が
                不正な場合。
        """
        super().__init__()
        if num_groups < 2:
            raise ValueError(f"num_groups must be at least 2, got {num_groups}")
        if not group_key:
            raise ValueError("group_key must not be empty")
        self.register_buffer("_class_weight", _group_class_weight_tensor(class_weight, num_groups=num_groups), persistent=False)
        self.num_groups = num_groups
        self.group_key = group_key

    def forward(self, inputs: ObjectiveInput) -> torch.Tensor:
        """観測されたgroupごとの平均lossを一様平均して返す。

        Args:
            inputs: logits・target・group ID を含む属性を持つ1バッチ分の入力

        Returns:
            torch.Tensor: スカラーの loss
        """
        group_ids = inputs.attributes[self.group_key].long()
        group_losses, observed = _group_losses(
            _per_sample_loss(inputs.logits, inputs.target, group_ids, self._class_weight),
            group_ids,
            num_groups=self.num_groups,
            group_key=self.group_key,
        )
        return group_losses[observed].mean()


class GroupDROTaskLoss(nn.Module):
    """固定hidden cohortに対するonline Group DROの学習目的を計算する。

    group loss は group ごとの重み付き平均 cross-entropy で、`class_weight` が
    その意味を決める。

    | class_weight | group loss | adversarial weight の寄り方 |
    | --- | --- | --- |
    | `None` | 素の平均 CE | group の陽性率とほぼ単調に対応するため、「識別が難しい group」ではなく「陽性が多い group」へ寄る |
    | `[num_groups, num_classes]` | `w[g,c] = 1 / (C * f_{g,c})` なら期待値が群内クラス平均 | 陽性率依存が消える |

    全 group 共通の `[num_classes]` は受け取らない。それでは陽性率依存が group 間に
    残るためで、重みを掛けるなら group ごとに掛ける。group ごとの重みは cohort の構成から
    決まるので、cohort を作り直すたびに解き直す。制約は `_group_class_weight_tensor` を参照する。
    """

    def __init__(
        self,
        num_groups: int,
        class_weight: Sequence[Sequence[float]] | None = None,
        step_size: float = 0.01,
        group_key: str = "group_id",
    ):
        """num_groups・step_size・group_key を検証し、一様初期化した adv_probs buffer を作る。

        Args:
            num_groups: 固定cohortのgroup数
            class_weight: `None`、または group ごとの `[num_groups, num_classes]`
            step_size: adversarial weightの指数勾配ステップ幅。group loss の大きさと
                1 epoch あたりの step 数に合わせる。class weight を変えると group loss の
                大きさも変わるため、独立に選べる値ではない
            group_key: ObjectiveInput.attributes 内のグループIDのキー

        Returns:
            None

        Raises:
            ValueError: num_groups が 2 未満、step_size が非正または非有限、group_key が
                空、または class_weight が不正な場合。
        """
        super().__init__()
        if num_groups < 2:
            raise ValueError(f"num_groups must be at least 2, got {num_groups}")
        if not math.isfinite(step_size) or step_size <= 0:
            raise ValueError(f"step_size must be finite and positive, got {step_size}")
        if not group_key:
            raise ValueError("group_key must not be empty")
        self.register_buffer("adv_probs", torch.full((num_groups,), 1.0 / num_groups))
        self.register_buffer("_class_weight", _group_class_weight_tensor(class_weight, num_groups=num_groups), persistent=False)
        self.num_groups = num_groups
        self.step_size = float(step_size)
        self.group_key = group_key

    def forward(self, inputs: ObjectiveInput) -> torch.Tensor:
        """group lossでexponentiated gradient更新したadv_probsとの加重和を返す。

        Args:
            inputs: logits・target・group ID を含む属性を持つ1バッチ分の入力

        Returns:
            torch.Tensor: スカラーの loss
        """
        group_ids = inputs.attributes[self.group_key].long()
        group_losses, _ = _group_losses(
            _per_sample_loss(inputs.logits, inputs.target, group_ids, self._class_weight),
            group_ids,
            num_groups=self.num_groups,
            group_key=self.group_key,
        )
        # exponentiated gradientでadv_probsを更新する（勾配はここに流さない）。
        # max を引いてから exp する。正規化後の値は同じで、overflow だけを避ける。
        with torch.no_grad():
            shifted = group_losses.detach()
            updated = self.adv_probs * torch.exp(self.step_size * (shifted - shifted.max()))
            self.adv_probs.copy_(updated / updated.sum())
        return torch.dot(self.adv_probs, group_losses)
