"""群・クラス条件付き統計と、標本抽出補正を共有する。"""

import math
from collections.abc import Sequence

import torch
from torch import nn
from torch.nn import functional as F

from .input import ObjectiveInput


class ConditionalObjective(nn.Module):
    """全セルが存在する二値分類バッチの母集団補正を保持する。

    counts は train だけから求める [G,2]。セル均等抽出を前提とする。
    """

    def __init__(self, num_groups: int, population_cell_counts: Sequence[Sequence[float]], group_key: str = "group_id", class_weight: Sequence[float] | None = None) -> None:
        """母集団確率と固定正規化定数を設定する。

        Args:
            num_groups: 群数。
            population_cell_counts: train の群・クラス別件数。
            group_key: 属性辞書の群キー。
            class_weight: 全群共通のクラス重み。

        Returns:
            None
        """
        super().__init__()
        counts = torch.as_tensor(population_cell_counts, dtype=torch.float64)
        if num_groups < 2 or counts.shape != (num_groups, 2) or not torch.isfinite(counts).all() or (counts <= 0).any():
            raise ValueError("population_cell_counts は全セルが正の有限値の [G,2] である必要がある")
        weight = torch.ones(2) if class_weight is None else torch.as_tensor(class_weight, dtype=torch.float32)
        if weight.shape != (2,) or not torch.isfinite(weight).all() or (weight <= 0).any():
            raise ValueError("class_weight は正の有限値の [2] である必要がある")
        self.num_groups, self.group_key = int(num_groups), group_key
        self.register_buffer("population", (counts / counts.sum()).float())
        self.register_buffer("class_weight", weight)
        self.register_buffer("update_count", torch.tensor(0, dtype=torch.long))
        self._pending = None
        self._diagnostics = {}

    def _cells(self, inputs: ObjectiveInput) -> tuple[torch.Tensor, list[list[torch.Tensor]]]:
        groups = inputs.attributes[self.group_key].long()
        if not torch.isfinite(inputs.logits).all():
            raise ValueError("logits に非有限値がある")
        if inputs.logits.shape != (inputs.target.numel(), 2) or groups.shape != inputs.target.shape:
            raise ValueError("二値 logits [B,2] と target/group [B] が必要")
        if ((groups < 0) | (groups >= self.num_groups)).any() or ((inputs.target < 0) | (inputs.target > 1)).any():
            raise ValueError("group/target が定義域外")
        masks = [[(groups == g) & (inputs.target == c) for c in range(2)] for g in range(self.num_groups)]
        if any(not mask.any() for row in masks for mask in row):
            raise ValueError("条件付き objective のバッチには全群・クラスセルが必要")
        return groups, masks

    def _task(self, inputs: ObjectiveInput, groups: torch.Tensor) -> torch.Tensor:
        ce = F.cross_entropy(inputs.logits.float(), inputs.target, reduction="none")
        rho = 2 * self.num_groups * self.population[groups, inputs.target]
        normalizer = (self.population * self.class_weight).sum()
        return (rho * self.class_weight[inputs.target] * ce).mean() / normalizer

    def batch_diagnostics(self) -> dict[str, torch.Tensor]:
        """直近の学習計算の診断値を返す。再計算・状態更新はしない。

        Args:
            なし

        Returns:
            dict: detached Tensor の辞書。
        """
        return self._diagnostics


def _positive_step(value: float, name: str) -> float:
    if not math.isfinite(value) or value <= 0:
        raise ValueError(f"{name} は正の有限値が必要")
    return float(value)
