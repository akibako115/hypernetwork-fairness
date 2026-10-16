"""セル均等抽出下で母集団 CE を保持する対照目的関数。"""

import torch
from torch.nn import functional as F

from .conditional import ConditionalObjective
from .input import ObjectiveInput


class SamplingCorrectedTaskLoss(ConditionalObjective):
    """rho=P/Q と固定 Z による母集団加重 CE。"""

    def forward(self, inputs: ObjectiveInput) -> torch.Tensor:
        """抽出確率を補正した task loss を返す。

        Args:
            inputs: 二値 logits と群・ラベル。

        Returns:
            Tensor: scalar loss。
        """
        groups, _ = self._cells(inputs)
        return self._task(inputs, groups)


class SamplingCorrectedGroupTaskLoss(ConditionalObjective):
    """群内クラス比を母集団へ戻した群別 CE を、全群等重みで平均する uniform-group 対照。

    群 g のリスクは sum_c w_c * pi(c|g) * (セル平均 CE)。pi(c|g) は train の群内クラス比なので、
    セル均等抽出のバッチから母集団の群リスクを偏りなく推定できる。群重みは 1/G で固定する。
    """

    def forward(self, inputs: ObjectiveInput) -> torch.Tensor:
        """母集団クラス比による群別 CE の等重み平均を返す。

        Args:
            inputs: 二値 logits と群・ラベル。

        Returns:
            Tensor: scalar loss。
        """
        _, cells = self._cells(inputs)
        ce = F.cross_entropy(inputs.logits.float(), inputs.target, reduction="none")
        means = torch.stack([torch.stack([ce[m].mean() for m in row]) for row in cells])
        risks = (means * self.class_weight * self.population / self.population.sum(dim=1, keepdim=True)).sum(dim=1)
        if self.training:
            self._diagnostics = {"risk": risks.detach()}
        return risks.mean()
