"""セル均等抽出下で母集団 CE を保持する対照目的関数。"""

from collections.abc import Sequence

import torch
from torch.nn import functional as F

from .conditional import ConditionalObjective, _positive_step
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
    """群内クラス比を母集団へ戻す uniform-group / Group DRO 対照。

    mode=dro のみモデル更新後に双対状態を更新する。
    """

    @property
    def dual_update(self) -> str | None:
        """mode=dro のときだけ step 単位で双対を更新する。

        Args:
            なし

        Returns:
            str | None: ``"step"`` または None。
        """
        return "step" if self.mode == "dro" else None

    def __init__(
        self,
        num_groups: int,
        population_cell_counts: Sequence[Sequence[float]],
        group_key: str = "group_id",
        class_weight: Sequence[float] | None = None,
        mode: str = "uniform",
        dual_step_size: float = 0.01,
    ) -> None:
        """群重みの種類を指定する。

        Args:
            num_groups: 群数。
            population_cell_counts: train の [G,2] 件数。
            group_key: 群属性キー。
            class_weight: クラス重み。
            mode: uniform または dro。
            dual_step_size: 指数勾配ステップ幅。

        Returns:
            None
        """
        super().__init__(num_groups, population_cell_counts, group_key, class_weight)
        if mode not in {"uniform", "dro"}:
            raise ValueError("mode は uniform/dro")
        self.mode = mode
        self.dual_step_size = _positive_step(dual_step_size, "dual_step_size")
        self.register_buffer("log_q", torch.full((num_groups,), -torch.log(torch.tensor(float(num_groups)))))

    def forward(self, inputs: ObjectiveInput) -> torch.Tensor:
        """母集団クラス比による群別 CE を旧 q で重み付けする。

        Args:
            inputs: 二値 logits と群・ラベル。

        Returns:
            Tensor: scalar loss。
        """
        _, cells = self._cells(inputs)
        ce = F.cross_entropy(inputs.logits.float(), inputs.target, reduction="none")
        means = torch.stack([torch.stack([ce[m].mean() for m in row]) for row in cells])
        risks = (means * self.class_weight * self.population / self.population.sum(dim=1, keepdim=True)).sum(dim=1)
        q = self.log_q.exp().detach().clone()
        if self.training:
            self._pending = risks.detach()
            self._diagnostics = {"risk": risks.detach(), "q_before": q}
        return torch.dot(q, risks)

    @torch.no_grad()
    def commit_update(self) -> None:
        """DRO のみ群重みを更新する。

        Args:
            なし

        Returns:
            None
        """
        if self._pending is not None:
            if self.mode == "dro":
                self.log_q.copy_(torch.log_softmax(self.log_q + self.dual_step_size * self._pending, dim=0))
                self.update_count.add_(1)
            self._pending = None
            self._diagnostics["q_after"] = self.log_q.exp().clone()
