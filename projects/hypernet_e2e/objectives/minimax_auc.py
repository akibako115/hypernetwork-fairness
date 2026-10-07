"""Yang et al. (AAAI 2023) Algorithm 1 の pairwise logistic minimax。"""

from collections.abc import Sequence

import torch
from torch.nn import functional as F

from .conditional import ConditionalObjective, _positive_step
from .input import ObjectiveInput


class MinimaxAUCTaskLoss(ConditionalObjective):
    """陽性群 g・陰性群 h の順位損失を指数勾配で最悪化する。

    forward は旧 q を使い、commit_update はモデル更新後に一度だけ呼ぶ。
    q 初期値は母集団の陽性群分布と陰性群分布の積。
    """

    def __init__(
        self,
        num_groups: int,
        population_cell_counts: Sequence[Sequence[float]],
        group_key: str = "group_id",
        dual_step_size: float = 0.01,
        pair_scope: str = "all",
        class_weight: Sequence[float] | None = None,
    ) -> None:
        """ペア集合と母集団由来の初期分布を設定する。

        Args:
            num_groups: 群数。
            population_cell_counts: train の [G,2] 件数。
            group_key: 群属性のキー。
            dual_step_size: q の指数勾配ステップ幅。
            pair_scope: all または intra。
            class_weight: 互換設定用。純 AUC では None のみ許可。

        Returns:
            None
        """
        if class_weight is not None:
            raise ValueError("純 AUC objective に class_weight は適用しない")
        super().__init__(num_groups, population_cell_counts, group_key)
        self.dual_step_size = _positive_step(dual_step_size, "dual_step_size")
        if pair_scope not in {"all", "intra"}:
            raise ValueError("pair_scope は all/intra")
        self.pairs = [(g, h) for g in range(num_groups) for h in range(num_groups) if pair_scope == "all" or g == h]
        p = self.population
        priors = torch.stack([p[g, 1] / p[:, 1].sum() * p[h, 0] / p[:, 0].sum() for g, h in self.pairs])
        self.register_buffer("log_q", (priors / priors.sum()).log())

    def forward(self, inputs: ObjectiveInput) -> torch.Tensor:
        """旧 q によるリスクを返し、学習時だけ更新用統計を保留する。

        Args:
            inputs: 二値 logits と群・ラベル。

        Returns:
            Tensor: scalar 順位損失。
        """
        _, cells = self._cells(inputs)
        score = inputs.logits[:, 1].float() - inputs.logits[:, 0].float()
        if not torch.isfinite(score).all():
            raise ValueError("AUC score に非有限値がある")
        risks = torch.stack([F.softplus(-(score[cells[g][1]][:, None] - score[cells[h][0]][None, :])).mean() for g, h in self.pairs])
        q = self.log_q.detach().exp().clone()
        if self.training:
            self._pending = risks.detach()
            pair_counts = torch.stack([cells[g][1].sum() * cells[h][0].sum() for g, h in self.pairs])
            self._diagnostics = {"risk": risks.detach(), "q_before": q, "cell_count": torch.stack([m.sum() for row in cells for m in row]), "pair_count": pair_counts}
        return torch.dot(q, risks)

    @torch.no_grad()
    def commit_update(self) -> None:
        """モデル更新後、同じ forward の損失で q を更新する。

        Args:
            なし

        Returns:
            None
        """
        if self._pending is not None:
            self.log_q.copy_(torch.log_softmax(self.log_q + self.dual_step_size * self._pending, dim=0))
            self.update_count.add_(1)
            self._pending = None
            self._diagnostics["q_after"] = self.log_q.exp().clone()
            self._diagnostics["q_entropy"] = -(self.log_q.exp() * self.log_q).sum()
            self._diagnostics["q_max"] = self.log_q.exp().max()


class PopulationAUCTaskLoss(MinimaxAUCTaskLoss):
    """母集団由来 q を固定した AUCMax 対照条件。"""

    @torch.no_grad()
    def commit_update(self) -> None:
        """双対分布を固定し、保留統計だけ破棄する。

        Args:
            なし

        Returns:
            None
        """
        self._pending = None
        self._diagnostics["q_after"] = self.log_q.exp().clone()
