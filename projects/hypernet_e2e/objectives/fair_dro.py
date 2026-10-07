"""Jung et al. (ICLR 2023) Algorithm 1 の classwise FairDRO。"""

import math
from collections.abc import Sequence

import torch
from torch.nn import functional as F

from .conditional import ConditionalObjective
from .input import ObjectiveInput


def chi_square_best_response(losses: torch.Tensor, radius: float) -> torch.Tensor:
    """χ² 球 Q_rho 上の内側最大化の閉形式解を返す（Prop. 2, 式 8）。

    Args:
        losses: [A] の群別損失。
        radius: rho。

    Returns:
        Tensor: [A] の quasi-probability。偏差がなければ一様。
    """
    size = losses.numel()
    deviation = losses - losses.mean()
    norm = torch.linalg.vector_norm(deviation)
    uniform = torch.full_like(losses, 1 / size)
    if norm <= 0:
        return uniform
    return uniform + math.sqrt(radius / size) * deviation / norm


class FairDROTaskLoss(ConditionalObjective):
    """クラスごとの群重み q^y で群別 CE を重み付けする。

    q は train 全件の 0-1 誤り率から epoch ごとに平滑化 IBR で更新する（式 10）。
    バッチはセル均等抽出（Alg. 1 l.6）なので母集団補正と class weight は使わない。
    """

    def __init__(
        self,
        num_groups: int,
        population_cell_counts: Sequence[Sequence[float]],
        group_key: str = "group_id",
        radius: float = 1.0,
        max_epochs: int = 30,
        class_weight: Sequence[float] | None = None,
    ) -> None:
        """一様な q と平滑化スケジュールを設定する。

        Args:
            num_groups: 群数。
            population_cell_counts: train の [G,2] 件数。セル存在の検証に使う。
            group_key: 群属性のキー。
            radius: χ² 球の半径 rho。
            max_epochs: eta_t = 1 - t/T の T。
            class_weight: 互換設定用。None のみ許可。

        Returns:
            None
        """
        if class_weight is not None:
            raise ValueError("FairDRO に class_weight は適用しない")
        super().__init__(num_groups, population_cell_counts, group_key)
        if not math.isfinite(radius) or radius <= 0:
            raise ValueError("radius は正の有限値が必要")
        if int(max_epochs) < 1:
            raise ValueError("max_epochs は 1 以上が必要")
        self.radius, self.max_epochs = float(radius), int(max_epochs)
        self.register_buffer("q", torch.full((2, num_groups), 1 / num_groups, dtype=torch.float64))

    def forward(self, inputs: ObjectiveInput) -> torch.Tensor:
        """現在の q による classwise 群 CE を返す（式 9）。

        Args:
            inputs: 二値 logits と群・ラベル。

        Returns:
            Tensor: scalar loss。
        """
        _, cells = self._cells(inputs)
        ce = F.cross_entropy(inputs.logits.float(), inputs.target, reduction="none")
        risks = torch.stack([torch.stack([ce[m].mean() for m in row]) for row in cells])
        q = self.q.detach().float().T
        if self.training:
            score = inputs.logits[:, 1].float() - inputs.logits[:, 0].float()
            rates = torch.stack([torch.stack([(score[m] > 0).float().mean() for m in row]) for row in cells])
            # q は epoch 内で一定なので module が epoch 末に名前付きで記録する。
            self._diagnostics = {"risk": risks.detach(), "rates": rates.detach(), "cell_count": torch.stack([m.sum() for row in cells for m in row])}
        return (q * risks).sum() / 2

    @torch.no_grad()
    def commit_update(self) -> None:
        """q は epoch 単位で更新するので step ごとには何もしない。

        Args:
            なし

        Returns:
            None
        """
        self._pending = None

    @torch.no_grad()
    def update_from_errors(self, errors: torch.Tensor) -> dict[str, torch.Tensor]:
        """train 全件の群・クラス別 0-1 誤り率で q を平滑化 IBR 更新する。

        Args:
            errors: [G,2] の誤り率。

        Returns:
            dict: q_star・eta・更新後 q。
        """
        errors = errors.to(self.q.device, torch.float64)
        if errors.shape != (self.num_groups, 2) or not torch.isfinite(errors).all():
            raise ValueError("errors は有限値の [G,2] が必要")
        step = int(self.update_count)
        if step >= self.max_epochs:
            raise RuntimeError("FairDRO の q 更新回数が max_epochs を超えた")
        eta = 1 - step / self.max_epochs
        q_star = torch.stack([chi_square_best_response(errors[:, c], self.radius) for c in range(2)])
        self.q.copy_((1 - eta) * self.q + eta * q_star)
        self.update_count.add_(1)
        return {"q_star": q_star, "eta": torch.tensor(eta, dtype=torch.float64), "q": self.q.clone()}
