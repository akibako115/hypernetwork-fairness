"""Cotter et al. (JMLR 2019) Algorithm 3 の proxy-Lagrangian EO。"""

from collections.abc import Sequence

import torch

from .conditional import ConditionalObjective, _positive_step
from .input import ObjectiveInput


def smooth_step(x: torch.Tensor) -> torch.Tensor:
    """コンパクトな C1 smoothstep を返す。

    Args:
        x: 任意形状の Tensor。

    Returns:
        Tensor: [0,1] の値。
    """
    t = x.clamp(0, 1)
    return t.square() * (3 - 2 * t)


PROXIES = ("hinge", "smoothstep")


def proxy_bounds(score: torch.Tensor, tau: float, proxy: str) -> tuple[torch.Tensor, torch.Tensor]:
    """指示関数 1(s>0) の上界 U と下界 D を返す。

    hinge は TFCO の既定 ``HingeLoss``（margin 1）と同じ形で、``U=(1+s/tau)_+``、
    ``D=1-(1-s/tau)_+``。負の係数がかかる rate を、補集合の hinge で評価したものが D になる。
    U は閾値の正側でも勾配を持つので、陽性と予測した標本を閾値の下へ押し戻せる。

    smoothstep は 2026-10-13 より前の run が使った自前の有界 ramp で、既存 run の再現のためにだけ残す。
    U は s>=0 で 1、D は s<=0 で 0 に張り付くため、予測を閾値の反対側へ動かす勾配を持たない。

    Args:
        score: 二値 logit 差 s。
        tau: proxy の幅。
        proxy: ``"hinge"`` または ``"smoothstep"``。

    Returns:
        tuple[Tensor, Tensor]: score と同じ形の (U, D)。全 s で D <= 1(s>0) <= U。

    Raises:
        ValueError: proxy が未知のとき。
    """
    if proxy == "hinge":
        return (1 + score / tau).clamp_min(0), 1 - (1 - score / tau).clamp_min(0)
    if proxy == "smoothstep":
        return smooth_step(1 + score / tau), smooth_step(score / tau)
    raise ValueError(f"proxy は {PROXIES} のいずれか: {proxy!r}")


class EqualizedOddsRateTaskLoss(ConditionalObjective):
    """hard EO 制約と、上下界 proxy によるモデル損失を分離する。

    列確率行列 M は uniform 初期化。更新は outer([0,hard_g],lambda)。
    陽性、陰性の順に全有向群ペアを列挙し、固定閾値 s>0 を使う。
    モデル側の proxy は既定で TFCO と同じ hinge 上下界を使う（``proxy_bounds``）。
    """

    dual_update = "step"

    def __init__(
        self,
        num_groups: int,
        population_cell_counts: Sequence[Sequence[float]],
        group_key: str = "group_id",
        class_weight: Sequence[float] | None = None,
        dual_step_size: float = 0.01,
        epsilon_pos: float = 0.1,
        epsilon_neg: float = 0.1,
        tau: float = 1.0,
        initial_task_weight: float | None = None,
        proxy: str = "hinge",
    ) -> None:
        """制約定義と swap-regret 状態を設定する。

        Args:
            num_groups: 群数。
            population_cell_counts: train の [G,2] 件数。
            group_key: 群属性のキー。
            class_weight: task CE のクラス重み。
            dual_step_size: M の指数勾配ステップ幅。
            epsilon_pos: TPR 差の許容値。
            epsilon_neg: FPR 差の許容値。
            tau: proxy の幅。
            initial_task_weight: lambda の初期 task 重み。None なら M を一様に置き、全座標が 1/(m+1)。
                未学習から始める 1 stage では、一様だと proxy が損失の大半を占め、全予測を片側へ
                寄せる自明解（制約は満たされ双対の圧力も消える）に落ちるので、task 側へ寄せて始める。
            proxy: モデル側の上下界。``"hinge"``（既定、TFCO と同じ）か、既存 run 再現用の ``"smoothstep"``。

        Returns:
            None
        """
        super().__init__(num_groups, population_cell_counts, group_key, class_weight)
        self.dual_step_size = _positive_step(dual_step_size, "dual_step_size")
        self.tau = _positive_step(tau, "tau")
        if proxy not in PROXIES:
            raise ValueError(f"proxy は {PROXIES} のいずれか: {proxy!r}")
        self.proxy = proxy
        if not all(0 <= x <= 1 for x in (epsilon_pos, epsilon_neg)):
            raise ValueError("epsilon は [0,1] が必要")
        self.constraints = [(c, g, h) for c in (1, 0) for g in range(num_groups) for h in range(num_groups) if g != h]
        self.register_buffer("epsilon", torch.tensor([epsilon_pos if c else epsilon_neg for c, _, _ in self.constraints]))
        dimension = len(self.constraints) + 1
        if initial_task_weight is None:
            initial = torch.full((dimension,), 1 / dimension, dtype=torch.float64)
        else:
            if not 0 < initial_task_weight < 1:
                raise ValueError("initial_task_weight は (0,1) が必要")
            initial = torch.full((dimension,), (1 - initial_task_weight) / (dimension - 1), dtype=torch.float64)
            initial[0] = initial_task_weight
        # 全列を同じ分布にすると、その分布が M の定常分布（初期 lambda）になる。
        self.register_buffer("log_M", initial.log()[:, None].expand(dimension, dimension).clone())

    def _stationary(self) -> tuple[torch.Tensor, torch.Tensor]:
        matrix = self.log_M.exp()
        distribution = torch.full((len(matrix),), 1 / len(matrix), device=matrix.device, dtype=torch.float64)
        for _ in range(1000):
            distribution = matrix @ distribution
            distribution = distribution / distribution.sum()
            residual = (matrix @ distribution - distribution).abs().sum()
            if residual <= 1e-8:
                return distribution, residual
        raise RuntimeError("EO stationary distribution が収束しない")

    def forward(self, inputs: ObjectiveInput) -> torch.Tensor:
        """旧 lambda でモデル損失を計算し、hard 制約を保留する。

        Args:
            inputs: 二値 logits と群・ラベル。

        Returns:
            Tensor: task と制約 proxy の scalar 和。
        """
        groups, cells = self._cells(inputs)
        score = inputs.logits[:, 1].float() - inputs.logits[:, 0].float()
        hard = torch.stack([torch.stack([(score[m] > 0).float().mean() for m in row]) for row in cells])
        upper, lower = proxy_bounds(score, self.tau, self.proxy)
        violations = torch.stack([hard[g, c] - hard[h, c] for c, g, h in self.constraints]) - self.epsilon
        proxy = torch.stack([upper[cells[g][c]].mean() - lower[cells[h][c]].mean() for c, g, h in self.constraints]) - self.epsilon
        with torch.no_grad():
            distribution, residual = self._stationary()
        weights = distribution.float().clone()
        task = self._task(inputs, groups)
        if self.training:
            self._pending = (violations.detach().double(), distribution.clone())
            self._diagnostics = {
                "lambda": weights,
                "hard_g": violations.detach(),
                "proxy_g": proxy.detach(),
                "rates": hard.detach(),
                "stationary_residual": residual,
                "task": task.detach(),
                "cell_count": torch.stack([m.sum() for row in cells for m in row]),
                "active_fraction": ((score.abs() < self.tau) & (score != 0)).float().mean().detach(),
            }
        return weights[0] * task + torch.dot(weights[1:], proxy)

    @torch.no_grad()
    def commit_update(self) -> None:
        """モデル更新後に hard 制約を使い列ごとの指数勾配を進める。

        Args:
            なし

        Returns:
            None
        """
        if self._pending is not None:
            violations, distribution = self._pending
            values = torch.cat([violations.new_zeros(1), violations])
            self.log_M.copy_(torch.log_softmax(self.log_M + self.dual_step_size * torch.outer(values, distribution), dim=0))
            self.update_count.add_(1)
            self._pending = None
