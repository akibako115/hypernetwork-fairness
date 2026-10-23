"""Cotter et al. (JMLR 2019) Algorithm 3 の proxy-Lagrangian EO。"""

from collections.abc import Sequence

import lightning as L
import torch

from .conditional import ConditionalObjective, StepDualUpdate, _positive_step
from .input import ObjectiveInput

CONSTRAINT_FORMS = ("pairwise", "overall")
# 率の名前と、その率を測るクラス（TPR は陽性、FPR は陰性）。並びが制約の並び。
RATE_CLASSES = {"tpr": 1, "fpr": 0}


def proxy_bounds(score: torch.Tensor, tau: float) -> tuple[torch.Tensor, torch.Tensor]:
    """指示関数 1(s>0) の hinge 上界 U と下界 D を返す。

    TFCO の既定 ``HingeLoss``（margin 1）と同じ形で、``U=(1+s/tau)_+``、``D=1-(1-s/tau)_+``。
    負の係数がかかる rate を、補集合の hinge で評価したものが D になる。
    U は閾値の正側でも勾配を持つので、陽性と予測した標本を閾値の下へ押し戻せる。

    Args:
        score: 二値 logit 差 s。
        tau: proxy の幅。

    Returns:
        tuple[Tensor, Tensor]: score と同じ形の (U, D)。全 s で D <= 1(s>0) <= U。
    """
    return (1 + score / tau).clamp_min(0), 1 - (1 - score / tau).clamp_min(0)


class EqualizedOddsRateTaskLoss(ConditionalObjective):
    """hard EO 制約と、上下界 proxy によるモデル損失を分離する。

    列確率行列 M は uniform 初期化。更新は outer([0,hard_g],lambda)。固定閾値 s>0 を使う。
    制約は陽性（TPR）、陰性（FPR）の順に並べる。どちらの率にかけるかは ``rates``、形は ``constraint_form`` で選ぶ。

    - ``"pairwise"``：全有向群ペアで ``rate_g - rate_h <= epsilon``。最大 − 最小 <= epsilon と同値で、
      評価指標（Eopp1・Eopp0）に直接対応する。制約は 2G(G-1) 本。
    - ``"overall"``：Cotter et al. の実験（COMPAS・Communities）にならい、各群を全体と比べる片側の制約。
      ``FPR_g - FPR_all <= epsilon_neg``（FPR が全体より高すぎない）は Communities と同じ向き。
      ``TPR_all - TPR_g <= epsilon_pos``（TPR が全体より低すぎない）は COMPAS の
      ``TPR_g <= TPR_all + 0.05`` と**逆向き**にしてある。COMPAS は再犯予測で TPR が高すぎる群を抑えたが、
      ここでは陽性を見落とす（TPR が低い）群を抑えたいため。全体の率は train の群構成で重み付けした
      群の率の和で、セル均等のバッチの単純平均ではない。制約は 2G 本。

    モデル側の proxy は TFCO と同じ hinge 上下界（``proxy_bounds``）。率に正の係数がかかる項は上界、
    負の係数がかかる項は下界で置き換える。
    """

    uses_held_out = True

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
        constraint_form: str = "pairwise",
        rates: Sequence[str] = ("tpr", "fpr"),
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
            constraint_form: 制約の形。``"pairwise"``（既定）か ``"overall"``（class docstring）。
            rates: 制約をかける率。既定は EO の ``("tpr", "fpr")``。Cotter et al. の実験は 1 種類だけにかける
                （COMPAS は TPR、Communities は FPR）。

        Returns:
            None
        """
        super().__init__(num_groups, population_cell_counts, group_key, class_weight)
        self.dual_step_size = _positive_step(dual_step_size, "dual_step_size")
        self.tau = _positive_step(tau, "tau")
        if not all(0 <= x <= 1 for x in (epsilon_pos, epsilon_neg)):
            raise ValueError("epsilon は [0,1] が必要")
        if constraint_form not in CONSTRAINT_FORMS:
            raise ValueError(f"constraint_form は {CONSTRAINT_FORMS} のいずれか: {constraint_form!r}")
        self.constraint_form = constraint_form
        if not rates or len(set(rates)) != len(rates) or not set(rates) <= set(RATE_CLASSES):
            raise ValueError(f"rates は {tuple(RATE_CLASSES)} から重複なく 1 つ以上: {rates!r}")
        classes = [c for name, c in RATE_CLASSES.items() if name in rates]
        if constraint_form == "pairwise":
            self.constraints = [(c, g, h) for c in classes for g in range(num_groups) for h in range(num_groups) if g != h]
        else:
            self.constraints = [(c, g, None) for c in classes for g in range(num_groups)]
            # pairwise の state_dict を変えないよう、overall のときだけ持つ（既存 checkpoint を strict に読める）。
            self.register_buffer("overall_coefficients", self._overall_coefficients())
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

    def _overall_coefficients(self) -> torch.Tensor:
        # 制約 k の値は sum_g A[k,g] rate[g,c_k]。全体の率は train の群構成 w_c で重み付けした和なので、
        # TPR は w_1 - e_g（全体 − 群）、FPR は e_g - w_0（群 − 全体）。
        share = self.population / self.population.sum(dim=0, keepdim=True)
        rows = []
        for c, g, _ in self.constraints:
            coefficient = share[:, c].clone()
            coefficient[g] -= 1
            rows.append(coefficient if c == 1 else -coefficient)
        return torch.stack(rows)

    def _versus_overall(self, hard: torch.Tensor, upper: torch.Tensor, lower: torch.Tensor, cells: list[list[torch.Tensor]]) -> tuple[torch.Tensor, torch.Tensor]:
        classes = torch.tensor([c for c, _, _ in self.constraints], device=hard.device)
        upper_mean = torch.stack([torch.stack([upper[m].mean() for m in row]) for row in cells])
        lower_mean = torch.stack([torch.stack([lower[m].mean() for m in row]) for row in cells])
        coefficients = self.overall_coefficients
        violations = (coefficients * hard[:, classes].T).sum(dim=1) - self.epsilon
        bound = coefficients.clamp_min(0) * upper_mean[:, classes].T + coefficients.clamp_max(0) * lower_mean[:, classes].T
        return violations, bound.sum(dim=1) - self.epsilon

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
        upper, lower = proxy_bounds(score, self.tau)
        if self.constraint_form == "pairwise":
            violations = torch.stack([hard[g, c] - hard[h, c] for c, g, h in self.constraints]) - self.epsilon
            proxy = torch.stack([upper[cells[g][c]].mean() - lower[cells[h][c]].mean() for c, g, h in self.constraints]) - self.epsilon
        else:
            violations, proxy = self._versus_overall(hard, upper, lower, cells)
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

    def callbacks(self) -> list[L.Callback]:
        """途中再開の拒否に、step ごとの双対の確定を足して返す。

        Args:
            なし

        Returns:
            list[Callback]: `RejectFitResume` と `StepDualUpdate`。
        """
        return [*super().callbacks(), StepDualUpdate()]

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
