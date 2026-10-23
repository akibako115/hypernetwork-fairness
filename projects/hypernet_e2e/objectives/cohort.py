"""LHCF の比較で使う群つき目的関数（最悪コホート損失・FIS・FaMI）。

いずれも群の id を `attributes[group_key]` から読み、群が属性でもコホート（画像埋め込みのクラスタ）でも同じに動く。
- `WorstCohortTaskLoss`：Masroor et al. (MICCAI 2026) Eq. 3 の ℓ_wst。CE + λ·max_k R_k
- `FISTaskLoss`：Luo et al. (FairVision) Eq. 3–5 の Fair Identity Scaling
- `FaMITaskLoss`：Sadri et al. (MICCAI 2025) Eq. 1–3 の相互情報量の勾配罰則。LHCF の文献 [34] は FairDomain を指すが、名前と説明はこの論文のもの
"""

from collections.abc import Callable, Iterator, Sequence
from typing import Any

import lightning as L
import torch
import torch.nn.functional as F

from .base import Objective
from .group import _group_losses
from .input import ObjectiveInput
from .task import _class_weight_tensor


def _weighted_ce(logits: torch.Tensor, target: torch.Tensor, weight: torch.Tensor | None) -> torch.Tensor:
    """global class weight を掛けた標本ごとの CE。"""
    losses = F.cross_entropy(logits.float(), target, reduction="none")
    return losses if weight is None else losses * weight[target.long()]


class _DiagnosticsMixin:
    """直近の train forward の診断値を `batch_diagnostics()` で渡す。

    `ObjectiveDiagnosticsRecorder` が epoch ごとに `metrics/objective_diagnostics.jsonl` へ書く。epoch 平均を取るので、
    値はどの batch でも同じ shape にする（群の次元は常に num_groups）。
    """

    _batch_diagnostics: dict[str, torch.Tensor]

    def batch_diagnostics(self) -> dict[str, torch.Tensor]:
        """直近の train forward の診断値を返す。再計算・状態更新はしない。

        Args:
            なし

        Returns:
            dict[str, Tensor]: detached Tensor の辞書。forward 前は空。
        """
        return self._batch_diagnostics


class WorstCohortTaskLoss(_DiagnosticsMixin, Objective):
    """CE に、バッチ内で平均損失が最大の群の損失を λ 倍して足す（LHCF の ℓ_wst）。

    診断値（いずれも `[num_groups]`）：`group_loss` は群の平均損失（バッチに居ない群は 0）、`group_present` は
    バッチに居たか（0/1）、`worst_group` は最悪の群の one-hot。epoch 平均で割ると、居た batch での平均損失と、
    各群が最悪になった割合が読める。
    """

    def __init__(self, num_groups: int, fairness_weight: float = 1.0, class_weight: Sequence[float] | None = None, group_key: str = "group_id"):
        """群数と λ を設定する。

        Args:
            num_groups: 群（コホート）の総数。
            fairness_weight: λ。LHCF の本文に値がないので既定は 1。
            class_weight: 全群共通の `[num_classes]` 重み。None なら重み付けしない。
            group_key: 群 id のキー。

        Returns:
            None
        """
        super().__init__()
        if num_groups < 2 or fairness_weight < 0:
            raise ValueError(f"num_groups >= 2 と fairness_weight >= 0 が必要: {num_groups}, {fairness_weight}")
        self.register_buffer("_class_weight", _class_weight_tensor(class_weight), persistent=False)
        self.num_groups = num_groups
        self.fairness_weight = float(fairness_weight)
        self.group_key = group_key
        self._batch_diagnostics = {}

    def forward(self, inputs: ObjectiveInput) -> torch.Tensor:
        group_ids = inputs.attributes[self.group_key].long()
        losses = _weighted_ce(inputs.logits, inputs.target, self._class_weight)
        group_losses, observed = _group_losses(losses, group_ids, num_groups=self.num_groups, group_key=self.group_key)
        worst = group_losses[observed].max()
        with torch.no_grad():
            worst_group = torch.where(observed, group_losses, -1.0).argmax()
            self._batch_diagnostics = {
                "group_loss": torch.where(observed, group_losses, 0.0).detach(),
                "group_present": observed.float(),
                "worst_group": F.one_hot(worst_group, self.num_groups).float(),
            }
        return losses.mean() + self.fairness_weight * worst


def _entropic_ot(x: torch.Tensor, y: torch.Tensor, eps: float, iterations: int) -> torch.Tensor:
    """一様重みの 1 次元点群どうしの、コスト |x − y|²/2 のエントロピー正則化 OT（log 領域 Sinkhorn の双対値）。"""
    cost = (x[:, None] - y[None, :]) ** 2 / 2
    log_a = torch.full_like(x, -torch.log(torch.tensor(float(len(x)))))
    log_b = torch.full_like(y, -torch.log(torch.tensor(float(len(y)))))
    f, g = torch.zeros_like(x), torch.zeros_like(y)
    # ε を大きい値から目標値へ下げて収束を速める（geomloss の ε-scaling と同じ考え方）
    start = max(float(cost.max()), eps)
    schedule = torch.logspace(torch.log10(torch.tensor(start)), torch.log10(torch.tensor(eps)), iterations).tolist()
    for e in schedule:
        f = -e * torch.logsumexp(log_b[None, :] + (g[None, :] - cost) / e, dim=1)
        g = -e * torch.logsumexp(log_a[:, None] + (f[:, None] - cost) / e, dim=0)
    return f.mean() + g.mean()


def sinkhorn_divergence(x: torch.Tensor, y: torch.Tensor, blur: float = 0.05, iterations: int = 50) -> torch.Tensor:
    """偏りを除いた Sinkhorn divergence（geomloss の `SamplesLoss("sinkhorn", p=2, blur)` と同じ定義）。

    Args:
        x: `[n]` の値。
        y: `[m]` の値。
        blur: ε = blur²。
        iterations: ε-scaling の段数。

    Returns:
        Tensor: 非負のスカラー。
    """
    eps = blur**2
    return _entropic_ot(x, y, eps, iterations) - (_entropic_ot(x, x, eps, iterations) + _entropic_ot(y, y, eps, iterations)) / 2


class FISTaskLoss(_DiagnosticsMixin, Objective):
    """Fair Identity Scaling：標本の重み s^I と群の重み s^G を c で混ぜて CE に掛ける（Luo et al. Eq. 3–5）。

    論文の式に従い、論文が書いていない所は公式実装（FairVision `src/modules.py` の `Fair_Loss_Scaler`、
    `scripts/train_*_fis.py`）に従う。

    - s^I_i：バッチ B^t の損失の softmax（Eq. 4）
    - s^G_a：B^t の損失分布と、群 a から別に引いたミニバッチ B^t_a の損失分布との OT を、群について取った softmax（Eq. 5）。
      標本 i には s^G_{a_i} を掛ける
    - 損失：(1/|B|) Σ_i ((1 − c) s^I_i + c s^G_{a_i}) ℓ_i（Eq. 3）

    論文と公式実装の食い違いは論文に合わせた。
    - OT の相手：論文は B^t と B^t_a。公式実装は B^t_a と全群の B^t_a の連結で、B^t を使わない。
    - s^G の softmax：論文は群について（和が 1）。公式実装はバッチの標本について。
      論文の形では s^G_{a_i} が 1/|A| 前後、s^I_i が 1/|B| 前後なので、c = 0.5 でも群の項が大きくなる。
      2026-10-13 から 10-15 までは公式実装の正規化にしていた。
    論文に記載がなく公式実装に従ったもの：OT は Sinkhorn divergence（geomloss の `SamplesLoss("sinkhorn", p=2, blur=0.1)`）、
    B^t_a は群あたり 64 件、温度と `right_asymptote` は 1、s^I には勾配を流し、s^G には流さない（公式実装の s^G は勾配なしの B^t_a だけから作られる）。

    診断値：`group_weight`・`group_distance`（`[num_groups]`、s^G と OT 距離）と、`group_term_share`（スカラー、
    標本の重みの総和のうち群の項 c·s^G が占める割合。c が実際にどれだけ群の項を効かせたかを読む）。

    B^t_a は `set_group_sample_source` で渡された関数から forward のたびに 1 つ受け取る。学習中は `GroupSampleSource`
    が DataModule の群ごとのバッチから用意する。
    """

    def __init__(
        self,
        num_groups: int,
        fusion: float = 0.5,
        blur: float = 0.1,
        group_sample_size: int = 64,
        class_weight: Sequence[float] | None = None,
        group_key: str = "group_id",
    ):
        """混合比 c と Sinkhorn・B^t_a の設定を保持する。

        Args:
            num_groups: 群の総数 |A|。
            fusion: c。論文の既定 0.5。
            blur: Sinkhorn の blur（ε = blur²）。公式実装の既定 0.1。
            group_sample_size: B^t_a の件数。公式実装の既定 64。
            class_weight: 全群共通の `[num_classes]` 重み。
            group_key: 群 id のキー。

        Returns:
            None
        """
        super().__init__()
        if num_groups < 2 or not 0 <= fusion <= 1 or group_sample_size < 1:
            raise ValueError(f"num_groups >= 2、0 <= fusion <= 1、group_sample_size >= 1 が必要: {num_groups}, {fusion}, {group_sample_size}")
        self.register_buffer("_class_weight", _class_weight_tensor(class_weight), persistent=False)
        self.num_groups = num_groups
        self.fusion = float(fusion)
        self.blur = float(blur)
        self.group_sample_size = int(group_sample_size)
        self.group_key = group_key
        self._batch_diagnostics = {}
        self._group_sample_source: Callable[[], ObjectiveInput] | None = None

    def set_group_sample_source(self, source: Callable[[], ObjectiveInput] | None) -> None:
        """forward のたびに B^t_a（群ごとに別に引いたバッチの、勾配なしの出力）を返す関数を設定する。

        Args:
            source: 引数なしで次の B^t_a を返す関数。None で外す。

        Returns:
            None
        """
        self._group_sample_source = source

    def callbacks(self) -> list[L.Callback]:
        """学習中に B^t_a を用意する callback を返す。

        Args:
            なし

        Returns:
            list[Callback]: `GroupSampleSource` だけのリスト。
        """
        return [GroupSampleSource()]

    def forward(self, inputs: ObjectiveInput) -> torch.Tensor:
        if self._group_sample_source is None:
            raise ValueError("FIS には群ごとに別に引いたバッチ（set_group_sample_source）が必要")
        group_ids = inputs.attributes[self.group_key].long()
        losses = _weighted_ce(inputs.logits, inputs.target, self._class_weight)
        samples = self._group_sample_source()
        sample_groups = samples.attributes[self.group_key].long()
        with torch.no_grad():
            sample_losses = _weighted_ce(samples.logits, samples.target, self._class_weight)
            batch = losses.detach()
            distance = torch.stack([sinkhorn_divergence(batch, sample_losses[sample_groups == g], self.blur) for g in range(self.num_groups)])
            group_weight = torch.softmax(distance, dim=0)
        individual = torch.softmax(losses, dim=0)
        group_term = self.fusion * group_weight[group_ids]
        weight = (1 - self.fusion) * individual + group_term
        self._batch_diagnostics = {
            "group_weight": group_weight,
            "group_distance": distance,
            "group_term_share": (group_term.sum() / weight.sum()).detach(),
        }
        return (weight * losses).mean()


class GroupSampleSource(L.Callback):
    """FIS の B^t_a を、DataModule の群ごとのバッチから用意して目的関数に渡す。

    バッチの流れは学習の開始時に `group_sample_batches(group_sample_size)` から作り、目的関数には「次の B^t_a を推論して
    返す関数」を渡す。推論は目的関数の forward の中、つまり学習バッチの forward の後に起きる。FIS の公式実装
    （FairVision `train_*_fis.py`）と同じく、model は train モードのまま no_grad で forward するので、BatchNorm の
    running 統計は学習バッチ、B^t_a の順に動く。
    """

    def __init__(self) -> None:
        """バッチの流れを持たない状態で作る。

        Args:
            なし

        Returns:
            None
        """
        self._batches: Iterator[Any] | None = None

    def on_train_start(self, trainer: L.Trainer, pl_module: L.LightningModule) -> None:
        """群ごとのバッチの流れを作り、推論する関数を目的関数に渡す。

        Args:
            trainer: Trainer。
            pl_module: 学習 module。

        Returns:
            None
        """
        objective = pl_module.training_objective
        self._batches = trainer.datamodule.group_sample_batches(objective.group_sample_size)

        @torch.no_grad()
        def draw() -> ObjectiveInput:
            image, attributes, target = pl_module.transfer_batch_to_device(next(self._batches), pl_module.device, 0)
            return ObjectiveInput(logits=pl_module(image, attributes).float(), target=target.long(), attributes=attributes)

        objective.set_group_sample_source(draw)

    def on_train_end(self, trainer: L.Trainer, pl_module: L.LightningModule) -> None:
        """学習が終わったら、module を参照する関数を目的関数から外す。

        Args:
            trainer: Trainer。
            pl_module: 学習 module。

        Returns:
            None
        """
        pl_module.training_objective.set_group_sample_source(None)
        self._batches = None


def _polynomial_basis(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """各次元の 0〜3 次の単項式（交差項なし）と、その各次元についての微分。

    Babaie-Zadeh & Jutten (2005) 式 (29)–(32) の基底を d 次元に広げたもの。

    Args:
        x: `[n, d]`。

    Returns:
        tuple: 基底 `[n, L]` と微分 `[d, n, L]`（L = 1 + 3d）。
    """
    n, d = x.shape
    ones, zeros = torch.ones(n, 1, dtype=x.dtype, device=x.device), torch.zeros(n, 1, dtype=x.dtype, device=x.device)
    basis = torch.cat([ones] + [torch.stack([x[:, j], x[:, j] ** 2, x[:, j] ** 3], 1) for j in range(d)], 1)
    derivatives = []
    for k in range(d):
        parts = [zeros]
        for j in range(d):
            parts.append(torch.stack([torch.ones_like(x[:, j]), 2 * x[:, j], 3 * x[:, j] ** 2], 1) if j == k else torch.zeros(n, 3, dtype=x.dtype, device=x.device))
        derivatives.append(torch.cat(parts, 1))
    return basis, torch.stack(derivatives)


def polynomial_log_density_gradient(x: torch.Tensor) -> torch.Tensor:
    """標本 x の分布の ∇_x ln p(x) を、多項式の最小二乗スコア推定で x 自身の点で評価する。

    Babaie-Zadeh & Jutten (2005) IV-A-2・IV-B-1：スコア φ(x) = −∇ ln p(x) を基底の線形結合 k(x)ᵀw で近似し、
    E{k kᵀ} w = E{∂k/∂x_d}（式 (28)）を標本平均で解く。正則化は論文どおり入れない。係数行列が退化したときに
    解が定まるよう、逆行列ではなく擬似逆行列を使う。

    Args:
        x: `[n, d]` の標本。

    Returns:
        Tensor: `[n, d]`。
    """
    x = x.double()
    basis, derivatives = _polynomial_basis(x)
    gram = basis.T @ basis / len(x)
    weight = torch.linalg.pinv(gram) @ derivatives.mean(1).T
    return -(basis @ weight)


class FaMITaskLoss(_DiagnosticsMixin, Objective):
    """CE + μ·I(u; v)。I の勾配を score function difference で与える（Sadri et al. MICCAI 2025 Eq. 1–3、Alg. 1）。

    v はモデルの出力（logits、`[B, num_classes]`）、u は群。λ(u, v) = ∇_v ln p(u|v) = ∇_v ln p(v|u) − ∇_v ln p(v) を、
    論文が引く多項式のスコア推定（Babaie-Zadeh & Jutten 2005、3 次・交差項なし・正則化なし）で、
    バッチ内の群ごとの v と、バッチ全体の v からそれぞれ推定する。`λ.detach() · v` の平均を罰則に足すと、
    その勾配が Alg. 1 の (1/|b|) Σ_i J_vᵢᵀ λ(u_i, v_i) になる。

    公式実装はない。u が離散なので p(u|v) のスコアを群ごとの条件付き密度と周辺密度の差で求める点、
    基底の数（1 + 3·num_classes）より件数の少ない群の λ を 0 にする点は、論文に記載がなくこちらで決めた。

    診断値：`penalty`（スカラー、μ を掛ける前の罰則 mean(λ·v)）、`lambda_norm`（`[num_groups]`、群ごとの ‖λ‖ の平均。
    バッチに居ないか件数不足で λ を 0 にした群は 0）、`lambda_active`（`[num_groups]`、λ を推定できた群なら 1）。
    依存が無くても ‖λ‖ は推定の揺らぎで 0 にならないので、件数の少ない設定では大きさそのものより群の間の差で読む。
    """

    def __init__(self, num_groups: int, mi_weight: float = 0.5, class_weight: Sequence[float] | None = None, group_key: str = "group_id"):
        """μ と群の設定を保持する。

        Args:
            num_groups: 群の総数。
            mi_weight: μ。論文の 0.5。
            class_weight: 全群共通の `[num_classes]` 重み。
            group_key: 群 id のキー。

        Returns:
            None
        """
        super().__init__()
        if num_groups < 2 or mi_weight < 0:
            raise ValueError(f"num_groups >= 2 と mi_weight >= 0 が必要: {num_groups}, {mi_weight}")
        self.register_buffer("_class_weight", _class_weight_tensor(class_weight), persistent=False)
        self.num_groups = num_groups
        self.mi_weight = float(mi_weight)
        self.group_key = group_key
        self._batch_diagnostics = {}

    def score_difference(self, v: torch.Tensor, group_ids: torch.Tensor) -> torch.Tensor:
        """各標本の λ(u_i, v_i)。基底の数より件数の少ない群は 0。

        Args:
            v: `[B, d]` の logits。
            group_ids: `[B]` の群 id。

        Returns:
            Tensor: `[B, d]`。
        """
        v = v.detach()
        minimum = 1 + 3 * v.shape[1]
        marginal = polynomial_log_density_gradient(v)
        out = torch.zeros_like(marginal)
        for g in group_ids.unique():
            mask = group_ids == g
            if mask.sum() >= minimum:
                out[mask] = polynomial_log_density_gradient(v[mask]) - marginal[mask]
        return out.to(v.dtype)

    def forward(self, inputs: ObjectiveInput) -> torch.Tensor:
        group_ids = inputs.attributes[self.group_key].long()
        v = inputs.logits.float()
        task = _weighted_ce(v, inputs.target, self._class_weight).mean()
        sfd = self.score_difference(v, group_ids)
        penalty = (sfd * v).sum(1).mean()
        with torch.no_grad():
            norms = sfd.norm(dim=1)
            active = torch.zeros(self.num_groups, device=v.device)
            lambda_norm = torch.zeros(self.num_groups, device=v.device)
            for g in range(self.num_groups):
                mask = group_ids == g
                if mask.any() and bool((norms[mask] > 0).any()):
                    active[g], lambda_norm[g] = 1.0, norms[mask].mean()
            self._batch_diagnostics = {"penalty": penalty.detach(), "lambda_norm": lambda_norm, "lambda_active": active}
        return task + self.mi_weight * penalty
