"""LHCF の比較で使う群つき目的関数（最悪コホート損失・FIS・FaMI）。

いずれも群の id を `attributes[group_key]` から読み、群が属性でもコホート（画像埋め込みのクラスタ）でも同じに動く。
- `WorstCohortTaskLoss`：Masroor et al. (MICCAI 2026) Eq. 3 の ℓ_wst。CE + λ·max_k R_k
- `FISTaskLoss`：Luo et al. (FairVision) Eq. 3–5 の Fair Identity Scaling
- `FaMITaskLoss`：Sadri et al. (MICCAI 2025) Eq. 1–3 の相互情報量の勾配罰則
"""

from collections.abc import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from .group import _group_losses
from .input import ObjectiveInput
from .task import _class_weight_tensor


def _weighted_ce(logits: torch.Tensor, target: torch.Tensor, weight: torch.Tensor | None) -> torch.Tensor:
    """global class weight を掛けた標本ごとの CE。"""
    losses = F.cross_entropy(logits.float(), target, reduction="none")
    return losses if weight is None else losses * weight[target.long()]


class WorstCohortTaskLoss(nn.Module):
    """CE に、バッチ内で平均損失が最大の群の損失を λ 倍して足す（LHCF の ℓ_wst）。"""

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

    def forward(self, inputs: ObjectiveInput) -> torch.Tensor:
        group_ids = inputs.attributes[self.group_key].long()
        losses = _weighted_ce(inputs.logits, inputs.target, self._class_weight)
        group_losses, observed = _group_losses(losses, group_ids, num_groups=self.num_groups, group_key=self.group_key)
        worst = group_losses[observed].max()
        self._batch_diagnostics = {"worst_group": torch.where(observed, group_losses, -1.0).argmax().detach()}
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


class FISTaskLoss(nn.Module):
    """Fair Identity Scaling：標本の重み s^I と群の重み s^G を c で混ぜて CE に掛ける（Eq. 3–5）。

    - s^I_i：バッチ内の損失の softmax（Eq. 4）
    - s^G_i：バッチ全体の損失分布と、標本 i が属す群のバッチ内の損失分布との Sinkhorn 距離を
      標本ごとに並べ、バッチについて取った softmax（Eq. 5）

    s^I と s^G はどちらもバッチ内の和が 1 なので、c がそのまま 2 つの重みの混合比になる。
    公式実装（FairVision `src/modules.py` の `Fair_Loss_Scaler`、temperature=1・right_asymptote=1）と同じ正規化。
    2026-10-13 より前は s^G を群について softmax してから標本に配っており、和が約 B/G になって
    c=0.5 でも群の項が支配していた。

    重みは損失から作るが勾配は流さない（公式実装は重みを detach しない）。論文の B^t_a
    （群 a から別に抽出したミニバッチ）は、追加の forward を避けてバッチ内の群 a の標本で代える。
    """

    def __init__(self, num_groups: int, fusion: float = 0.5, blur: float = 0.05, class_weight: Sequence[float] | None = None, group_key: str = "group_id"):
        """混合比 c と Sinkhorn の設定を保持する。

        Args:
            num_groups: 群の総数。
            fusion: c。論文の既定 0.5。
            blur: Sinkhorn の blur（ε = blur²）。
            class_weight: 全群共通の `[num_classes]` 重み。
            group_key: 群 id のキー。

        Returns:
            None
        """
        super().__init__()
        if num_groups < 2 or not 0 <= fusion <= 1:
            raise ValueError(f"num_groups >= 2 と 0 <= fusion <= 1 が必要: {num_groups}, {fusion}")
        self.register_buffer("_class_weight", _class_weight_tensor(class_weight), persistent=False)
        self.num_groups = num_groups
        self.fusion = float(fusion)
        self.blur = float(blur)
        self.group_key = group_key

    def forward(self, inputs: ObjectiveInput) -> torch.Tensor:
        group_ids = inputs.attributes[self.group_key].long()
        losses = _weighted_ce(inputs.logits, inputs.target, self._class_weight)
        with torch.no_grad():
            detached = losses.detach()
            individual = torch.softmax(detached, dim=0)
            distance = torch.full((self.num_groups,), float("-inf"), device=losses.device)
            for g in group_ids.unique():
                distance[g] = sinkhorn_divergence(detached, detached[group_ids == g], self.blur)
            sample_group_weight = torch.softmax(distance[group_ids], dim=0)
            # 群あたりの重みの合計。バッチ内の件数の差も含む（公式実装と同じ）。
            group_mass = torch.zeros(self.num_groups, device=losses.device).index_add_(0, group_ids, sample_group_weight)
            self._batch_diagnostics = {"group_weight_mass": group_mass}
            weight = (1 - self.fusion) * individual + self.fusion * sample_group_weight
        return (weight * losses).mean()


def _kde_score(points: torch.Tensor, data: torch.Tensor, bandwidth: torch.Tensor) -> torch.Tensor:
    """Gaussian KDE の対数密度の微分 d/dv ln p̂(v) を `points` で評価する。"""
    diff = points[:, None] - data[None, :]
    log_kernel = -(diff**2) / (2 * bandwidth**2)
    weight = torch.softmax(log_kernel, dim=1)
    return -(weight * diff).sum(1) / bandwidth**2


class FaMITaskLoss(nn.Module):
    """CE + μ·I(u; v)。I の勾配を score function difference で与える（Eq. 1–3）。

    v は 2 クラスの logit 差（softmax の予測を決める 1 次元）。λ(u, v) = ∇_v ln p(v|u) − ∇_v ln p(v) を
    バッチ内の Gaussian KDE（Silverman の幅）で推定し、`λ.detach() · v` の平均を罰則に足す。
    この項の勾配が Eq. 2 の J_v^T λ になる。論文の多項式カーネル推定は、本文に式がないので KDE で代える。
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

    def score_difference(self, v: torch.Tensor, group_ids: torch.Tensor) -> torch.Tensor:
        """各標本の λ(u_i, v_i)。バッチ内に 2 件未満の群は 0。

        Args:
            v: `[B]` の logit 差。
            group_ids: `[B]` の群 id。

        Returns:
            Tensor: `[B]`。
        """
        v = v.detach()
        bandwidth = (1.06 * v.std() * len(v) ** (-0.2)).clamp_min(1e-3)
        marginal = _kde_score(v, v, bandwidth)
        out = torch.zeros_like(v)
        for g in group_ids.unique():
            mask = group_ids == g
            if mask.sum() >= 2:
                out[mask] = _kde_score(v[mask], v[mask], bandwidth) - marginal[mask]
        return out

    def forward(self, inputs: ObjectiveInput) -> torch.Tensor:
        group_ids = inputs.attributes[self.group_key].long()
        logits = inputs.logits.float()
        v = logits[:, 1] - logits[:, 0]
        task = _weighted_ce(logits, inputs.target, self._class_weight).mean()
        sfd = self.score_difference(v, group_ids)
        return task + self.mi_weight * (sfd * v).mean()
