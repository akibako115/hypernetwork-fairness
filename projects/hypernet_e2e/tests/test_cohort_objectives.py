"""LHCF の比較に使う目的関数（ℓ_wst・FIS・FaMI）と SWAD の谷の判定のテスト。"""

import pytest
import torch
import torch.nn.functional as F

from projects.hypernet_e2e.callbacks.swad import SWADCallback, _Segment
from projects.hypernet_e2e.objectives.cohort import FaMITaskLoss, FISTaskLoss, WorstCohortTaskLoss, sinkhorn_divergence
from projects.hypernet_e2e.objectives.input import ObjectiveInput


def _inputs(logits: torch.Tensor, target: list[int], groups: list[int]) -> ObjectiveInput:
    return ObjectiveInput(logits=logits, target=torch.tensor(target), attributes={"group_id": torch.tensor(groups)})


def test_worst_cohort_adds_the_largest_group_mean() -> None:
    logits = torch.tensor([[2.0, 0.0], [0.0, 2.0], [2.0, 0.0], [0.0, 0.0]])
    inputs = _inputs(logits, [0, 1, 1, 1], [0, 0, 1, 1])
    losses = F.cross_entropy(logits, inputs.target, reduction="none")
    expected = losses.mean() + 0.5 * max(losses[:2].mean(), losses[2:].mean())
    assert torch.isclose(WorstCohortTaskLoss(3, fairness_weight=0.5)(inputs), expected)


def test_worst_cohort_with_zero_weight_is_plain_cross_entropy() -> None:
    logits = torch.randn(6, 2)
    inputs = _inputs(logits, [0, 1, 0, 1, 0, 1], [0, 1, 2, 0, 1, 2])
    assert torch.isclose(WorstCohortTaskLoss(3, fairness_weight=0.0)(inputs), F.cross_entropy(logits, inputs.target))


def test_sinkhorn_divergence_is_zero_for_identical_sets_and_grows_with_shift() -> None:
    x = torch.linspace(0, 1, 20)
    assert sinkhorn_divergence(x, x).abs() < 1e-4
    near, far = sinkhorn_divergence(x, x + 0.2), sinkhorn_divergence(x, x + 1.0)
    # 1 次元の平行移動では W2²/2 = shift²/2 に近い
    assert 0 < near < far
    assert far == pytest.approx(0.5, rel=0.1)


def test_fis_matches_equation_three() -> None:
    """公式実装（Fair_Loss_Scaler）と同じく、群の重みは標本ごとに並べた距離のバッチについての softmax。"""
    torch.manual_seed(0)
    logits = torch.randn(8, 2)
    target, groups = [0, 1] * 4, [0] * 6 + [1] * 2
    inputs = _inputs(logits, target, groups)
    loss = FISTaskLoss(2, fusion=0.5)(inputs)
    losses = F.cross_entropy(logits, inputs.target, reduction="none")
    g = torch.tensor(groups)
    distance = torch.stack([sinkhorn_divergence(losses, losses[g == k]) for k in range(2)])
    weight = 0.5 * torch.softmax(losses, 0) + 0.5 * torch.softmax(distance[g], 0)
    assert torch.isclose(loss, (weight * losses).mean(), atol=1e-6)


def test_fis_individual_and_group_weights_have_the_same_total() -> None:
    """c が混合比として働くには、2 つの重みの合計がそろっている必要がある。"""
    torch.manual_seed(0)
    logits = torch.randn(64, 2)
    inputs = _inputs(logits, [0, 1] * 32, [k % 4 for k in range(64)])
    objective = FISTaskLoss(4, fusion=1.0)
    objective(inputs)
    assert torch.isclose(objective._batch_diagnostics["group_weight_mass"].sum(), torch.tensor(1.0))


def test_fis_weights_carry_no_gradient() -> None:
    logits = torch.randn(6, 2, requires_grad=True)
    inputs = _inputs(logits, [0, 1, 0, 1, 0, 1], [0, 0, 0, 1, 1, 1])
    FISTaskLoss(2)(inputs).backward()
    assert logits.grad is not None and torch.isfinite(logits.grad).all()


def test_fami_score_difference_is_zero_when_groups_are_identically_distributed() -> None:
    v = torch.tensor([-1.0, 0.0, 1.0, -1.0, 0.0, 1.0])
    sfd = FaMITaskLoss(2).score_difference(v, torch.tensor([0, 0, 0, 1, 1, 1]))
    assert torch.allclose(sfd, torch.zeros(6), atol=1e-5)


def test_fami_penalty_pulls_separated_groups_together() -> None:
    v = torch.tensor([-2.0, -1.5, -1.0, 1.0, 1.5, 2.0], requires_grad=True)
    logits = torch.stack([torch.zeros_like(v), v], 1)
    loss = FaMITaskLoss(2, mi_weight=1.0)
    groups = torch.tensor([0, 0, 0, 1, 1, 1])
    penalty = (loss.score_difference(v, groups) * v).mean()
    penalty.backward()
    # 降下方向（−勾配）が、低い群を上へ、高い群を下へ動かす
    assert (v.grad[:3] < 0).all() and (v.grad[3:] > 0).all()
    assert torch.isfinite(loss(_inputs(logits, [0, 0, 0, 1, 1, 1], groups.tolist())))


def _segments(losses: list[float]) -> list[_Segment]:
    return [_Segment(epoch, {"w": torch.tensor([float(epoch)])}, 10, loss) for epoch, loss in enumerate(losses)]


def test_swad_averages_from_the_valley_start_until_it_dies() -> None:
    swad = SWADCallback(n_converge=3, n_tolerance=6, tolerance_ratio=0.3)
    # epoch 2 で最小（以後 2 区間はそれより高い）→ epoch 2 から平均。epoch 10 以降はしきい値を超え続ける
    losses = [1.0, 0.8, 0.5, 0.55, 0.6, 0.52, 0.58, 0.6, 0.62, 0.64] + [2.0] * 8
    for segment in _segments(losses):
        swad.update(segment)
    swad.finalize()
    assert swad.converge_epoch == 2
    assert swad.threshold == pytest.approx((0.5 + 0.55 + 0.6) / 3 * 1.3)
    assert swad.dead
    assert swad.averaged_epochs == list(range(2, 10))
    assert swad.total["w"] / swad.total_steps == pytest.approx(sum(range(2, 10)) / 8)


def test_swad_adds_the_remaining_valley_at_the_end_of_training() -> None:
    swad = SWADCallback(n_converge=3, n_tolerance=6, tolerance_ratio=0.3)
    for segment in _segments([1.0, 0.5, 0.6, 0.6, 0.55, 0.6]):
        swad.update(segment)
    swad.finalize()
    assert swad.converge_epoch == 1
    assert swad.averaged_epochs == [1, 2, 3, 4, 5]
