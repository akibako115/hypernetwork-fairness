import pytest
import torch
import torch.nn.functional as F

from projects.hypernet_iterative.loss import (
    GroupDROTaskLoss,
    ObjectiveInput,
    TaskLoss,
    UniformGroupTaskLoss,
)


def test_task_loss_uses_objective_input() -> None:
    logits = torch.tensor([[2.0, -1.0], [-0.5, 1.5]])
    target = torch.tensor([0, 1])
    inputs = ObjectiveInput(
        logits=logits,
        target=target,
        attributes={"unused": torch.tensor([10, 20])},
    )

    loss = TaskLoss(class_weight=[1.0, 2.0])(inputs)

    expected = F.cross_entropy(
        logits,
        target,
        weight=torch.tensor([1.0, 2.0]),
    )
    assert torch.allclose(loss, expected)


@pytest.mark.parametrize("class_weight", [[], [1.0, float("nan")], [1.0, -1.0]])
def test_task_loss_rejects_invalid_class_weight(class_weight: list[float]) -> None:
    with pytest.raises(ValueError, match="class_weight"):
        TaskLoss(class_weight=class_weight)


def test_uniform_group_task_loss_averages_observed_group_losses() -> None:
    logits = torch.tensor([[3.0, 0.0], [0.0, 3.0], [2.0, 0.0]])
    target = torch.tensor([0, 0, 1])
    inputs = ObjectiveInput(logits=logits, target=target, attributes={"group_id": torch.tensor([0, 1, 1])})

    loss = UniformGroupTaskLoss(num_groups=2)(inputs)

    per_sample = F.cross_entropy(logits, target, reduction="none")
    expected = torch.stack((per_sample[0], per_sample[1:].mean())).mean()
    assert torch.allclose(loss, expected)


@pytest.mark.parametrize("objective_class", [UniformGroupTaskLoss, GroupDROTaskLoss])
def test_group_objectives_accept_and_apply_class_weight(objective_class: type[torch.nn.Module]) -> None:
    logits = torch.tensor([[3.0, 0.0], [0.0, 3.0]])
    target = torch.tensor([0, 1])
    inputs = ObjectiveInput(logits=logits, target=target, attributes={"group_id": torch.tensor([0, 1])})

    objective = objective_class(num_groups=2, class_weight=[[1.0, 2.0], [1.0, 2.0]])
    loss = objective(inputs)

    per_sample = F.cross_entropy(logits, target, weight=torch.tensor([1.0, 2.0]), reduction="none")
    if objective_class is UniformGroupTaskLoss:
        expected = per_sample.mean()
    else:
        expected_q = torch.softmax(0.01 * per_sample, dim=0)
        expected = torch.dot(expected_q, per_sample)
    assert torch.allclose(loss, expected)


def test_group_dro_updates_persistent_weights_and_ignores_absent_groups() -> None:
    objective = GroupDROTaskLoss(num_groups=3, step_size=1.0)
    logits = torch.tensor([[4.0, 0.0], [0.0, 0.0]])
    target = torch.tensor([0, 1])
    inputs = ObjectiveInput(logits=logits, target=target, attributes={"group_id": torch.tensor([0, 1])})

    loss = objective(inputs)

    assert loss.ndim == 0
    assert objective.adv_probs.sum().item() == pytest.approx(1.0)
    assert objective.adv_probs[1] > objective.adv_probs[0]
    assert objective.adv_probs[2] > 0
    assert "adv_probs" in objective.state_dict()


def test_group_dro_restores_adversarial_weights_from_checkpoint_state() -> None:
    objective = GroupDROTaskLoss(num_groups=3, step_size=1.0)
    inputs = ObjectiveInput(
        logits=torch.tensor([[4.0, 0.0], [0.0, 0.0]]),
        target=torch.tensor([0, 1]),
        attributes={"group_id": torch.tensor([0, 1])},
    )
    objective(inputs)
    checkpoint_state = objective.state_dict()

    resumed_objective = GroupDROTaskLoss(num_groups=3, step_size=1.0)
    resumed_objective.load_state_dict(checkpoint_state)

    assert torch.allclose(resumed_objective.adv_probs, objective.adv_probs)


def test_group_dro_uses_full_group_q_for_a_batch_with_missing_groups() -> None:
    objective = GroupDROTaskLoss(num_groups=3, step_size=1.0)
    objective.adv_probs.copy_(torch.tensor([0.9, 0.1, 0.0]))
    logits = torch.tensor([[0.0, 0.0]])
    target = torch.tensor([1])
    inputs = ObjectiveInput(logits=logits, target=target, attributes={"group_id": torch.tensor([1])})

    loss = objective(inputs)

    group_loss = torch.tensor(torch.log(torch.tensor(2.0)))
    expected_q = torch.tensor([0.9, 0.1 * torch.exp(group_loss), 0.0])
    expected_q = expected_q / expected_q.sum()
    assert torch.allclose(objective.adv_probs, expected_q)
    assert torch.allclose(loss, expected_q[1] * group_loss)


def test_group_losses_reject_mismatched_group_id_shape() -> None:
    """group ID の範囲は CohortImageDataModule がロード時に保証するため、ここでは形状だけ見る。"""
    inputs = ObjectiveInput(
        logits=torch.randn(2, 2),
        target=torch.tensor([0, 1]),
        attributes={"group_id": torch.tensor([[0], [1]])},
    )
    with pytest.raises(ValueError, match="shape"):
        GroupDROTaskLoss(num_groups=2)(inputs)


def _imbalanced_cohort_batch() -> ObjectiveInput:
    """クラス別のlogitsは同一で陽性率だけが違う2 groupのbatchを作る。

    group 0 は 2陰性 + 2陽性（陽性率 50%）、group 1 は 5陰性 + 1陽性（陽性率 17%）。
    クラス平均を取れば両者の group loss は厳密に一致する。
    """
    negative_logits = [3.0, 0.0]
    positive_logits = [0.0, 1.0]
    logits = torch.tensor([negative_logits] * 2 + [positive_logits] * 2 + [negative_logits] * 5 + [positive_logits])
    target = torch.tensor([0, 0, 1, 1, 0, 0, 0, 0, 0, 1])
    group_id = torch.tensor([0, 0, 0, 0, 1, 1, 1, 1, 1, 1])
    return ObjectiveInput(logits=logits, target=target, attributes={"group_id": group_id})


def _balanced_group_class_weight(inputs: ObjectiveInput, *, num_groups: int, num_classes: int) -> list[list[float]]:
    """batch の構成から w[g, c] = 1 / (C * f_{g,c}) を作る。"""
    group_id = inputs.attributes["group_id"]
    weights = []
    for group in range(num_groups):
        in_group = group_id == group
        weights.append([float(in_group.sum()) / (num_classes * float((in_group & (inputs.target == c)).sum())) for c in range(num_classes)])
    return weights


def test_group_class_weight_makes_the_group_loss_invariant_to_group_prevalence() -> None:
    """陽性率だけが違う2 groupが同じ group loss になる。class weight 無しとの差はここに出る。"""
    inputs = _imbalanced_cohort_batch()
    weight = _balanced_group_class_weight(inputs, num_groups=2, num_classes=2)
    objective = GroupDROTaskLoss(num_groups=2, class_weight=weight, step_size=1.0)

    objective(inputs)

    # group loss が一致するので、adversarial weight は一様のまま動かない。
    assert torch.allclose(objective.adv_probs, torch.full((2,), 0.5))


def test_group_class_weight_reproduces_the_within_group_class_mean() -> None:
    """w[g, c] = 1 / (C * f_{g,c}) の group loss が群内クラス平均に一致する。"""
    inputs = _imbalanced_cohort_batch()
    weight = _balanced_group_class_weight(inputs, num_groups=2, num_classes=2)
    objective = GroupDROTaskLoss(num_groups=2, class_weight=weight, step_size=1e-12)

    loss = objective(inputs)

    per_sample = F.cross_entropy(inputs.logits, inputs.target, reduction="none")
    group_id = inputs.attributes["group_id"]
    class_means = torch.stack([torch.stack([per_sample[(group_id == group) & (inputs.target == c)].mean() for c in range(2)]).mean() for group in range(2)])
    assert torch.allclose(loss, torch.dot(objective.adv_probs, class_means))


def test_an_unweighted_group_loss_keeps_the_prevalence_dependence() -> None:
    """重み無しでは陽性率の差が group loss に残り、adversarial weight が動く。"""
    inputs = _imbalanced_cohort_batch()
    objective = GroupDROTaskLoss(num_groups=2, step_size=1.0)

    objective(inputs)

    assert not torch.allclose(objective.adv_probs, torch.full((2,), 0.5))


@pytest.mark.parametrize("objective_class", [UniformGroupTaskLoss, GroupDROTaskLoss])
def test_group_objectives_reject_a_class_weight_shared_by_every_group(objective_class: type[torch.nn.Module]) -> None:
    """全 group 共通の重みでは陽性率依存が残る。掛けるなら group ごと、を型で固定する。"""
    with pytest.raises(ValueError, match=r"\[num_groups, num_classes\]"):
        objective_class(num_groups=2, class_weight=[1.0, 2.0])


def test_uniform_group_applies_the_group_class_weight() -> None:
    inputs = _imbalanced_cohort_batch()
    weight = _balanced_group_class_weight(inputs, num_groups=2, num_classes=2)

    loss = UniformGroupTaskLoss(num_groups=2, class_weight=weight)(inputs)

    per_sample = F.cross_entropy(inputs.logits, inputs.target, reduction="none")
    group_id = inputs.attributes["group_id"]
    class_means = torch.stack([torch.stack([per_sample[(group_id == group) & (inputs.target == c)].mean() for c in range(2)]).mean() for group in range(2)])
    assert torch.allclose(loss, class_means.mean())


@pytest.mark.parametrize("objective_class", [UniformGroupTaskLoss, GroupDROTaskLoss])
@pytest.mark.parametrize(
    ("class_weight", "message"),
    [
        ([[1.0, 2.0]], "3 行"),
        ([[1.0, 2.0], [1.0, 2.0], [1.0]], "同じ長さ"),
        ([[1.0, 2.0], [1.0, 2.0], [1.0, -2.0]], "positive"),
        ([[1.0, 2.0], [1.0, 2.0], [1.0, float("inf")]], "finite"),
    ],
)
def test_group_objectives_reject_an_invalid_group_class_weight(
    objective_class: type[torch.nn.Module],
    class_weight: list[list[float]],
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        objective_class(num_groups=3, class_weight=class_weight)


def test_group_dro_backpropagates_into_logits_through_the_group_class_weight() -> None:
    inputs = _imbalanced_cohort_batch()
    logits = inputs.logits.clone().requires_grad_(True)
    weight = _balanced_group_class_weight(inputs, num_groups=2, num_classes=2)

    GroupDROTaskLoss(num_groups=2, class_weight=weight, step_size=1.0)(ObjectiveInput(logits=logits, target=inputs.target, attributes=inputs.attributes)).backward()

    assert logits.grad is not None
    assert torch.isfinite(logits.grad).all()
