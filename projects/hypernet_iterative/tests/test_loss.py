import pytest
import torch
import torch.nn.functional as F

from projects.hypernet_iterative.loss import (
    GlobalClassWeightedGroupDROTaskLoss,
    ObjectiveInput,
    SubgroupClassWeightedGroupDROTaskLoss,
    TaskLoss,
    UniformGroupTaskLoss,
)


def test_task_loss_uses_objective_input() -> None:
    logits = torch.tensor([[2.0, -1.0], [-0.5, 1.5]])
    target = torch.tensor([0, 1])
    inputs = ObjectiveInput(logits=logits, target=target, attributes={"unused": torch.tensor([10, 20])})

    assert torch.allclose(TaskLoss(class_weight=[1.0, 2.0])(inputs), F.cross_entropy(logits, target, weight=torch.tensor([1.0, 2.0])))


@pytest.mark.parametrize("class_weight", [[], [1.0, float("nan")], [1.0, -1.0]])
def test_task_loss_rejects_invalid_class_weight(class_weight: list[float]) -> None:
    with pytest.raises(ValueError, match="class_weight"):
        TaskLoss(class_weight=class_weight)


def _inputs() -> ObjectiveInput:
    return ObjectiveInput(
        logits=torch.tensor([[3.0, 0.0], [0.0, 3.0]]),
        target=torch.tensor([0, 1]),
        attributes={"group_id": torch.tensor([0, 1])},
    )


@pytest.mark.parametrize(
    ("objective_class", "class_weight"),
    [
        (GlobalClassWeightedGroupDROTaskLoss, [1.0, 2.0]),
        (SubgroupClassWeightedGroupDROTaskLoss, [[1.0, 2.0], [1.0, 2.0]]),
    ],
)
def test_class_weighted_group_dro_applies_its_required_weight(
    objective_class: type[torch.nn.Module],
    class_weight: list[float] | list[list[float]],
) -> None:
    inputs = _inputs()
    objective = objective_class(num_groups=2, class_weight=class_weight)
    per_sample = F.cross_entropy(inputs.logits, inputs.target, weight=torch.tensor([1.0, 2.0]), reduction="none")

    assert torch.allclose(objective(inputs), torch.dot(torch.softmax(0.01 * per_sample, dim=0), per_sample))


@pytest.mark.parametrize(
    ("objective_class", "class_weight", "message"),
    [
        (GlobalClassWeightedGroupDROTaskLoss, None, "global Group DRO"),
        (GlobalClassWeightedGroupDROTaskLoss, [[1.0, 2.0], [1.0, 2.0]], r"\[num_classes\]"),
        (SubgroupClassWeightedGroupDROTaskLoss, None, "subgroup-wise Group DRO"),
        (SubgroupClassWeightedGroupDROTaskLoss, [1.0, 2.0], r"\[num_groups, num_classes\]"),
    ],
)
def test_class_weighted_group_dro_rejects_missing_or_wrong_weight_shape(
    objective_class: type[torch.nn.Module],
    class_weight: list[float] | list[list[float]] | None,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        objective_class(num_groups=2, class_weight=class_weight)


def test_subgroup_class_weighted_group_dro_balances_prevalence() -> None:
    logits = torch.tensor([[3.0, 0.0]] * 2 + [[0.0, 1.0]] * 2 + [[3.0, 0.0]] * 5 + [[0.0, 1.0]])
    target = torch.tensor([0, 0, 1, 1, 0, 0, 0, 0, 0, 1])
    group_id = torch.tensor([0, 0, 0, 0, 1, 1, 1, 1, 1, 1])
    objective = SubgroupClassWeightedGroupDROTaskLoss(num_groups=2, class_weight=[[1.0, 1.0], [0.6, 3.0]], step_size=1.0)

    objective(ObjectiveInput(logits=logits, target=target, attributes={"group_id": group_id}))

    assert torch.allclose(objective.adv_probs, torch.full((2,), 0.5))


def test_global_class_weighted_group_dro_preserves_adversarial_state() -> None:
    objective = GlobalClassWeightedGroupDROTaskLoss(num_groups=2, class_weight=[1.0, 2.0])
    objective(_inputs())
    resumed = GlobalClassWeightedGroupDROTaskLoss(num_groups=2, class_weight=[1.0, 2.0])
    resumed.load_state_dict(objective.state_dict())

    assert torch.allclose(resumed.adv_probs, objective.adv_probs)


def test_uniform_group_task_loss_remains_optional_for_subgroup_weight() -> None:
    inputs = _inputs()
    assert torch.isfinite(UniformGroupTaskLoss(num_groups=2)(inputs))


@pytest.mark.parametrize("subgroup", [False, True])
@pytest.mark.parametrize("groups", [[0, 1], [0], [0, 1, 2]])
def test_group_dro_matches_official_update_and_gradient(subgroup: bool, groups: list[int]) -> None:
    objective = (
        SubgroupClassWeightedGroupDROTaskLoss(3, class_weight=[[1.0, 2.0], [3.0, 1.0], [2.0, 4.0]], step_size=0.5)
        if subgroup
        else GlobalClassWeightedGroupDROTaskLoss(3, class_weight=[1.0, 2.0], step_size=0.5)
    )
    expected_q = torch.full((3,), 1 / 3)
    for step in range(3):
        group_ids = torch.tensor(groups)
        target = torch.arange(len(groups)) % 2
        logits = torch.tensor([[2.0 - step, 0.0], [0.0, 1.0], [1.0, 0.0]])[: len(groups)].requires_grad_()
        reference = logits.detach().clone().requires_grad_()
        per_sample = F.cross_entropy(reference, target, reduction="none")
        weights = torch.tensor([[1.0, 2.0], [3.0, 1.0], [2.0, 4.0]])[group_ids, target] if subgroup else torch.tensor([1.0, 2.0])[target]
        per_sample = per_sample * weights
        risks = torch.stack([per_sample[group_ids == g].mean() if g in groups else reference.sum() * 0 for g in range(3)])
        expected_q = expected_q * torch.exp(0.5 * risks.detach())
        expected_q /= expected_q.sum()
        expected = expected_q @ risks
        actual = objective(ObjectiveInput(logits=logits, target=target, attributes={"group_id": group_ids}))
        assert torch.allclose(objective.adv_probs, expected_q)
        assert torch.allclose(actual, expected)
        assert torch.allclose(torch.autograd.grad(actual, logits)[0], torch.autograd.grad(expected, reference)[0])
    if 2 not in groups:
        assert objective.adv_probs[2] < 1 / 3
