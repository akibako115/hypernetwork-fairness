import pytest
import torch
import torch.nn.functional as F

from projects.hypernet_e2e.loss import (
    GlobalClassWeightedGroupDROTaskLoss,
    ObjectiveInput,
    SubgroupClassWeightedGroupDROTaskLoss,
    TaskLoss,
    UniformGroupTaskLoss,
)

_ONE_CATEGORICAL_ATTRIBUTE = {"categorical": ["sex"], "continuous": []}


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


@pytest.mark.parametrize(
    ("objective_class", "weights"),
    [
        (GlobalClassWeightedGroupDROTaskLoss, {"class_weight": [1.0, 2.0]}),
        (SubgroupClassWeightedGroupDROTaskLoss, {"class_weight": [1.0, 2.0], "q_class_weight": [[1.0, 2.0], [1.0, 2.0]]}),
    ],
)
def test_class_weighted_group_dro_applies_its_required_weight(
    objective_class: type[torch.nn.Module],
    weights: dict[str, list[float] | list[list[float]]],
) -> None:
    logits = torch.tensor([[3.0, 0.0], [0.0, 3.0]])
    target = torch.tensor([0, 1])
    inputs = ObjectiveInput(logits=logits, target=target, attributes={"group_id": torch.tensor([0, 1])})
    objective = objective_class(num_groups=2, **weights)
    loss = objective(inputs)
    per_sample = F.cross_entropy(logits, target, weight=torch.tensor([1.0, 2.0]), reduction="none")
    assert torch.allclose(loss, torch.dot(torch.softmax(0.01 * per_sample, dim=0), per_sample))


@pytest.mark.parametrize(
    ("objective_class", "weights", "message"),
    [
        (GlobalClassWeightedGroupDROTaskLoss, {"class_weight": None}, "Group DRO には"),
        (GlobalClassWeightedGroupDROTaskLoss, {"class_weight": [[1.0, 2.0], [1.0, 2.0]]}, r"\[num_classes\]"),
        (SubgroupClassWeightedGroupDROTaskLoss, {"class_weight": [1.0, 2.0], "q_class_weight": None}, "q_class_weight"),
        (SubgroupClassWeightedGroupDROTaskLoss, {"class_weight": [1.0, 2.0], "q_class_weight": [1.0, 2.0]}, r"\[num_groups, num_classes\]"),
        (SubgroupClassWeightedGroupDROTaskLoss, {"class_weight": None, "q_class_weight": [[1.0, 2.0], [1.0, 2.0]]}, "Group DRO には"),
        (SubgroupClassWeightedGroupDROTaskLoss, {"class_weight": [[1.0, 2.0], [1.0, 2.0]], "q_class_weight": [[1.0, 2.0], [1.0, 2.0]]}, r"\[num_classes\]"),
    ],
)
def test_class_weighted_group_dro_rejects_missing_or_wrong_weight_shape(
    objective_class: type[torch.nn.Module],
    weights: dict[str, list[float] | list[list[float]] | None],
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        objective_class(num_groups=2, **weights)


def test_subgroup_class_weighted_group_dro_moves_q_by_groupwise_class_balance() -> None:
    """群内の陽性・陰性の CE が群をまたいで同じなら、陽性率が違っても q は動かない。"""
    logits = torch.tensor([[3.0, 0.0]] * 2 + [[0.0, 1.0]] * 2 + [[3.0, 0.0]] * 5 + [[0.0, 1.0]])
    target = torch.tensor([0, 0, 1, 1, 0, 0, 0, 0, 0, 1])
    group_id = torch.tensor([0, 0, 0, 0, 1, 1, 1, 1, 1, 1])
    inputs = ObjectiveInput(logits=logits, target=target, attributes={"group_id": group_id})
    objective = SubgroupClassWeightedGroupDROTaskLoss(num_groups=2, class_weight=[1.0, 1.0], q_class_weight=[[1.0, 1.0], [0.6, 3.0]], step_size=1.0)
    objective(inputs)
    assert torch.allclose(objective.adv_probs, torch.full((2,), 0.5))
    # モデル更新側の群損失は全群共通の重みなので、陽性率の違いで群 0 と群 1 の値が分かれる。
    model_losses = objective.batch_diagnostics()["loss_sum"] / objective.batch_diagnostics()["count"]
    assert not torch.allclose(model_losses[0], model_losses[1])


def test_global_class_weighted_group_dro_restores_adversarial_state() -> None:
    inputs = ObjectiveInput(logits=torch.tensor([[4.0, 0.0], [0.0, 0.0]]), target=torch.tensor([0, 1]), attributes={"group_id": torch.tensor([0, 1])})
    objective = GlobalClassWeightedGroupDROTaskLoss(num_groups=3, class_weight=[1.0, 2.0], step_size=1.0)
    objective(inputs)
    resumed = GlobalClassWeightedGroupDROTaskLoss(num_groups=3, class_weight=[1.0, 2.0], step_size=1.0)
    resumed.load_state_dict(objective.state_dict())
    assert torch.allclose(resumed.adv_probs, objective.adv_probs)


def _group_means(per_sample: torch.Tensor, group_ids: torch.Tensor, groups: list[int], reference: torch.Tensor) -> torch.Tensor:
    # バッチに現れない群の損失は、グラフにつながった 0 にする（公式と同じ扱い）。
    return torch.stack([per_sample[group_ids == g].mean() if g in groups else reference.sum() * 0 for g in range(3)])


@pytest.mark.parametrize("subgroup", [False, True])
@pytest.mark.parametrize("groups", [[0, 1], [0], [0, 1, 2]])
def test_group_dro_matches_official_update_and_gradient(subgroup: bool, groups: list[int]) -> None:
    """q は q 用の重みの群損失で公式どおり更新し、勾配は全群共通の重みの群損失を q で重み付けたものになる。"""
    objective = (
        SubgroupClassWeightedGroupDROTaskLoss(3, class_weight=[1.0, 2.0], q_class_weight=[[1.0, 2.0], [3.0, 1.0], [2.0, 4.0]], step_size=0.5)
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
        q_weights = torch.tensor([[1.0, 2.0], [3.0, 1.0], [2.0, 4.0]])[group_ids, target] if subgroup else torch.tensor([1.0, 2.0])[target]
        model_weights = torch.tensor([1.0, 2.0])[target]
        q_risks = _group_means(per_sample * q_weights, group_ids, groups, reference)
        risks = _group_means(per_sample * model_weights, group_ids, groups, reference)
        expected_q = expected_q * torch.exp(0.5 * q_risks.detach())
        expected_q /= expected_q.sum()
        expected = expected_q @ risks
        actual = objective(ObjectiveInput(logits=logits, target=target, attributes={"group_id": group_ids}))
        assert torch.allclose(objective.adv_probs, expected_q)
        assert torch.allclose(actual, expected)
        assert torch.allclose(torch.autograd.grad(actual, logits)[0], torch.autograd.grad(expected, reference)[0])
    if 2 not in groups:
        assert objective.adv_probs[2] < 1 / 3
