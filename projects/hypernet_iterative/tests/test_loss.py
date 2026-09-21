import pytest
import torch
import torch.nn.functional as F

from projects.hypernet_iterative.loss import (
    ClassBalancedGroupDROTaskLoss,
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

    objective = objective_class(num_groups=2, class_weight=[1.0, 2.0])
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


def test_class_balanced_group_dro_group_loss_is_invariant_to_group_prevalence() -> None:
    """陽性率だけが違う2 groupが同じgroup lossになる。GroupDROTaskLossとの差はここに出る。"""
    inputs = _imbalanced_cohort_batch()
    objective = ClassBalancedGroupDROTaskLoss(num_groups=2, step_size=1.0, loss_ema_momentum=1.0)

    objective(inputs)

    balanced = objective.cell_loss_ema.mean(dim=1)
    assert balanced[0] == pytest.approx(balanced[1].item())
    assert torch.allclose(objective.adv_probs, torch.tensor([0.5, 0.5]))

    plain = GroupDROTaskLoss(num_groups=2, step_size=1.0)
    plain(inputs)
    assert plain.adv_probs[0] > plain.adv_probs[1]


def test_class_balanced_group_dro_returns_q_weighted_class_balanced_loss() -> None:
    inputs = _imbalanced_cohort_batch()
    objective = ClassBalancedGroupDROTaskLoss(num_groups=2, step_size=1.0, loss_ema_momentum=1.0)

    loss = objective(inputs)

    per_sample = F.cross_entropy(inputs.logits, inputs.target, reduction="none")
    expected_group_loss = torch.stack(
        (
            torch.stack((per_sample[:2].mean(), per_sample[2:4].mean())).mean(),
            torch.stack((per_sample[4:9].mean(), per_sample[9])).mean(),
        )
    )
    assert torch.allclose(loss, torch.dot(objective.adv_probs, expected_group_loss))


def test_class_balanced_group_dro_falls_back_to_the_observed_class_in_a_batch() -> None:
    """片方のクラスしか含まないgroupは、そのクラスの平均lossをgroup lossにする。"""
    logits = torch.tensor([[3.0, 0.0], [1.0, 0.0], [0.0, 2.0]])
    target = torch.tensor([0, 0, 1])
    inputs = ObjectiveInput(logits=logits, target=target, attributes={"group_id": torch.tensor([0, 0, 1])})
    objective = ClassBalancedGroupDROTaskLoss(num_groups=2, step_size=1.0, loss_ema_momentum=1.0)

    loss = objective(inputs)

    per_sample = F.cross_entropy(logits, target, reduction="none")
    expected_group_loss = torch.stack((per_sample[:2].mean(), per_sample[2]))
    assert torch.allclose(loss, torch.dot(objective.adv_probs, expected_group_loss))


def test_class_balanced_group_dro_bias_corrects_the_loss_ema_after_one_step() -> None:
    """momentumで縮んだEMAをそのまま指数勾配へ渡さない（Adamのstep補正と同じ扱い）。"""
    inputs = _imbalanced_cohort_batch()
    objective = ClassBalancedGroupDROTaskLoss(num_groups=2, step_size=1.0, loss_ema_momentum=0.01)

    objective(inputs)

    corrected = objective.cell_loss_ema / objective.cell_ema_weight
    reference = ClassBalancedGroupDROTaskLoss(num_groups=2, step_size=1.0, loss_ema_momentum=1.0)
    reference(inputs)
    assert torch.allclose(corrected, reference.cell_loss_ema)
    assert torch.allclose(objective.adv_probs, reference.adv_probs)


def test_class_balanced_group_dro_updates_absent_groups_from_the_loss_ema() -> None:
    """batchに現れなかったgroupも、過去のEMAに基づいてadversarial weightを更新する。"""
    objective = ClassBalancedGroupDROTaskLoss(num_groups=2, step_size=1.0, loss_ema_momentum=1.0)
    first = ObjectiveInput(
        logits=torch.tensor([[8.0, 0.0], [0.0, 8.0], [0.0, 0.0], [0.0, 0.0]]),
        target=torch.tensor([0, 1, 0, 1]),
        attributes={"group_id": torch.tensor([0, 0, 1, 1])},
    )
    objective(first)
    weight_after_first = objective.adv_probs[1].clone()

    second = ObjectiveInput(
        logits=torch.tensor([[8.0, 0.0], [0.0, 8.0]]),
        target=torch.tensor([0, 1]),
        attributes={"group_id": torch.tensor([0, 0])},
    )
    objective(second)

    # group 1 は second batch に不在だが、EMA 上の loss が group 0 より高いので重みは増える。
    assert objective.adv_probs[1] > weight_after_first
    assert objective.adv_probs.sum().item() == pytest.approx(1.0)
    assert torch.equal(objective.cell_ema_weight[1], torch.ones(2))


def test_class_balanced_group_dro_restores_all_adversarial_state_from_checkpoint() -> None:
    objective = ClassBalancedGroupDROTaskLoss(num_groups=2, step_size=1.0, loss_ema_momentum=0.5)
    objective(_imbalanced_cohort_batch())
    checkpoint_state = objective.state_dict()

    resumed = ClassBalancedGroupDROTaskLoss(num_groups=2, step_size=1.0, loss_ema_momentum=0.5)
    resumed.load_state_dict(checkpoint_state)

    for name in ("adv_probs", "cell_loss_ema", "cell_ema_weight"):
        assert name in checkpoint_state
        assert torch.allclose(getattr(resumed, name), getattr(objective, name))


def test_class_balanced_group_dro_applies_class_weight() -> None:
    logits = torch.tensor([[3.0, 0.0], [0.0, 3.0]])
    target = torch.tensor([0, 1])
    inputs = ObjectiveInput(logits=logits, target=target, attributes={"group_id": torch.tensor([0, 1])})
    objective = ClassBalancedGroupDROTaskLoss(num_groups=2, class_weight=[1.0, 2.0], step_size=1.0, loss_ema_momentum=1.0)

    loss = objective(inputs)

    per_sample = F.cross_entropy(logits, target, weight=torch.tensor([1.0, 2.0]), reduction="none")
    assert torch.allclose(loss, torch.dot(objective.adv_probs, per_sample))


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"num_groups": 1}, "num_groups must be at least 2"),
        ({"num_groups": 2, "num_classes": 1}, "num_classes must be at least 2"),
        ({"num_groups": 2, "step_size": 0.0}, "step_size must be finite and positive"),
        ({"num_groups": 2, "loss_ema_momentum": 0.0}, "loss_ema_momentum must be in"),
        ({"num_groups": 2, "loss_ema_momentum": 1.5}, "loss_ema_momentum must be in"),
        ({"num_groups": 2, "group_key": ""}, "group_key must not be empty"),
    ],
)
def test_class_balanced_group_dro_rejects_invalid_configuration(kwargs: dict, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        ClassBalancedGroupDROTaskLoss(**kwargs)


def test_class_balanced_group_dro_rejects_mismatched_group_id_shape() -> None:
    inputs = ObjectiveInput(
        logits=torch.randn(2, 2),
        target=torch.tensor([0, 1]),
        attributes={"group_id": torch.tensor([[0], [1]])},
    )
    with pytest.raises(ValueError, match="shape"):
        ClassBalancedGroupDROTaskLoss(num_groups=2)(inputs)


def test_class_balanced_group_dro_backpropagates_into_logits() -> None:
    """scatter_add と index_put を経由しても勾配が logits まで届く。"""
    inputs = _imbalanced_cohort_batch()
    logits = inputs.logits.clone().requires_grad_(True)
    objective = ClassBalancedGroupDROTaskLoss(num_groups=2, step_size=1.0, loss_ema_momentum=1.0)

    objective(ObjectiveInput(logits=logits, target=inputs.target, attributes=inputs.attributes)).backward()

    assert logits.grad is not None
    assert torch.isfinite(logits.grad).all()
    assert (logits.grad.abs().sum(dim=1) > 0).all()
    # group 1 の陽性は1件でセル平均を独占するため、5件で割られる陰性より勾配が大きい。
    assert logits.grad[9].abs().sum() > logits.grad[4].abs().sum()
