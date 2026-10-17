import pytest
import torch
import torch.nn.functional as F

from projects.hypernet_e2e.loss import (
    AlternatingAttributeInvariantTaskLoss,
    AttributeInvariantTaskLoss,
    GlobalClassWeightedGroupDROTaskLoss,
    ObjectiveInput,
    SubgroupClassWeightedGroupDROTaskLoss,
    TaskLoss,
    UniformGroupTaskLoss,
)
from projects.hypernet_e2e.objectives.attribute_invariance import _GradientReverse

_ONE_CATEGORICAL_ATTRIBUTE = {"categorical": ["sex"], "continuous": []}


def _attribute_invariant_loss(**kwargs: object) -> AttributeInvariantTaskLoss:
    """1個の categorical 属性を使う小さな attribute-invariance objective を作る。"""
    return AttributeInvariantTaskLoss(
        TaskLoss(),
        feature_dim=2,
        input_attribute_names=_ONE_CATEGORICAL_ATTRIBUTE,
        adversarial_attribute_names=_ONE_CATEGORICAL_ATTRIBUTE,
        categorical_cardinalities=[2],
        **kwargs,
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


def test_gradient_reverse_negates_only_the_feature_gradient() -> None:
    features = torch.tensor([[1.0, -2.0]], requires_grad=True)
    weights = torch.tensor([[3.0, 4.0]])
    (_GradientReverse.apply(features, 0.5) * weights).sum().backward()

    assert torch.equal(features.grad, -0.5 * weights)


def test_dann_gradient_schedule_warms_up_monotonically() -> None:
    objective = _attribute_invariant_loss(
        gradient_schedule={"name": "dann", "gamma": 10.0, "max_scale": 1.0},
    )
    objective.set_training_progress(0.0)
    start = objective.adversary_scale
    objective.set_training_progress(0.5)
    middle = objective.adversary_scale
    objective.set_training_progress(1.0)
    end = objective.adversary_scale
    assert start == pytest.approx(0.0)
    assert 0.0 < middle < end <= 1.0


def test_alfr_adversary_loss_updates_only_the_adversary() -> None:
    objective = AlternatingAttributeInvariantTaskLoss(
        TaskLoss(),
        feature_dim=2,
        input_attribute_names=_ONE_CATEGORICAL_ATTRIBUTE,
        adversarial_attribute_names=_ONE_CATEGORICAL_ATTRIBUTE,
        categorical_cardinalities=[2],
    )
    features = torch.randn(4, 2, requires_grad=True)
    inputs = ObjectiveInput(
        logits=torch.randn(4, 2, requires_grad=True),
        target=torch.tensor([0, 1, 0, 1]),
        attributes={
            "categorical": torch.tensor([[0], [1], [0], [1]]),
            "categorical_missing": torch.zeros(4, 1, dtype=torch.bool),
        },
        features=features,
    )
    objective.adversary_loss(inputs).backward()
    assert features.grad is None
    assert any(parameter.grad is not None for parameter in objective.adversary_parameters())


def test_attribute_invariant_task_loss_combines_task_and_observed_attribute_losses() -> None:
    objective = _attribute_invariant_loss(hidden_dim=2, attribute_adversary_weight=0.5)
    for parameter in objective.attribute_adversary.parameters():
        parameter.data.zero_()
    inputs = ObjectiveInput(
        logits=torch.tensor([[2.0, 0.0], [0.0, 2.0]], requires_grad=True),
        target=torch.tensor([0, 1]),
        attributes={
            "categorical": torch.tensor([[0], [1]]),
            "categorical_missing": torch.zeros(2, 1, dtype=torch.bool),
        },
        features=torch.randn(2, 2, requires_grad=True),
    )

    loss = objective(inputs)

    assert torch.allclose(loss, F.cross_entropy(inputs.logits, inputs.target) + 0.5 * torch.log(torch.tensor(2.0)))
    loss.backward()
    assert inputs.features.grad is not None


def test_attribute_invariant_task_loss_requires_features() -> None:
    objective = _attribute_invariant_loss()
    inputs = ObjectiveInput(
        logits=torch.randn(2, 2),
        target=torch.tensor([0, 1]),
        attributes={"categorical": torch.tensor([[0], [1]]), "categorical_missing": torch.zeros(2, 1, dtype=torch.bool)},
    )

    with pytest.raises(ValueError, match="features"):
        objective(inputs)


def test_attribute_invariant_task_loss_keeps_all_missing_batch_backwardable() -> None:
    objective = _attribute_invariant_loss()
    features = torch.randn(2, 2, requires_grad=True)
    inputs = ObjectiveInput(
        logits=torch.randn(2, 2, requires_grad=True),
        target=torch.tensor([0, 1]),
        attributes={"categorical": torch.tensor([[0], [1]]), "categorical_missing": torch.ones(2, 1, dtype=torch.bool)},
        features=features,
    )

    objective(inputs).backward()

    assert features.grad is not None


def test_attribute_invariant_task_loss_uses_only_selected_attribute_columns() -> None:
    input_names = {
        "categorical": ["sex", "race", "ethnicity", "frontal_lateral", "ap_pa"],
        "continuous": ["age"],
    }
    objective = AttributeInvariantTaskLoss(
        TaskLoss(),
        feature_dim=2,
        input_attribute_names=input_names,
        adversarial_attribute_names={"categorical": ["sex", "race", "ethnicity"], "continuous": ["age"]},
        categorical_cardinalities=[2, 3, 2, 2, 2],
        num_continuous=1,
        hidden_dim=2,
        attribute_adversary_weight=1.0,
    )
    for parameter in objective.attribute_adversary.parameters():
        parameter.data.zero_()
    base = ObjectiveInput(
        logits=torch.tensor([[2.0, 0.0], [0.0, 2.0]]),
        target=torch.tensor([0, 1]),
        attributes={
            "categorical": torch.tensor([[0, 0, 1, 0, 1], [1, 2, 0, 1, 0]]),
            "categorical_missing": torch.zeros(2, 5, dtype=torch.bool),
            "continuous": torch.tensor([[1.0], [2.0]]),
            "continuous_missing": torch.zeros(2, 1, dtype=torch.bool),
        },
        features=torch.randn(2, 2),
    )
    changed_excluded = ObjectiveInput(
        logits=base.logits,
        target=base.target,
        attributes={
            **base.attributes,
            "categorical": torch.tensor([[0, 0, 1, 1, 0], [1, 2, 0, 0, 1]]),
            "categorical_missing": torch.tensor([[False, False, False, True, True], [False, False, False, True, True]]),
        },
        features=base.features,
    )

    assert len(objective.attribute_adversary.categorical_heads) == 3
    assert objective.attribute_adversary.continuous_head is not None
    assert torch.allclose(objective(base), objective(changed_excluded))


def _sex_race_age_inputs() -> ObjectiveInput:
    """sex / race / age を全行観測した 4 行の入力を返す。"""
    return ObjectiveInput(
        logits=torch.randn(4, 2),
        target=torch.tensor([0, 1, 0, 1]),
        attributes={
            "categorical": torch.tensor([[0, 2], [1, 0], [0, 1], [1, 2]]),
            "categorical_missing": torch.zeros(4, 2, dtype=torch.bool),
            "continuous": torch.randn(4, 1),
            "continuous_missing": torch.zeros(4, 1, dtype=torch.bool),
        },
        features=torch.rand(4, 3),
    )


def _sex_race_age_loss(**kwargs: object) -> AttributeInvariantTaskLoss:
    """sex / race / age の 3 属性を GRL 対象にする objective を作る。"""
    names = {"categorical": ["sex", "race"], "continuous": ["age"]}
    return AttributeInvariantTaskLoss(
        TaskLoss(),
        feature_dim=3,
        input_attribute_names=names,
        adversarial_attribute_names=names,
        categorical_cardinalities=[2, 3],
        num_continuous=1,
        **kwargs,
    )


def test_default_adversary_keeps_the_single_shared_trunk_parameter_names() -> None:
    adversary = _sex_race_age_loss(hidden_dim=4).attribute_adversary

    assert sorted(name for name, _ in adversary.named_parameters()) == [
        "categorical_heads.0.bias",
        "categorical_heads.0.weight",
        "categorical_heads.1.bias",
        "categorical_heads.1.weight",
        "continuous_head.bias",
        "continuous_head.weight",
        "trunk.0.bias",
        "trunk.0.weight",
    ]


def test_adversary_stacks_one_trunk_layer_per_hidden_dim() -> None:
    adversary = _sex_race_age_loss(hidden_dim=[5, 4]).attribute_adversary

    widths = [layer.out_features for layer in adversary.trunk if isinstance(layer, torch.nn.Linear)]
    assert widths == [5, 4]
    assert adversary.categorical_heads[1].in_features == 4
    assert adversary.continuous_head.in_features == 4


def test_per_attribute_trunks_receive_only_their_own_attribute_loss() -> None:
    objective = _sex_race_age_loss(hidden_dim=[5, 4], shared_trunk=False)
    adversary = objective.attribute_adversary
    assert adversary.trunk is None
    assert len(adversary.categorical_trunks) == 2
    assert len(adversary.continuous_trunks) == 1

    objective.loss_components(_sex_race_age_inputs())["attribute_adversary/race"].backward()

    def has_gradient(module: torch.nn.Module) -> bool:
        return any(parameter.grad is not None and parameter.grad.abs().sum() > 0 for parameter in module.parameters())

    assert has_gradient(adversary.categorical_trunks[1])
    assert not has_gradient(adversary.categorical_trunks[0])
    assert not has_gradient(adversary.continuous_trunks[0])


def test_per_attribute_trunks_match_separate_single_attribute_adversaries() -> None:
    objective = _sex_race_age_loss(hidden_dim=4, shared_trunk=False)
    adversary = objective.attribute_adversary
    inputs = _sex_race_age_inputs()

    components = objective.loss_components(inputs)

    features = inputs.features
    sex = F.cross_entropy(adversary.categorical_heads[0](adversary.categorical_trunks[0](features)), inputs.attributes["categorical"][:, 0])
    age = F.mse_loss(adversary.continuous_head(adversary.continuous_trunks[0](features))[:, 0], inputs.attributes["continuous"][:, 0])
    assert torch.allclose(components["attribute_adversary/sex"], sex)
    assert torch.allclose(components["attribute_adversary/age"], age)


@pytest.mark.parametrize(
    ("adversarial_names", "match"),
    [
        ({"categorical": [], "continuous": []}, "少なくとも1つ"),
        ({"categorical": ["sex", "sex"], "continuous": []}, "重複"),
        ({"categorical": ["age"], "continuous": []}, "ない属性"),
    ],
)
def test_attribute_invariant_task_loss_rejects_invalid_attribute_selection(adversarial_names: dict[str, list[str]], match: str) -> None:
    with pytest.raises(ValueError, match=match):
        AttributeInvariantTaskLoss(
            TaskLoss(),
            feature_dim=2,
            input_attribute_names=_ONE_CATEGORICAL_ATTRIBUTE,
            adversarial_attribute_names=adversarial_names,
            categorical_cardinalities=[2],
        )


def test_attribute_invariant_task_loss_rejects_attribute_under_the_wrong_kind() -> None:
    with pytest.raises(ValueError, match="別種別"):
        AttributeInvariantTaskLoss(
            TaskLoss(),
            feature_dim=2,
            input_attribute_names={"categorical": ["sex"], "continuous": ["age"]},
            adversarial_attribute_names={"categorical": ["age"], "continuous": []},
            categorical_cardinalities=[2],
            num_continuous=1,
        )


def test_attribute_invariant_task_loss_rejects_full_attribute_shape_mismatch() -> None:
    objective = _attribute_invariant_loss()
    inputs = ObjectiveInput(
        logits=torch.randn(2, 2),
        target=torch.tensor([0, 1]),
        attributes={
            "categorical": torch.tensor([[0, 1], [1, 0]]),
            "categorical_missing": torch.zeros(2, 2, dtype=torch.bool),
        },
        features=torch.randn(2, 2),
    )

    with pytest.raises(ValueError, match="input_attribute_names"):
        objective(inputs)


def test_attribute_invariant_task_loss_validates_an_unselected_full_attribute_kind() -> None:
    objective = AttributeInvariantTaskLoss(
        TaskLoss(),
        feature_dim=2,
        input_attribute_names={"categorical": ["sex"], "continuous": ["age"]},
        adversarial_attribute_names={"categorical": ["sex"], "continuous": []},
        categorical_cardinalities=[2],
        num_continuous=1,
    )
    inputs = ObjectiveInput(
        logits=torch.randn(2, 2),
        target=torch.tensor([0, 1]),
        attributes={
            "categorical": torch.tensor([[0], [1]]),
            "categorical_missing": torch.zeros(2, 1, dtype=torch.bool),
        },
        features=torch.randn(2, 2),
    )

    with pytest.raises(ValueError, match="continuous"):
        objective(inputs)


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


def test_adversary_trunk_uses_leaky_relu_by_default_and_keeps_relu_for_old_runs() -> None:
    """ReLU の trunk は GRL 下で全 unit を負側へ押し込まれると head への勾配ごと消える。"""
    leaky = _attribute_invariant_loss()
    legacy = _attribute_invariant_loss(activation="relu")
    assert isinstance(leaky.attribute_adversary.trunk[1], torch.nn.LeakyReLU)
    assert isinstance(legacy.attribute_adversary.trunk[1], torch.nn.ReLU)
    assert leaky.state_dict().keys() == legacy.state_dict().keys()

    features = torch.full((4, 2), -5.0, requires_grad=True)
    attributes = {"categorical": torch.tensor([[0], [1], [0], [1]]), "categorical_missing": torch.zeros(4, 1, dtype=torch.bool)}
    for objective, expect_gradient in ((leaky, True), (legacy, False)):
        with torch.no_grad():
            objective.attribute_adversary.trunk[0].weight.fill_(1.0)
            objective.attribute_adversary.trunk[0].bias.zero_()
        features.grad = None
        objective.attribute_adversary(features, attributes).backward()
        assert (features.grad.abs().sum() > 0) == expect_gradient


@pytest.mark.parametrize("objective_type", [AttributeInvariantTaskLoss, AlternatingAttributeInvariantTaskLoss])
@pytest.mark.parametrize("training", [True, False])
@pytest.mark.parametrize("missing", ["race", "age", "all", "none"])
def test_adversary_handles_missing_selected_attributes(objective_type, training: bool, missing: str) -> None:
    torch.manual_seed(7)
    names = {"categorical": ["sex", "race"], "continuous": ["age"]}
    objective = objective_type(TaskLoss(), feature_dim=3, input_attribute_names=names, adversarial_attribute_names=names, categorical_cardinalities=[2, 3], num_continuous=1)
    objective.train(training)
    inputs = _sex_race_age_inputs()
    inputs.features.requires_grad_()
    inputs.logits.requires_grad_()
    if missing in {"race", "all"}:
        inputs.attributes["categorical_missing"][:, 1] = True
    if missing in {"age", "all"}:
        inputs.attributes["continuous_missing"][:] = True
    if missing == "all":
        inputs.attributes["categorical_missing"][:] = True
    components = objective.loss_components(inputs)
    observed = set(names["categorical"] + names["continuous"]) - ({missing} if missing != "all" else {"sex", "race", "age"})
    keys = {f"attribute_adversary/{name}" for name in observed}
    assert set(components) == {"task", "attribute_adversary"} | keys
    expected = torch.stack([components[key] for key in sorted(keys)]).mean() if keys else inputs.features.sum() * 0
    assert torch.allclose(components["attribute_adversary"], expected)
    components["attribute_adversary"].backward()
    assert torch.isfinite(inputs.features.grad).all()
    assert bool(inputs.features.grad.abs().sum() > 0) == bool(keys)
    assert torch.isfinite(objective(inputs))
