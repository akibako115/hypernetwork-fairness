import math

import pytest
import torch

from projects.hypernet_two_stage.utils.metrics import build_eval_attributes, compute_fairness_metrics, compute_metrics_by_attribute


def test_binary_fairness_excludes_missing_values_and_uses_attribute_names() -> None:
    logits = torch.tensor([[0.0, 2.0], [2.0, 0.0], [0.0, 2.0], [2.0, 0.0]])
    targets = torch.tensor([1, 0, 1, 0])
    attributes = {"categorical": torch.tensor([[0], [0], [1], [1]]), "categorical_missing": torch.tensor([[False], [False], [False], [True]])}

    metrics = compute_fairness_metrics(logits, targets, attributes, {"categorical": ["sex"]})

    assert math.isnan(metrics["sex"]["Eopp0"])
    assert metrics["sex"]["Eopp1"] == 0.0
    assert math.isnan(metrics["sex"]["Eodds"])


def test_attribute_accuracy_and_generated_evaluation_names_follow_the_contract() -> None:
    logits = torch.tensor([[2.0, 0.0], [0.0, 2.0], [2.0, 0.0]])
    targets = torch.tensor([0, 0, 0])
    attributes, names = build_eval_attributes(torch.tensor([[0, 1], [0, 1], [1, 0]]))

    assert names == {"categorical": ["attr[0]", "attr[1]"]}
    assert compute_metrics_by_attribute(logits, targets, attributes, names) == {
        "attr[0]": {0: 0.5, 1: 1.0},
        "attr[1]": {0: 1.0, 1: 0.5},
    }


def test_fairness_rejects_incompatible_attribute_width() -> None:
    with pytest.raises(ValueError, match="must contain 2 names"):
        compute_fairness_metrics(
            torch.tensor([[1.0, 0.0], [0.0, 1.0]]),
            torch.tensor([0, 1]),
            {"categorical": torch.tensor([[0, 1], [1, 0]])},
            {"categorical": ["sex"]},
        )


def test_fairness_metrics_include_group_performance_gaps_and_worst_group_values() -> None:
    logits = torch.tensor([[4.0, 0.0], [0.0, 4.0], [0.0, 4.0], [4.0, 0.0], [0.0, 4.0], [4.0, 0.0]])
    targets = torch.tensor([0, 1, 0, 1, 1, 0])
    attributes = {"categorical": torch.tensor([[0], [0], [0], [0], [1], [1]])}

    metrics = compute_fairness_metrics(logits, targets, attributes, {"categorical": ["sex"]})["sex"]

    assert metrics["auroc_gap"] == 0.5
    assert metrics["bacc_gap"] == 0.5
    assert metrics["worst_group_auroc"] == 0.5
    assert metrics["worst_group_bacc"] == 0.5


def test_fairness_metrics_exclude_single_class_groups_from_performance_aggregates() -> None:
    logits = torch.tensor([[4.0, 0.0], [0.0, 4.0], [0.0, 4.0], [0.0, 4.0]])
    targets = torch.tensor([0, 1, 1, 1])
    attributes = {"categorical": torch.tensor([[0], [0], [1], [1]])}

    metrics = compute_fairness_metrics(logits, targets, attributes, {"categorical": ["sex"]})["sex"]

    assert math.isnan(metrics["auroc_gap"])
    assert math.isnan(metrics["worst_group_bacc"])
