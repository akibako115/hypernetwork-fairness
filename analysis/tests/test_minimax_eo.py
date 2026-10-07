"""独立した固定データで EO mixture と AUC の意味を検算する。"""

import numpy as np

from analysis.minimax_auc_vs_eo.evaluation import (
    age_metrics,
    eo_mixture,
    paired_patient_bootstrap,
    randomized_positive_probability,
)


def test_ties_and_pair_direction():
    row = age_metrics([-2, 1, 3, -1], [0, 1, 0, 1], [0, 0, 1, 1])
    assert row["pair_auroc"] == [1.0, 0.0, 1.0, 0.0]
    assert row["tpr"] == [1.0, 0.0] and row["tnr"] == [1.0, 0.0]
    assert age_metrics([0, 0, 0, 0], [0, 1, 0, 1], [0, 0, 1, 1])["pair_auroc"] == [0.5] * 4


def test_lp_mixture_and_randomized_prediction_not_average_logits():
    parent = dict(auroc=0.8, worst_age_auroc=0.7, tpr=[0.5, 0.5], tnr=[0.5, 0.5])
    rows = [
        dict(parent, epoch=1, checkpoint="a", weighted_ce=0.4, tpr=[1.0, 0.0], tnr=[1.0, 0.0]),
        dict(parent, epoch=2, checkpoint="b", weighted_ce=0.4, tpr=[0.0, 1.0], tnr=[0.0, 1.0]),
    ]
    result = eo_mixture(rows, parent, epsilon=0)
    assert result["status"] == "selected"
    weights = [row["probability"] for row in result["components"]]
    assert np.allclose(weights, [0.5, 0.5])
    probability = randomized_positive_probability([[100, -1], [-1, 100]], weights)
    assert np.allclose(probability, [0.5, 0.5])
    assert np.all((np.array([[100, -1], [-1, 100]]).mean(axis=0) > 0) == 1)
    assert eo_mixture(rows[:1], parent, epsilon=0)["status"] == "infeasible"


def test_bootstrap_pairs_all_images_of_patient():
    def difference(index):
        assert (index == 0).sum() == (index == 1).sum()
        return float(np.mean(index))

    result = paired_patient_bootstrap(["a", "a", "b"], difference, repetitions=100)
    assert result["estimate"] == 1.0


def test_seed_average_differences_keep_seed_models_separate():
    from analysis.minimax_auc_vs_eo.evaluation import bootstrap_seed_difference

    labels = {
        "target": np.array([0, 1, 0, 1] * 3),
        "groups": np.array([0, 0, 1, 1] * 3),
        "patient_ids": np.array([0] * 4 + [1] * 4 + [2] * 4),
    }
    candidate = {"seeds": [42, 43], "scores": np.array([[-1, 1, -1, 1] * 3, [-1, 1, -1, 1] * 3])}
    reference = {"seeds": [42, 43], "scores": np.array([[1, -1, 1, -1] * 3, [-1, 1, -1, 1] * 3])}
    result = bootstrap_seed_difference(candidate, reference, labels, "auroc", repetitions=50)
    assert result["estimate"] == 0.5
    assert result["confidence_interval"] == [0.5, 0.5]


def test_vector_bootstrap_matches_scalar_and_reuses_auc_computation(monkeypatch):
    import analysis.minimax_auc_vs_eo.evaluation as evaluation

    labels = {
        "target": np.array([0, 1, 0, 1] * 3),
        "groups": np.array([0, 0, 1, 1] * 3),
        "patient_ids": np.repeat([0, 1, 2], 4),
    }
    candidate = {"seeds": [42, 43], "scores": np.array([[-1, 1, -1, 1] * 3, [-2, 2, -2, 2] * 3])}
    reference = {"seeds": [42, 43], "scores": np.array([[1, -1, 1, -1] * 3, [-1, 1, -1, 1] * 3])}
    metrics = ["auroc", "worst_age_auroc", "min_pair_auroc", "tpr_gap", "fpr_gap"]
    expected = {
        metric: evaluation.bootstrap_seed_difference(candidate, reference, labels, metric, repetitions=20)
        for metric in metrics
    }
    original = evaluation.roc_auc_score
    calls = []

    def auc(*args, **kwargs):
        calls.append(1)
        return original(*args, **kwargs)

    monkeypatch.setattr(evaluation, "roc_auc_score", auc)
    result = evaluation.bootstrap_seed_differences(candidate, reference, labels, metrics, repetitions=20)
    assert result == expected
    assert len(calls) == 21 * 2 * 2 * 5  # point estimate + 20 draws; each seed/condition evaluated once
    calls.clear()
    evaluation.bootstrap_seed_differences(candidate, reference, labels, ["tpr_gap", "fpr_gap"], repetitions=20)
    assert not calls


def test_patient_grouping_preserves_order_and_all_repeated_images():
    from analysis.minimax_auc_vs_eo.evaluation import _patient_index_groups

    groups = _patient_index_groups(["b", "a", "b", "c", "a"])
    assert [group.tolist() for group in groups] == [[1, 4], [0, 2], [3]]
