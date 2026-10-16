"""率・寄与・症例変更の分解を手計算ケースで固定する。"""

import numpy as np
import pandas as pd
import pytest

from analysis.archive.race_known_architecture_objective_comparison.comparison import evaluate
from analysis.archive.race_known_architecture_objective_comparison.operating_point import (
    gap_extrema,
    paired_group_changes,
    prediction_transitions,
    summarize_best_q,
)


def test_changes_and_transitions_reproduce_manual_bacc_decomposition():
    # 第0群は陽性2/陰性1、第1群は陽性1/陰性3。他6群は陽性/陰性各1。
    groups = np.array([0, 0, 0, 1, 1, 1, 1] + [g for g in range(2, 8) for _ in range(2)])
    target = np.array([1, 1, 0, 1, 0, 0, 0] + [1, 0] * 6)
    before = target.copy()
    before[0] = 0
    before[4] = 1
    after = before.copy()
    after[0] = 1  # 第0群TPR +1/2。
    after[2] = 1  # 第0群TNR -1。
    after[4] = 0  # 第1群TNR +1/3。
    frame = pd.DataFrame(
        {
            "target": target,
            "race_group": groups % 4,
            "age": np.where(groups < 4, 40, 70),
            "age_missing": False,
            "race_group_missing": False,
        }
    )
    caches = {
        (condition, 42): {
            "target": target,
            "predictions": prediction,
            "probabilities": np.column_stack((1 - prediction, prediction)),
        }
        for condition, prediction in [("A", before), ("B", after)]
    }
    summary, subgroup = evaluate(caches, frame)
    changes = paired_group_changes(subgroup, [("A", "B")]).set_index("group_id")
    assert changes.loc[0, "delta_tpr"] == pytest.approx(0.5)
    assert changes.loc[0, "delta_tnr"] == pytest.approx(-1)
    assert changes.loc[1, "delta_tnr"] == pytest.approx(1 / 3)
    assert changes.tpr_contribution.sum() == pytest.approx(1 / 18)
    assert changes.tnr_contribution.sum() == pytest.approx(0)
    overall = summary.set_index("condition")
    assert changes.bacc_contribution.sum() == pytest.approx(overall.loc["B", "bacc"] - overall.loc["A", "bacc"])
    assert changes.macro_bacc_contribution.sum() == pytest.approx((-0.25 + 1 / 6) / 8)
    transitions = prediction_transitions(caches, frame, [("A", "B")], [42]).set_index("group_id")
    assert transitions.loc[0, "FN_to_TP"] == 1
    assert transitions.loc[0, "TN_to_FP"] == 1
    assert transitions.loc[1, "FP_to_TN"] == 1
    for metric, label in [("tpr", "positive"), ("tnr", "negative")]:
        np.testing.assert_allclose(transitions[f"net_{label}"] / transitions[f"n_{label}"], changes[f"delta_{metric}"])
    extremes = gap_extrema(subgroup).set_index(["condition", "metric"])
    assert extremes.loc[("B", "tnr"), "minimum"] == 0
    assert extremes.loc[("B", "tnr"), "minimum_groups"] == "<65 × White"
    assert ";" in extremes.loc[("B", "tnr"), "maximum_groups"]


def test_group_changes_reject_unmatched_seeds_and_extrema_preserve_undefined():
    data = pd.DataFrame({"condition": ["A", "B"], "seed": [42, 43], "group_id": [0, 0]})
    with pytest.raises(ValueError, match="対応"):
        paired_group_changes(data, [("A", "B")])
    groups = pd.DataFrame(
        {"condition": ["A", "A"], "seed": [42, 42], "group": ["one", "two"], "tpr": [0.8, np.nan], "tnr": [0.7, 0.9]}
    )
    extrema = gap_extrema(groups).set_index("metric")
    assert np.isnan(extrema.loc["tpr", "gap"])


def test_q_summary_excludes_post_best_epochs_and_keeps_seed_specific_cutoffs():
    rows = []
    for seed, best in [(42, 0), (43, 1)]:
        for epoch in range(3):
            rows.append(
                {
                    "condition": "A",
                    "seed": seed,
                    "group_id": 0,
                    "group": "one",
                    "epoch": epoch,
                    "best_epoch": best,
                    "q_mean": [0.2, 0.4, 0.9][epoch],
                    "q_end": [0.3, 0.5, 1.0][epoch],
                    "loss": 0.8,
                    "count": 10,
                }
            )
    history = pd.DataFrame(rows)
    summary = summarize_best_q(history).set_index("seed")
    assert summary.loc[42, "q_mean_to_best"] == pytest.approx(0.2)
    assert summary.loc[43, "q_mean_to_best"] == pytest.approx(0.3)
    assert summary.loc[43, "q_end_at_best"] == pytest.approx(0.5)
    assert summary.loc[42, "epochs_to_best"] == 1
    with pytest.raises(ValueError, match="epoch"):
        summarize_best_q(history[~((history.seed == 43) & (history.epoch == 0))])
