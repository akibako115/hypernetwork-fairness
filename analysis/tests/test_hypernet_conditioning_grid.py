"""柱 A の格子の指標が、評価の定義どおりに計算されることを確かめる。"""

import numpy as np
import pandas as pd
import pytest

from analysis.hypernet_conditioning_grid.compute import condition
from analysis.hypernet_conditioning_grid.grid_metrics import (
    MIN_CELL,
    group_rates,
    patient_bootstrap_weights,
    summary,
)


def _toy(n: int = MIN_CELL):
    """群 0 と群 1 に、陽性・陰性を n 件ずつ置く。群 0 は全員を陽性と判定、群 1 は全員を陰性と判定。"""
    group = np.repeat([0, 0, 1, 1], n)
    target = np.tile(np.repeat([1, 0], n), 2)
    probability = np.where(group == 0, 0.9, 0.1)
    # 順位は群の中で正しく付ける（AUROC 1）。動作点だけが群で違う。
    probability = probability + np.where(target == 1, 0.05, -0.05)
    return probability, target, group


def test_rates_use_the_fixed_threshold_and_skip_cells_below_the_floor() -> None:
    probability, target, group = _toy()
    rates = group_rates(probability, target, group)

    assert rates.tpr[:2].tolist() == [1.0, 0.0]
    assert rates.fpr[:2].tolist() == [1.0, 0.0]
    # 件数の無い群 2〜7 は率を出さない。
    assert np.isnan(rates.tpr[2:]).all() and np.isnan(rates.fpr[2:]).all()


def test_summary_takes_gaps_over_evaluable_groups_and_reports_their_count() -> None:
    probability, target, group = _toy()
    result = summary(probability, target, group)

    assert result["eopp1"] == 1.0 and result["eopp0"] == 1.0 and result["eodds"] == 1.0
    assert result["worst_fpr"] == 1.0 and result["min_tpr"] == 0.0
    assert result["worst_group_auroc"] == 1.0 and result["auroc_gap"] == 0.0
    assert result["groups_tpr"] == 2 and result["groups_fpr"] == 2


def test_a_cell_one_short_of_the_floor_is_not_evaluated() -> None:
    probability, target, group = _toy(MIN_CELL - 1)
    rates = group_rates(probability, target, group)

    assert np.isnan(rates.tpr).all()


def test_bootstrap_draws_whole_patients() -> None:
    patient = pd.Series(["a", "a", "b", "c", "c", "c"])
    weights = patient_bootstrap_weights(patient, n_resamples=50, seed=1)

    # 同じ患者の行は常に同じ回数だけ選ばれ、1 回の resample で患者数ぶん選ぶ。
    assert (weights[:, 0] == weights[:, 1]).all() and (weights[:, 3] == weights[:, 5]).all()
    assert (weights[:, [0, 2, 3]].sum(axis=1) == 3).all()


def test_bootstrap_weights_reweight_counts_like_duplicated_rows() -> None:
    probability, target, group = _toy()
    weight = np.ones_like(probability)
    weight[: MIN_CELL // 2] = 2.0
    duplicated = np.concatenate([np.arange(len(probability)), np.arange(MIN_CELL // 2)])

    weighted = summary(probability, target, group, weight)
    expanded = summary(probability[duplicated], target[duplicated], group[duplicated])

    assert weighted == pytest.approx(expanded)


@pytest.mark.parametrize(
    ("experiment", "objective"),
    [
        ("resnet-chexpert-age-race-eo-rate-dual-holdout", "eo_rate"),
        ("resnet-chexpert-age-race-eo-fpr-overall-dual-holdout", "eo_fpr_overall"),
        ("spatial-lora-chexpert-fc-from-resnet-age-race-eo-fpr-overall-dual-holdout", "eo_fpr_overall"),
        ("spatial-lora-chexpert-from-resnet-race-known", "erm"),
    ],
)
def test_condition_does_not_read_the_overall_eo_as_erm(experiment: str, objective: str) -> None:
    """Cotter et al. の形の EO は ID に eo-rate を含まない。ERM として集計されないことを見る。"""
    assert condition(experiment)["objective"] == objective
