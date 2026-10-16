"""HAM10000・FairVLMedのモデル構成比較で、群別指標が静かに間違う経路を固定する。"""

import numpy as np
import pandas as pd
import pytest

from analysis.fairvlmed_architecture_erm import comparison as fairvlmed
from analysis.ham10000_architecture_erm import comparison as ham


def _cell(n_pos: int, n_neg: int, tp: int, tn: int) -> tuple[np.ndarray, np.ndarray]:
    """陽性n_pos件のうちtp件、陰性n_neg件のうちtn件を正しく判定する確率を作る。"""
    target = np.r_[np.ones(n_pos), np.zeros(n_neg)].astype(int)
    score = np.r_[np.full(tp, 0.9), np.full(n_pos - tp, 0.1), np.full(tn, 0.1), np.full(n_neg - tn, 0.9)]
    return target, score


def test_cells_below_the_minimum_count_are_left_undefined() -> None:
    """陽性19件の群はTPR・AUROC・BACCを持たず、陰性側のTNRだけを持つ。"""
    target, score = _cell(19, 40, 10, 30)

    metrics = ham.cell_metrics(target, score)

    assert np.isnan(metrics["tpr"]) and np.isnan(metrics["auroc"]) and np.isnan(metrics["bacc"])
    assert metrics["tnr"] == pytest.approx(0.75)


def test_gaps_and_equalized_odds_follow_their_definitions() -> None:
    """female：TPR 0.5・TNR 0.9、male：TPR 0.8・TNR 0.6。Eopp1 0.3、Eopp0 0.3、Eodds 0.3。"""
    female_t, female_s = _cell(20, 20, 10, 18)
    male_t, male_s = _cell(20, 20, 16, 12)
    target, score = np.r_[female_t, male_t], np.r_[female_s, male_s]
    sex = np.r_[np.zeros(40), np.ones(40)].astype(int)

    summary, rows = ham.evaluate(target, score, {"sex": sex})

    assert summary["sex/Eopp1"] == pytest.approx(0.3)
    assert summary["sex/Eopp0"] == pytest.approx(0.3)
    assert summary["sex/Eodds"] == pytest.approx(0.3)
    assert summary["sex/bacc_gap"] == pytest.approx(abs((0.5 + 0.9) / 2 - (0.8 + 0.6) / 2))
    assert [row["group"] for row in rows] == ["female", "male"]


def test_worst_and_gap_need_two_evaluable_groups() -> None:
    """片方の群が最小セル未満なら、残る1群だけでworstやgapを作らない。"""
    large_t, large_s = _cell(30, 30, 20, 20)
    small_t, small_s = _cell(5, 30, 3, 20)
    target, score = np.r_[large_t, small_t], np.r_[large_s, small_s]
    sex = np.r_[np.zeros(60), np.ones(35)].astype(int)

    summary, _ = ham.evaluate(target, score, {"sex": sex})

    assert np.isnan(summary["sex/worst_auroc"]) and np.isnan(summary["sex/auroc_gap"])
    assert np.isnan(summary["sex/Eopp1"])
    assert not np.isnan(summary["sex/Eopp0"])


def test_ham_groups_drop_rows_missing_sex_or_age_and_order_the_intersection() -> None:
    """交差群はfemale <65・female ≥65・male <65・male ≥65の順。欠損行は全軸で-1。"""
    frame = pd.DataFrame(
        {
            "sex": [0, 0, 1, 1, -1, 0],
            "sex_missing": [0, 0, 0, 0, 1, 0],
            "age": [40.0, 70.0, 40.0, 65.0, 50.0, np.nan],
            "age_missing": [0, 0, 0, 0, 0, 1],
        }
    )

    ids = ham.group_ids(frame)

    assert ids["sex × age"].tolist() == [0, 1, 2, 3, -1, -1]
    assert ids["age_group_65"].tolist() == [0, 1, 0, 1, -1, -1]
    assert ids["sex"].tolist() == [0, 0, 1, 1, -1, -1]


def test_fairvlmed_drops_a_missing_attribute_only_from_its_own_axis() -> None:
    """ethnicityがunknownの行も、race軸とage軸には残る。"""
    frame = pd.DataFrame(
        {
            "race": [0, 1, 2],
            "race_missing": [0, 0, 0],
            "gender": [0, 1, 0],
            "gender_missing": [0, 0, 0],
            "ethnicity": [0, -1, 1],
            "ethnicity_missing": [0, 1, 0],
            "language": [0, 2, -1],
            "language_missing": [0, 0, 1],
            "age": [64.9, 65.0, 80.0],
        }
    )

    ids = fairvlmed.group_ids(frame)

    assert ids["ethnicity"].tolist() == [0, -1, 1]
    assert ids["language"].tolist() == [0, 2, -1]
    assert ids["race"].tolist() == [0, 1, 2]
    assert ids["age_group_65"].tolist() == [0, 1, 1]


def test_paired_bootstrap_reports_zero_for_identical_predictions() -> None:
    """全モデルが同じ予測なら、どの再標本でも差は0になる（再標本がモデル間で共有されている）。"""
    rng = np.random.default_rng(0)
    n = 200
    frame = pd.DataFrame(
        {
            "target": rng.integers(0, 2, n),
            "sex": rng.integers(0, 2, n),
            "sex_missing": 0,
            "age": rng.choice([40.0, 70.0], n),
            "age_missing": 0,
        }
    )
    scores = {}
    for seed in ham.SEEDS:
        score = rng.random(n)
        for model in ham.MODELS:
            scores[model, seed] = score

    boot = ham.paired_bootstrap(scores, frame, np.ones(n, bool), np.arange(n), n_boot=5)
    overall = boot[boot["metric"] == "overall/auroc"]

    assert set(overall["model"]) == {"spatial_lora", "hyperadapt"}
    assert np.allclose(overall[["delta", "boot_se", "ci_low", "ci_high"]].to_numpy(), 0)


def test_paired_bootstrap_restricted_to_a_subset_matches_the_seed_mean_difference() -> None:
    """評価集合で行を絞ったときも、点推定は同じ行で評価したseed別差の平均に一致する。"""
    rng = np.random.default_rng(1)
    n = 300
    target = rng.integers(0, 2, n)
    frame = pd.DataFrame(
        {"target": target, "sex": rng.integers(0, 2, n), "sex_missing": 0, "age": 40.0, "age_missing": 0}
    )
    mask = np.arange(n) % 3 != 0
    scores = {}
    for seed in ham.SEEDS:
        noise = rng.random(n)
        scores["resnet", seed] = 0.5 * target + noise
        scores["spatial_lora", seed] = 0.8 * target + noise
        scores["hyperadapt", seed] = noise

    boot = ham.paired_bootstrap(scores, frame, mask, np.arange(n), n_boot=3)
    summary, _ = ham.evaluate_all(scores, frame, mask)
    expected = ham.seed_deltas(summary).groupby("model")["overall/auroc"].mean()
    point = boot[boot["metric"] == "overall/auroc"].set_index("model")["delta"]

    assert point["spatial_lora"] == pytest.approx(expected["spatial_lora"])
    assert point["hyperadapt"] == pytest.approx(expected["hyperadapt"])
