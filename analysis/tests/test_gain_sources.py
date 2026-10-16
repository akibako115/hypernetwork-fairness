"""`analysis/dataset_gain_sources/metrics.py` のうち、落ちずに間違う経路を確かめる。"""

from __future__ import annotations

import numpy as np
from sklearn.metrics import roc_auc_score

from analysis.dataset_gain_sources.metrics import boot_delta, group_members, within_between


def test_within_between_recombines_to_auroc():
    """セル内とセル間の一致率を組数で重み付けると、元の AUROC に戻る。"""
    rng = np.random.default_rng(1)
    y = rng.integers(0, 2, 400)
    cell = rng.integers(0, 5, 400)
    score = y + 0.5 * cell + rng.normal(size=400)
    within, between, share = within_between(score, y, cell)
    assert np.isclose(share * within + (1 - share) * between, roc_auc_score(y, score))


def test_within_ignores_a_pure_cell_shift():
    """セルごとに定数を足しても、セル内の一致率は変わらない（事前確率の注入はセル間だけを動かす）。"""
    rng = np.random.default_rng(2)
    y = rng.integers(0, 2, 300)
    cell = rng.integers(0, 4, 300)
    score = y + rng.normal(size=300)
    shifted = score + 3.0 * cell
    assert np.isclose(within_between(score, y, cell)[0], within_between(shifted, y, cell)[0])


def test_group_members_keeps_patients_together():
    groups = np.array(["b", "a", "b", "c", "a"])
    members = group_members(groups)
    assert sorted(sorted(groups[m].tolist()) for m in members) == [["a", "a"], ["b", "b"], ["c"]]
    assert sorted(np.concatenate(members).tolist()) == list(range(5))


def test_boot_delta_of_identical_models_is_zero():
    rng = np.random.default_rng(3)
    y = rng.integers(0, 2, 200)
    score = y + rng.normal(size=200)
    result = boot_delta([score], [score.copy()], y, np.arange(200), n_boot=50)
    assert result["delta"] == 0 and result["lo"] == 0 and result["hi"] == 0
