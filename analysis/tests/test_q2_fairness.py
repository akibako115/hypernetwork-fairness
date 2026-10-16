"""Q2 の公平性損失の run：held-out の切り方、セル均等抽出、群の重みの読み方を検証する。"""

import numpy as np
import pandas as pd

from analysis.archive.latent_bias_hypernet.probes.train_adapter_fair import PER_CELL, G, cell_batches, split_holdout
from analysis.archive.latent_bias_hypernet.q2_fairness_loss import weight_summary
from projects.hypernet_e2e.objectives.equalized_odds import EqualizedOddsRateTaskLoss


def test_holdout_is_patient_disjoint() -> None:
    """同じ患者の画像は、学習側と held-out の片方にだけ入る。"""
    images = [f"train/patient{p:05d}/study{s}/view1_frontal.jpg" for p in range(500) for s in range(1 + p % 3)]
    fit, held = split_holdout(pd.DataFrame({"image": images}))
    patients = [set(f.image.str.split("/").str[1]) for f in (fit, held)]
    assert not patients[0] & patients[1]
    assert len(patients[1]) == 100


def test_cell_batches_cover_every_cell_equally() -> None:
    """どのバッチも 16 セルから PER_CELL 件ずつ取る（公平性損失は全セルが無いと定義できない）。"""
    rng = np.random.default_rng(0)
    group, y = rng.integers(0, G, 5000), rng.integers(0, 2, 5000)
    for batch in cell_batches(group, y, 5, rng):
        counts = np.bincount(group[batch] * 2 + y[batch], minlength=2 * G)
        assert (counts == PER_CELL).all()


def test_eo_weight_summary_reads_constraints_in_loss_order() -> None:
    """λ の並びを損失の実装と同じ順に読む。

    TPR が最も低い群を引き上げる制約だけに重みを置くと、全量がそこに数えられる。
    """
    loss = EqualizedOddsRateTaskLoss(G, [[10.0, 10.0]] * G)
    worst = 3
    lam = np.zeros(len(loss.constraints) + 1)
    lifting = [i for i, (c, _a, b) in enumerate(loss.constraints) if c == 1 and b == worst]
    lam[[i + 1 for i in lifting]] = 1 / len(lifting)
    rates = np.full((G, 2), 0.5)
    rates[worst, 1] = 0.1
    out = weight_summary("eo", {"lambda": lam.tolist(), "holdout_rates": rates.tolist()})
    assert out["worst_tpr_group"] == worst
    assert np.isclose(out["mass_lifting_worst_tpr"], 1.0)
