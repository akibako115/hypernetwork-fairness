"""ΔW の Gram・二元の主効果の除去・セル内の一致率を、素朴な計算と突き合わせて固定する。"""

import numpy as np
import torch
from sklearn.metrics import roc_auc_score

from analysis.archive.conditioned_subgroup_discovery import delta_probe, rule_tests
from projects.hypernet_e2e.models.spatial_hypernet.blocks import SpatialLoRAConv


def test_factor_gram_matches_materialized_delta():
    torch.manual_seed(0)
    layer = SpatialLoRAConv(5, torch.nn.Conv2d(3, 4, 3, bias=False), rank=2)
    # 既定の A=0 では ΔW が全部 0 になり、検算にならない。
    for parameter in layer.parameters():
        torch.nn.init.normal_(parameter)
    condition = torch.randn(6, 5)
    with torch.no_grad():
        gram = delta_probe._factor_gram(layer, condition).numpy()
        delta = layer.weight_delta(condition).flatten(1).double().numpy()
    # weight_delta は float32 で計算するので、その丸めの分だけ許す。
    np.testing.assert_allclose(gram, delta @ delta.T, rtol=1e-6)


def test_two_way_centered_is_weighted_fixed_effects_residual():
    rng = np.random.default_rng(0)
    values = rng.normal(size=(5, 4))
    weights = rng.integers(1, 9, size=5).astype(float)
    # 行・列のダミーへの重み付き最小二乗の残差と一致すること（全体平均の扱いを誤ると T 倍ずれる）。
    rows, columns = np.meshgrid(np.arange(5), np.arange(4), indexing="ij")
    design = np.concatenate([np.eye(5)[rows.ravel()], np.eye(4)[columns.ravel()][:, 1:]], 1)
    w = np.sqrt(np.repeat(weights, 4))
    coef, *_ = np.linalg.lstsq(design * w[:, None], values.ravel() * w, rcond=None)
    residual = (values.ravel() - design @ coef).reshape(5, 4)
    np.testing.assert_allclose(rule_tests.two_way_centered(values, weights), residual, atol=1e-10)


def test_within_cell_counts_pool_only_same_cell_pairs():
    rng = np.random.default_rng(1)
    cells = np.array([0, 0, 0, 1, 1, 1, 1, 2])
    target = np.array([1, 0, 0, 1, 1, 0, 0, 1])
    margins = rng.normal(size=(3, 8))
    margins[:, 1] = margins[:, 0]  # 同点は 0.5 と数える
    concordant, pairs = rule_tests.within_cell_counts(margins, cells, target)
    np.testing.assert_array_equal(pairs, [2, 4, 0])
    for t in range(3):
        for cell in (0, 1):
            rows = cells == cell
            expected = roc_auc_score(target[rows], margins[t, rows]) * pairs[cell]
            assert concordant[cell, t] == expected
    # セル内の全員に同じ切片を足しても一致数は変わらない。
    shifted = margins + np.where(cells == 1, 5.0, 0.0)
    np.testing.assert_array_equal(rule_tests.within_cell_counts(shifted, cells, target)[0], concordant)
