"""2-stage Spatial LoRA診断の分解・属性組・反実仮想を、素朴な計算と突き合わせて固定する。"""

import numpy as np
import pandas as pd
import pytest
import torch

from analysis.archive.race_known_architecture_objective_comparison import spatial_lora_inspection as inspection
from projects.hypernet_e2e.models.spatial_hypernet.blocks import SpatialLoRAConv

ATTRIBUTE_NAMES = {"categorical": ["sex", "race"], "continuous": ["age"]}


def _adapter(seed=0):
    torch.manual_seed(seed)
    adapter = SpatialLoRAConv(3, torch.nn.Conv2d(2, 3, 3, bias=False), rank=2)
    # 既定のA=0では差分が全部0になり、分解を検算できない。
    for parameter in adapter.parameters():
        torch.nn.init.normal_(parameter)
    return adapter


def test_delta_decomposition_matches_materialized_moments():
    adapter, base = _adapter(), torch.randn(3, 2, 3, 3)
    condition = torch.randn(7, 3)
    weights = np.array([1, 2, 3, 1, 5, 2, 1])
    groups = np.array([0, 0, 1, 1, 2, -1, 2])
    row, relative, deviations = inspection.delta_decomposition(
        adapter, base, condition, weights, {"group": groups}, chunk=3
    )
    with torch.no_grad():
        delta = adapter.weight_delta(condition).flatten(1).double().numpy()
    base_sq = (base.double() ** 2).sum().item()
    mean = (weights[:, None] * delta).sum(0) / weights.sum()
    np.testing.assert_allclose(relative, np.sqrt((delta**2).sum(1) / base_sq), rtol=1e-6)
    assert row["rms_rel"] ** 2 == pytest.approx(row["shared_rel"] ** 2 + row["varying_rel"] ** 2, rel=1e-6)
    assert row["shared_rel"] == pytest.approx(np.sqrt((mean**2).sum() / base_sq), rel=1e-6)
    # 欠損群（-1）はμ_gを作らず、群間項の重みからも外す。全体平均μには含める。
    kept = groups >= 0
    between = 0.0
    for group in (0, 1, 2):
        mask = groups == group
        centroid = (weights[mask, None] * delta[mask]).sum(0) / weights[mask].sum()
        np.testing.assert_allclose(deviations["group"][group].numpy(), centroid - mean, rtol=1e-5, atol=1e-6)
        between += weights[mask].sum() * ((centroid - mean) ** 2).sum()
    assert set(deviations["group"]) == {0, 1, 2}
    assert row["between_group_rel"] == pytest.approx(np.sqrt(between / weights[kept].sum() / base_sq), rel=1e-6)


def test_constant_condition_has_no_attribute_specific_part():
    adapter, base = _adapter(1), torch.randn(3, 2, 3, 3)
    condition = torch.randn(1, 3).repeat(4, 1)
    row, _, _ = inspection.delta_decomposition(adapter, base, condition, np.ones(4), {"group": np.arange(4)})
    assert row["varying_rel"] == pytest.approx(0, abs=1e-6)
    assert row["between_group_rel"] == pytest.approx(0, abs=1e-6)
    assert row["shared_rel"] == pytest.approx(row["rms_rel"], rel=1e-6)


def test_unique_conditions_weight_by_rows_and_keep_groups():
    frame = pd.DataFrame(
        {
            "sex": [0, 0, 1, 0],
            "race": [0, 0, 2, 0],
            "age": [0.5, 0.5, -1.0, 0.7],
            "sex_missing": False,
            "race_missing": False,
            "age_missing": False,
            "group_id": [4, 4, 1, 4],
            "race_group": [0, 0, 1, 0],
        }
    )
    table = inspection.unique_conditions(frame, ATTRIBUTE_NAMES)
    assert table["count"].sum() == len(frame) and len(table) == 3
    assert table.set_index("age").loc[0.5, "count"] == 2
    assert table.set_index("age").loc[-1.0, ["group_id", "race_group", "age_group"]].tolist() == [1, 1, 0]
    frame.loc[1, "group_id"] = 0
    with pytest.raises(ValueError, match="8群"):
        inspection.unique_conditions(frame, ATTRIBUTE_NAMES)


def test_counterfactuals_change_only_the_named_attribute():
    attributes = {
        "categorical": torch.tensor([[1, 5], [0, 2]]),
        "categorical_missing": torch.tensor([[False, True], [False, False]]),
        "continuous": torch.tensor([[0.3], [-0.2]]),
        "continuous_missing": torch.tensor([[True], [False]]),
    }
    variants = inspection.counterfactual_attributes(attributes, race_column=1, standardize_age=lambda age: age / 100)
    assert list(variants) == [*inspection.RACE_CODES, *inspection.COUNTERFACTUAL_AGES]
    black = variants["Black"]
    assert black["categorical"][:, 1].tolist() == [3, 3] and black["categorical"][:, 0].tolist() == [1, 0]
    assert not black["categorical_missing"][:, 1].any() and black["categorical_missing"][0, 0].item() is False
    assert torch.equal(black["continuous"], attributes["continuous"])
    old = variants["age 75"]
    assert old["continuous"].flatten().tolist() == pytest.approx([0.75, 0.75])
    assert not old["continuous_missing"].any() and torch.equal(old["categorical"], attributes["categorical"])
    # 元の属性dictは書き換えない。
    assert attributes["categorical"][0, 1].item() == 5 and attributes["categorical_missing"][0, 1].item()
