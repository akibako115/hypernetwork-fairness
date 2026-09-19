"""fairness 評価専用の年齢群生成を検証する。"""

import pandas as pd
import pytest

from projects.hypernet_e2e.data.evaluation_attributes import add_age_groups


def test_add_age_groups_uses_raw_age_boundary_and_preserves_missingness() -> None:
    frame = pd.DataFrame({"age": [64.0, 65.0, float("nan")], "age_missing": [False, False, True]})

    actual = add_age_groups(frame, {"age_group_65": {"source": "age", "boundaries": [65]}})

    assert actual["age_group_65"].tolist() == [0, 1, 0]
    assert actual["age_group_65_missing"].tolist() == [False, False, True]


def test_add_age_groups_rejects_unsorted_boundaries() -> None:
    frame = pd.DataFrame({"age": [50.0], "age_missing": [False]})

    with pytest.raises(ValueError, match="昇順"):
        add_age_groups(frame, {"age_group": {"source": "age", "boundaries": [65, 40]}})
