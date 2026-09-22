"""群別の行が、予測 cache の正しい行を見ていることを固定する。

`demographics_of` は split CSV の行番号を index に残したまま欠損行を落とす。`group_rows` は
その index をそのまま cache の添字として使う。ここがずれると、**群の割り当てが全部ずれた表**が
落ちずに出る。
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from analysis.iterative_probe.groups import GROUPING_NAMES, group_metrics, group_rows

# split CSV 8 行のうち、欠損で 4 行が落ちた状態を作る。
KEPT = [1, 2, 5, 6]


def _demographics() -> pd.DataFrame:
    """index を元の行番号のまま残した属性表。"""
    return pd.DataFrame(
        {
            "age": ["<65", ">=65", "<65", ">=65"],
            "sex": ["Male", "Male", "Female", "Female"],
            "race": ["White", "White", "White", "White"],
        },
        index=KEPT,
    )


def _model() -> dict[str, Any]:
    """落ちた行にだけ極端な値を入れた cache を持つ model。"""
    probabilities = np.tile(np.array([[0.5, 0.5]]), (8, 1))
    probabilities[[0, 3, 4, 7]] = [1.0, 0.0]
    predictions = np.zeros(8, dtype=np.int64)
    predictions[KEPT] = [1, 0, 1, 0]
    target = np.zeros(8, dtype=np.int64)
    target[KEPT] = [1, 0, 1, 0]
    cache = {"probabilities": probabilities, "predictions": predictions, "target": target}
    return {"run_id": "r", "label": "model", "modulation": "fc", "step_size": 0.001, "cache": cache}


def test_dropped_rows_never_enter_a_group() -> None:
    """落とした行の予測が混ざっていないことを、行数と値の両方で見る。"""
    rows = group_rows(_model(), _demographics())

    age_rows = [row for row in rows if row["grouping"] == "age"]
    assert sorted(row["group"] for row in age_rows) == ["<65", ">=65"]
    assert [row["n"] for row in age_rows] == [2, 2]


def test_a_group_sees_exactly_the_cache_rows_of_its_members() -> None:
    """index は split CSV の行番号であり、絞ったあとの連番ではない。"""
    model = _model()
    rows = {row["group"]: row for row in group_rows(model, _demographics()) if row["grouping"] == "sex"}
    cache = model["cache"]
    # "Male" は元の行 1 と 2。連番として読むと行 0 と 1 を見てしまう。
    index = np.array([1, 2])
    expected = group_metrics(cache["target"][index], cache["probabilities"][index, 1], cache["predictions"][index])

    assert rows["Male"]["bacc"] == expected["bacc"]
    assert rows["Male"]["tpr"] == expected["tpr"]


def test_every_grouping_is_expanded_for_each_model() -> None:
    """単独属性から 3 属性の交差まで、同じ母集団で並べるのがこの package の判断。"""
    rows = group_rows(_model(), _demographics())

    assert {row["grouping"] for row in rows} == set(GROUPING_NAMES)
    assert {row["model"] for row in rows} == {"model"}
