"""split CSV の属性復号に関するテスト。"""

from __future__ import annotations

import pandas as pd
import pytest

from projects.foundation_linear_probe.chexpert import (
    MISSING_LABEL,
    add_group_columns,
    group_summary,
)


def _frame() -> pd.DataFrame:
    """欠損が code 0 へ補完された状態を再現した最小の split。"""
    return pd.DataFrame(
        {
            "image": ["a.jpg", "b.jpg", "c.jpg", "d.jpg"],
            "target": [1, 0, 0, 1],
            "sex": [0, 1, 0, 1],
            "race": [0, 3, 0, 2],
            "race_missing": [0, 0, 1, 0],
            "age": [25.0, 55.0, 61.0, 90.0],
            "age_missing": [0, 0, 1, 0],
        }
    )


def test_missing_flag_is_separated_from_the_imputed_majority_group() -> None:
    """欠損を補完先の White に混ぜず Unknown として分離する。"""
    grouped = add_group_columns(_frame(), ["race"])
    assert grouped["race_group"].tolist() == ["White", "Black", MISSING_LABEL, "Asian"]


def test_age_is_binned_and_missing_ages_are_not_binned() -> None:
    """年齢 bin は左閉区間で、補完値は Unknown になる。"""
    grouped = add_group_columns(_frame(), ["age"])
    assert grouped["age_group"].tolist() == ["0-39", "40-59", MISSING_LABEL, "80+"]


def test_unknown_category_code_raises() -> None:
    """定義にないコードは黙って落とさず例外にする。"""
    frame = _frame()
    frame.loc[0, "race"] = 9
    with pytest.raises(ValueError, match="race has codes outside"):
        add_group_columns(frame, ["race"])


def test_group_summary_reports_size_and_positive_rate() -> None:
    """件数不足の群を判断できるよう、群ごとの n と陽性率を返す。"""
    summary = group_summary(_frame(), "sex").set_index("sex_group")
    assert summary.loc["Male", "n"] == 2
    assert summary.loc["Male", "n_positive"] == 1
    assert summary.loc["Female", "positive_rate"] == pytest.approx(0.5)
