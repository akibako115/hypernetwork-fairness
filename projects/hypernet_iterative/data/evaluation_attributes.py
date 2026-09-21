"""fairness 評価専用のカテゴリ属性を split DataFrame に追加する。"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
import pandas as pd


def add_age_groups(frame: pd.DataFrame, definitions: Mapping[str, Mapping[str, Any]] | None) -> pd.DataFrame:
    """連続年齢と境界値から評価専用の年齢群列・欠損 flag を作って返す。

    各 definition は ``source`` と昇順の ``boundaries`` を持つ。境界 ``[65]`` のとき、
    source が 65 未満なら 0、65 以上なら 1 とする。source の ``*_missing`` が真、または
    値が NaN の行は、値を 0 として対応する ``<name>_missing`` を真にする。
    """
    if not definitions:
        return frame
    result = frame.copy()
    for name, definition in definitions.items():
        source, boundaries = _validate_definition(name, definition)
        missing_name = f"{source}_missing"
        if source not in result or missing_name not in result:
            raise ValueError(f"年齢群 {name!r} の source {source!r} と {missing_name!r} が split に必要")
        values = pd.to_numeric(result[source], errors="raise")
        missing = result[missing_name].astype(bool) | values.isna()
        groups = np.searchsorted(np.asarray(boundaries, dtype=float), values.fillna(0.0).to_numpy(), side="right")
        result[name] = groups.astype("int64")
        result[f"{name}_missing"] = missing
    return result


def _validate_definition(name: str, definition: Mapping[str, Any]) -> tuple[str, list[float]]:
    source = definition.get("source")
    boundaries = definition.get("boundaries")
    if not isinstance(source, str) or not source:
        raise ValueError(f"年齢群 {name!r} の source は空でない列名である必要がある")
    if not isinstance(boundaries, Sequence) or isinstance(boundaries, str) or not boundaries:
        raise ValueError(f"年齢群 {name!r} の boundaries は1個以上の数値境界である必要がある")
    numeric_boundaries = [float(value) for value in boundaries]
    if numeric_boundaries != sorted(set(numeric_boundaries)):
        raise ValueError(f"年齢群 {name!r} の boundaries は重複しない昇順である必要がある")
    return source, numeric_boundaries
