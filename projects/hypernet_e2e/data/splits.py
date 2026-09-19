"""split CSV の名前と必須スキーマに関する共有定義。

`train` / `val` / `test` という split 名と、split CSV が満たすべき列
（`image`, `target`, 属性列とその欠損フラグ）は DataModule・cohort sidecar・
cohort 生成スクリプト・分析 artifact で共通なので、定義をここ 1 箇所に置く。
"""

from collections.abc import Mapping, Sequence
from typing import Literal, get_args

import pandas as pd

from .attribute_utils import ATTRIBUTE_KINDS, attribute_columns, missing_columns

Split = Literal["train", "val", "test"]
SPLITS: tuple[Split, ...] = get_args(Split)


def required_split_columns(attribute_names: Mapping[str, Sequence[str]] | None) -> set[str]:
    """split CSV に必須の列名集合（`image` / `target` / 属性列 / 欠損フラグ列）を返す。"""
    attributes = [name for kind in ATTRIBUTE_KINDS for name in attribute_columns(attribute_names, kind)]
    return {"image", "target", *attributes, *missing_columns(attributes)}


def validate_split_frame(
    frame: pd.DataFrame,
    attribute_names: Mapping[str, Sequence[str]] | None,
    *,
    split: str,
) -> None:
    """split CSV に必須列が揃い、`image` が一意であることを検証する。

    Args:
        frame: 検証対象の split DataFrame
        attribute_names: `categorical` / `continuous` の列名
        split: エラーメッセージに出す split 名

    Raises:
        ValueError: 必須列が欠けている場合、または `image` が重複している場合。
    """
    missing = required_split_columns(attribute_names) - set(frame.columns)
    if missing:
        raise ValueError(f"{split} CSV missing columns: {sorted(missing)}")
    if not frame["image"].astype(str).is_unique:
        raise ValueError(f"{split} CSV contains duplicate image values")
