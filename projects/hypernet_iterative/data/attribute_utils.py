"""属性列の抽出・テンソル化・標準化・spec 検証を担う共有ヘルパ。

`categorical` / `continuous` とその欠損フラグという属性の contract は、
Dataset・cohort 生成スクリプト・分析 artifact のすべてがこのモジュールを経由して扱う。
"""

from collections.abc import Iterable, Mapping, Sequence
from typing import Any, Literal

import numpy as np
import pandas as pd
import torch

# 属性の種別と、値テンソル / 値配列に使う dtype。欠損フラグは常に bool。
# このキー集合は spatial_hypernet の属性条件付きモデルの入力契約と一致させる。
AttributeKind = Literal["categorical", "continuous"]
ATTRIBUTE_KINDS: tuple[AttributeKind, ...] = ("categorical", "continuous")
_VALUE_DTYPES: dict[AttributeKind, str] = {"categorical": "int64", "continuous": "float32"}
_TENSOR_DTYPES: dict[AttributeKind, torch.dtype] = {"categorical": torch.long, "continuous": torch.float32}


def missing_columns(columns: Iterable[str]) -> list[str]:
    """属性列名から対応する欠損フラグ列名（`{col}_missing`）のリストを作る。"""
    return [f"{column}_missing" for column in columns]


def attribute_columns(attribute_names: Mapping[str, Sequence[str]] | None, kind: AttributeKind) -> list[str]:
    """attribute_names から指定種別の列名リストを取り出す。未指定なら空リスト。"""
    if attribute_names is None:
        return []
    return list(attribute_names.get(kind, []))


def categorical_columns(attribute_names: Mapping[str, Sequence[str]] | None) -> list[str]:
    """attribute_names から categorical 列名のリストを取り出す。"""
    return attribute_columns(attribute_names, "categorical")


def continuous_columns(attribute_names: Mapping[str, Sequence[str]] | None) -> list[str]:
    """attribute_names から continuous 列名のリストを取り出す。"""
    return attribute_columns(attribute_names, "continuous")


def attribute_arrays(
    source: pd.DataFrame,
    attribute_names: Mapping[str, Sequence[str]] | None,
    *,
    include_empty: bool = False,
) -> dict[str, np.ndarray]:
    """属性列と欠損フラグ列を `[n_rows, n_attributes]` の numpy 配列へまとめる。

    Args:
        source: `image` / `target` と属性列・欠損フラグ列を持つ DataFrame
        attribute_names: `categorical` / `continuous` の列名
        include_empty: 列が空の種別も `[n_rows, 0]` の配列として含めるか。
            npz へ固定スキーマで保存する用途では True にする。

    Returns:
        dict[str, np.ndarray]: `categorical` / `categorical_missing` /
            `continuous` / `continuous_missing` のうち、対象列を持つものだけの配列。
    """
    result: dict[str, np.ndarray] = {}
    # 種別ごとに値列と欠損フラグ列を切り出し、固定 dtype の 2 次元配列へ変換する
    for kind in ATTRIBUTE_KINDS:
        columns = attribute_columns(attribute_names, kind)
        if not columns:
            if include_empty:
                result[kind] = np.empty((len(source), 0), dtype=_VALUE_DTYPES[kind])
                result[f"{kind}_missing"] = np.empty((len(source), 0), dtype="bool")
            continue
        result[kind] = source[columns].to_numpy(dtype=_VALUE_DTYPES[kind])
        result[f"{kind}_missing"] = source[missing_columns(columns)].to_numpy(dtype="bool")
    return result


def attribute_tensors(
    source: pd.DataFrame | pd.Series,
    attribute_names: Mapping[str, Sequence[str]] | None,
) -> dict[str, torch.Tensor]:
    """属性列と欠損フラグ列をモデル入力の属性 dict へ変換する。

    欠損（NaN）の補完はデータ前処理側で行うため、この関数のスコープ外とする。

    Args:
        source: 属性列と欠損フラグ列を持つ DataFrame（batch 単位）または Series（1 行分）
        attribute_names: `categorical` / `continuous` の列名。None なら空 dict を返す

    Returns:
        dict[str, torch.Tensor]: `categorical`（`torch.long`）/ `continuous`（`torch.float32`）と
            対応する `*_missing`（`torch.bool`）。DataFrame を渡した場合は
            `[n_rows, n_attributes]`、Series を渡した場合は `[n_attributes]` になる。
    """
    result: dict[str, torch.Tensor] = {}
    # 種別ごとに値列と欠損フラグ列をテンソル化する（categorical があれば必ず categorical_missing も作る）
    for kind in ATTRIBUTE_KINDS:
        columns = attribute_columns(attribute_names, kind)
        if not columns:
            continue
        result[kind] = torch.as_tensor(source[columns].to_numpy(dtype=_VALUE_DTYPES[kind]), dtype=_TENSOR_DTYPES[kind])
        result[f"{kind}_missing"] = torch.as_tensor(source[missing_columns(columns)].to_numpy(dtype="bool"), dtype=torch.bool)
    return result


def standardize_continuous_columns(
    train_df: pd.DataFrame,
    *split_dfs: pd.DataFrame,
    columns: Sequence[str],
) -> tuple[pd.DataFrame, ...]:
    """
    train split の非欠損行の統計量で連続属性を標準化する。

    Args:
        train_df: 統計量（mean/std）の算出元となる train split
        *split_dfs: 標準化対象の DataFrame（train 自身を含めてよい）
        columns: 標準化する連続属性の列名

    Returns:
        tuple[pd.DataFrame, ...]: split_dfs と同じ順序・件数で標準化した DataFrame
    """
    if not columns:
        return tuple(split_df.copy() for split_df in split_dfs)

    means: dict[str, float] = {}
    stds: dict[str, float] = {}
    # train split の非欠損値から列ごとの mean/std を算出する
    for col in columns:
        missing_col = f"{col}_missing"
        train_values = pd.to_numeric(train_df[col], errors="raise")
        valid_train_values = train_values.mask(train_df[missing_col].astype(bool))
        mean = valid_train_values.mean()
        std = valid_train_values.std(ddof=0)
        if pd.isna(mean) or pd.isna(std):
            raise ValueError(f"continuous 属性 {col} に train split の非欠損値が存在しない")
        means[col] = float(mean)
        stds[col] = 1.0 if std == 0 else float(std)

    standardized_dfs = []
    # 算出した mean/std で各 split の対象列を標準化する
    for split_df in split_dfs:
        out = split_df.copy()
        for col in columns:
            out[col] = (pd.to_numeric(out[col], errors="raise") - means[col]) / stds[col]
        standardized_dfs.append(out)
    return tuple(standardized_dfs)


def validate_attribute_spec(
    attribute_names: Mapping[str, Sequence[str]] | None,
    attribute_spec: Mapping[str, Any] | None,
) -> None:
    """attribute_spec の cardinality 数・continuous_dim が attribute_names と要素数一致するか検証する。"""
    if attribute_spec is None or attribute_names is None:
        return
    categorical_names = categorical_columns(attribute_names)
    cardinalities = list(attribute_spec.get("categorical_cardinalities", []))
    if len(cardinalities) != len(categorical_names):
        raise ValueError(f"attribute_spec.categorical_cardinalities の要素数（{len(cardinalities)}）が attribute_names.categorical の要素数（{len(categorical_names)}）と一致しない")
    continuous_names = continuous_columns(attribute_names)
    continuous_dim = attribute_spec.get("continuous_dim", 0)
    if continuous_dim != len(continuous_names):
        raise ValueError(f"attribute_spec.continuous_dim（{continuous_dim}）が attribute_names.continuous の要素数（{len(continuous_names)}）と一致しない")
