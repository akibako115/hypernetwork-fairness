"""CheXpert / CheXpert Plus の split と属性を読み込む。

split CSV は `../fairness` の `src/create_cv_chexpert.py` が生成したもので、
患者単位に分割済み。target は No Finding present を 1 とする二値ラベル
（1 = 所見なし = 正常、0 = 何らかの所見あり）。

カテゴリカル列は整数コードで保存されており、欠損は code 0 に補完されたうえで
`<attr>_missing` フラグが立つ。したがって code 0 をそのまま群として扱うと
欠損例が最大群に混ざる。`add_group_columns` は欠損を `Unknown` として分離する。
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import pandas as pd

from projects.foundation_linear_probe.paths import chexpert_root

SPLITS: tuple[str, ...] = ("train", "val", "test")

TARGET_NAME = "No Finding"
"""target=1 が意味するラベル。陽性は「所見なし」であり疾患ではない。"""

MISSING_LABEL = "Unknown"

ATTRIBUTE_LEVELS: dict[str, tuple[str, ...]] = {
    "sex": ("Male", "Female"),
    "race": ("White", "Other", "Asian", "Black", "Pacific Islander", "Native American"),
    "ethnicity": ("Non-Hispanic/Non-Latino", "Hispanic/Latino"),
    "frontal_lateral": ("Frontal", "Lateral"),
    "ap_pa": ("AP", "PA"),
    "insurance_type": ("Medicare", "Private Insurance", "Medicaid", "Other"),
    "interpreter_needed": ("No", "Yes"),
    "deceased": ("No", "Yes"),
}
"""整数コードから元のカテゴリ名への対応。出典は `../fairness/src/create_cv_chexpert.py`。"""

AGE_BIN_EDGES: tuple[float, ...] = (0.0, 40.0, 60.0, 80.0, 200.0)
AGE_BIN_LABELS: tuple[str, ...] = ("0-39", "40-59", "60-79", "80+")


def split_path(split: str, root: Path | None = None) -> Path:
    """split CSV のパスを返す。"""
    if split not in SPLITS:
        raise ValueError(f"split must be one of {SPLITS}, got {split!r}")
    return chexpert_root(root) / "splits" / f"{split}.csv"


def load_split(split: str, root: Path | None = None) -> pd.DataFrame:
    """split CSV を読み、画像の絶対パスを `image_path` 列として追加する。"""
    base = chexpert_root(root)
    df = pd.read_csv(split_path(split, root))
    missing_columns = {"image", "target"} - set(df.columns)
    if missing_columns:
        raise ValueError(f"{split}.csv lacks required columns: {sorted(missing_columns)}")
    df["split"] = split
    df["image_path"] = df["image"].map(lambda rel: str(base / "images" / rel))
    return df


def _decode_categorical(df: pd.DataFrame, attribute: str) -> pd.Series:
    """整数コード列をカテゴリ名へ復号し、欠損フラグ付きの行を `Unknown` にする。"""
    levels = ATTRIBUTE_LEVELS[attribute]
    decoded = df[attribute].map(dict(enumerate(levels)))
    if decoded.isna().any():
        unknown_codes = sorted(df.loc[decoded.isna(), attribute].unique().tolist())
        raise ValueError(f"{attribute} has codes outside {list(range(len(levels)))}: {unknown_codes}")
    missing_column = f"{attribute}_missing"
    if missing_column in df.columns:
        decoded = decoded.mask(df[missing_column].astype(int) == 1, MISSING_LABEL)
    return decoded.astype("string")


def _decode_age(df: pd.DataFrame) -> pd.Series:
    """年齢を bin へ変換する。補完値（`age_missing=1`）は `Unknown` にする。"""
    binned = pd.cut(df["age"], bins=list(AGE_BIN_EDGES), labels=list(AGE_BIN_LABELS), right=False).astype("string")
    if "age_missing" in df.columns:
        binned = binned.mask(df["age_missing"].astype(int) == 1, MISSING_LABEL)
    return binned.fillna(MISSING_LABEL)


def add_group_columns(df: pd.DataFrame, attributes: Sequence[str] | None = None) -> pd.DataFrame:
    """属性ごとに `<attr>_group` 列を追加した複製を返す。

    欠損は最大群へ補完されているため、ここで `Unknown` として分離する。
    公平性指標を計算するときは `Unknown` を独立した群として扱うか、明示的に除外する。
    """
    targets = list(attributes) if attributes is not None else [*ATTRIBUTE_LEVELS, "age"]
    out = df.copy()
    for attribute in targets:
        if attribute == "age":
            out["age_group"] = _decode_age(out)
            continue
        if attribute not in ATTRIBUTE_LEVELS:
            raise ValueError(f"unknown attribute {attribute!r}")
        if attribute not in out.columns:
            raise ValueError(f"column {attribute!r} is not present in the split")
        out[f"{attribute}_group"] = _decode_categorical(out, attribute)
    return out


def group_summary(df: pd.DataFrame, attribute: str) -> pd.DataFrame:
    """群ごとの件数・陽性数・陽性率を返す。件数不足の群を判断するために使う。"""
    column = f"{attribute}_group"
    if column not in df.columns:
        df = add_group_columns(df, [attribute])
    grouped = df.groupby(column, observed=True)["target"]
    summary = grouped.agg(n="size", n_positive="sum").reset_index()
    summary["positive_rate"] = summary["n_positive"] / summary["n"]
    return summary.sort_values("n", ascending=False, ignore_index=True)
