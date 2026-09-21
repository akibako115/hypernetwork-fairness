"""ISIC 2019 の train/val/test split CSV を生成する。

`data/isic2019/splits_fairness/{train,val,test}.csv` を書き出す。旧 pipeline の `splits/` とは
別物で、そちらは上書きしない。列は CheXpert の split と同じ契約（`image` / `target` / 属性列 /
`{属性}_missing`）で、`projects/*/data/splits.py` の `validate_split_frame` が
そのまま通る形にする。

CheXpert に倣い 80/10/10 で分け、**同一病変の画像が split をまたがないよう `lesion_id` で
group 分割**する。CheXpert が患者単位で分離しているのと同じ理由で、同じ病変の別アングルが
train と test に散ると test が楽観側へ歪む。`lesion_id` を持たない画像は、同一病変である
証拠が無いので1枚ずつ独立した group として扱う（この 2,084 枚の中に同一病変の重複が
含まれる可能性は残る）。

層化はクラスで行う。`lesion_id` 内でラベルが割れる病変は存在しないため、group 単位の
層化がそのまま画像単位のクラス比を保つ。

`--cv` は同じ group 分割で 5-fold を作り、`splits_fairness_cv/fold{k}/` へ書き出す。稀少クラス
（DF 239 枚 / VASC 253 枚）は単一 test split（10% = 24 枚前後）だと属性で割った時点で群あたり
10 枚程度になり、群間 gap がサンプリング変動に埋もれる。全画像が 1 回ずつ test に回る CV なら
プールした評価セルが実数のまま使える。fold 内は 3:1:1（train 3 fold / val 1 fold / test 1 fold）で、
ISIC 文献の 6:2:2 に一致する。単一 split の 80/10/10 は MEDFAIR の皮膚科データセットに合わせた値。

Usage:
    uv run python -m data_pipeline.isic2019_splits
    uv run python -m data_pipeline.isic2019_splits --cv
"""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

# ISIC 2019 の 8 クラス。GroundTruth CSV の one-hot 列順をそのまま target の整数値にする。
CLASSES: tuple[str, ...] = ("MEL", "NV", "BCC", "AK", "BKL", "DF", "VASC", "SCC")

# 文字列 → 整数。旧 repo の scripts/encode_isic2019_splits.py と同じアルファベット順を保つ。
SITE_TO_INT: dict[str, int] = {
    "anterior torso": 0,
    "head/neck": 1,
    "lateral torso": 2,
    "lower extremity": 3,
    "oral/genital": 4,
    "palms/soles": 5,
    "posterior torso": 6,
    "upper extremity": 7,
}
SEX_TO_INT: dict[str, int] = {"female": 0, "male": 1}

# 元データセット。archive の attribution 文字列がそのまま BCN20000 / HAM10000 / MSK に対応する。
# 患者の属性ではないので fairness の評価軸には使わず、色かぶりと肌色を切り分けるための
# 共変量として持つ。archive で引けない画像（MSK の一部）は欠損になる。
ATTRIBUTION_TO_INT: dict[str, int] = {
    "Anonymous": 0,
    "Hospital Clínic de Barcelona": 1,
    "MILK study team": 2,
}

CATEGORICAL: tuple[str, ...] = ("sex", "anatom_site_general")
CONTINUOUS: tuple[str, ...] = ("age_approx",)

# 共変量として CSV に残すだけのカテゴリ属性。`attribute_names` にも
# `fairness_attribute_names` にも入れない。
COVARIATE_CATEGORICAL: tuple[str, ...] = ("attribution",)

# 欠損 categorical の格納値。MetadataEncoder は `categorical_missing` フラグで
# 埋め込み index を差し替えるので値自体は使われないが、実カテゴリと取り違えられない
# 値を入れて、生の CSV を読む側が欠損と分かるようにする。
MISSING_CATEGORY = -1

# 単一 split の比率。MEDFAIR が皮膚科データセット（HAM10000 / Fitzpatrick17k）で使う 80/10/10 に合わせる。
SPLIT_FRACTIONS: dict[str, float] = {"train": 0.8, "val": 0.1, "test": 0.1}

# CV の fold 数。ISIC 2019 チャレンジ優勝の Gessert et al. と同じ 5-fold。
DEFAULT_FOLDS = 5


def load_source(data_dir: Path) -> pd.DataFrame:
    """ラベル・属性・元データセットを画像単位の1枚の表にまとめる。

    Args:
        data_dir: `data/isic2019`

    Returns:
        pd.DataFrame: `image` / `target` / 属性 / `lesion_id` / `attribution` を持つ表

    Raises:
        ValueError: GroundTruth の one-hot が排他でない場合、または UNK を含む場合。
    """
    truth = pd.read_csv(data_dir / "ISIC_2019_Training_GroundTruth.csv")
    if int(truth["UNK"].sum()):
        raise ValueError("UNK を持つ画像は 8 クラス分類に含められない")
    if not (truth[[*CLASSES, "UNK"]].sum(axis=1) == 1).all():
        raise ValueError("GroundTruth の one-hot が排他ではない")
    labels = pd.DataFrame({"image": truth["image"], "target": truth[list(CLASSES)].to_numpy().argmax(axis=1)})

    meta = pd.read_csv(data_dir / "ISIC_2019_Training_Metadata.csv")
    frame = labels.merge(meta[["image", "sex", "age_approx", "anatom_site_general", "lesion_id"]], on="image", how="left")

    # attribution（元データセット）は共変量として持つ。2019 の metadata には無いので archive 側から引く。
    archive = pd.read_csv(data_dir / "isic_all_metadata.csv", low_memory=False, usecols=["isic_id", "attribution"])
    frame = frame.merge(archive.rename(columns={"isic_id": "image"}), on="image", how="left")
    return frame


def encode_attributes(frame: pd.DataFrame) -> pd.DataFrame:
    """属性を整数・実数へ符号化し、`{属性}_missing` フラグを付ける。

    Args:
        frame: `load_source` の戻り値

    Returns:
        pd.DataFrame: split CSV の列順（image / target / categorical / その missing /
            continuous / その missing）に並べた表。`lesion_id` と `attribution` も残す

    Raises:
        ValueError: 未知の sex / anatom_site_general の値がある場合。
    """
    result = frame[["image", "target"]].copy()

    for column, mapping in (("sex", SEX_TO_INT), ("anatom_site_general", SITE_TO_INT)):
        raw = frame[column]
        unknown = set(raw.dropna().unique()) - set(mapping)
        if unknown:
            raise ValueError(f"{column} に未知の値がある: {sorted(unknown)}")
        result[column] = raw.map(mapping).fillna(MISSING_CATEGORY).astype(int)
    for column in CATEGORICAL:
        result[f"{column}_missing"] = frame[column].isna().astype(int)

    for column in CONTINUOUS:
        # 連続値の欠損は 0.0 で埋め、同時に渡す missing フラグで見分ける。
        # `age_approx` は 0 が実在する（乳児 54 件）ため、CSV 上は欠損と区別が付かない。
        # `standardize_continuous_columns` は欠損行も同じ式で標準化するので、両者は
        # 標準化後も同じ値になる。見分けるのは `age_approx_missing` だけである。
        result[column] = frame[column].fillna(0.0).astype(float)
        result[f"{column}_missing"] = frame[column].isna().astype(int)

    raw_attribution = frame["attribution"]
    unknown = set(raw_attribution.dropna().unique()) - set(ATTRIBUTION_TO_INT)
    if unknown:
        raise ValueError(f"attribution に未知の値がある: {sorted(unknown)}")
    result["attribution"] = raw_attribution.map(ATTRIBUTION_TO_INT).fillna(MISSING_CATEGORY).astype(int)
    result["attribution_missing"] = raw_attribution.isna().astype(int)

    # lesion_id は split の group 分割の根拠として残す。属性ではないので符号化しない。
    result["lesion_id"] = frame["lesion_id"]
    return result


def assign_groups(frame: pd.DataFrame) -> pd.Series:
    """split 分割の単位となる group ID を返す。

    `lesion_id` がある画像は同じ病変を1 group にまとめ、無い画像は1枚ずつ独立させる。
    """
    lesion = frame["lesion_id"]
    return lesion.where(lesion.notna(), pd.Series([f"__image_{image}" for image in frame["image"]], index=frame.index))


def _assign_bins(frame: pd.DataFrame, groups: pd.Series, seed: int, fractions: Sequence[float], labels: Sequence[Any]) -> pd.Series:
    """group をクラス層化して、指定比率の bin へ割り当てる。

    group 内でラベルは一意なので、group の代表ラベルで層化すれば画像単位のクラス比も保たれる。
    group の大きさは 1〜31 枚とばらつくため、各クラス内で group をシャッフルしてから
    **累積画像数**が目標比に達するまで先頭の bin から順に詰める。group 数ではなく画像数で
    切ることで、大きな group が偏っても比率が崩れにくい。

    Args:
        frame: `encode_attributes` の戻り値
        groups: `assign_groups` の戻り値
        seed: group のシャッフルに使う乱数種
        fractions: bin ごとの目標比。合計 1.0
        labels: bin の名前。fractions と同じ長さ

    Returns:
        pd.Series: 各行の bin ラベル
    """
    rng = np.random.default_rng(seed)
    assignment: dict[str, Any] = {}
    table = pd.DataFrame({"group": groups, "target": frame["target"]})
    for _, class_rows in table.groupby("target", sort=True):
        sizes = class_rows.groupby("group").size()
        order = rng.permutation(sizes.index.to_numpy())
        total = int(sizes.sum())
        # 最後の bin は余りを受けるので境界を持たない
        boundaries = np.cumsum(fractions[:-1]) * total
        cumulative = 0
        for group in order:
            index = int(np.searchsorted(boundaries, cumulative, side="right"))
            assignment[group] = labels[index]
            cumulative += int(sizes[group])
    return groups.map(assignment)


def split_groups(frame: pd.DataFrame, groups: pd.Series, seed: int) -> pd.Series:
    """group をクラス層化して train/val/test へ割り当てる。

    Args:
        frame: `encode_attributes` の戻り値
        groups: `assign_groups` の戻り値
        seed: group のシャッフルに使う乱数種

    Returns:
        pd.Series: 各行の split 名
    """
    labels = list(SPLIT_FRACTIONS)
    return _assign_bins(frame, groups, seed, [SPLIT_FRACTIONS[label] for label in labels], labels)


def assign_folds(frame: pd.DataFrame, groups: pd.Series, seed: int, folds: int) -> pd.Series:
    """group をクラス層化して `0..folds-1` の fold へ均等に割り当てる。

    Args:
        frame: `encode_attributes` の戻り値
        groups: `assign_groups` の戻り値
        seed: group のシャッフルに使う乱数種
        folds: fold 数

    Returns:
        pd.Series: 各行の fold 番号
    """
    return _assign_bins(frame, groups, seed, [1.0 / folds] * folds, list(range(folds)))


def build(data_dir: Path, seed: int) -> dict[str, pd.DataFrame]:
    """split 名から split CSV の DataFrame への辞書を作る。"""
    frame = encode_attributes(load_source(data_dir))
    groups = assign_groups(frame)
    frame["split"] = split_groups(frame, groups, seed)
    frame["image"] = frame["image"] + ".jpg"
    return {split: part.drop(columns=["split"]).reset_index(drop=True) for split, part in frame.groupby("split")}


def build_cv(data_dir: Path, seed: int, folds: int = DEFAULT_FOLDS) -> dict[int, dict[str, pd.DataFrame]]:
    """fold 番号から split CSV の DataFrame への辞書を作る。

    fold k は test に fold k、val に fold k+1（巡回）、train に残りを使う。全画像がちょうど
    1 回ずつ test に回り、fold 内の比率は 3:1:1 になる。

    Args:
        data_dir: `data/isic2019`
        seed: group のシャッフルに使う乱数種
        folds: fold 数

    Returns:
        dict[int, dict[str, pd.DataFrame]]: fold 番号 → split 名 → DataFrame

    Raises:
        ValueError: folds が 3 未満の場合（train / val / test を分けられない）。
    """
    if folds < 3:
        raise ValueError(f"folds は 3 以上が必要だが {folds} が指定された")
    frame = encode_attributes(load_source(data_dir))
    fold_of = assign_folds(frame, assign_groups(frame), seed, folds)
    frame["image"] = frame["image"] + ".jpg"
    result: dict[int, dict[str, pd.DataFrame]] = {}
    for fold in range(folds):
        membership = {"test": fold_of == fold, "val": fold_of == (fold + 1) % folds}
        membership["train"] = ~(membership["test"] | membership["val"])
        result[fold] = {split: frame[mask].reset_index(drop=True) for split, mask in membership.items()}
    return result


def _print_breakdown(splits: dict[str, pd.DataFrame], label: str) -> None:
    """split ごとの枚数とクラス比を表示する。"""
    total = sum(len(part) for part in splits.values())
    print(f"{label} 画像 {total:,} 枚")
    for split in ("train", "val", "test"):
        part = splits[split]
        shares = ", ".join(f"{CLASSES[i]} {(part['target'] == i).mean() * 100:4.1f}%" for i in range(len(CLASSES)))
        print(f"  {split:5s} {len(part):6,d} ({len(part) / total * 100:4.1f}%)  {shares}")


def _write(splits: dict[str, pd.DataFrame], output_dir: Path) -> None:
    """split CSV を書き出す。"""
    output_dir.mkdir(parents=True, exist_ok=True)
    for split, part in splits.items():
        part.to_csv(output_dir / f"{split}.csv", index=False)
    print(f"書き出し: {output_dir}")


def main() -> None:
    """split CSV を書き出し、結果の内訳を表示する。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data/isic2019"))
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument("--cv", action="store_true", help="単一 split ではなく k-fold を作る")
    parser.add_argument("--folds", type=int, default=DEFAULT_FOLDS, help="--cv のときの fold 数")
    parser.add_argument("--dry-run", action="store_true", help="書き出さずに内訳だけ表示する")
    args = parser.parse_args()

    if not args.cv:
        splits = build(args.data_dir, args.seed)
        _print_breakdown(splits, "単一 split")
        if not args.dry_run:
            _write(splits, args.data_dir / "splits_fairness")
        return

    for fold, splits in build_cv(args.data_dir, args.seed, args.folds).items():
        _print_breakdown(splits, f"fold {fold}")
        if not args.dry_run:
            _write(splits, args.data_dir / "splits_fairness_cv" / f"fold{fold}")


if __name__ == "__main__":
    main()
