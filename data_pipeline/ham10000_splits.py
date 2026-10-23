"""HAM10000 の画像を展開し、train/val/test split CSV を生成する。

`data/ham10000/raw/` に Harvard Dataverse（doi:10.7910/DVN/DBW86T, v4.0）の配布物を置いた状態で
実行し、次を書き出す。

- `images/ISIC_XXXXXXX.jpg`: HAM10000 本体 10,015 枚と ISIC2018 Task3 test 1,511 枚。ID は重ならない
  ので 1 つの directory にまとめる。既に存在する JPEG は書き直さない
- `splits/{train,val,test}.csv`: CheXpert / ISIC 2019 と同じ契約（`image` / `target` / 属性列 /
  `{属性}_missing`）で、`projects/*/data/splits.py` の `validate_split_frame` が通る形

**test は ISIC2018 Task3 の公式 test をそのまま使う**（画像の無い 1 枚だけ除く）。 HAM10000 本体とは画像も病変も重ならず
（`lesion_id` は `HAMTEST_` 系列で別管理）、病変単位の漏洩が起こらない。本体は `lesion_id` 単位で
train/val を 85/15 に分ける。層化は 7 クラスの `dx` で行い、2 値の target の比も同時に保つ。

target は MEDFAIR と同じ悪性 2 値（mel / akiec = 1、bcc は良性側）。7 クラスのままでは test の
属性別セルが 1 桁になり、群間 gap がサンプリング変動で決まるため。7 クラスのラベルは `dx` 列に残す。

train/val からは sex か age が欠損した行（本体の 67 枚）を外す。fairness の評価軸と Group DRO の group を
決める属性で、全手法が同じ学習データを使えるよう split の段階で落とす。test は全行を残し、全体の性能は
1,511 枚、fairness 指標は属性の揃った行で計算する。

属性の欠損は配布 CSV で `unknown` または空欄で表されており、ISIC 2019 と同じく categorical には
`-1` を格納して `{属性}_missing` を立てる。age の群分けは split に焼き込まず、CheXpert と同じく
config の `age_group_65`（`boundaries: [65]`）で raw の `age` から作る。

Usage:
    uv run python -m data_pipeline.ham10000_splits
    uv run python -m data_pipeline.ham10000_splits --skip-images  # split CSV だけ作り直す
"""

from __future__ import annotations

import argparse
import zipfile
from pathlib import Path, PurePosixPath

import pandas as pd

from data_pipeline.isic2019_splits import _assign_bins

# 7 クラス。アルファベット順を `dx` の整数値にする。
DX_CLASSES: tuple[str, ...] = ("akiec", "bcc", "bkl", "df", "mel", "nv", "vasc")
# 2 値 target で陽性（悪性）とするクラス。MEDFAIR（Zong et al., ICLR 2023）の公式前処理
# （notebooks/HAM10000-example.ipynb）に合わせる。MEDFAIR は Maron et al. (2019) に従い、bcc を
# 良性側に置く。臨床的には bcc も悪性腫瘍だが、先行研究との比較可能性を優先する。
MALIGNANT: frozenset[str] = frozenset({"akiec", "mel"})

SEX_TO_INT: dict[str, int] = {"female": 0, "male": 1}
# 本体と test に現れる部位をアルファベット順に並べる。
LOCALIZATION_TO_INT: dict[str, int] = {
    "abdomen": 0,
    "acral": 1,
    "back": 2,
    "chest": 3,
    "ear": 4,
    "face": 5,
    "foot": 6,
    "genital": 7,
    "hand": 8,
    "lower extremity": 9,
    "neck": 10,
    "scalp": 11,
    "trunk": 12,
    "upper extremity": 13,
}
# 撮影元。患者の属性ではないので fairness の評価軸には使わず、共変量として列だけ残す。
# `external` は test にだけ現れる HAM10000 外の施設で、age / sex の欠損がここに集中する。
DATASET_TO_INT: dict[str, int] = {"external": 0, "rosendahl": 1, "vidir_modern": 2, "vidir_molemax": 3, "vienna_dias": 4}
# 診断の根拠。histo（病理）以外はラベルの確度が下がるので、分析用に共変量として残す。
DX_TYPE_TO_INT: dict[str, int] = {"confocal": 0, "consensus": 1, "follow_up": 2, "histo": 3}

CATEGORY_MAPPINGS: dict[str, dict[str, int]] = {"sex": SEX_TO_INT, "localization": LOCALIZATION_TO_INT}
COVARIATE_MAPPINGS: dict[str, dict[str, int]] = {"dataset": DATASET_TO_INT, "dx_type": DX_TYPE_TO_INT}

CATEGORICAL: tuple[str, ...] = tuple(CATEGORY_MAPPINGS)
CONTINUOUS: tuple[str, ...] = ("age",)
# 共変量として CSV に残すだけのカテゴリ属性。`attribute_names` にも `fairness_attribute_names` にも入れない。
COVARIATE_CATEGORICAL: tuple[str, ...] = tuple(COVARIATE_MAPPINGS)

# 配布 CSV での欠損表記と、split CSV に格納する値。
UNKNOWN = "unknown"
MISSING_CATEGORY = -1

# 本体の train/val 比。test は公式 test が別にあるので、本体は学習と model selection にだけ使う。
TRAIN_VAL_FRACTIONS: dict[str, float] = {"train": 0.85, "val": 0.15}

# train/val から欠損行を外す属性。fairness の評価軸と Group DRO の group を作る属性で、欠損すると
# group が決まらない。欠損を独立 group にすると数十行の group を DRO が過大に重み付けし、run ごとに
# 欠損行の有無を変えると手法間で学習データが揃わない。test は全行を残し、fairness 指標の計算時に
# 欠損フラグで外す。
REQUIRED_TRAIN_ATTRIBUTES: tuple[str, ...] = ("sex", "age")

METADATA_CSV = "HAM10000_metadata.csv"
TEST_CSV = "ISIC2018_Task3_Test_GroundTruth.csv"
# 公式 test の GroundTruth にはあるが画像 zip に無い画像。dx 以外の metadata（dataset を含む）も
# すべて空欄で、配布時点で取り下げられたものとみなして test から外す。
EXCLUDED_TEST_IMAGES: frozenset[str] = frozenset({"ISIC_0035068"})

IMAGE_ZIPS: tuple[str, ...] = ("HAM10000_images_part_1.zip", "HAM10000_images_part_2.zip", "ISIC2018_Task3_Test_Images.zip")


def encode(metadata: pd.DataFrame) -> pd.DataFrame:
    """配布 metadata を split CSV の列へ符号化する。

    Args:
        metadata: `HAM10000_metadata.csv` または `ISIC2018_Task3_Test_GroundTruth.csv` の DataFrame

    Returns:
        pd.DataFrame: `image` / `target` / categorical / その missing / continuous / その missing /
            `dx` / 共変量 / その missing / `lesion_id` の順に並べた表。`image` は `images/` からの
            相対 path（`.jpg`）

    Raises:
        ValueError: 未知の dx・カテゴリ値がある場合。
    """
    unknown_dx = set(metadata["dx"].unique()) - set(DX_CLASSES)
    if unknown_dx:
        raise ValueError(f"dx に未知の値がある: {sorted(unknown_dx)}")
    result = pd.DataFrame({"image": metadata["image_id"] + ".jpg", "target": metadata["dx"].isin(MALIGNANT).astype(int)})

    def _categorical(column: str, mapping: dict[str, int]) -> tuple[pd.Series, pd.Series]:
        raw = metadata[column]
        missing = raw.isna() | (raw == UNKNOWN)
        unknown = set(raw[~missing].unique()) - set(mapping)
        if unknown:
            raise ValueError(f"{column} に未知の値がある: {sorted(unknown)}")
        return raw.map(mapping).where(~missing, MISSING_CATEGORY).astype(int), missing.astype(int)

    encoded = {column: _categorical(column, mapping) for column, mapping in CATEGORY_MAPPINGS.items()}
    for column in CATEGORICAL:
        result[column] = encoded[column][0]
    for column in CATEGORICAL:
        result[f"{column}_missing"] = encoded[column][1]

    for column in CONTINUOUS:
        # 連続値の欠損は 0.0 で埋め、同時に渡す missing フラグで見分ける。age は 0 が実在する
        # （本体 39 件）ため、CSV 上の値だけでは欠損と区別が付かない。
        result[column] = pd.to_numeric(metadata[column]).fillna(0.0).astype(float)
        result[f"{column}_missing"] = metadata[column].isna().astype(int)

    result["dx"] = metadata["dx"].map({name: index for index, name in enumerate(DX_CLASSES)}).astype(int)
    for column, mapping in COVARIATE_MAPPINGS.items():
        result[column], result[f"{column}_missing"] = _categorical(column, mapping)

    # lesion_id は split の group 分割の根拠として残す。属性ではないので符号化しない。
    result["lesion_id"] = metadata["lesion_id"]
    return result


def split_train_val(frame: pd.DataFrame, seed: int) -> pd.Series:
    """本体を `lesion_id` 単位・`dx` 層化で train/val へ割り当てる。

    `lesion_id` 内で dx は割れないので、group の代表ラベルで層化すれば画像単位のクラス比も保たれる。
    割り当ては ISIC 2019 と同じく、累積画像数で比率に達するまで詰める。

    Args:
        frame: `encode` の戻り値（本体）
        seed: group のシャッフルに使う乱数種

    Returns:
        pd.Series: 各行の split 名

    Raises:
        ValueError: `lesion_id` が欠けている、または 1 病変内で dx が割れている場合。
    """
    if frame["lesion_id"].isna().any():
        raise ValueError("lesion_id の無い画像がある")
    if (frame.groupby("lesion_id")["dx"].nunique() > 1).any():
        raise ValueError("1 病変内で dx が割れている")
    labels = list(TRAIN_VAL_FRACTIONS)
    return _assign_bins(frame.assign(target=frame["dx"]), frame["lesion_id"], seed, [TRAIN_VAL_FRACTIONS[label] for label in labels], labels)


def build(metadata: pd.DataFrame, test_metadata: pd.DataFrame, seed: int) -> dict[str, pd.DataFrame]:
    """split 名から split CSV の DataFrame への辞書を作る。

    Raises:
        ValueError: 本体と test で画像 ID または `lesion_id` が重なる場合。

    train/val からは `REQUIRED_TRAIN_ATTRIBUTES` のいずれかが欠損した行を外す。test は全行を残す。
    """
    frame = encode(metadata)
    test = encode(test_metadata[~test_metadata["image_id"].isin(EXCLUDED_TEST_IMAGES)].reset_index(drop=True))
    for column in ("image", "lesion_id"):
        shared = set(frame[column].dropna()) & set(test[column].dropna())
        if shared:
            raise ValueError(f"本体と test で {column} が {len(shared)} 件重なる（先頭: {sorted(shared)[:3]}）")
    # 病変の割り当ては全行で決めてから欠損行を外す。外す行の有無で他の病変の所属が動かないようにする。
    split = split_train_val(frame, seed)
    complete = ~frame[[f"{column}_missing" for column in REQUIRED_TRAIN_ATTRIBUTES]].astype(bool).any(axis=1)
    result = {name: frame[(split == name) & complete].reset_index(drop=True) for name in TRAIN_VAL_FRACTIONS}
    result["test"] = test
    return result


def extract_images(zip_paths: list[Path], image_dir: Path, expected: set[str]) -> int:
    """zip から JPEG を `image_dir` 直下へ書き出し、split に載る画像が揃っていることを確かめる。

    zip 内の directory 構成は捨て、ファイル名だけで置く。macOS の resource fork（`__MACOSX/`、
    `._*`）と JPEG 以外（test zip の説明ファイル）は無視する。

    Args:
        zip_paths: 画像 zip
        image_dir: JPEG の書き出し先
        expected: split CSV の `image` 列の値の集合

    Returns:
        int: 新たに書き出した JPEG の枚数

    Raises:
        ValueError: 同じファイル名が複数の zip / directory にある場合、または split に載る画像が
            zip に無い場合。
    """
    image_dir.mkdir(parents=True, exist_ok=True)
    members: dict[str, tuple[Path, str]] = {}
    for zip_path in zip_paths:
        with zipfile.ZipFile(zip_path) as archive:
            for name in archive.namelist():
                path = PurePosixPath(name)
                if path.suffix.lower() != ".jpg" or path.name.startswith("._") or "__MACOSX" in path.parts:
                    continue
                if path.name in members:
                    raise ValueError(f"{path.name} が zip 内で重複している: {members[path.name][1]}, {name}")
                members[path.name] = (zip_path, name)
    absent = sorted(expected - set(members))
    if absent:
        raise ValueError(f"zip に無い画像が {len(absent)} 件ある（先頭: {absent[:3]}）")

    written = 0
    for zip_path in zip_paths:
        with zipfile.ZipFile(zip_path) as archive:
            for filename, (source, name) in members.items():
                out = image_dir / filename
                if source != zip_path or filename not in expected or out.exists():
                    continue
                out.write_bytes(archive.read(name))
                written += 1
    return written


def _print_breakdown(splits: dict[str, pd.DataFrame]) -> None:
    """split ごとの枚数・悪性率・属性欠損を表示する。"""
    for split, part in splits.items():
        missing = ", ".join(f"{column} {int(part[f'{column}_missing'].sum())}" for column in (*CATEGORICAL, *CONTINUOUS))
        print(f"  {split:5s} {len(part):6,d}  病変 {part['lesion_id'].nunique():5,d}  悪性 {part['target'].mean() * 100:4.1f}%  欠損: {missing}")


def main() -> None:
    """画像を展開して split CSV を書き出し、結果の内訳を表示する。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data/ham10000"))
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument("--skip-images", action="store_true", help="JPEG の展開を省く")
    parser.add_argument("--dry-run", action="store_true", help="何も書き出さずに内訳だけ表示する")
    args = parser.parse_args()

    raw = args.data_dir / "raw"
    splits = build(pd.read_csv(raw / METADATA_CSV), pd.read_csv(raw / TEST_CSV), args.seed)
    _print_breakdown(splits)
    if args.dry_run:
        return

    if not args.skip_images:
        expected = set(pd.concat([part["image"] for part in splits.values()]))
        written = extract_images([raw / name for name in IMAGE_ZIPS], args.data_dir / "images", expected)
        print(f"JPEG 書き出し: {written:,} 枚（既存分は省略）")

    output_dir = args.data_dir / "splits"
    output_dir.mkdir(parents=True, exist_ok=True)
    for split, part in splits.items():
        part.to_csv(output_dir / f"{split}.csv", index=False)
    print(f"書き出し: {output_dir}")


if __name__ == "__main__":
    main()
