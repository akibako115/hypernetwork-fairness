"""Harvard-FairVLMed の画像を PNG へ展開し、train/val/test split CSV を生成する。

`data/fairvlmed/` に HF（`harvardairobotics/FairVLMed`）の `dataset.zip` と `data_summary.csv`
を置いた状態で実行し、次を書き出す。

- `images/data_XXXXX.png`: npz の `slo_fundus`（SLO 眼底画像）。`ImageDataset` は PIL で
  画像を開くので、npz のままでは読めない。既に存在する PNG は書き直さない
- `splits/{train,val,test}.csv`: CheXpert / ISIC 2019 と同じ契約（`image` / `target` /
  属性列 / `{属性}_missing`）で、`projects/*/data/splits.py` の `validate_split_frame` が通る形

**split は配布元の `use` 列（7,000 / 1,000 / 2,000）をそのまま使い、再分割しない。**
FairCLIP をはじめ先行研究の数値はすべてこの split 上のもので、比較可能性を優先する。
1 患者 1 枚（10,000 患者 10,000 枚）なので、患者単位の漏洩は起こらない。

target は `glaucoma`（yes=1）。属性の欠損は `unknown` で表されており、ISIC 2019 と同じく
`-1` を格納して `{属性}_missing` を立てる。npz 側にも同じ属性が整数で入っているので、
展開時に CSV の符号化と突き合わせ、食い違えば止める。

Usage:
    uv run python -m data_pipeline.fairvlmed_splits
    uv run python -m data_pipeline.fairvlmed_splits --skip-images  # split CSV だけ作り直す
"""

from __future__ import annotations

import argparse
import io
import zipfile
from pathlib import Path, PurePosixPath

import numpy as np
import pandas as pd
from PIL import Image

# 文字列 → 整数。npz 側の整数符号（HF の dataset card の Data Fields）と同じ値にする。
GENDER_TO_INT: dict[str, int] = {"female": 0, "male": 1}
RACE_TO_INT: dict[str, int] = {"asian": 0, "black": 1, "white": 2}
ETHNICITY_TO_INT: dict[str, int] = {"non-hispanic": 0, "hispanic": 1}
LANGUAGE_TO_INT: dict[str, int] = {"english": 0, "spanish": 1, "other": 2}
MARITALSTATUS_TO_INT: dict[str, int] = {
    "married or partnered": 0,
    "single": 1,
    "divorced": 2,
    "widowed": 3,
    "legally separated": 4,
}
CATEGORY_MAPPINGS: dict[str, dict[str, int]] = {
    "gender": GENDER_TO_INT,
    "race": RACE_TO_INT,
    "ethnicity": ETHNICITY_TO_INT,
    "language": LANGUAGE_TO_INT,
    "maritalstatus": MARITALSTATUS_TO_INT,
}
TARGET_TO_INT: dict[str, int] = {"no": 0, "yes": 1}

CATEGORICAL: tuple[str, ...] = tuple(CATEGORY_MAPPINGS)
CONTINUOUS: tuple[str, ...] = ("age",)

# 配布 CSV での欠損表記と、split CSV に格納する値。npz 側の欠損も -1 で一致する。
UNKNOWN = "unknown"
MISSING_CATEGORY = -1

USE_TO_SPLIT: dict[str, str] = {"training": "train", "validation": "val", "test": "test"}
# zip 内の directory 名。`use` 列と食い違わないことを展開時に確かめる。
ZIP_DIR_TO_SPLIT: dict[str, str] = {"Training": "train", "Validation": "val", "Test": "test"}

# 展開時に CSV の符号化と突き合わせる npz のキー
NPZ_CHECK_KEYS: tuple[str, ...] = ("glaucoma", *CATEGORICAL)

# npz の language は 2（Other）と -1（Unknown）が入れ替わっている。配布 CSV では other 76 件・
# unknown 498 件で、論文の "0.8% prefer other languages, and 5.0% remain unknown" と一致する。
# 一方 npz では -1 が 76 件・2 が 498 件になり、他の属性は全件一致する。よって CSV を正とし、
# 突き合わせでは npz の値をこの対応で読み替える。FairCLIP 公式実装は npz の -1 を unknown として
# 評価から外しているため、実際には other を外し unknown を Others 群として評価していることになる。
NPZ_LANGUAGE_TO_CSV: dict[int, int] = {2: MISSING_CATEGORY, MISSING_CATEGORY: 2}


def encode(summary: pd.DataFrame) -> pd.DataFrame:
    """`data_summary.csv` を split CSV の列へ符号化し、`split` 列を付ける。

    Args:
        summary: `data_summary.csv` の DataFrame

    Returns:
        pd.DataFrame: `image` / `target` / categorical / その missing / continuous / その missing /
            `split` の順に並べた表。`image` は `images/` からの相対 path（`.png`）

    Raises:
        ValueError: 未知のカテゴリ値・target 値・`use` 値がある場合、または age が欠損している場合。
    """
    result = pd.DataFrame({"image": summary["filename"].str.replace(r"\.npz$", ".png", regex=True)})

    unknown_target = set(summary["glaucoma"].unique()) - set(TARGET_TO_INT)
    if unknown_target:
        raise ValueError(f"glaucoma に未知の値がある: {sorted(unknown_target)}")
    result["target"] = summary["glaucoma"].map(TARGET_TO_INT).astype(int)

    for column, mapping in CATEGORY_MAPPINGS.items():
        raw = summary[column]
        unknown = set(raw.unique()) - set(mapping) - {UNKNOWN}
        if unknown:
            raise ValueError(f"{column} に未知の値がある: {sorted(unknown)}")
        result[column] = raw.map(mapping).fillna(MISSING_CATEGORY).astype(int)
    for column in CATEGORICAL:
        result[f"{column}_missing"] = (summary[column] == UNKNOWN).astype(int)

    for column in CONTINUOUS:
        # 配布データに age の欠損は無い。欠損が現れたら符号化の前提が崩れているので止める。
        if summary[column].isna().any():
            raise ValueError(f"{column} に欠損がある")
        result[column] = summary[column].astype(float)
        result[f"{column}_missing"] = 0

    unknown_use = set(summary["use"].unique()) - set(USE_TO_SPLIT)
    if unknown_use:
        raise ValueError(f"use に未知の値がある: {sorted(unknown_use)}")
    result["split"] = summary["use"].map(USE_TO_SPLIT)
    return result


def build(summary: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """split 名から split CSV の DataFrame への辞書を作る。"""
    frame = encode(summary)
    return {split: frame[frame["split"] == split].drop(columns=["split"]).reset_index(drop=True) for split in USE_TO_SPLIT.values()}


def _npz_members(archive: zipfile.ZipFile) -> dict[str, str]:
    """zip 内の npz を、ファイル名（`data_XXXXX.npz`）から member 名への辞書にする。

    Raises:
        ValueError: 同じファイル名の npz が複数の directory に入っている場合。
    """
    members: dict[str, str] = {}
    for name in archive.namelist():
        path = PurePosixPath(name)
        if path.suffix != ".npz" or path.name.startswith("._"):
            continue
        if path.name in members:
            raise ValueError(f"{path.name} が zip 内で重複している: {members[path.name]}, {name}")
        members[path.name] = name
    return members


def fundus_to_image(fundus: np.ndarray) -> Image.Image:
    """`slo_fundus` 配列を 8 bit grayscale の PIL Image にする。

    Raises:
        ValueError: 2 次元でない場合、または uint8 でない場合。
    """
    if fundus.ndim != 2:
        raise ValueError(f"slo_fundus は 2 次元を想定しているが shape={fundus.shape}")
    if fundus.dtype != np.uint8:
        raise ValueError(f"slo_fundus は uint8 を想定しているが dtype={fundus.dtype}")
    return Image.fromarray(fundus, mode="L")


def extract_images(zip_path: Path, image_dir: Path, encoded: pd.DataFrame) -> int:
    """zip 内の npz から PNG を書き出し、npz の属性と置き場所が CSV と一致することを確かめる。

    zip には npz と同じ画像の RGB JPEG（`slo_fundus_XXXXX.jpg`）も入っているが、非可逆圧縮
    なので使わず、npz の配列から可逆の PNG を作る。

    Args:
        zip_path: `dataset.zip`
        image_dir: PNG の書き出し先
        encoded: `encode` の戻り値。突き合わせの基準に使う

    Returns:
        int: 新たに書き出した PNG の枚数

    Raises:
        ValueError: CSV にある画像が zip に無い場合、npz の属性が CSV と食い違う場合、または
            npz の置かれた directory が CSV の `use` と食い違う場合。
    """
    image_dir.mkdir(parents=True, exist_ok=True)
    expected = encoded.assign(glaucoma=encoded["target"]).set_index(encoded["image"].str.replace(r"\.png$", ".npz", regex=True))
    written = 0
    with zipfile.ZipFile(zip_path) as archive:
        members = _npz_members(archive)
        absent = sorted(set(expected.index) - set(members))
        if absent:
            raise ValueError(f"zip に無い画像が {len(absent)} 件ある（先頭: {absent[:3]}）")
        for filename, row in expected.iterrows():
            directory = PurePosixPath(members[filename]).parent.name
            if ZIP_DIR_TO_SPLIT.get(directory) != row["split"]:
                raise ValueError(f"{filename} は zip の {directory}/ にあるが CSV の split は {row['split']}")
            with archive.open(members[filename]) as handle, np.load(io.BytesIO(handle.read()), allow_pickle=False) as npz:
                actual = {key: int(npz[key]) for key in NPZ_CHECK_KEYS}
                actual["language"] = NPZ_LANGUAGE_TO_CSV.get(actual["language"], actual["language"])
                wanted = {key: int(row[key]) for key in NPZ_CHECK_KEYS}
                if actual != wanted:
                    raise ValueError(f"{filename} の npz 属性が CSV と食い違う: npz={actual} csv={wanted}")
                out = image_dir / row["image"]
                if out.exists():
                    continue
                fundus_to_image(npz["slo_fundus"]).save(out)
                written += 1
    return written


def _print_breakdown(splits: dict[str, pd.DataFrame]) -> None:
    """split ごとの枚数・陽性率・race 分布を表示する。"""
    for split, part in splits.items():
        races = ", ".join(f"{name} {(part['race'] == code).sum():,}" for name, code in RACE_TO_INT.items())
        print(f"  {split:5s} {len(part):6,d}  glaucoma {part['target'].mean() * 100:4.1f}%  {races}")


def main() -> None:
    """PNG を展開して split CSV を書き出し、結果の内訳を表示する。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data/fairvlmed"))
    parser.add_argument("--skip-images", action="store_true", help="PNG の展開と npz の突き合わせを省く")
    parser.add_argument("--dry-run", action="store_true", help="何も書き出さずに内訳だけ表示する")
    args = parser.parse_args()

    summary = pd.read_csv(args.data_dir / "data_summary.csv")
    splits = build(summary)
    _print_breakdown(splits)
    if args.dry_run:
        return

    if not args.skip_images:
        written = extract_images(args.data_dir / "dataset.zip", args.data_dir / "images", encode(summary))
        print(f"PNG 書き出し: {written:,} 枚（既存分は省略）")

    output_dir = args.data_dir / "splits"
    output_dir.mkdir(parents=True, exist_ok=True)
    for split, part in splits.items():
        part.to_csv(output_dir / f"{split}.csv", index=False)
    print(f"書き出し: {output_dir}")


if __name__ == "__main__":
    main()
