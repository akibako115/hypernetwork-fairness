"""CheXpert の既存 split から race 欠損を除き、評価用4区分を追加する。

元CSVは変更せず、splits_race_known/ に CSV と入力 hash・件数の manifest を出力する。
Usage: uv run python -m data_pipeline.chexpert_race_known_splits
"""

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd

SPLITS = ("train", "val", "test")
RACE_MAPPING = {0: 0, 2: 1, 3: 2, 1: 3, 4: 3, 5: 3}
RACE_LABELS = {0: "White", 1: "Asian", 2: "Black", 3: "Others"}


def race_known_frame(frame: pd.DataFrame) -> pd.DataFrame:
    """欠損フラグで行を除外し、元の race を保って評価用属性を追加する。

    Args:
        frame: race と race_missing を持つ元 split。
    Returns:
        pd.DataFrame: 元の行順・列を保持した非欠損行と追加の race_group 列。
    Raises:
        ValueError: 必須列、不正なフラグ・既知 race、追加列との衝突がある場合。
    """
    if not {"race", "race_missing"}.issubset(frame.columns):
        raise ValueError("race と race_missing が必要")
    if {"race_group", "race_group_missing"}.intersection(frame.columns):
        raise ValueError("入力には race_group 列を含めない")
    if not frame.race_missing.isin([0, 1]).all():
        raise ValueError("race_missing は非欠損の 0/1 または bool が必要")
    result = frame.loc[~frame.race_missing.astype(bool)].copy()
    if not result.race.isin(RACE_MAPPING).all():
        raise ValueError("非欠損 race は整数コード 0〜5 が必要")
    result["race_group"] = result.race.map(RACE_MAPPING).astype("int64")
    result["race_group_missing"] = False
    return result


def create_splits(input_dir: Path, output_dir: Path) -> dict:
    """既存 split を検証し、未作成の出力ディレクトリへ派生 split を書く。

    Args:
        input_dir: 元の train/val/test.csv があるディレクトリ。
        output_dir: 新規作成する出力ディレクトリ。
    Returns:
        dict: 入力 hash、除外件数、クラス分布、群対応の manifest。
    Raises:
        FileExistsError: 出力先がすでに存在する場合。
        ValueError: 入力スキーマや画像の一意性が不正な場合。
    """
    if output_dir.exists():
        raise FileExistsError(f"既存の出力は上書きしない: {output_dir}")
    frames = {}
    manifest = {"input_dir": str(input_dir.resolve()), "race_mapping": RACE_MAPPING, "race_group_labels": RACE_LABELS, "splits": {}}
    for split in SPLITS:
        path = input_dir / f"{split}.csv"
        source = pd.read_csv(path)
        if not {"image", "target"}.issubset(source.columns) or not source.image.is_unique:
            raise ValueError(f"{split}: image と target、一意な image が必要")
        result = race_known_frame(source)
        frames[split] = result
        manifest["splits"][split] = {
            "input_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "input_rows": len(source),
            "removed_rows": len(source) - len(result),
            "output_rows": len(result),
            "input_target_counts": source.target.value_counts().sort_index().to_dict(),
            "output_target_counts": result.target.value_counts().sort_index().to_dict(),
        }
    output_dir.mkdir(parents=True, exist_ok=False)
    for split, frame in frames.items():
        frame.to_csv(output_dir / f"{split}.csv", index=False)
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest


def main() -> None:
    """CLI 引数から派生 split を生成し、件数を表示する。

    Args: なし
    Returns: None
    """
    root = Path(__file__).resolve().parent.parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=root / "data/chexpert/splits")
    parser.add_argument("--output-dir", type=Path, default=root / "data/chexpert/splits_race_known")
    args = parser.parse_args()
    manifest = create_splits(args.input_dir, args.output_dir)
    for split, counts in manifest["splits"].items():
        print(f"{split}: {counts['input_rows']} → {counts['output_rows']} (除外 {counts['removed_rows']})")


if __name__ == "__main__":
    main()
