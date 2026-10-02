"""FairVLMed の split 生成が、split CSV の契約と配布 split・npz との整合を満たすことを検証する。"""

import io
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from PIL import Image

from data_pipeline.fairvlmed_splits import (
    CATEGORICAL,
    CONTINUOUS,
    MISSING_CATEGORY,
    build,
    encode,
    extract_images,
)
from projects.hypernet_e2e.data.splits import validate_split_frame

_ATTRIBUTE_NAMES = {"categorical": list(CATEGORICAL), "continuous": list(CONTINUOUS)}


def _row(index: int, **overrides) -> dict:
    """`data_summary.csv` の 1 行を作る。"""
    row = {
        "filename": f"data_{index:05d}.npz",
        "age": 60.0,
        "gender": "female",
        "race": "white",
        "ethnicity": "non-hispanic",
        "language": "english",
        "maritalstatus": "single",
        "glaucoma": "yes",
        "use": "training",
    }
    return row | overrides


def _npz_bytes(row: dict, fundus: np.ndarray) -> bytes:
    """配布 npz と同じく、language の 2 と -1 を入れ替えた符号で npz を作る。"""
    codes = encode(pd.DataFrame([row])).iloc[0]
    attributes = {key: int(codes[key]) for key in CATEGORICAL}
    attributes["language"] = {2: -1, -1: 2}.get(attributes["language"], attributes["language"])
    buffer = io.BytesIO()
    np.savez(buffer, slo_fundus=fundus, glaucoma=codes["target"], **attributes)
    return buffer.getvalue()


def test_unknown_is_stored_as_minus_one_with_its_missing_flag() -> None:
    """欠損は実カテゴリと衝突しない値と flag の組で表す。"""
    encoded = encode(pd.DataFrame([_row(1), _row(2, ethnicity="unknown", language="unknown", maritalstatus="unknown")]))

    assert encoded.loc[0, list(CATEGORICAL)].tolist() == [0, 2, 0, 0, 1]
    assert encoded.loc[1, ["ethnicity", "language", "maritalstatus"]].tolist() == [MISSING_CATEGORY] * 3
    assert encoded.loc[1, [f"{name}_missing" for name in ("ethnicity", "language", "maritalstatus")]].tolist() == [1, 1, 1]
    assert encoded.loc[0, [f"{name}_missing" for name in (*CATEGORICAL, *CONTINUOUS)]].sum() == 0


def test_encode_rejects_an_unknown_category() -> None:
    """未知の値を黙って欠損に落とすと、符号化ミスが学習まで通ってしまう。"""
    with pytest.raises(ValueError, match="race"):
        encode(pd.DataFrame([_row(1, race="hispanic")]))


def test_build_keeps_the_distributed_split_and_satisfies_the_contract() -> None:
    """先行研究と比較できるよう、配布元の `use` 列をそのまま split にする。"""
    rows = [_row(1, use="training"), _row(2, use="validation", glaucoma="no"), _row(3, use="test")]

    splits = build(pd.DataFrame(rows))

    assert {split: part["image"].tolist() for split, part in splits.items()} == {
        "train": ["data_00001.png"],
        "val": ["data_00002.png"],
        "test": ["data_00003.png"],
    }
    assert splits["val"]["target"].tolist() == [0]
    for split, part in splits.items():
        validate_split_frame(part, _ATTRIBUTE_NAMES, split=split)


def test_extract_writes_png_and_rejects_npz_that_disagrees_with_the_csv(tmp_path: Path) -> None:
    """npz と CSV の属性・split が食い違うなら、どちらかが誤っているので止める。"""
    rows = [_row(1, language="other"), _row(2, race="black", language="unknown")]
    fundus = np.arange(12, dtype=np.uint8).reshape(3, 4)
    zip_path = tmp_path / "dataset.zip"
    with zipfile.ZipFile(zip_path, "w") as archive:
        for row in rows:
            archive.writestr(f"Training/{row['filename']}", _npz_bytes(row, fundus))

    encoded = encode(pd.DataFrame(rows))
    assert extract_images(zip_path, tmp_path / "images", encoded) == 2
    with Image.open(tmp_path / "images" / "data_00001.png") as image:
        assert np.array_equal(np.asarray(image), fundus)
    assert extract_images(zip_path, tmp_path / "images", encoded) == 0

    mismatched = encode(pd.DataFrame([rows[0], rows[1] | {"race": "asian"}]))
    with pytest.raises(ValueError, match="data_00002.npz"):
        extract_images(zip_path, tmp_path / "images", mismatched)

    moved = encode(pd.DataFrame([rows[0], rows[1] | {"use": "test"}]))
    with pytest.raises(ValueError, match="Training/"):
        extract_images(zip_path, tmp_path / "images", moved)
