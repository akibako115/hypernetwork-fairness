"""HAM10000 の split 生成が、split CSV の契約と漏洩防止を満たすことを検証する。"""

import zipfile
from pathlib import Path

import pandas as pd
import pytest

from data_pipeline.ham10000_splits import (
    CATEGORICAL,
    CONTINUOUS,
    COVARIATE_CATEGORICAL,
    DX_CLASSES,
    MISSING_CATEGORY,
    build,
    encode,
    extract_images,
)
from projects.hypernet_e2e.data.splits import validate_split_frame

_ATTRIBUTE_NAMES = {"categorical": list(CATEGORICAL), "continuous": list(CONTINUOUS)}


def _row(index: int, **overrides) -> dict:
    """配布 metadata の 1 行を作る。"""
    row = {
        "lesion_id": f"HAM_{index:07d}",
        "image_id": f"ISIC_{index:07d}",
        "dx": "nv",
        "dx_type": "histo",
        "age": 50.0,
        "sex": "female",
        "localization": "back",
        "dataset": "rosendahl",
    }
    return row | overrides


def test_target_is_malignant_and_dx_keeps_the_seven_classes() -> None:
    """2 値の target と 7 クラスの dx を同時に持つ。"""
    encoded = encode(pd.DataFrame([_row(index, dx=dx) for index, dx in enumerate(DX_CLASSES)]))

    assert dict(zip(DX_CLASSES, encoded["target"], strict=True)) == {"akiec": 1, "bcc": 0, "bkl": 0, "df": 0, "mel": 1, "nv": 0, "vasc": 0}
    assert encoded["dx"].tolist() == list(range(len(DX_CLASSES)))


def test_unknown_and_blank_are_stored_as_minus_one_with_their_missing_flag() -> None:
    """配布 CSV の `unknown` と空欄はどちらも欠損で、実カテゴリと衝突しない値と flag の組で表す。"""
    rows = [_row(1), _row(2, sex="unknown", localization="unknown", age=None), _row(3, sex=None, localization=None, dataset=None)]

    encoded = encode(pd.DataFrame(rows))

    assert encoded.loc[0, ["sex", "localization", "age", "dataset"]].tolist() == [0, 2, 50.0, 1]
    assert encoded.loc[1, ["sex", "localization"]].tolist() == [MISSING_CATEGORY] * 2
    assert encoded.loc[1, ["sex_missing", "localization_missing", "age_missing"]].tolist() == [1, 1, 1]
    assert encoded.loc[2, ["sex_missing", "localization_missing", "dataset_missing"]].tolist() == [1, 1, 1]
    assert encoded.loc[0, [f"{name}_missing" for name in (*CATEGORICAL, *CONTINUOUS, *COVARIATE_CATEGORICAL)]].sum() == 0


def test_encode_rejects_an_unknown_category() -> None:
    """未知の値を黙って欠損に落とすと、符号化ミスが学習まで通ってしまう。"""
    with pytest.raises(ValueError, match="localization"):
        encode(pd.DataFrame([_row(1, localization="left elbow")]))


def test_images_of_one_lesion_never_cross_train_and_val() -> None:
    """同じ病変の別アングルが train と val に散ると model selection が楽観側へ歪む。"""
    rows = [_row(lesion * 10 + view, lesion_id=f"HAM_{lesion:07d}", dx=DX_CLASSES[lesion % 7]) for lesion in range(140) for view in range(lesion % 3 + 1)]
    test_rows = [_row(9_000_000 + index, lesion_id=f"HAMTEST_{index:07d}") for index in range(5)]

    splits = build(pd.DataFrame(rows), pd.DataFrame(test_rows), seed=0)

    assert set(splits["train"]["lesion_id"]).isdisjoint(splits["val"]["lesion_id"])
    assert len(splits["train"]) + len(splits["val"]) == len(rows)
    assert splits["test"]["image"].tolist() == [f"ISIC_{9_000_000 + index:07d}.jpg" for index in range(5)]
    for split, part in splits.items():
        validate_split_frame(part, _ATTRIBUTE_NAMES, split=split)


def test_rows_missing_sex_or_age_are_dropped_from_train_and_val_but_kept_in_test() -> None:
    """group を決められない行は学習から外し、test では全体性能のために残す。"""
    rows = [_row(1), _row(2, sex="unknown"), _row(3, age=None), _row(4, localization="unknown")]
    test_rows = [_row(9_000_000, lesion_id="HAMTEST_0000000", sex="unknown", age=None)]

    splits = build(pd.DataFrame(rows), pd.DataFrame(test_rows), seed=0)

    kept = pd.concat([splits["train"], splits["val"]])["image"].tolist()
    assert sorted(kept) == ["ISIC_0000001.jpg", "ISIC_0000004.jpg"]
    assert splits["test"][["sex_missing", "age_missing"]].values.tolist() == [[1, 1]]


def test_build_rejects_an_image_shared_with_the_official_test() -> None:
    """本体と公式 test が重なれば test は外部 test ではなくなる。"""
    with pytest.raises(ValueError, match="image"):
        build(pd.DataFrame([_row(1)]), pd.DataFrame([_row(1, lesion_id="HAMTEST_0000001")]), seed=0)


def test_build_drops_the_test_image_missing_from_the_distribution() -> None:
    """配布 zip に無い公式 test の画像は、split に載せると学習時に読めずに落ちる。"""
    test_rows = [_row(9_000_000, lesion_id="HAMTEST_0000000"), _row(35068, image_id="ISIC_0035068", lesion_id="HAMTEST_0000496")]

    splits = build(pd.DataFrame([_row(1)]), pd.DataFrame(test_rows), seed=0)

    assert splits["test"]["image"].tolist() == ["ISIC_9000000.jpg"]


def test_extract_flattens_directories_and_skips_macos_forks(tmp_path: Path) -> None:
    """zip ごとの directory 構成を捨て、`images/` 直下に JPEG だけを置く。"""
    with zipfile.ZipFile(tmp_path / "a.zip", "w") as archive:
        archive.writestr("ISIC_0000001.jpg", b"one")
    with zipfile.ZipFile(tmp_path / "b.zip", "w") as archive:
        archive.writestr("Test_Images/ISIC_0000002.jpg", b"two")
        archive.writestr("Test_Images/LICENSE.txt", b"license")
        archive.writestr("__MACOSX/Test_Images/._ISIC_0000002.jpg", b"fork")
    zips = [tmp_path / "a.zip", tmp_path / "b.zip"]
    expected = {"ISIC_0000001.jpg", "ISIC_0000002.jpg"}

    assert extract_images(zips, tmp_path / "images", expected) == 2
    assert sorted(path.name for path in (tmp_path / "images").iterdir()) == sorted(expected)
    assert (tmp_path / "images" / "ISIC_0000002.jpg").read_bytes() == b"two"
    assert extract_images(zips, tmp_path / "images", expected) == 0

    with pytest.raises(ValueError, match="ISIC_0000003.jpg"):
        extract_images(zips, tmp_path / "images", expected | {"ISIC_0000003.jpg"})
