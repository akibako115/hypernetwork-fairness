"""ISIC 2019 の split 生成が、split CSV の契約と漏洩防止を満たすことを検証する。"""

from pathlib import Path

import pandas as pd
import pytest

from data_pipeline.isic2019_splits import (
    CATEGORICAL,
    CLASSES,
    CONTINUOUS,
    COVARIATE_CATEGORICAL,
    DEFAULT_FOLDS,
    MISSING_CATEGORY,
    assign_folds,
    assign_groups,
    build,
    build_cv,
    encode_attributes,
    split_groups,
)
from projects.hypernet_e2e.data.splits import validate_split_frame

_ATTRIBUTE_NAMES = {"categorical": list(CATEGORICAL), "continuous": list(CONTINUOUS)}


def _source(rows: list[dict]) -> pd.DataFrame:
    """`load_source` が返す形の最小 DataFrame を作る。"""
    return pd.DataFrame(rows, columns=["image", "target", "sex", "age_approx", "anatom_site_general", "lesion_id", "attribution"])


def test_encode_marks_missing_attributes_without_inventing_a_category() -> None:
    """欠損は専用の値と flag で表し、実カテゴリと衝突させない。"""
    frame = _source(
        [
            {"image": "a", "target": 0, "sex": "female", "age_approx": 30.0, "anatom_site_general": "head/neck", "lesion_id": "L1", "attribution": "Anonymous"},
            {"image": "b", "target": 1, "sex": None, "age_approx": None, "anatom_site_general": None, "lesion_id": None, "attribution": None},
        ]
    )

    encoded = encode_attributes(frame)

    assert encoded.loc[0, ["sex", "anatom_site_general", "age_approx", "attribution"]].tolist() == [0, 1, 30.0, 0]
    assert encoded.loc[1, ["sex", "anatom_site_general", "attribution"]].tolist() == [MISSING_CATEGORY] * 3
    assert encoded.loc[1, [f"{name}_missing" for name in (*CATEGORICAL, *CONTINUOUS, *COVARIATE_CATEGORICAL)]].tolist() == [1, 1, 1, 1]
    assert encoded.loc[0, [f"{name}_missing" for name in CATEGORICAL]].tolist() == [0, 0]


def test_encode_rejects_an_unknown_category() -> None:
    """未知の値を黙って欠損に落とすと、符号化ミスが学習まで通ってしまう。"""
    frame = _source([{"image": "a", "target": 0, "sex": "female", "age_approx": 30.0, "anatom_site_general": "left elbow", "lesion_id": "L1", "attribution": "Anonymous"}])

    with pytest.raises(ValueError, match="anatom_site_general"):
        encode_attributes(frame)


def test_images_of_one_lesion_never_cross_a_split() -> None:
    """同じ病変の別アングルが train と test に散ると test が楽観側へ歪む。"""
    rows = []
    for lesion in range(60):
        for index in range(3):
            rows.append(
                {
                    "image": f"L{lesion}_{index}",
                    "target": lesion % len(CLASSES),
                    "sex": "male",
                    "age_approx": 40.0,
                    "anatom_site_general": "head/neck",
                    "lesion_id": f"L{lesion}",
                    "attribution": "Anonymous",
                }
            )
    frame = encode_attributes(_source(rows))
    groups = assign_groups(frame)

    frame["split"] = split_groups(frame, groups, seed=0)

    per_lesion = frame.groupby("lesion_id")["split"].nunique()
    assert per_lesion.max() == 1


def test_images_without_a_lesion_id_become_independent_groups() -> None:
    """同一病変である証拠が無い画像を1つの group に束ねると、split の比率が崩れる。"""
    rows = [{"image": f"x{i}", "target": 0, "sex": "male", "age_approx": 40.0, "anatom_site_general": "head/neck", "lesion_id": None, "attribution": "Anonymous"} for i in range(5)]

    groups = assign_groups(encode_attributes(_source(rows)))

    assert groups.nunique() == 5


@pytest.mark.parametrize("split", ["train", "val", "test"])
def test_generated_splits_satisfy_the_split_contract(split: str) -> None:
    """DataModule が読む契約を、生成時点で満たしていることを確かめる。"""
    data_dir = Path("data/isic2019")
    if not (data_dir / "ISIC_2019_Training_GroundTruth.csv").is_file():
        pytest.skip("ISIC 2019 の raw metadata が無い環境")

    frame = build(data_dir, seed=12345)[split]

    validate_split_frame(frame, _ATTRIBUTE_NAMES, split=split)
    assert frame["image"].str.endswith(".jpg").all()
    assert frame["target"].between(0, len(CLASSES) - 1).all()


def _lesion_rows(lesions: int, per_lesion: int) -> list[dict]:
    """1 病変あたり `per_lesion` 枚の最小行を作る。"""
    return [
        {
            "image": f"L{lesion}_{index}",
            "target": lesion % len(CLASSES),
            "sex": "male",
            "age_approx": 40.0,
            "anatom_site_general": "head/neck",
            "lesion_id": f"L{lesion}",
            "attribution": "Anonymous",
        }
        for lesion in range(lesions)
        for index in range(per_lesion)
    ]


def test_one_lesion_stays_inside_one_fold() -> None:
    """病変が fold をまたぐと、その fold で train と test に同じ病変が入る。"""
    frame = encode_attributes(_source(_lesion_rows(80, 3)))

    frame["fold"] = assign_folds(frame, assign_groups(frame), seed=0, folds=DEFAULT_FOLDS)

    assert frame.groupby("lesion_id")["fold"].nunique().max() == 1


def test_folds_cover_every_image_exactly_once() -> None:
    """fold が重なると、プールした評価セルが同じ画像を二重に数える。"""
    frame = encode_attributes(_source(_lesion_rows(80, 3)))

    folds = assign_folds(frame, assign_groups(frame), seed=0, folds=DEFAULT_FOLDS)

    assert sorted(folds.unique()) == list(range(DEFAULT_FOLDS))
    assert len(folds) == len(frame)


def test_cv_needs_enough_folds_to_separate_three_splits() -> None:
    """2 fold では val と test のどちらかが train と重なる。"""
    with pytest.raises(ValueError, match="folds"):
        build_cv(Path("data/isic2019"), seed=0, folds=2)


@pytest.mark.parametrize("fold", range(DEFAULT_FOLDS))
def test_each_cv_fold_satisfies_the_split_contract(fold: int) -> None:
    """DataModule は fold ディレクトリを単一 split と同じ契約で読む。"""
    data_dir = Path("data/isic2019")
    if not (data_dir / "ISIC_2019_Training_GroundTruth.csv").is_file():
        pytest.skip("ISIC 2019 の raw metadata が無い環境")

    splits = build_cv(data_dir, seed=12345)[fold]

    for split, part in splits.items():
        validate_split_frame(part, _ATTRIBUTE_NAMES, split=f"fold{fold}/{split}")
    assert set(splits["train"]["image"]).isdisjoint(splits["test"]["image"])
    assert set(splits["val"]["image"]).isdisjoint(splits["test"]["image"])
    assert set(splits["train"]["image"]).isdisjoint(splits["val"]["image"])


def test_cv_uses_every_image_as_test_exactly_once() -> None:
    """プールして評価するので、test が重なると稀少クラスの枚数を過大に数える。"""
    data_dir = Path("data/isic2019")
    if not (data_dir / "ISIC_2019_Training_GroundTruth.csv").is_file():
        pytest.skip("ISIC 2019 の raw metadata が無い環境")

    folds = build_cv(data_dir, seed=12345)

    tests = [set(folds[fold]["test"]["image"]) for fold in range(DEFAULT_FOLDS)]
    pooled = set().union(*tests)
    assert sum(len(part) for part in tests) == len(pooled) == len(build(data_dir, seed=12345)["train"]) + len(build(data_dir, seed=12345)["val"]) + len(build(data_dir, seed=12345)["test"])
