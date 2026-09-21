import pandas as pd
import pytest

from projects.hypernet_iterative.data.splits import SPLITS, required_split_columns, validate_split_frame


def test_required_split_columns_include_only_configured_attributes() -> None:
    assert required_split_columns(None) == {"image", "target"}
    assert required_split_columns({"categorical": ["sex"], "continuous": ["age"]}) == {
        "image",
        "target",
        "sex",
        "sex_missing",
        "age",
        "age_missing",
    }


def test_validate_split_frame_accepts_the_configured_schema() -> None:
    frame = pd.DataFrame(
        {
            "image": ["a.jpg", "b.jpg"],
            "target": [0, 1],
            "sex": [0, 1],
            "sex_missing": [False, False],
        }
    )

    validate_split_frame(frame, {"categorical": ["sex"], "continuous": []}, split="train")
    assert SPLITS == ("train", "val", "test")


def test_validate_split_frame_rejects_missing_columns_and_duplicate_image_values() -> None:
    with pytest.raises(ValueError, match="train CSV missing columns: \['sex_missing'\]"):
        validate_split_frame(
            pd.DataFrame({"image": ["a.jpg"], "target": [0], "sex": [0]}),
            {"categorical": ["sex"]},
            split="train",
        )

    with pytest.raises(ValueError, match="val CSV contains duplicate image values"):
        validate_split_frame(
            pd.DataFrame({"image": [1, "1"], "target": [0, 1]}),
            None,
            split="val",
        )
