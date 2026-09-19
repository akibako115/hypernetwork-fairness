import pandas as pd
import pytest
import torch

from projects.hypernet_e2e.data.attribute_utils import (
    attribute_arrays,
    attribute_tensors,
    missing_columns,
    standardize_continuous_columns,
    validate_attribute_spec,
)


def _attribute_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "sex": [0, 1],
            "sex_missing": [False, False],
            "age": [20.0, 30.0],
            "age_missing": [False, True],
        }
    )


def test_attribute_arrays_and_tensors_follow_the_attribute_contract() -> None:
    attribute_names = {"categorical": ["sex"], "continuous": ["age"]}

    arrays = attribute_arrays(_attribute_frame(), attribute_names)
    tensors = attribute_tensors(_attribute_frame(), attribute_names)

    assert missing_columns(["sex", "age"]) == ["sex_missing", "age_missing"]
    assert arrays["categorical"].dtype.name == "int64"
    assert arrays["continuous"].dtype.name == "float32"
    assert arrays["categorical_missing"].dtype.name == "bool"
    assert arrays["continuous"].shape == (2, 1)
    assert tensors["categorical"].dtype == torch.long
    assert tensors["continuous"].dtype == torch.float32
    assert tensors["continuous_missing"].dtype == torch.bool
    assert tensors["categorical"].shape == (2, 1)


def test_attribute_arrays_can_include_empty_kinds() -> None:
    arrays = attribute_arrays(pd.DataFrame(index=range(2)), None, include_empty=True)

    assert {name: values.shape for name, values in arrays.items()} == {
        "categorical": (2, 0),
        "categorical_missing": (2, 0),
        "continuous": (2, 0),
        "continuous_missing": (2, 0),
    }


def test_continuous_standardization_uses_non_missing_train_values_only() -> None:
    train = pd.DataFrame({"age": [10.0, 20.0, 999.0], "age_missing": [False, False, True]})
    validation = pd.DataFrame({"age": [15.0], "age_missing": [False]})

    standardized_train, standardized_validation = standardize_continuous_columns(train, train, validation, columns=["age"])

    assert standardized_train["age"].tolist() == [-1.0, 1.0, 196.8]
    assert standardized_validation["age"].tolist() == [0.0]
    assert train["age"].tolist() == [10.0, 20.0, 999.0]


def test_validate_attribute_spec_rejects_mismatched_dimensions() -> None:
    attribute_names = {"categorical": ["sex"], "continuous": ["age"]}

    with pytest.raises(ValueError, match="categorical_cardinalities"):
        validate_attribute_spec(attribute_names, {"categorical_cardinalities": [], "continuous_dim": 1})
    with pytest.raises(ValueError, match="continuous_dim"):
        validate_attribute_spec(attribute_names, {"categorical_cardinalities": [2], "continuous_dim": 0})
