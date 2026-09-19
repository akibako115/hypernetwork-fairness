from pathlib import Path

import pandas as pd
import pytest
import torch
from PIL import Image

from projects.hypernet_two_stage.data.dataset import ImageDataset


def _write_image(directory: Path, filename: str, *, mode: str = "RGB") -> None:
    Image.new(mode, (8, 6), color=128).save(directory / filename)


def test_dataset_loads_rgb_image_applies_transform_and_returns_attributes(tmp_path: Path) -> None:
    image_dir = tmp_path / "images"
    image_dir.mkdir()
    _write_image(image_dir, "sample.png", mode="L")
    frame = pd.DataFrame(
        {
            "image": ["sample.png"],
            "target": [1],
            "sex": [0],
            "sex_missing": [False],
            "age": [40.0],
            "age_missing": [True],
        }
    )
    dataset = ImageDataset(
        frame,
        str(image_dir),
        transform=lambda image: (image.mode, image.size),
        attribute_names={"categorical": ["sex"], "continuous": ["age"]},
    )

    image, attributes, target = dataset[0]

    assert image == ("RGB", (8, 6))
    assert target == 1
    assert attributes["categorical"].dtype == torch.long
    assert attributes["continuous"].dtype == torch.float32
    assert attributes["categorical"].tolist() == [0]
    assert attributes["continuous_missing"].tolist() == [True]


def test_dataset_resets_the_input_frame_index_and_supports_no_attributes(tmp_path: Path) -> None:
    image_dir = tmp_path / "images"
    image_dir.mkdir()
    _write_image(image_dir, "sample.png")
    frame = pd.DataFrame({"image": ["ignored.png", "sample.png"], "target": [0, 0]}).iloc[[1]]
    dataset = ImageDataset(frame, str(image_dir))

    image, attributes, target = dataset[0]

    assert image.mode == "RGB"
    assert attributes == {}
    assert target == 0


def test_dataset_uses_the_image_column_as_an_exact_relative_path(tmp_path: Path) -> None:
    image_dir = tmp_path / "images"
    image_dir.mkdir()
    _write_image(image_dir, "sample.png")

    assert ImageDataset(pd.DataFrame({"image": ["sample.png"], "target": [0]}), str(image_dir))[0][2] == 0
    with pytest.raises(FileNotFoundError):
        ImageDataset(pd.DataFrame({"image": ["sample"], "target": [0]}), str(image_dir))[0]
