from pathlib import Path

import pandas as pd
import pytest
import torch
from PIL import Image
from torch.utils.data import RandomSampler, SequentialSampler, WeightedRandomSampler
from torchvision.transforms import transforms

from projects.hypernet_iterative.data.datamodule import ImageDataModule


def _write_fixture(tmp_path: Path) -> tuple[Path, Path]:
    image_dir = tmp_path / "images"
    split_dir = tmp_path / "splits"
    image_dir.mkdir()
    split_dir.mkdir()
    rows_by_split = {
        "train": [("train_0.png", 0, 20.0), ("train_1.png", 1, 40.0), ("train_2.png", 0, 20.0)],
        "val": [("val_0.png", 1, 30.0)],
        "test": [("test_0.png", 0, 50.0)],
    }
    for split, rows in rows_by_split.items():
        frame_rows = []
        for image_name, target, age in rows:
            Image.new("RGB", (8, 8), color=128).save(image_dir / image_name)
            frame_rows.append({"image": image_name, "target": target, "sex": target, "sex_missing": False, "age": age, "age_missing": False})
        pd.DataFrame(frame_rows).to_csv(split_dir / f"{split}.csv", index=False)
    return image_dir, split_dir


def _datamodule(image_dir: Path, split_dir: Path, **kwargs: object) -> ImageDataModule:
    return ImageDataModule(
        data_dir=str(image_dir),
        cv_splits_dir=str(split_dir),
        num_classes=2,
        batch_size=2,
        num_workers=0,
        pin_memory=False,
        persistent_workers=False,
        attribute_names={"categorical": ["sex"], "continuous": ["age"]},
        train_transform=transforms.Compose([transforms.ToTensor()]),
        val_transform=transforms.Compose([transforms.ToTensor()]),
        **kwargs,
    )


def test_setup_loads_only_the_datasets_needed_by_each_stage(tmp_path: Path) -> None:
    dm = _datamodule(*_write_fixture(tmp_path))

    dm.setup("fit")
    assert dm.data_train is not None and dm.data_val is not None
    assert dm.data_test is None

    dm.setup("test")
    assert dm.data_test is not None


def test_train_batch_preserves_the_image_attribute_target_contract(tmp_path: Path) -> None:
    dm = _datamodule(*_write_fixture(tmp_path))
    dm.setup("fit")

    images, attributes, targets = next(iter(dm.train_dataloader()))

    assert images.shape == (2, 3, 8, 8)
    assert targets.dtype == torch.long
    assert attributes["categorical"].dtype == torch.long
    assert attributes["continuous"].dtype == torch.float32
    assert attributes["continuous_missing"].dtype == torch.bool


def test_inverse_frequency_sampling_and_evaluation_loader_have_distinct_sampling_contracts(tmp_path: Path) -> None:
    dm = _datamodule(*_write_fixture(tmp_path), train_sampling="inverse_frequency")
    dm.setup("fit")

    sampler = dm.train_dataloader().sampler
    assert isinstance(sampler, WeightedRandomSampler)
    assert sampler.weights.tolist() == pytest.approx([0.5, 1.0, 0.5])
    assert isinstance(dm.evaluation_dataloader("train").sampler, SequentialSampler)


def test_uniform_train_loader_shuffles_and_evaluation_uses_validation_transform(tmp_path: Path) -> None:
    image_dir, split_dir = _write_fixture(tmp_path)
    dm = ImageDataModule(
        data_dir=str(image_dir),
        cv_splits_dir=str(split_dir),
        num_classes=2,
        batch_size=1,
        num_workers=0,
        pin_memory=False,
        persistent_workers=False,
        train_transform=transforms.Lambda(lambda _: torch.zeros((3, 4, 4))),
        val_transform=transforms.Lambda(lambda _: torch.ones((3, 4, 4))),
    )
    dm.setup("fit")

    assert isinstance(dm.train_dataloader().sampler, RandomSampler)
    images, attributes, _ = next(iter(dm.evaluation_dataloader("train")))
    assert torch.equal(images, torch.ones_like(images))
    assert attributes == {}


def test_standardization_uses_train_statistics_and_invalid_sampler_is_rejected(tmp_path: Path) -> None:
    image_dir, split_dir = _write_fixture(tmp_path)
    dm = _datamodule(image_dir, split_dir, standardize_continuous=True)

    _, attributes, _ = next(iter(dm.evaluation_dataloader("val")))
    assert attributes["continuous"].item() == pytest.approx(0.35355338)

    with pytest.raises(ValueError, match="train_sampling"):
        _datamodule(image_dir, split_dir, train_sampling="invalid")


def test_evaluation_attributes_keep_raw_age_groups_while_model_age_is_standardized(tmp_path: Path) -> None:
    dm = _datamodule(
        *_write_fixture(tmp_path),
        standardize_continuous=True,
        fairness_attribute_names={"categorical": ["sex", "age_group_65"]},
        fairness_age_groups={"age_group_65": {"source": "age", "boundaries": [65]}},
    )

    _, attributes, _ = next(iter(dm.evaluation_dataloader("val")))

    assert attributes["continuous"].item() == pytest.approx(0.35355338)
    assert attributes["evaluation_categorical"].tolist() == [[1, 0]]
    assert attributes["evaluation_categorical_missing"].tolist() == [[False, False]]


def test_train_attributes_cover_every_train_row_without_reading_images(tmp_path: Path) -> None:
    """Var(c) の推定は属性列だけで決まる。画像を消しても train_attributes は成立する。"""
    image_dir, split_dir = _write_fixture(tmp_path)
    dm = _datamodule(image_dir, split_dir)
    dm.setup("fit")
    for image in image_dir.iterdir():
        image.unlink()

    attributes = dm.train_attributes()

    assert attributes["categorical"].tolist() == [[0], [1], [0]]
    assert attributes["categorical_missing"].tolist() == [[False], [False], [False]]
    assert attributes["continuous"].squeeze(1).tolist() == pytest.approx([20.0, 40.0, 20.0])


def test_train_attributes_use_the_standardized_continuous_values(tmp_path: Path) -> None:
    dm = _datamodule(*_write_fixture(tmp_path), standardize_continuous=True)
    dm.setup("fit")

    values = dm.train_attributes()["continuous"].squeeze(1)

    assert values.mean().item() == pytest.approx(0.0, abs=1e-6)
    assert values.tolist() == pytest.approx([-0.70710678, 1.41421356, -0.70710678])


def test_train_attributes_require_the_fit_datasets(tmp_path: Path) -> None:
    dm = _datamodule(*_write_fixture(tmp_path))

    with pytest.raises(RuntimeError, match="setup"):
        dm.train_attributes()
