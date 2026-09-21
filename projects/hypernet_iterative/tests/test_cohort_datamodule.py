"""固定 cohort の sidecar を扱うDataModuleの契約テスト。"""

from pathlib import Path

import pandas as pd
import pytest
from PIL import Image

from projects.hypernet_iterative.data.cohort_datamodule import CohortImageDataModule


def _write_fixture(tmp_path: Path) -> tuple[Path, Path]:
    image_dir = tmp_path / "images"
    split_dir = tmp_path / "splits"
    image_dir.mkdir()
    split_dir.mkdir()

    splits = {
        "train": ["train_0.jpg", "train_1.jpg", "train_2.jpg"],
        "val": ["val_0.jpg"],
        "test": ["test_0.jpg"],
    }
    for split, images in splits.items():
        pd.DataFrame({"image": images, "target": range(len(images))}).to_csv(split_dir / f"{split}.csv", index=False)
        for image in images:
            Image.new("RGB", (16, 16), color=(128, 128, 128)).save(image_dir / image)
    return image_dir, split_dir


def _assignments(*, extra: list[dict] | None = None) -> pd.DataFrame:
    rows = [
        {"split": "train", "image": "train_0.jpg", "group_id": 1},
        {"split": "train", "image": "train_1.jpg", "group_id": 2},
        {"split": "train", "image": "train_2.jpg", "group_id": 0},
        {"split": "val", "image": "val_0.jpg", "group_id": 1},
        {"split": "test", "image": "test_0.jpg", "group_id": 2},
    ]
    return pd.DataFrame(rows + (extra or []))


def _make_datamodule(image_dir: Path, split_dir: Path, assignment_path: Path, *, group_key: str = "group_id") -> CohortImageDataModule:
    return CohortImageDataModule(
        data_dir=str(image_dir),
        cv_splits_dir=str(split_dir),
        num_classes=3,
        batch_size=1,
        num_workers=0,
        pin_memory=False,
        persistent_workers=False,
        group_assignment_path=str(assignment_path),
        num_groups=3,
        group_key=group_key,
    )


def _write_assignments(tmp_path: Path, assignments: pd.DataFrame) -> Path:
    assignment_path = tmp_path / "assignments.parquet"
    assignments.to_parquet(assignment_path, index=False)
    return assignment_path


def test_sidecar_is_propagated_to_all_splits(tmp_path: Path) -> None:
    image_dir, split_dir = _write_fixture(tmp_path)
    assignment_path = _write_assignments(tmp_path, _assignments())
    dm = _make_datamodule(image_dir, split_dir, assignment_path, group_key="cohort_id")

    dm.setup(stage="fit")
    assert dm.data_train is not None and dm.data_val is not None
    assert dm.data_train[0][1]["cohort_id"].item() == 1
    assert dm.data_val[0][1]["cohort_id"].item() == 1

    dm.setup(stage="test")
    assert dm.data_test is not None
    assert dm.data_test[0][1]["cohort_id"].item() == 2


def test_sidecar_rejects_images_absent_from_base_split(tmp_path: Path) -> None:
    image_dir, split_dir = _write_fixture(tmp_path)
    assignment_path = _write_assignments(
        tmp_path,
        _assignments(extra=[{"split": "train", "image": "not_in_train.csv", "group_id": 0}]),
    )

    with pytest.raises(ValueError, match="absent from train CSV"):
        _make_datamodule(image_dir, split_dir, assignment_path).setup(stage="fit")


def test_sidecar_rejects_unknown_split(tmp_path: Path) -> None:
    image_dir, split_dir = _write_fixture(tmp_path)
    assignment_path = _write_assignments(
        tmp_path,
        _assignments(extra=[{"split": "holdout", "image": "outside.jpg", "group_id": 0}]),
    )

    with pytest.raises(ValueError, match="unknown splits"):
        _make_datamodule(image_dir, split_dir, assignment_path).setup(stage="fit")


def test_sidecar_rejects_non_integer_group_ids(tmp_path: Path) -> None:
    image_dir, split_dir = _write_fixture(tmp_path)
    assignments = _assignments()
    assignments["group_id"] = assignments["group_id"].astype(float)
    assignments.loc[0, "group_id"] = 1.5
    assignment_path = _write_assignments(tmp_path, assignments)

    with pytest.raises(ValueError, match="must be an integer"):
        _make_datamodule(image_dir, split_dir, assignment_path).setup(stage="fit")


def test_sidecar_rejects_out_of_range_group_ids(tmp_path: Path) -> None:
    """group ID の被覆検証はここが唯一の担当。GroupDRO 側では step ごとに再検証しない。"""
    image_dir, split_dir = _write_fixture(tmp_path)
    assignments = _assignments()
    assignments.loc[0, "group_id"] = 3
    assignment_path = _write_assignments(tmp_path, assignments)

    with pytest.raises(ValueError, match="must cover exactly groups"):
        _make_datamodule(image_dir, split_dir, assignment_path).setup(stage="fit")


def test_sidecar_rejects_group_missing_from_train_even_if_present_elsewhere(tmp_path: Path) -> None:
    image_dir, split_dir = _write_fixture(tmp_path)
    assignments = _assignments()
    assignments.loc[assignments["image"] == "train_2.jpg", "group_id"] = 1
    assignments.loc[assignments["image"] == "val_0.jpg", "group_id"] = 0
    assignment_path = _write_assignments(tmp_path, assignments)

    with pytest.raises(ValueError, match="train split must cover exactly groups"):
        _make_datamodule(image_dir, split_dir, assignment_path)


def test_sidecar_requires_an_assignment_path(tmp_path: Path) -> None:
    image_dir, split_dir = _write_fixture(tmp_path)

    with pytest.raises(ValueError, match="group_assignment_path"):
        CohortImageDataModule(
            data_dir=str(image_dir),
            cv_splits_dir=str(split_dir),
            num_classes=3,
            batch_size=1,
            num_workers=0,
            pin_memory=False,
            persistent_workers=False,
            group_assignment_path=None,
            num_groups=3,
        )


def test_sidecar_rejects_existing_group_column(tmp_path: Path) -> None:
    image_dir, split_dir = _write_fixture(tmp_path)
    train_path = split_dir / "train.csv"
    train_frame = pd.read_csv(train_path)
    train_frame["group_id"] = 0
    train_frame.to_csv(train_path, index=False)
    assignment_path = _write_assignments(tmp_path, _assignments())

    with pytest.raises(ValueError, match="already contains 'group_id'"):
        _make_datamodule(image_dir, split_dir, assignment_path).setup(stage="fit")
