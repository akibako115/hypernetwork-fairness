"""固定 cohort sidecar を結合する実験用のデータ拡張。"""

from collections.abc import Mapping, Sequence
from typing import Any

import pandas as pd
import torch
from torchvision.transforms import transforms

from .datamodule import ImageDataModule
from .dataset import ImageDataset
from .splits import SPLITS

_GROUP_ASSIGNMENT_COLUMNS = {"split", "image", "group_id"}


class CohortImageDataset(ImageDataset):
    """固定 cohort の group ID を追加属性として返す画像 Dataset。"""

    def __init__(
        self,
        df: pd.DataFrame,
        img_dir: str,
        group_key: str,
        transform: transforms.Compose | None = None,
        attribute_names: Mapping[str, Sequence[str]] | None = None,
        fairness_attribute_names: Mapping[str, Sequence[str]] | None = None,
    ):
        """親 Dataset に group ID の属性キーを追加する。"""
        super().__init__(
            df,
            img_dir,
            transform=transform,
            attribute_names=attribute_names,
            fairness_attribute_names=fairness_attribute_names,
        )
        self.group_key = group_key

    def __getitem__(self, idx: int):
        """親の `(image, attributes, target)` に group ID を追加して返す。"""
        image, attributes, target = super().__getitem__(idx)
        attributes[self.group_key] = torch.tensor(self.df[self.group_key].iloc[idx], dtype=torch.long)
        return image, attributes, target


class CohortImageDataModule(ImageDataModule):
    """固定 cohort sidecar を結合し group ID を供給する DataModule。"""

    def __init__(
        self,
        data_dir: str,
        cv_splits_dir: str,
        num_classes: int,
        batch_size: int,
        num_workers: int,
        pin_memory: bool,
        persistent_workers: bool,
        group_assignment_path: str | None,
        num_groups: int,
        group_key: str = "group_id",
        prefetch_factor: int | None = None,
        train_sampling: str = "uniform",
        attribute_names: Mapping[str, Sequence[str]] | None = None,
        attribute_spec: Mapping[str, Any] | None = None,
        fairness_attribute_names: Mapping[str, Sequence[str]] | None = None,
        fairness_age_groups: Mapping[str, Mapping[str, Any]] | None = None,
        standardize_continuous: bool = False,
        train_transform: transforms.Compose | None = None,
        val_transform: transforms.Compose | None = None,
    ):
        """sidecar を検証し、固定 cohort 用の DataModule を初期化する。"""
        if not group_assignment_path:
            raise ValueError("fixed cohort training requires data.group_assignment_path")
        super().__init__(
            data_dir=data_dir,
            cv_splits_dir=cv_splits_dir,
            num_classes=num_classes,
            batch_size=batch_size,
            num_workers=num_workers,
            pin_memory=pin_memory,
            persistent_workers=persistent_workers,
            prefetch_factor=prefetch_factor,
            train_sampling=train_sampling,
            attribute_names=attribute_names,
            attribute_spec=attribute_spec,
            fairness_attribute_names=fairness_attribute_names,
            fairness_age_groups=fairness_age_groups,
            standardize_continuous=standardize_continuous,
            train_transform=train_transform,
            val_transform=val_transform,
        )
        self._assignments: pd.DataFrame | None = None
        # 検証: 学習前に sidecar のスキーマと group 集合を一度だけ確認する。
        self._load_assignments()

    def read_split_dataframe(self, split: str) -> pd.DataFrame:
        """base CSV と sidecar を 1 対 1 で結合する。"""
        frame = super().read_split_dataframe(split)
        if self.hparams.group_key in frame.columns:
            raise ValueError(f"{split} CSV already contains {self.hparams.group_key!r}; cannot merge a group assignment sidecar")
        # 抽出: 対象 split の検証済み assignment を取り出す。
        split_assignments = self._assignments_for_split(split)
        # 結合: base CSV に存在しない image と sidecar の欠落を拒否する。
        frame = frame.copy()
        frame["image"] = frame["image"].astype(str)
        extra_images = sorted(set(split_assignments["image"]) - set(frame["image"]))
        if extra_images:
            raise ValueError(f"group assignment contains images absent from {split} CSV: {extra_images[:5]}")
        merged = frame.merge(split_assignments, on="image", how="left")
        if merged[self.hparams.group_key].isna().any():
            missing_images = merged.loc[merged[self.hparams.group_key].isna(), "image"].head(5).tolist()
            raise ValueError(f"group assignment is missing images for split={split!r}: {missing_images}")
        return merged

    def _create_dataset(self, frame: pd.DataFrame, transform: transforms.Compose | None) -> CohortImageDataset:
        """group ID を追加属性として返す cohort Dataset を生成する。"""
        return CohortImageDataset(
            frame,
            self.hparams.data_dir,
            group_key=self.hparams.group_key,
            transform=transform,
            attribute_names=self.hparams.attribute_names,
            fairness_attribute_names=self.hparams.fairness_attribute_names,
        )

    def _assignments_for_split(self, split: str) -> pd.DataFrame:
        """検証済み sidecar から指定 split を取り出し、image 重複を拒否する。"""
        assignments = self._load_assignments()
        split_assignments = assignments.loc[assignments["split"] == split, ["image", "group_id"]].copy()
        if not split_assignments["image"].is_unique:
            raise ValueError(f"group assignment has duplicate images for split={split!r}")
        return split_assignments.rename(columns={"group_id": self.hparams.group_key})

    def _load_assignments(self) -> pd.DataFrame:
        """sidecar schema・split 名・group ID の被覆を検証してキャッシュする。"""
        if self._assignments is not None:
            return self._assignments
        # 抽出: parquet を読み、必要な schema を検証する。
        assignments = pd.read_parquet(self.hparams.group_assignment_path).copy()
        missing_columns = _GROUP_ASSIGNMENT_COLUMNS - set(assignments.columns)
        if missing_columns:
            raise ValueError(f"group assignment is missing columns: {sorted(missing_columns)}")
        # 変換: key を文字列、group ID を整数に揃える。
        assignments["split"] = assignments["split"].astype(str)
        assignments["image"] = assignments["image"].astype(str)
        unknown_splits = sorted(set(assignments["split"]) - set(SPLITS))
        if unknown_splits:
            raise ValueError(f"group assignment contains unknown splits: {unknown_splits}")
        group_ids = pd.to_numeric(assignments["group_id"], errors="coerce")
        if not group_ids.notna().all() or not (group_ids == group_ids.round()).all():
            raise ValueError("group assignment group_id must be an integer")
        # 絞り込み: config と sidecar の group 集合が過不足なく一致することを確認する。
        observed_groups = set(group_ids.astype("int64"))
        expected_groups = set(range(self.hparams.num_groups))
        if observed_groups != expected_groups:
            raise ValueError(f"group assignment must cover exactly groups 0..{self.hparams.num_groups - 1}, got {sorted(observed_groups)}")
        train_groups = set(group_ids.loc[assignments["split"] == "train"].astype("int64"))
        if train_groups != expected_groups:
            raise ValueError(f"group assignment train split must cover exactly groups 0..{self.hparams.num_groups - 1}, got {sorted(train_groups)}")
        assignments["group_id"] = group_ids.astype("int64")
        self._assignments = assignments
        return assignments
