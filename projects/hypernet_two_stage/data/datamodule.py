"""split CSV から学習・評価用 DataLoader を構築する Lightning DataModule。"""

from collections.abc import Mapping, Sequence
from typing import Any

import pandas as pd
import torch
from lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torchvision.transforms import transforms

from .attribute_utils import attribute_tensors, continuous_columns, standardize_continuous_columns, validate_attribute_spec
from .dataset import ImageDataset
from .evaluation_attributes import add_age_groups
from .splits import Split


class ImageDataModule(LightningDataModule):
    """split CSV を画像 Dataset と train・評価用 DataLoader に変換する。

    `setup` は stage に必要な split だけを読み込む。連続属性を標準化する場合、統計量は常に
    train split の非欠損行から算出する。sample は `(image, attributes, target)` であり、
    `attribute_names` が model 入力、`fairness_attribute_names` と `fairness_age_groups` が
    評価専用の `evaluation_categorical` を決める。`evaluation_dataloader` は train sampling に
    かかわらず評価 transform と CSV 順を使う。
    """

    def __init__(
        self,
        data_dir: str,
        cv_splits_dir: str,
        num_classes: int,
        batch_size: int,
        num_workers: int,
        pin_memory: bool,
        persistent_workers: bool,
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
        """DataModule の設定を検証し、Dataset 構築前の状態で保持する。"""
        super().__init__()
        if train_sampling not in {"uniform", "inverse_frequency"}:
            raise ValueError(f"train_sampling は 'uniform' または 'inverse_frequency' である必要があるが、{train_sampling!r} が指定された")
        validate_attribute_spec(attribute_names, attribute_spec)

        self.save_hyperparameters(logger=False, ignore=["train_transform", "val_transform"])
        self.batch_size_per_device = batch_size
        self.train_transform = train_transform
        self.val_transform = val_transform

        self.data_train: Dataset | None = None
        self.data_val: Dataset | None = None
        self.data_test: Dataset | None = None

    def read_split_dataframe(self, split: str) -> pd.DataFrame:
        """指定 split の CSV を標準化前の DataFrame として読む。"""
        return pd.read_csv(f"{self.hparams.cv_splits_dir}/{split}.csv")

    def standardized_dataframes(self, *splits: Split) -> tuple[pd.DataFrame, ...]:
        """指定 split を読み込み、必要なら train 統計量で連続属性を標準化して返す。"""
        needed = dict.fromkeys(("train", *splits)) if self.hparams.standardize_continuous else dict.fromkeys(splits)
        frames = {split: add_age_groups(self.read_split_dataframe(split), self.hparams.fairness_age_groups) for split in needed}
        if not self.hparams.standardize_continuous:
            return tuple(frames[split] for split in splits)

        return standardize_continuous_columns(
            frames["train"],
            *(frames[split] for split in splits),
            columns=continuous_columns(self.hparams.attribute_names),
        )

    def _create_dataset(self, frame: pd.DataFrame, transform: transforms.Compose | None) -> ImageDataset:
        """DataFrame から画像 Dataset を生成する。"""
        return ImageDataset(
            frame,
            self.hparams.data_dir,
            transform=transform,
            attribute_names=self.hparams.attribute_names,
            fairness_attribute_names=self.hparams.fairness_attribute_names,
        )

    def setup(self, stage: str) -> None:
        """stage に応じて split CSV を読み込み、標準化を適用した Dataset を構築する。"""
        if self.trainer is not None:
            if self.hparams.batch_size % self.trainer.world_size != 0:
                raise RuntimeError(f"batch_size（{self.hparams.batch_size}）が devices 数（{self.trainer.world_size}）で割り切れない")
            self.batch_size_per_device = self.hparams.batch_size // self.trainer.world_size

        if stage == "fit":
            train_df, val_df = self.standardized_dataframes("train", "val")
            self.data_train = self._create_dataset(train_df, self.train_transform)
            self.data_val = self._create_dataset(val_df, self.val_transform)

        if stage == "validate":
            (val_df,) = self.standardized_dataframes("val")
            self.data_val = self._create_dataset(val_df, self.val_transform)

        if stage == "test":
            (test_df,) = self.standardized_dataframes("test")
            self.data_test = self._create_dataset(test_df, self.val_transform)

    def train_attributes(self) -> dict[str, torch.Tensor]:
        """train split 全行の model 入力属性を `[n_rows, n_attributes]` のテンソル辞書で返す。

        データ依存の初期化（Spatial LoRA の Var(c) 推定など）が使う。`setup(stage="fit")` が
        構築した Dataset の DataFrame をそのまま読むので、連続属性は標準化済みの値になる。
        画像は読まない。
        """
        if self.data_train is None:
            raise RuntimeError("train_attributes() を呼ぶ前に setup(stage='fit') を呼び出す必要がある")
        return attribute_tensors(self.data_train.df, self.hparams.attribute_names)

    def _train_sampler(self) -> WeightedRandomSampler | None:
        """inverse-frequency sampling 時だけ class 頻度の逆数で重み付けした sampler を作る。"""
        if self.hparams.train_sampling == "uniform":
            return None
        if self.data_train is None:
            raise RuntimeError("train_dataloader() を呼ぶ前に setup(stage='fit') を呼び出す必要がある")

        targets = torch.as_tensor(self.data_train.df["target"].to_numpy(), dtype=torch.long)
        if torch.any(targets < 0) or torch.any(targets >= self.hparams.num_classes):
            raise ValueError("train split の target は [0, num_classes) の範囲でなければならない")

        class_counts = torch.bincount(targets, minlength=self.hparams.num_classes)
        if torch.any(class_counts == 0):
            missing_classes = torch.where(class_counts == 0)[0].tolist()
            raise ValueError(f"inverse_frequency sampling には train split に全 class が必要だが、{missing_classes} が欠けている")

        sample_weights = class_counts.to(dtype=torch.double).reciprocal()[targets]
        return WeightedRandomSampler(sample_weights, num_samples=len(targets), replacement=True)

    def train_dataloader(self) -> DataLoader[Any]:
        """学習用 DataLoader を返す。"""
        sampler = self._train_sampler()
        return DataLoader(
            dataset=self.data_train,
            batch_size=self.batch_size_per_device,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=sampler is None,
            sampler=sampler,
            persistent_workers=self.hparams.persistent_workers,
            prefetch_factor=self.hparams.prefetch_factor,
        )

    def val_dataloader(self) -> DataLoader[Any]:
        """検証用 DataLoader を返す。"""
        return DataLoader(
            dataset=self.data_val,
            batch_size=self.batch_size_per_device,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=False,
            persistent_workers=self.hparams.persistent_workers,
            prefetch_factor=self.hparams.prefetch_factor,
        )

    def test_dataloader(self) -> DataLoader[Any]:
        """テスト用 DataLoader を返す。"""
        return DataLoader(
            dataset=self.data_test,
            batch_size=self.batch_size_per_device,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=False,
            persistent_workers=self.hparams.persistent_workers,
            prefetch_factor=self.hparams.prefetch_factor,
        )

    def evaluation_dataloader(self, split: Split) -> DataLoader[Any]:
        """指定 split を評価 transform・固定順で返す。"""
        (frame,) = self.standardized_dataframes(split)
        dataset = self._create_dataset(frame, self.val_transform)
        return DataLoader(
            dataset=dataset,
            batch_size=self.batch_size_per_device,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=False,
            persistent_workers=self.hparams.persistent_workers,
            prefetch_factor=self.hparams.prefetch_factor,
        )
