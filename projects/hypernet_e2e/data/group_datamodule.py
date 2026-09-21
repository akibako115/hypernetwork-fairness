"""demographic 属性の組を固定 group ID にまとめる実験用のデータ拡張。"""

from collections.abc import Mapping, Sequence
from math import prod
from typing import Any

import pandas as pd
import torch
from torchvision.transforms import transforms

from .datamodule import ImageDataModule
from .dataset import ImageDataset


def demographic_group_ids(
    frame: pd.DataFrame,
    attribute_names: Sequence[str],
    cardinalities: Sequence[int],
    *,
    group_key: str,
) -> pd.Series:
    """属性値の組を mixed-radix で 1 つの group ID に畳み、`[0, prod(cardinalities))` を返す。

    `attribute_names` の先頭ほど上位桁になる。欠損は group を持たないので、欠損 flag が立った行が
    1 行でもあれば例外を投げる。0 に丸めて先頭 group へ混ぜると、その group の loss だけが
    静かに汚れて原因が追えなくなるため。

    Args:
        frame: group 属性列とその `*_missing` 列を持つ split DataFrame
        attribute_names: group を構成する属性列名。先頭ほど上位桁になる
        cardinalities: 各属性が取りうる値の数。attribute_names と同じ長さ
        group_key: エラーメッセージに出す group ID の列名

    Returns:
        pd.Series: frame と同じ index の `int64` group ID

    Raises:
        ValueError: 属性列かその `*_missing` 列が無い場合、欠損した行がある場合、
            または属性値が `[0, cardinality)` の外にある場合。
    """
    group_ids = pd.Series(0, index=frame.index, dtype="int64")
    for name, cardinality in zip(attribute_names, cardinalities, strict=True):
        missing_name = f"{name}_missing"
        if name not in frame or missing_name not in frame:
            raise ValueError(f"group 属性 {name!r} には {name!r} と {missing_name!r} の列が必要")
        values = pd.to_numeric(frame[name], errors="coerce")
        missing = frame[missing_name].astype(bool) | values.isna()
        if missing.any():
            raise ValueError(f"group 属性 {name!r} が欠損した行は {group_key} を決められない: {frame.loc[missing, 'image'].head(5).tolist()}")
        values = values.astype("int64")
        out_of_range = (values < 0) | (values >= cardinality)
        if out_of_range.any():
            raise ValueError(f"group 属性 {name!r} は [0, {cardinality}) である必要があるが、{sorted(set(values[out_of_range]))} が含まれる")
        group_ids = group_ids * cardinality + values
    return group_ids


class GroupImageDataset(ImageDataset):
    """固定 group の ID を追加属性として返す画像 Dataset。"""

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

    def group_counts(self, num_groups: int) -> torch.Tensor:
        """`[num_groups]` の行数を返す。

        Args:
            num_groups: 数える group 数。出現しない group は 0 になる

        Returns:
            torch.Tensor: `[num_groups]` の `int64` 行数
        """
        return torch.bincount(torch.as_tensor(self.df[self.group_key].to_numpy(), dtype=torch.long), minlength=num_groups)


class GroupImageDataModule(ImageDataModule):
    """demographic 属性の組から group ID を作り、属性辞書へ供給する DataModule。

    sample は親と同じ `(image, attributes, target)` で、`attributes[group_key]` に
    `[0, num_groups)` の long scalar が加わる。`loss.py` の group 系目的関数はこのキーだけを読み、
    ID の範囲を step ごとには再検証しない。範囲の保証はこの DataModule が Dataset 構築時に持つ。

    `group_attribute_names` は split の列名で、`age_group_65` のような評価専用の年齢群も
    `fairness_age_groups` が生成したあとなので指定できる。`num_groups` は
    `group_cardinalities` の積と一致する必要がある。Hydra は乗算できないので config 側で明示し、
    ここで突き合わせる。train split は全 group を含む必要がある。1 つも現れない group は
    Group DRO の adversarial weight が初期値のまま動かず、条件名だけが残った run になるため。
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
        group_attribute_names: Sequence[str],
        group_cardinalities: Sequence[int],
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
        """group の定義を検証したうえで DataModule を初期化する。"""
        names = list(group_attribute_names)
        cardinalities = [int(value) for value in group_cardinalities]
        if not names:
            raise ValueError("group_attribute_names は1個以上の属性名である必要がある")
        if len(names) != len(cardinalities):
            raise ValueError(f"group_attribute_names と group_cardinalities は同じ長さである必要があるが、{len(names)} と {len(cardinalities)} が指定された")
        if any(value < 2 for value in cardinalities):
            raise ValueError(f"group_cardinalities は 2 以上である必要があるが、{cardinalities} が指定された")
        if prod(cardinalities) != int(num_groups):
            raise ValueError(f"num_groups は group_cardinalities の積 {prod(cardinalities)} と一致する必要があるが、{num_groups} が指定された")
        if not group_key:
            raise ValueError("group_key は空にできない")
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

    def setup(self, stage: str) -> None:
        """親の Dataset を構築し、fit の train split が全 group を含むことを確かめる。

        Args:
            stage: Lightning が渡す stage 名。全 group の確認は `fit` でだけ行う

        Returns:
            None

        Raises:
            ValueError: train split に1行も現れない group がある場合。
        """
        super().setup(stage)
        if stage != "fit":
            return
        counts = self.data_train.group_counts(self.hparams.num_groups)
        empty = torch.where(counts == 0)[0].tolist()
        if empty:
            raise ValueError(f"train split には全 group が必要だが、{empty} が空である")

    def _create_dataset(self, frame: pd.DataFrame, transform: transforms.Compose | None) -> GroupImageDataset:
        """group ID 列を付けた DataFrame から Dataset を生成する。

        `age_group_65` のような評価専用の年齢群は `standardized_dataframes` が作るので、
        group ID を決められるのはこの時点以降になる。
        """
        if self.hparams.group_key in frame.columns:
            raise ValueError(f"split CSV はすでに {self.hparams.group_key!r} を含んでいる")
        frame = frame.copy()
        frame[self.hparams.group_key] = demographic_group_ids(
            frame,
            self.hparams.group_attribute_names,
            self.hparams.group_cardinalities,
            group_key=self.hparams.group_key,
        )
        return GroupImageDataset(
            frame,
            self.hparams.data_dir,
            group_key=self.hparams.group_key,
            transform=transform,
            attribute_names=self.hparams.attribute_names,
            fairness_attribute_names=self.hparams.fairness_attribute_names,
        )
