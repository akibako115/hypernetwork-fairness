"""群・クラスセルを均等に復元抽出する。"""

import math
from collections.abc import Iterator, Sequence
from typing import Any

import torch
from numpy.typing import NDArray
from torch.utils.data import DataLoader, Sampler

from .group_datamodule import GroupImageDataModule


class GroupClassBatchSampler(Sampler[list[int]]):
    """各セル同数のフルバッチを seed+epoch により再現する。

    各セル内 IID 復元抽出。最後のバッチもフルサイズ。
    """

    def __init__(self, group_ids: Sequence[int] | NDArray | torch.Tensor, targets: Sequence[int] | NDArray | torch.Tensor, num_groups: int, batch_size: int, seed: int = 42) -> None:
        """セル index とバッチサイズを検証する。

        Args:
            group_ids: 行ごとの群 ID。
            targets: 行ごとの二値ラベル。
            num_groups: 群数。
            batch_size: 2G の倍数。
            seed: 抽出 seed。

        Returns:
            None
        """
        groups, labels = torch.as_tensor(group_ids), torch.as_tensor(targets)
        if groups.ndim != 1 or groups.shape != labels.shape or not len(groups):
            raise ValueError("非空の同形状 group/target が必要")
        if ((groups < 0) | (groups >= num_groups)).any() or ((labels < 0) | (labels > 1)).any():
            raise ValueError("group/target が範囲外")
        if batch_size <= 0 or batch_size % (2 * num_groups):
            raise ValueError("batch_size は 2G の正の倍数が必要")
        self.cells = [torch.where((groups == g) & (labels == c))[0] for g in range(num_groups) for c in range(2)]
        if any(not len(cell) for cell in self.cells):
            raise ValueError("train に空の群・クラスセルがある")
        self.seed, self.epoch = int(seed), 0
        self.batch_size = int(batch_size)
        self.cell_size = batch_size // (2 * num_groups)
        self.batches = math.ceil(len(groups) / batch_size)

    def set_epoch(self, epoch: int) -> None:
        """次の抽出に使う epoch を設定する。

        Args:
            epoch: 0 始まりの epoch。

        Returns:
            None
        """
        self.epoch = int(epoch)

    def __iter__(self) -> Iterator[list[int]]:
        """epoch の抽出順と抽出件数の診断を生成する。

        Args:
            なし

        Returns:
            Iterator: バッチごとの行 index。診断はprefetchを含む抽出数。
        """
        generator = torch.Generator().manual_seed(self.seed + self.epoch)
        self.draw_counts = torch.zeros(sum(len(cell) for cell in self.cells), dtype=torch.long)
        for _ in range(self.batches):
            indices = torch.cat([cell[torch.randint(len(cell), (self.cell_size,), generator=generator)] for cell in self.cells])
            self.draw_counts.scatter_add_(0, indices, torch.ones_like(indices))
            yield indices[torch.randperm(self.batch_size, generator=generator)].tolist()

    def __len__(self) -> int:
        """フルバッチの数を返す。

        Args:
            なし

        Returns:
            int: ceil(N/B)。
        """
        return self.batches


class GroupClassImageDataModule(GroupImageDataModule):
    """train のみセル均等抽出を使い、val/test は自然分布を保つ。"""

    def train_dataloader(self) -> DataLoader[Any]:
        """群・クラス別に均等なバッチを返す。

        Args:
            なし

        Returns:
            DataLoader: 復元抽出 train loader。
        """
        frame = self.data_train.df
        self.group_class_sampler = GroupClassBatchSampler(
            frame[self.hparams.group_key].to_numpy(), frame.target.to_numpy(), self.hparams.num_groups, self.batch_size_per_device, self.hparams.sampling_seed
        )
        self.group_class_sampler.set_epoch(getattr(getattr(self, "trainer", None), "current_epoch", 0))
        return DataLoader(
            self.data_train,
            batch_sampler=self.group_class_sampler,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            persistent_workers=self.hparams.persistent_workers,
            prefetch_factor=self.hparams.prefetch_factor,
        )

    def __init__(self, sampling_seed: int = 42, **kwargs: Any) -> None:
        """親のデータ設定と抽出 seed を保持する。

        Args:
            sampling_seed: seed+epoch に使う seed。
            **kwargs: GroupImageDataModule の設定。

        Returns:
            None
        """
        requested = kwargs.pop("train_sampling", "group_class_balanced")
        if requested != "group_class_balanced":
            raise ValueError("GroupClassImageDataModule は group_class_balanced のみ対応")
        super().__init__(train_sampling="uniform", **kwargs)
        self.hparams.train_sampling = requested
        self.hparams.sampling_seed = int(sampling_seed)
