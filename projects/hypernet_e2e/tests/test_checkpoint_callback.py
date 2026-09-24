from pathlib import Path

import lightning as L
import torch
from lightning.pytorch.callbacks import ModelCheckpoint
from torch.utils.data import DataLoader, TensorDataset

from projects.hypernet_e2e.callbacks.checkpoint import LastEpochModelCheckpoint

_MAX_EPOCHS = 4


class _PeakAtFirstEpoch(L.LightningModule):
    """val/auroc が epoch 0 で最大になり、以後は更新されない module。"""

    def __init__(self) -> None:
        super().__init__()
        self.layer = torch.nn.Linear(1, 1)

    def training_step(self, batch: list[torch.Tensor], batch_idx: int) -> torch.Tensor:
        return self.layer(batch[0]).pow(2).mean()

    def validation_step(self, batch: list[torch.Tensor], batch_idx: int) -> None:
        self.log("val/auroc", 1.0 - 0.1 * self.current_epoch)

    def configure_optimizers(self) -> torch.optim.Optimizer:
        return torch.optim.SGD(self.parameters(), lr=0.1)


def _fit(callback: ModelCheckpoint, root: Path) -> None:
    loader = DataLoader(TensorDataset(torch.ones(4, 1)), batch_size=2)
    trainer = L.Trainer(
        max_epochs=_MAX_EPOCHS,
        callbacks=[callback],
        logger=False,
        enable_progress_bar=False,
        enable_model_summary=False,
        default_root_dir=str(root),
        accelerator="cpu",
    )
    trainer.fit(_PeakAtFirstEpoch(), train_dataloaders=loader, val_dataloaders=loader)


def _checkpoint(callback_class: type[ModelCheckpoint], dirpath: Path) -> ModelCheckpoint:
    return callback_class(
        dirpath=str(dirpath),
        filename="best_val_auroc_{epoch:03d}",
        monitor="val/auroc",
        mode="max",
        save_top_k=1,
        save_last=True,
        auto_insert_metric_name=False,
    )


def _epoch(path: str | Path) -> int:
    return int(torch.load(path, map_location="cpu", weights_only=False)["epoch"])


def test_last_checkpoint_follows_the_final_epoch_when_best_stops_improving(tmp_path: Path) -> None:
    callback = _checkpoint(LastEpochModelCheckpoint, tmp_path / "checkpoints")

    _fit(callback, tmp_path)

    assert Path(callback.best_model_path).name == "best_val_auroc_000.ckpt"
    assert _epoch(callback.best_model_path) == 0
    assert Path(callback.last_model_path).name == "last.ckpt"
    assert _epoch(callback.last_model_path) == _MAX_EPOCHS - 1


def test_stock_model_checkpoint_still_freezes_last_at_the_best_epoch(tmp_path: Path) -> None:
    """LastEpochModelCheckpoint が要る理由を固定する。

    Lightning がこの挙動を直したら、このテストが落ちる。そのときは subclass を外してよい。
    """
    callback = _checkpoint(ModelCheckpoint, tmp_path / "checkpoints")

    _fit(callback, tmp_path)

    assert _epoch(callback.last_model_path) == 0


def test_last_checkpoint_writes_nothing_extra_without_save_last(tmp_path: Path) -> None:
    callback = LastEpochModelCheckpoint(dirpath=str(tmp_path / "checkpoints"), monitor="val/auroc", mode="max", save_top_k=1, save_last=False)

    _fit(callback, tmp_path)

    assert not callback.last_model_path
    assert [path.name for path in (tmp_path / "checkpoints").iterdir()] == [Path(callback.best_model_path).name]
