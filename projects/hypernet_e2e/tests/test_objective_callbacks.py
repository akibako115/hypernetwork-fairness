"""目的関数の callback を LitModule が Trainer に渡す約束を固定する。"""

import lightning as L
import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

from projects.hypernet_e2e.module import LitModule
from projects.hypernet_e2e.objectives.conditional import RejectFitResume, StepDualUpdate
from projects.hypernet_e2e.objectives.minimax_auc import MinimaxAUCTaskLoss
from projects.hypernet_e2e.objectives.task import TaskLoss

COUNTS = [[30, 10], [10, 50]]


class _Data(TensorDataset):
    def __getitem__(self, index):
        x, group, target = super().__getitem__(index)
        return x, {"group_id": group}, int(target)


def _data() -> _Data:
    return _Data(torch.randn(16, 2), torch.tensor([0, 0, 1, 1] * 4), torch.tensor([0, 1, 0, 1] * 4))


class _DM(L.LightningDataModule):
    def __init__(self, holdout: bool = False) -> None:
        super().__init__()
        self.data_holdout = _data() if holdout else None

    def train_dataloader(self):
        return DataLoader(_data(), batch_size=8)

    def val_dataloader(self):
        return DataLoader(_data(), batch_size=8)

    def held_out_batches(self):
        while True:
            yield from DataLoader(self.data_holdout, batch_size=8)


class _Reader(L.Callback):
    """step の終わりに、双対の確定回数を読む（診断を記録する callback の代わり）。"""

    def __init__(self) -> None:
        self.seen: list[int] = []

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx) -> None:
        self.seen.append(int(pl_module.training_objective.update_count))


def _trainer(tmp_path, **kwargs) -> L.Trainer:
    kwargs.setdefault("max_epochs", 1)
    kwargs.setdefault("enable_checkpointing", False)
    return L.Trainer(
        accelerator="cpu",
        logger=False,
        enable_model_summary=False,
        enable_progress_bar=False,
        num_sanity_val_steps=0,
        default_root_dir=tmp_path,
        **kwargs,
    )


def _module(objective) -> LitModule:
    return LitModule(torch.nn.Linear(2, 2), objective, optimizer=lambda params: torch.optim.SGD(params, lr=0.1), scheduler=None)


def test_objective_callbacks_run_before_the_trainer_callbacks_and_are_not_duplicated(tmp_path) -> None:
    reader = _Reader()
    module = _module(MinimaxAUCTaskLoss(2, COUNTS))
    trainer = _trainer(tmp_path, callbacks=[reader])
    trainer.fit(module, datamodule=_DM())

    # 確定は step の終わりの他の callback より先に済んでいる（2 step で 1, 2 を読む）。
    assert reader.seen == [1, 2]
    kinds = [type(callback) for callback in trainer.callbacks]
    assert kinds.index(StepDualUpdate) < kinds.index(_Reader)
    assert kinds.count(StepDualUpdate) == 1 and kinds.count(RejectFitResume) == 1
    # 同じ Trainer で評価しても、目的関数の callback は置き換わるだけで増えない。
    trainer.validate(module, datamodule=_DM(), verbose=False)
    kinds = [type(callback) for callback in trainer.callbacks]
    assert kinds.count(StepDualUpdate) == 1 and kinds.count(_Reader) == 1
    assert len({id(callback) for callback in trainer.callbacks}) == len(trainer.callbacks)


def test_parent_and_child_checkpoint_types_are_neither_duplicated_nor_dropped(tmp_path) -> None:
    """HAM10000 の公平性 preset は LastEpochModelCheckpoint と、その親の ModelCheckpoint を両方持つ。"""
    from lightning.pytorch.callbacks import ModelCheckpoint

    from projects.hypernet_e2e.callbacks.checkpoint import LastEpochModelCheckpoint

    reader = _Reader()
    last = LastEpochModelCheckpoint(dirpath=tmp_path / "last", save_last=True)
    worst = ModelCheckpoint(dirpath=tmp_path / "worst", filename="w{epoch}")
    module = _module(MinimaxAUCTaskLoss(2, COUNTS))
    trainer = _trainer(tmp_path, callbacks=[reader, last, worst], enable_checkpointing=True)
    for _ in range(2):
        trainer.fit(module, datamodule=_DM())
        assert [id(callback) for callback in trainer.callbacks].count(id(last)) == 1
        assert [id(callback) for callback in trainer.callbacks].count(id(worst)) == 1
        assert len({id(callback) for callback in trainer.callbacks}) == len(trainer.callbacks)
        assert isinstance(trainer.callbacks[0], RejectFitResume) and isinstance(trainer.callbacks[1], StepDualUpdate)
        assert trainer.callbacks[-2:] == [last, worst]
        trainer.fit_loop.max_epochs += 1


def test_objective_without_callbacks_leaves_the_trainer_callbacks_alone(tmp_path) -> None:
    marker = L.Callback()
    trainer = _trainer(tmp_path, callbacks=[marker])
    before = list(trainer.callbacks)
    trainer.fit(_module(TaskLoss()), datamodule=_DM())
    assert trainer.callbacks == before


def test_held_out_is_rejected_for_an_objective_that_does_not_use_it(tmp_path) -> None:
    with pytest.raises(ValueError, match="held-out"):
        _trainer(tmp_path).fit(_module(TaskLoss()), datamodule=_DM(holdout=True))


def test_step_dual_update_requires_one_update_per_step(tmp_path) -> None:
    with pytest.raises(ValueError, match="accumulation=1"):
        _trainer(tmp_path, accumulate_grad_batches=2).fit(_module(MinimaxAUCTaskLoss(2, COUNTS)), datamodule=_DM())


def test_held_out_diagnostics_reach_the_module_after_each_step(tmp_path) -> None:
    module = _module(MinimaxAUCTaskLoss(2, COUNTS))
    _trainer(tmp_path).fit(module, datamodule=_DM(holdout=True))
    assert set(module.held_out_diagnostics) >= {"risk", "q_before"}
    # 目的関数の診断値は学習バッチのものに戻っている。
    assert not torch.equal(module.held_out_diagnostics["risk"], module.training_objective.batch_diagnostics()["risk"])
