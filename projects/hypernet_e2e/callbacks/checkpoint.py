"""最終 epoch の状態を `last.ckpt` に残す checkpoint callback。"""

from __future__ import annotations

import lightning.pytorch as pl
from lightning.pytorch.callbacks import ModelCheckpoint


class LastEpochModelCheckpoint(ModelCheckpoint):
    """`save_last=True` の `last.ckpt` を、top-k を保存しなかった epoch でも更新する。

    Lightning 2.6 の `ModelCheckpoint` は、その step で top-k を保存したときにしか `last.ckpt` を
    書かない。`save_top_k=1` と組み合わせると `last.ckpt` は最後に best が更新された epoch で止まり、
    best と同じ中身になる。`run.json` の `last` を「fit を終えた時点の重み」として読めるよう、
    保存の判定を通った epoch では必ず `last.ckpt` を書く。`save_last` が `True` 以外なら
    Lightning の挙動のままにする。

    Lightning の private method に依存するので、挙動は
    `tests/test_checkpoint_callback.py` の実 fit で固定している。
    """

    def on_train_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        """train epoch の終わりで保存する設定なら、top-k の判定の後に `last.ckpt` を更新する。

        Args:
            trainer: 実行中の Trainer
            pl_module: 学習中の module

        Returns:
            None
        """
        due = not self._should_skip_saving_checkpoint(trainer) and self._should_save_on_train_epoch_end(trainer)
        super().on_train_epoch_end(trainer, pl_module)
        if due:
            self._save_last_if_missed(trainer)

    def on_validation_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        """validation の終わりで保存する設定なら、top-k の判定の後に `last.ckpt` を更新する。

        Args:
            trainer: 実行中の Trainer
            pl_module: 学習中の module

        Returns:
            None
        """
        due = not self._should_skip_saving_checkpoint(trainer) and not self._should_save_on_train_epoch_end(trainer)
        super().on_validation_end(trainer, pl_module)
        if due:
            self._save_last_if_missed(trainer)

    def _save_last_if_missed(self, trainer: pl.Trainer) -> None:
        # top-k を保存した step では Lightning が `last.ckpt` も書いているので、二重に書かない。
        if self.save_last is True and self._last_global_step_saved != trainer.global_step:
            self._save_last_checkpoint(trainer, self._monitor_candidates(trainer))
