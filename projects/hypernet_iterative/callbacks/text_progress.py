"""epoch ごとの進捗を1行のテキストとして run log へ書き出す callback。

親 workflow が stage を独立 process で回すため、進捗は tty ではなくファイルに残す。
"""

from __future__ import annotations

import logging
import sys
from typing import Any

import lightning as L
import torch

log = logging.getLogger(__name__)


class TextProgressLogger(L.Callback):
    """TTY 非依存のテキスト進捗ログ。nohup 実行時にも epoch ごとの進捗を残す。"""

    def __init__(self) -> None:
        """プロセス全体の例外 hook を復元できるよう、差し替え前の hook を保持する。"""
        self._previous_excepthook: Any = None

    def setup(self, trainer: L.Trainer, pl_module: L.LightningModule, stage: str) -> None:
        """`sys.excepthook` を差し替え、nohup 実行時でも未捕捉例外をログへ残す。

        Args:
            trainer: 呼び出し元の Trainer
            pl_module: 呼び出し元の LightningModule
            stage: Lightning が渡す stage 名

        Returns:
            None
        """
        if self._previous_excepthook is None:
            self._previous_excepthook = sys.excepthook
            sys.excepthook = self._excepthook

    def teardown(self, trainer: L.Trainer, pl_module: L.LightningModule, stage: str) -> None:
        """この callback が登録した hook だけを差し替え前の状態へ戻す。

        Args:
            trainer: 呼び出し元の Trainer
            pl_module: 呼び出し元の LightningModule
            stage: Lightning が渡す stage 名

        Returns:
            None
        """
        if self._previous_excepthook is not None and sys.excepthook == self._excepthook:
            sys.excepthook = self._previous_excepthook
        self._previous_excepthook = None

    def _excepthook(self, exc_type, exc_value, exc_tb):
        """未捕捉例外を critical ログへ記録してから、元の excepthook に処理を委譲する。"""
        log.critical("Uncaught exception", exc_info=(exc_type, exc_value, exc_tb))
        hook = self._previous_excepthook or sys.__excepthook__
        hook(exc_type, exc_value, exc_tb)

    @staticmethod
    def _format_metric(value: object) -> str:
        """scalar Tensor と Python float を同じ小数精度で表示する。"""
        if isinstance(value, torch.Tensor) and value.numel() == 1:
            return f"{value.item():.4f}"
        if isinstance(value, float):
            return f"{value:.4f}"
        return str(value)

    def on_train_epoch_end(self, trainer: L.Trainer, pl_module: L.LightningModule) -> None:
        """train epoch 末に callback_metrics をテキスト整形し、1行のログとして出力する。

        Args:
            trainer: `callback_metrics` と epoch 番号の取得元
            pl_module: 呼び出し元の LightningModule（出力には使わない）

        Returns:
            None
        """
        # 内部用（"_" 始まり）のキーを除外し、epoch サマリとしてログ出力する対象を絞り込む
        metrics = {k: v for k, v in trainer.callback_metrics.items() if not k.startswith("_")}
        parts = [f"{key}={self._format_metric(value)}" for key, value in sorted(metrics.items())]
        log.info("Epoch %03d/%03d  %s", trainer.current_epoch, trainer.max_epochs, "  ".join(parts))
