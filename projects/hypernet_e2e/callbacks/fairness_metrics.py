"""属性別の公平性指標を epoch 単位で記録する callback。"""

import lightning as L
import torch

from projects.hypernet_e2e.utils.metrics import build_eval_attributes, compute_fairness_metrics


class FairnessMetricsCallback(L.Callback):
    """validation / test の出力を epoch ごとに集約して属性別公平性指標を記録する。

    step output は `logits`、`target`、`attributes` を持つ dict とする。`attributes` に
    `evaluation_categorical` があればそれを優先し、なければ `categorical` を使う。二値分類では
    Eopp0 / Eopp1 / Eodds と group 性能由来の gap・worst-group 指標をログする。
    """

    def __init__(self) -> None:
        """検証用・テスト用の batch 出力バッファを初期化する。"""
        self._val_buffer: list[dict] = []
        self._test_buffer: list[dict] = []

    def on_validation_epoch_start(self, trainer: L.Trainer, pl_module: L.LightningModule) -> None:
        """検証 epoch 開始時に出力バッファをクリアする。

        Args:
            trainer: 呼び出し元の Trainer
            pl_module: ログ先の LightningModule

        Returns:
            None
        """
        self._val_buffer = []

    def on_test_epoch_start(self, trainer: L.Trainer, pl_module: L.LightningModule) -> None:
        """テスト epoch 開始時に出力バッファをクリアする。

        Args:
            trainer: 呼び出し元の Trainer
            pl_module: ログ先の LightningModule

        Returns:
            None
        """
        self._test_buffer = []

    def on_validation_batch_end(self, trainer: L.Trainer, pl_module: L.LightningModule, outputs, batch, batch_idx: int, dataloader_idx: int = 0) -> None:
        """検証 batch の step output を保持する。None は属性なしとして無視する。

        Args:
            trainer: 呼び出し元の Trainer
            pl_module: 呼び出し元の LightningModule（バッファリングには使わない）
            outputs: step が返した `logits` / `target` / `attributes` を持つ dict
            batch: 呼び出し元が渡す batch（集計には使わない）
            batch_idx: batch の index（集計には使わない）
            dataloader_idx: 複数 dataloader 時の index

        Returns:
            None
        """
        if outputs is not None:
            self._val_buffer.append(outputs)

    def on_test_batch_end(self, trainer: L.Trainer, pl_module: L.LightningModule, outputs, batch, batch_idx: int, dataloader_idx: int = 0) -> None:
        """テスト batch の step output を保持する。None は属性なしとして無視する。

        Args:
            trainer: 呼び出し元の Trainer
            pl_module: 呼び出し元の LightningModule（バッファリングには使わない）
            outputs: step が返した `logits` / `target` / `attributes` を持つ dict
            batch: 呼び出し元が渡す batch（集計には使わない）
            batch_idx: batch の index（集計には使わない）
            dataloader_idx: 複数 dataloader 時の index

        Returns:
            None
        """
        if outputs is not None:
            self._test_buffer.append(outputs)

    def on_validation_epoch_end(self, trainer: L.Trainer, pl_module: L.LightningModule) -> None:
        """検証 epoch の公平性指標を `val/<attribute>/<metric>` に記録する。

        Args:
            trainer: 呼び出し元の Trainer
            pl_module: ログ先の LightningModule

        Returns:
            None
        """
        self._log_attribute_metrics(pl_module, "val", self._val_buffer)

    def on_test_epoch_end(self, trainer: L.Trainer, pl_module: L.LightningModule) -> None:
        """テスト epoch の公平性指標を `test/<attribute>/<metric>` に記録する。

        Args:
            trainer: 呼び出し元の Trainer
            pl_module: ログ先の LightningModule

        Returns:
            None
        """
        self._log_attribute_metrics(pl_module, "test", self._test_buffer)

    def _log_attribute_metrics(self, pl_module: L.LightningModule, prefix: str, buffer: list[dict]) -> None:
        if not buffer:
            return
        logits = torch.cat([output["logits"] for output in buffer])
        targets = torch.cat([output["target"] for output in buffer])
        attr_key = "evaluation_categorical" if "evaluation_categorical" in buffer[0]["attributes"] else "categorical"
        categorical = torch.cat([output["attributes"][attr_key] for output in buffer])
        missing_key = f"{attr_key}_missing"
        categorical_missing = torch.cat([output["attributes"][missing_key] for output in buffer]) if missing_key in buffer[0]["attributes"] else None
        if attr_key == "evaluation_categorical":
            fairness_names = getattr(pl_module, "fairness_attribute_names", None)
            attr_names = fairness_names.get("categorical") if fairness_names is not None else (None if pl_module.attribute_names is None else pl_module.attribute_names.get(attr_key))
        else:
            attr_names = None if pl_module.attribute_names is None else pl_module.attribute_names.get("categorical")
        eval_attributes, eval_attribute_names = build_eval_attributes(categorical, attr_names, categorical_missing)
        for attribute_name, metrics in compute_fairness_metrics(logits, targets, eval_attributes, eval_attribute_names).items():
            for metric_name, value in metrics.items():
                pl_module.log(f"{prefix}/{attribute_name}/{metric_name}", value)
