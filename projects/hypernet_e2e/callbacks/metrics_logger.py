"""全体性能の epoch 集計を Lightning のログへ接続する callback。"""

from __future__ import annotations

import lightning as L
from torchmetrics import MeanMetric
from torchmetrics.classification import MulticlassAccuracy, MulticlassAUROC


class MetricsLogger(L.Callback):
    """loss / acc / bacc / auroc のバッチ集計ログを担当する汎用コールバック。"""

    def __init__(self, num_classes: int) -> None:
        """train/val/test の各 phase 用に loss/acc/bacc/auroc の torchmetrics オブジェクトを用意する。"""
        super().__init__()
        self._losses: dict[str, MeanMetric] = {}
        self._accuracies: dict[str, MulticlassAccuracy] = {}
        self._baccs: dict[str, MulticlassAccuracy] = {}
        self._aurocs: dict[str, MulticlassAUROC] = {}

        for phase in ("train", "val", "test"):
            self._losses[phase] = MeanMetric()
            self._accuracies[phase] = MulticlassAccuracy(num_classes=num_classes, average="micro")
            self._baccs[phase] = MulticlassAccuracy(num_classes=num_classes, average="macro")
            self._aurocs[phase] = MulticlassAUROC(num_classes=num_classes, average="macro")

    def setup(self, trainer: L.Trainer, pl_module: L.LightningModule, stage: str) -> None:
        """metric を pl_module と同じ device へ移動し、`pl_module.log()` が `metric_attribute` で
        参照できるよう `_ml_<metric>_<phase>` の名前で pl_module の属性として登録する。
        """
        device = pl_module.device
        for phase in ("train", "val", "test"):
            self._losses[phase] = self._losses[phase].to(device)
            self._accuracies[phase] = self._accuracies[phase].to(device)
            self._baccs[phase] = self._baccs[phase].to(device)
            self._aurocs[phase] = self._aurocs[phase].to(device)
            setattr(pl_module, f"_ml_loss_{phase}", self._losses[phase])
            setattr(pl_module, f"_ml_acc_{phase}", self._accuracies[phase])
            setattr(pl_module, f"_ml_bacc_{phase}", self._baccs[phase])
            setattr(pl_module, f"_ml_auroc_{phase}", self._aurocs[phase])

    def on_fit_start(self, trainer: L.Trainer, pl_module: L.LightningModule) -> None:
        """WandbLogger 使用時、全メトリクスの x 軸を step ではなく epoch に統一する。"""
        from lightning.pytorch.loggers import WandbLogger

        if isinstance(pl_module.logger, WandbLogger):
            pl_module.logger.experiment.define_metric("epoch")
            pl_module.logger.experiment.define_metric("*", step_metric="epoch")

    def on_train_batch_end(
        self,
        trainer: L.Trainer,
        pl_module: L.LightningModule,
        outputs,
        batch,
        batch_idx: int,
    ) -> None:
        """train バッチの step output を集計し、`train/*` としてログする。"""
        self._step(pl_module, outputs, "train")

    def on_validation_batch_end(
        self,
        trainer: L.Trainer,
        pl_module: L.LightningModule,
        outputs,
        batch,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        """検証バッチの step output を集計し、`val/*` としてログする。"""
        self._step(pl_module, outputs, "val")

    def on_test_batch_end(
        self,
        trainer: L.Trainer,
        pl_module: L.LightningModule,
        outputs,
        batch,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        """テストバッチの step output を集計し、`test/*` としてログする。"""
        self._step(pl_module, outputs, "test")

    def _step(self, pl_module: L.LightningModule, outputs: dict | None, phase: str) -> None:
        """1バッチ分の step output を loss/acc/bacc/auroc の各 torchmetrics に投入し、
        エポック集計として `<phase>/<metric>` でログする。

        Args:
            pl_module: ログ先の LightningModule。
            outputs:   step output（`loss`/`preds`/`target`/`logits` を含む）。None なら何もしない。
            phase:     "train" / "val" / "test"。
        """
        if outputs is None:
            return

        prog_bar_bacc = phase in ("val", "test")

        # loss をサンプル数で重み付けして集計し、progress bar にも表示する
        self._losses[phase](outputs["loss"], weight=outputs["target"].numel())
        pl_module.log(
            f"{phase}/loss",
            self._losses[phase],
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            metric_attribute=f"_ml_loss_{phase}",
        )

        # micro 平均の accuracy を集計する
        self._accuracies[phase](outputs["preds"], outputs["target"])
        pl_module.log(
            f"{phase}/acc",
            self._accuracies[phase],
            on_step=False,
            on_epoch=True,
            prog_bar=False,
            metric_attribute=f"_ml_acc_{phase}",
        )

        # macro 平均の balanced accuracy を集計し、val/test では progress bar にも表示する
        self._baccs[phase](outputs["preds"], outputs["target"])
        pl_module.log(
            f"{phase}/bacc",
            self._baccs[phase],
            on_step=False,
            on_epoch=True,
            prog_bar=prog_bar_bacc,
            metric_attribute=f"_ml_bacc_{phase}",
        )

        # macro 平均の AUROC を集計する
        self._aurocs[phase](outputs["logits"], outputs["target"])
        pl_module.log(
            f"{phase}/auroc",
            self._aurocs[phase],
            on_step=False,
            on_epoch=True,
            prog_bar=False,
            metric_attribute=f"_ml_auroc_{phase}",
        )
