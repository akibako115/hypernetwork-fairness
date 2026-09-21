"""反復 run を構成する単一 stage の学習を Lightning へ接続する module。"""

from __future__ import annotations

import logging
from collections.abc import Callable, Mapping, Sequence
from typing import Any

import lightning as L
import torch
import torch.nn as nn
import torch.nn.functional as F

from projects.hypernet_iterative.loss import ObjectiveInput
from projects.hypernet_iterative.models.utils import load_compatible_state_dict

log = logging.getLogger(__name__)


class LitModule(L.LightningModule):
    """iterative run の1 stage における forward・loss・optimizer 構築を所有する。

    batch は ``(image, attributes, target)``、各 step の戻り値は MetricsLogger と
    FairnessMetricsCallback が読む ``loss/logits/preds/target/attributes`` を持つ dict とする。
    ``attribute_names`` は model 入力、``fairness_attribute_names`` は評価専用 categorical 属性の
    名前を表す。
    train は ``loss_fn``、validation/test は実験条件をまたいで比較できる通常の
    cross-entropy を用いる。``use_attributes=True`` のときだけ attributes を net に渡す。
    warm-start は直前 stage の ``net.*`` tensor だけを厳密一致で読むため、optimizer、scheduler、
    GroupDRO の状態はこの新しい stage に引き継がれない。
    """

    def __init__(
        self,
        net: nn.Module,
        loss_fn: nn.Module,
        optimizer: Callable[..., torch.optim.Optimizer] | None,
        scheduler: Callable[..., Any] | None,
        compile: bool = False,
        use_attributes: bool = False,
        backbone_checkpoint_path: str | None = None,
        warm_start_checkpoint_path: str | None = None,
        freeze_backbone: bool = False,
        attribute_names: Mapping[str, Sequence[str]] | None = None,
        fairness_attribute_names: Mapping[str, Sequence[str]] | None = None,
    ) -> None:
        """モデル、目的関数、および単一 stage の初期化設定を保持する。"""
        super().__init__()
        # optimizer / scheduler factory は lambda や Hydra partial を取り得る。hparams に残すと
        # checkpoint に pickle され、load 側が同じ import を解決できることを要求してしまう。
        # ignore して実行中の module だけが保持し、Lightning のシリアライズ経路から切り離す。
        self.save_hyperparameters(logger=False, ignore=["net", "loss_fn", "optimizer", "scheduler"])
        self.net = net
        self.loss_fn = loss_fn
        self._optimizer_factory = optimizer
        self._scheduler_factory = scheduler
        self.use_attributes = use_attributes
        self.attribute_names = attribute_names
        self.fairness_attribute_names = fairness_attribute_names

    @property
    def training_objective(self) -> nn.Module:
        """train phase で最適化する注入済みの目的関数を返す。

        Args:
            なし

        Returns:
            nn.Module: `ObjectiveInput` を受け取り scalar loss を返す目的関数
        """
        return self.loss_fn

    def forward(
        self,
        image: torch.Tensor,
        attributes: Mapping[str, torch.Tensor] | None = None,
    ) -> torch.Tensor:
        """画像から ``(batch_size, num_classes)`` の logits を返す。

        Args:
            image: `[B, 3, H, W]` の画像 batch
            attributes: `use_attributes` が真のとき必須の属性辞書

        Returns:
            torch.Tensor: `[B, num_classes]` の logits

        Raises:
            ValueError: `use_attributes` が真なのに attributes が None の場合。
        """
        if not self.use_attributes:
            return self.net(image)
        if attributes is None:
            raise ValueError("use_attributes=True の場合 attributes が必要")
        return self.net(image, attributes)

    def training_step(self, batch: tuple[Any, Mapping[str, torch.Tensor], torch.Tensor], batch_idx: int) -> dict[str, Any]:
        """注入された学習目的関数で loss を計算する。

        Args:
            batch: `(image, attributes, target)`
            batch_idx: batch の index（計算には使わない）

        Returns:
            dict[str, Any]: `loss` / `logits` / `preds` / `target` / `attributes`。
                callback がこれを epoch 集計に使う。`loss` は backward 可能なまま返す
        """
        logits, preds, target, attributes = self._shared_step(batch)
        loss = self.training_objective(ObjectiveInput(logits=logits, target=target, attributes=attributes))
        return self._step_output(loss, logits, preds, target, attributes, detach_loss=False)

    def validation_step(self, batch: tuple[Any, Mapping[str, torch.Tensor], torch.Tensor], batch_idx: int) -> dict[str, Any]:
        """比較用の通常 cross-entropy で validation loss を計算する。

        train の目的関数が group 目的関数でも、この loss は常に素の cross-entropy である。

        Args:
            batch: `(image, attributes, target)`
            batch_idx: batch の index（計算には使わない）

        Returns:
            dict[str, Any]: `loss` / `logits` / `preds` / `target` / `attributes`。
                callback がこれを epoch 集計に使う
        """
        return self._evaluation_step(batch)

    def test_step(self, batch: tuple[Any, Mapping[str, torch.Tensor], torch.Tensor], batch_idx: int) -> dict[str, Any]:
        """比較用の通常 cross-entropy で test loss を計算する。

        train の目的関数が group 目的関数でも、この loss は常に素の cross-entropy である。

        Args:
            batch: `(image, attributes, target)`
            batch_idx: batch の index（計算には使わない）

        Returns:
            dict[str, Any]: `loss` / `logits` / `preds` / `target` / `attributes`。
                callback がこれを epoch 集計に使う
        """
        return self._evaluation_step(batch)

    def setup(self, stage: str) -> None:
        """fit 開始時だけ、stage に必要な初期化を順に行う。

        backbone checkpoint の読み込み、freeze、データ依存初期化、warm-start の順に実行する。
        warm-start をデータ依存初期化より後に置くのは、参照 run の net を最終状態とするため。

        Args:
            stage: Lightning が渡す stage 名。`fit` 以外では何もしない

        Returns:
            None

        Raises:
            RuntimeError: backbone checkpoint から1つも重みを読めなかった場合。
            ValueError: warm-start checkpoint の net state が現在の構成と一致しない場合。
        """
        if stage != "fit":
            return
        if self.hparams.backbone_checkpoint_path is not None:
            result = self.load_backbone_checkpoint(self.hparams.backbone_checkpoint_path)
            self._require_loaded_backbone(result, self.hparams.backbone_checkpoint_path)
        if self.hparams.freeze_backbone:
            self._freeze_backbone()
        if hasattr(self.net, "initialize_with_attributes"):
            self.net.initialize_with_attributes(self.trainer.datamodule.train_attributes())
        # warm-start は optimizer / trainer state を復元せず、net の重みだけを引き継ぐ。
        # データ依存初期化より後に実行し、参照 run の net を最終状態とする。
        if self.hparams.warm_start_checkpoint_path is not None:
            loaded = self.load_warm_start_checkpoint(self.hparams.warm_start_checkpoint_path)
            log.info("Warm-started %s net tensors from %s", loaded, self.hparams.warm_start_checkpoint_path)
        if self.hparams.compile:
            self.net = torch.compile(self.net)

    def on_train_epoch_start(self) -> None:
        """凍結した backbone の BatchNorm 等を eval 状態に保つ。

        Lightning は epoch ごとに module を train モードへ戻すので、毎 epoch 掛け直す。

        Args:
            なし

        Returns:
            None
        """
        if self.hparams.freeze_backbone and (backbone := getattr(self.net, "backbone", None)) is not None:
            backbone.eval()

    def load_backbone_checkpoint(self, checkpoint_path: str, load_fc: bool = True) -> dict[str, list[str]]:
        """checkpoint から backbone 相当の互換する重みだけを読み込む。

        Args:
            checkpoint_path: 読み込む checkpoint の path
            load_fc: False なら `fc.` で始まるキーを読み込まない

        Returns:
            dict[str, list[str]]: loaded_keys / skipped_keys / missing_keys / unexpected_keys

        Raises:
            TypeError: checkpoint が state_dict でも `state_dict` キーを持つ mapping でもない場合。
        """
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        raw_state = checkpoint.get("state_dict", checkpoint)
        if not isinstance(raw_state, Mapping):
            raise TypeError("backbone checkpoint は state_dict、または 'state_dict' キーを持つ mapping である必要がある")
        state_dict = {key.removeprefix("net."): value for key, value in raw_state.items()}
        if hasattr(self.net, "load_base_state_dict"):
            return self.net.load_base_state_dict(state_dict)
        if hasattr(self.net, "backbone"):
            backbone_dict = {key.removeprefix("backbone."): value for key, value in state_dict.items()}
            return load_compatible_state_dict(self.net.backbone, backbone_dict, load_fc=load_fc)
        return load_compatible_state_dict(self.net, state_dict, load_fc=load_fc)

    def load_warm_start_checkpoint(self, checkpoint_path: str) -> int:
        """直前 stage checkpoint の net tensor だけを完全一致で読み込む。

        optimizer、scheduler、loss function の buffer は読まない。stage ごとの状態 reset は
        workflow が新しい Lightning process を起動することで保証し、この method はモデルの
        初期値だけを担当する。

        Args:
            checkpoint_path: 直前 stage が書いた checkpoint の path

        Returns:
            int: 読み込んだ net tensor の数

        Raises:
            TypeError: checkpoint が state_dict でも `state_dict` キーを持つ mapping でもない場合。
            ValueError: `net.` で始まるキーが無い場合、または net state が現在の構成と
                完全一致しない場合。
        """
        raw_state = self._checkpoint_state_dict(checkpoint_path, purpose="warm-start")
        source_state = {key.removeprefix("net."): value for key, value in raw_state.items() if isinstance(key, str) and key.startswith("net.")}
        if not source_state:
            raise ValueError("warm-start checkpoint に 'net.' で始まる state_dict key が存在しない")
        target_state = self.net.state_dict()
        missing = sorted(set(target_state) - set(source_state))
        unexpected = sorted(set(source_state) - set(target_state))
        shape_mismatches = sorted(key for key in set(target_state) & set(source_state) if target_state[key].shape != source_state[key].shape)
        if missing or unexpected or shape_mismatches:
            raise ValueError(f"warm-start の net state が現在のモデル構成と完全一致しない: missing={missing[:5]}, unexpected={unexpected[:5]}, shape_mismatches={shape_mismatches[:5]}")
        self.net.load_state_dict(source_state, strict=True)
        return len(source_state)

    @staticmethod
    def _checkpoint_state_dict(checkpoint_path: str, *, purpose: str) -> Mapping[str, Any]:
        """trusted checkpoint から state_dict container を取り出す。"""
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        raw_state = checkpoint.get("state_dict", checkpoint)
        if not isinstance(raw_state, Mapping):
            raise TypeError(f"{purpose} checkpoint は state_dict、または 'state_dict' キーを持つ mapping である必要がある")
        return raw_state

    def configure_optimizers(self) -> dict[str, Any]:
        """学習可能な net parameter から optimizer と任意 scheduler を構築する。

        Args:
            なし

        Returns:
            dict[str, Any]: `optimizer` と、scheduler を注入した場合は `val/loss` を monitor
                する `lr_scheduler`

        Raises:
            ValueError: optimizer factory が未指定の場合、または trainable parameter が
                1つも無い場合。
        """
        if self._optimizer_factory is None:
            raise ValueError("optimizer を指定する必要がある")
        parameters = [parameter for parameter in self.net.parameters() if parameter.requires_grad]
        if not parameters:
            raise ValueError("optimizer 構成に使う trainable parameter が見つからない")
        optimizer = self._optimizer_factory(params=parameters)
        if self._scheduler_factory is None:
            return {"optimizer": optimizer}
        scheduler = self._scheduler_factory(optimizer=optimizer)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "monitor": "val/loss", "interval": "epoch", "frequency": 1},
        }

    def _shared_step(self, batch: tuple[Any, Mapping[str, torch.Tensor], torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, Mapping[str, torch.Tensor]]:
        image, attributes, target = batch
        logits = self(image, attributes)
        target = target.long()
        return logits, torch.argmax(logits, dim=1), target, attributes

    def _evaluation_step(self, batch: tuple[Any, Mapping[str, torch.Tensor], torch.Tensor]) -> dict[str, Any]:
        logits, preds, target, attributes = self._shared_step(batch)
        return self._step_output(F.cross_entropy(logits, target), logits, preds, target, attributes, detach_loss=True)

    @staticmethod
    def _step_output(
        loss: torch.Tensor,
        logits: torch.Tensor,
        preds: torch.Tensor,
        target: torch.Tensor,
        attributes: Mapping[str, torch.Tensor],
        *,
        detach_loss: bool,
    ) -> dict[str, Any]:
        return {
            "loss": loss.detach() if detach_loss else loss,
            "logits": logits.detach(),
            "preds": preds.detach(),
            "target": target.detach(),
            "attributes": {key: value.detach() for key, value in attributes.items()},
        }

    @staticmethod
    def _require_loaded_backbone(result: Mapping[str, list[str]], checkpoint_path: str) -> None:
        """部分ロードの結果を記録し、1 key も一致しなかった run を失敗させる。

        backbone checkpoint は LoRA 生成器を持たないので部分一致が正常な状態である。ただし
        key が 1 つも一致しない場合、ランダム初期化のまま学習が進み、artifact 上は成功した run
        として残ってしまう。silent no-op をここで止める。
        """
        if not result["loaded_keys"]:
            raise RuntimeError(f"backbone checkpoint と shape まで一致する parameter が 1 つも無い: {checkpoint_path}")
        log.info(
            "Loaded %d tensors from %s (skipped %d, missing %d, unexpected %d)",
            len(result["loaded_keys"]),
            checkpoint_path,
            len(result["skipped_keys"]),
            len(result["missing_keys"]),
            len(result["unexpected_keys"]),
        )

    def _freeze_backbone(self) -> None:
        if hasattr(self.net, "freeze_base_model"):
            self.net.freeze_base_model()
            return
        backbone = getattr(self.net, "backbone", None)
        if backbone is None:
            raise TypeError("freeze_backbone=True の場合 net は 'backbone' 属性を持つ必要がある")
        for parameter in backbone.parameters():
            parameter.requires_grad = False
        backbone.eval()
