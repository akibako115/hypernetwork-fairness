"""hypernet_e2e の単段学習を Lightning へ接続する module。"""

from __future__ import annotations

import json
import logging
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

import lightning as L
import torch
import torch.nn as nn
import torch.nn.functional as F
from lightning.pytorch.trainer.states import TrainerFn

from projects.hypernet_e2e.loss import ObjectiveInput
from projects.hypernet_e2e.models.utils import load_compatible_state_dict
from projects.hypernet_e2e.objectives.conditional import ConditionalObjective

log = logging.getLogger(__name__)


class LitModule(L.LightningModule):
    """e2e の1回の fit における forward・loss・optimizer 構築を所有する。

    batch は ``(image, attributes, target)``、各 step の戻り値は MetricsLogger と
    FairnessMetricsCallback が読む ``loss/logits/preds/target/attributes`` を持つ dict とする。
    ``attribute_names`` は model 入力、``fairness_attribute_names`` は評価専用 categorical 属性の
    名前を表す。
    train は ``loss_fn``、validation/test は実験条件をまたいで比較できる通常の
    cross-entropy を用いる。``use_attributes=True`` のときだけ attributes を net に渡す。
    ``freeze_backbone=True`` は ``backbone_checkpoint_path`` を伴う必要がある。
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
        freeze_backbone: bool = False,
        attribute_names: Mapping[str, Sequence[str]] | None = None,
        fairness_attribute_names: Mapping[str, Sequence[str]] | None = None,
    ) -> None:
        """モデル、目的関数、および単段 fit の初期化設定を保持する。

        Args:
            net: logits を返す分類モデル。
            loss_fn: `ObjectiveInput` から task loss を計算する目的関数。
            optimizer: trainable parameter を受けて optimizer を作る factory。
            scheduler: optimizer を受けて scheduler を作る任意の factory。
            compile: `torch.compile` を fit 開始時に使うか。
            use_attributes: net の forward に属性辞書を渡すか。
            backbone_checkpoint_path: 初期化または第2段用に読む checkpoint の path。
            freeze_backbone: checkpoint 読み込み後に base model を凍結するか。
            attribute_names: model 入力属性の列名定義。
            fairness_attribute_names: 評価専用属性の列名定義。

        Returns:
            None

        Raises:
            ValueError: checkpoint 無しで backbone 凍結を指定した場合。
        """
        super().__init__()
        # backbone の重みは backbone_checkpoint_path からしか入らない。path 無しで凍結すると
        # ランダム初期化のまま固定された backbone で学習が完走し、artifact 上は成功した run に
        # 見えてしまう。2 段目 run の設定ミスをここで止める。
        if freeze_backbone and backbone_checkpoint_path is None:
            raise ValueError("freeze_backbone=True には backbone_checkpoint_path が必要")
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
        # 双対の更新に使う held-out のバッチの流れ。DataModule が held-out を持つときだけ on_train_start で作る。
        self._held_out = None
        self._group_samples = None
        # 直近の step で held-out から測った objective の診断値。held-out を使わないときは空。
        self.held_out_diagnostics: dict[str, torch.Tensor] = {}

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
        inputs = ObjectiveInput(logits=logits, target=target, attributes=attributes, group_samples=self._group_sample_input())
        loss = self.training_objective(inputs)
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
        """fit 開始時だけ、e2e モデルに必要な初期化を順に行う。

        backbone checkpoint の読み込み、freeze、データ依存初期化の順に実行する。

        Args:
            stage: Lightning が渡す stage 名。`fit` 以外では何もしない

        Returns:
            None

        Raises:
            RuntimeError: backbone checkpoint から1つも重みを読めなかった場合。
        """
        if stage != "fit":
            return
        if hasattr(self.training_objective, "commit_update"):
            if self.trainer.world_size != 1 or self.trainer.accumulate_grad_batches != 1:
                raise ValueError("条件付き公平性 objective は single device / accumulation=1 が必要")
        if self.hparams.backbone_checkpoint_path is not None:
            result = self.load_backbone_checkpoint(self.hparams.backbone_checkpoint_path)
            self._require_loaded_backbone(result, self.hparams.backbone_checkpoint_path)
        if self.hparams.freeze_backbone:
            self._freeze_backbone()
        if hasattr(self.net, "initialize_with_attributes"):
            self.net.initialize_with_attributes(self.trainer.datamodule.train_attributes())
        if self.hparams.compile:
            self.net = torch.compile(self.net)

    def on_load_checkpoint(self, checkpoint: dict[str, Any]) -> None:
        """条件付き公平性のfit再開を、状態を復元する前に拒否する。

        Args:
            checkpoint: Lightningが復元しようとするcheckpoint。

        Returns:
            None
        """
        if self._trainer is not None and self.trainer.state.fn == TrainerFn.FITTING and isinstance(self.training_objective, ConditionalObjective):
            raise ValueError("条件付き公平性 objective の途中再開は未対応")

    def on_train_epoch_start(self) -> None:
        """凍結した backbone の BatchNorm 等を eval 状態に保つ。

        Lightning は epoch ごとに module を train モードへ戻すので、毎 epoch 掛け直す。

        Args:
            なし

        Returns:
            None
        """
        sampler = getattr(getattr(self._trainer, "datamodule", None), "group_class_sampler", None)
        if sampler is not None:
            sampler.set_epoch(self.current_epoch)
        if self.hparams.freeze_backbone and (backbone := getattr(self.net, "backbone", None)) is not None:
            backbone.eval()

    def optimizer_step(self, *args, **kwargs) -> None:
        """モデル更新が完了した後に保留された双対更新を適用する。

        Args:
            *args: Lightning の optimizer step 引数。
            **kwargs: Lightning の optimizer step 引数。

        Returns:
            None
        """
        super().optimizer_step(*args, **kwargs)
        objective = self.training_objective
        if hasattr(objective, "commit_update"):
            if self._held_out is not None and getattr(objective, "dual_update", None) == "step":
                # held-out の forward は診断値も上書きするので、学習バッチの値と分けて残す。
                train_diagnostics = dict(objective.batch_diagnostics())
                self._measure_on_held_out(next(self._held_out))
                self.held_out_diagnostics = dict(objective.batch_diagnostics())
                objective._diagnostics = train_diagnostics
            objective.commit_update()

    def on_train_start(self) -> None:
        """DataModule が held-out を持てば、双対の更新に使うバッチの流れを用意する。

        step 単位の objective には held-out のセル均等バッチを毎 step 流す。epoch 単位の objective
        （FairDRO）は epoch 末に held-out 全件を測るので、ここでは流れを作らない。

        Args:
            なし

        Returns:
            None

        Raises:
            ValueError: 双対を更新しない objective に held-out を設定した場合。
        """
        datamodule = getattr(self.trainer, "datamodule", None)
        uses_held_out = getattr(datamodule, "data_holdout", None) is not None
        dual_update = getattr(self.training_objective, "dual_update", None)
        if uses_held_out and dual_update not in ("step", "epoch"):
            raise ValueError("held-out は双対変数を更新する条件付き objective でだけ使う")
        self._held_out = datamodule.held_out_batches() if uses_held_out and dual_update == "step" else None
        per_group = getattr(self.training_objective, "group_sample_size", None)
        self._group_samples = datamodule.group_sample_batches(per_group) if per_group else None

    @torch.no_grad()
    def _group_sample_input(self) -> ObjectiveInput | None:
        """群ごとに別に引いたバッチを勾配なしで forward し、目的関数に渡す入力にする。

        FIS の公式実装（FairVision `train_*_fis.py`）と同じく、model は train モードのまま no_grad で forward する。

        Args:
            なし

        Returns:
            ObjectiveInput | None: 群ごとのバッチを使わない目的関数なら None。
        """
        if self._group_samples is None:
            return None
        image, attributes, target = self.transfer_batch_to_device(next(self._group_samples), self.device, 0)
        return ObjectiveInput(logits=self(image, attributes).float(), target=target.long(), attributes=attributes)

    @torch.no_grad()
    def _measure_on_held_out(self, batch: tuple[Any, Mapping[str, torch.Tensor], torch.Tensor]) -> None:
        # 群の成績を学習に使わない画像で測り直し、保留中の双対の更新をその統計で置き換える。
        # 学習バッチで測ると、少数セルを記憶して差が消えたように見え、重みが戻る（Cotter et al. 2019 の
        # 2 データセット方式）。BatchNorm の統計を held-out で動かさないよう、net は eval で forward する。
        image, attributes, target = self.transfer_batch_to_device(batch, self.device, 0)
        self.net.eval()
        try:
            with self.trainer.precision_plugin.forward_context():
                logits = self(image, attributes).float()
        finally:
            self.net.train()
            if self.hparams.freeze_backbone and (backbone := getattr(self.net, "backbone", None)) is not None:
                backbone.eval()
        self.training_objective(ObjectiveInput(logits=logits, target=target.long(), attributes=attributes))

    def on_train_epoch_end(self) -> None:
        """epoch 単位で双対を更新する objective に全件の 0-1 誤り率を渡す。

        FairDRO の Alg. 1 l.10。評価 transform・sampler なしの split を eval モードで推論する。
        DataModule が held-out を持てば held-out 全件、持たなければ原著どおり train 全件で測る。

        Args:
            なし

        Returns:
            None
        """
        objective = self.training_objective
        if not hasattr(objective, "update_from_errors"):
            return
        datamodule = self.trainer.datamodule
        if getattr(datamodule, "data_holdout", None) is not None:
            source, loader = "holdout", datamodule.held_out_evaluation_dataloader()
        else:
            source, loader = "train", datamodule.evaluation_dataloader("train", persistent_workers=False)
        errors, counts = self._split_errors(loader, objective.group_key, objective.num_groups)
        update = objective.update_from_errors(errors)
        for c in range(2):
            for g in range(objective.num_groups):
                self.log(f"train/fair_dro/q/y{c}_g{g}", update["q"][c, g].float())
                self.log(f"train/fair_dro/error/g{g}_y{c}", errors[g, c].float())
        self.log("train/fair_dro/q_min", update["q_min"].float())
        row = {"epoch": self.current_epoch, "update": int(objective.update_count), "eta": float(update["eta"]), "source": source, "q_min": float(update["q_min"])}
        row.update({key: update[key].tolist() for key in ("q_star", "q")}, error=errors.tolist(), count=counts.tolist())
        path = Path(self.trainer.default_root_dir) / "metrics" / "fair_dro_q.jsonl"
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a") as stream:
            stream.write(json.dumps(row) + "\n")

    @torch.no_grad()
    def _split_errors(self, loader: Any, group_key: str, num_groups: int) -> tuple[torch.Tensor, torch.Tensor]:
        wrong = torch.zeros((num_groups, 2), dtype=torch.float64, device=self.device)
        counts = torch.zeros_like(wrong)
        was_training = self.training
        self.eval()
        try:
            for batch in loader:
                image, attributes, target = self.transfer_batch_to_device(batch, self.device, 0)
                with self.trainer.precision_plugin.forward_context():
                    logits = self(image, attributes).float()
                index = attributes[group_key].long() * 2 + target.long()
                miss = ((logits[:, 1] - logits[:, 0] > 0).long() != target.long()).double()
                wrong.view(-1).index_add_(0, index, miss)
                counts.view(-1).index_add_(0, index, torch.ones_like(miss))
        finally:
            self.train(was_training)
            if self.hparams.freeze_backbone and (backbone := getattr(self.net, "backbone", None)) is not None:
                backbone.eval()
        if len(loader.dataset) != int(counts.sum()) or (counts == 0).any():
            raise ValueError(f"全件の誤り集計に空セルか欠損がある: {counts.tolist()}")
        return wrong / counts, counts

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

    def configure_optimizers(self) -> dict[str, Any]:
        """学習可能な model / loss parameter から optimizer と任意 scheduler を構築する。

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
        parameters = [parameter for parameter in self.parameters() if parameter.requires_grad]
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
