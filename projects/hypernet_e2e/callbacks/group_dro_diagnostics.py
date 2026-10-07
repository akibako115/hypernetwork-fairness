"""Group DROの学習診断をepochごとにCSV・外部loggerへ接続する。"""

import json
from collections.abc import Sequence
from pathlib import Path

import lightning as L
import torch


class GroupDRODiagnostics(L.Callback):
    """更新後qの終端値・batch平均と、標本平均の群loss・件数を記録する。"""

    def __init__(self, group_labels: Sequence[str] | None = None) -> None:
        """群IDに対応する表示ラベルを受け取る。

        Args:
            group_labels: 群ID順のラベル。未指定時はgroup_0等を使う。

        Returns:
            None
        """
        self.group_labels = list(group_labels) if group_labels is not None else None
        self._active = False

    def on_fit_start(self, trainer: L.Trainer, pl_module: L.LightningModule) -> None:
        """DROの診断契約を確認し、群対応とq初期値をrun内へ保存する。

        Args:
            trainer: runの出力先を持つTrainer。
            pl_module: 学習目的関数を持つmodule。

        Returns:
            None
        """
        objective = pl_module.training_objective
        self._active = hasattr(objective, "adv_probs") and callable(getattr(objective, "batch_diagnostics", None))
        if not self._active:
            return
        if trainer.world_size != 1:
            raise ValueError("Group DROのq診断は単一device学習に対応。分散qの集約仕様は未定義")
        self._num_groups = objective.num_groups
        labels = self.group_labels or [f"group_{i}" for i in range(self._num_groups)]
        if len(labels) != self._num_groups or len(set(labels)) != len(labels):
            raise ValueError("group_labelsは群数と一致する重複なしのID順ラベルが必要")
        data = getattr(trainer.datamodule, "hparams", {})
        metadata = {
            "schema_version": 1,
            "groups": [{"id": i, "label": label} for i, label in enumerate(labels)],
            "group_key": objective.group_key,
            "group_attributes": list(data.get("group_attribute_names", [])),
            "group_cardinalities": list(data.get("group_cardinalities", [])),
            "initial_q": objective.adv_probs.detach().cpu().tolist(),
            "q_mean_definition": "batch平均（各train forwardの更新後q、batchは等重み）",
            "loss_definition": "class weight適用後のper-sample CEを群内標本数で平均",
            "missing_group_loss": "epoch全体で未観測の群はNaN、countは0",
        }
        path = Path(trainer.default_root_dir) / "metrics" / "group_dro_groups.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n")

    def on_train_epoch_start(self, trainer: L.Trainer, pl_module: L.LightningModule) -> None:
        """epoch診断の集計だけをリセットし、目的関数のqは保持する。

        Args:
            trainer: 呼び出し元のTrainer。
            pl_module: 集計tensorのdeviceを持つmodule。

        Returns:
            None
        """
        if not self._active:
            return
        self._q_sum = torch.zeros(self._num_groups, dtype=torch.float64, device=pl_module.device)
        self._loss_sum = torch.zeros_like(self._q_sum)
        self._counts = torch.zeros_like(self._q_sum)
        self._num_batches = 0

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx) -> None:
        """実際のtrain forwardが作った診断値だけを集計する。

        Args:
            trainer: 呼び出し元のTrainer。
            pl_module: 診断値を公開する目的関数を持つmodule。
            outputs: training_stepの出力。
            batch: 学習batch。
            batch_idx: batch番号。

        Returns:
            None
        """
        if not self._active:
            return
        diagnostics = pl_module.training_objective.batch_diagnostics()
        self._q_end = diagnostics["q"].detach().clone()
        self._q_sum += self._q_end
        self._loss_sum += diagnostics["loss_sum"]
        self._counts += diagnostics["count"]
        self._num_batches += 1

    def on_train_epoch_end(self, trainer: L.Trainer, pl_module: L.LightningModule) -> None:
        """診断scalarをepoch軸で全loggerへ送る。

        Args:
            trainer: 呼び出し元のTrainer。
            pl_module: CSV・W&Bへlogするmodule。

        Returns:
            None
        """
        if not self._active or not self._num_batches:
            return
        losses = torch.where(self._counts > 0, self._loss_sum / self._counts, float("nan"))
        values = {"q_end": self._q_end, "q_mean": self._q_sum / self._num_batches, "loss": losses, "count": self._counts}
        for metric, vector in values.items():
            for group_id, value in enumerate(vector):
                pl_module.log(f"train/group_dro/{metric}/g{group_id}", value, on_step=False, on_epoch=True, batch_size=1, sync_dist=False)
