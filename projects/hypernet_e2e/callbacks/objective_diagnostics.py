"""条件付き公平性 objective の診断値を epoch ごとに JSON Lines へ残す。"""

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import lightning as L
import torch


class _EpochMean:
    """batch ごとの診断値を key 別に足し込み、epoch の平均を返す。"""

    def __init__(self) -> None:
        self.sums: dict[str, torch.Tensor] = {}
        self.batches = 0

    def add(self, diagnostics: Mapping[str, Any]) -> None:
        values = {key: value.detach().double().cpu() for key, value in diagnostics.items() if isinstance(value, torch.Tensor)}
        if not values:
            return
        for key, value in values.items():
            self.sums[key] = self.sums[key] + value if key in self.sums else value.clone()
        self.batches += 1

    def mean(self) -> dict[str, Any]:
        return {key: (value / self.batches).tolist() for key, value in self.sums.items()}


class ObjectiveDiagnosticsRecorder(L.Callback):
    """`training_objective.batch_diagnostics()` を epoch ごとに 1 行ずつ書く。

    FairDRO・minimax AUC・EO の群の重み（双対・q）と群別の risk を、群の数によらず残す。
    出力は `<run_dir>/metrics/objective_diagnostics.jsonl`。各行の形は次のとおり。

    - `<診断名>`：epoch 最後の train batch の値（q・lambda など、epoch 末の状態を読む用途）。
    - `epoch_mean`：全 train batch の平均。率や risk の傾向はこちらで読む。最後の 1 batch は
      セルあたり数件しかなく、epoch の傾向を代表しない。
    - `held_out_epoch_mean`：held-out で双対を更新する run だけ。双対の更新に実際に使った held-out の統計の平均。
    - `batches` / `held_out_batches`：平均に使った batch 数。

    `batch_diagnostics` を持たない objective では何もしない。
    """

    def __init__(self) -> None:
        """epoch 内の集計を初期化する。

        Args:
            なし

        Returns:
            None
        """
        self._train = _EpochMean()
        self._held_out = _EpochMean()

    def on_train_epoch_start(self, trainer: L.Trainer, pl_module: L.LightningModule) -> None:
        """epoch の集計を空にする。

        Args:
            trainer: Trainer。
            pl_module: module。

        Returns:
            None
        """
        self._train, self._held_out = _EpochMean(), _EpochMean()

    def on_train_batch_end(self, trainer: L.Trainer, pl_module: L.LightningModule, outputs: Any, batch: Any, batch_idx: int) -> None:
        """optimizer step 後の診断値を足し込む。

        Args:
            trainer: Trainer。
            pl_module: 学習目的関数を持つ module。
            outputs: training_step の出力（使わない）。
            batch: batch（使わない）。
            batch_idx: batch 番号（使わない）。

        Returns:
            None
        """
        objective = getattr(pl_module, "training_objective", None)
        if not callable(getattr(objective, "batch_diagnostics", None)):
            return
        self._train.add(objective.batch_diagnostics())
        self._held_out.add(getattr(pl_module, "held_out_diagnostics", {}))

    def on_train_epoch_end(self, trainer: L.Trainer, pl_module: L.LightningModule) -> None:
        """最後の batch の値と epoch 平均を 1 行追記する。

        Args:
            trainer: 出力先と epoch を持つ Trainer。
            pl_module: 学習目的関数を持つ module。

        Returns:
            None
        """
        objective = getattr(pl_module, "training_objective", None)
        if not callable(getattr(objective, "batch_diagnostics", None)):
            return
        row: dict[str, Any] = {"epoch": trainer.current_epoch}
        for key, value in objective.batch_diagnostics().items():
            row[key] = value.detach().cpu().tolist() if isinstance(value, torch.Tensor) else value
        row["epoch_mean"], row["batches"] = self._train.mean(), self._train.batches
        if self._held_out.batches:
            row["held_out_epoch_mean"], row["held_out_batches"] = self._held_out.mean(), self._held_out.batches
        path = Path(trainer.default_root_dir) / "metrics" / "objective_diagnostics.jsonl"
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a") as handle:
            handle.write(json.dumps(row) + "\n")
