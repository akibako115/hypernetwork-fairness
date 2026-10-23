"""群・クラス条件付き統計と、標本抽出補正を共有する。"""

import math
from collections.abc import Iterator, Sequence
from typing import Any

import lightning as L
import torch
from lightning.pytorch.trainer.states import TrainerFn
from torch.nn import functional as F

from .base import Objective
from .input import ObjectiveInput


class ConditionalObjective(Objective):
    """全セルが存在する二値分類バッチの母集団補正を保持する。

    counts は train だけから求める [G,2]。セル均等抽出を前提とする。双対状態を持つ子 class は、その更新を
    `callbacks()` で返す callback に任せる（step 単位は `StepDualUpdate`、FairDRO の epoch 単位は `FairDROEpochUpdate`）。
    """

    def __init__(self, num_groups: int, population_cell_counts: Sequence[Sequence[float]], group_key: str = "group_id", class_weight: Sequence[float] | None = None) -> None:
        """母集団確率と固定正規化定数を設定する。

        Args:
            num_groups: 群数。
            population_cell_counts: train の群・クラス別件数。
            group_key: 属性辞書の群キー。
            class_weight: 全群共通のクラス重み。

        Returns:
            None
        """
        super().__init__()
        counts = torch.as_tensor(population_cell_counts, dtype=torch.float64)
        if num_groups < 2 or counts.shape != (num_groups, 2) or not torch.isfinite(counts).all() or (counts <= 0).any():
            raise ValueError("population_cell_counts は全セルが正の有限値の [G,2] である必要がある")
        weight = torch.ones(2) if class_weight is None else torch.as_tensor(class_weight, dtype=torch.float32)
        if weight.shape != (2,) or not torch.isfinite(weight).all() or (weight <= 0).any():
            raise ValueError("class_weight は正の有限値の [2] である必要がある")
        self.num_groups, self.group_key = int(num_groups), group_key
        self.register_buffer("population", (counts / counts.sum()).float())
        self.register_buffer("class_weight", weight)
        self.register_buffer("update_count", torch.tensor(0, dtype=torch.long))
        self._pending = None
        self._diagnostics = {}

    def _cells(self, inputs: ObjectiveInput) -> tuple[torch.Tensor, list[list[torch.Tensor]]]:
        groups = inputs.attributes[self.group_key].long()
        if not torch.isfinite(inputs.logits).all():
            raise ValueError("logits に非有限値がある")
        if inputs.logits.shape != (inputs.target.numel(), 2) or groups.shape != inputs.target.shape:
            raise ValueError("二値 logits [B,2] と target/group [B] が必要")
        if ((groups < 0) | (groups >= self.num_groups)).any() or ((inputs.target < 0) | (inputs.target > 1)).any():
            raise ValueError("group/target が定義域外")
        masks = [[(groups == g) & (inputs.target == c) for c in range(2)] for g in range(self.num_groups)]
        if any(not mask.any() for row in masks for mask in row):
            raise ValueError("条件付き objective のバッチには全群・クラスセルが必要")
        return groups, masks

    def _task(self, inputs: ObjectiveInput, groups: torch.Tensor) -> torch.Tensor:
        ce = F.cross_entropy(inputs.logits.float(), inputs.target, reduction="none")
        rho = 2 * self.num_groups * self.population[groups, inputs.target]
        normalizer = (self.population * self.class_weight).sum()
        return (rho * self.class_weight[inputs.target] * ce).mean() / normalizer

    def batch_diagnostics(self) -> dict[str, torch.Tensor]:
        """直近の学習計算の診断値を返す。再計算・状態更新はしない。

        Args:
            なし

        Returns:
            dict: detached Tensor の辞書。
        """
        return self._diagnostics

    def callbacks(self) -> list[L.Callback]:
        """fit の途中再開を拒否する callback を返す。双対状態を持つ子 class はこれに足す。

        Args:
            なし

        Returns:
            list[Callback]: `RejectFitResume` だけのリスト。
        """
        return [RejectFitResume()]


def require_single_device(trainer: L.Trainer) -> None:
    """双対状態を 1 プロセス・1 step 1 更新で進める前提を確かめる。

    Args:
        trainer: fit を始める Trainer。

    Returns:
        None

    Raises:
        ValueError: 複数 device か、勾配を累積する設定の場合。
    """
    if trainer.world_size != 1 or trainer.accumulate_grad_batches != 1:
        raise ValueError("条件付き公平性 objective は single device / accumulation=1 が必要")


class RejectFitResume(L.Callback):
    """条件付き目的関数の fit を checkpoint から途中再開させない。

    保留中の双対の統計や held-out のバッチの流れは checkpoint に残らないので、再開すると原著の更新列から外れる。
    Lightning は `callbacks` を持たない checkpoint（`save_weights_only`）では callback のこの hook を呼ばないが、
    その checkpoint からの fit の再開は Lightning 自身が optimizer の状態が無いことで止める。
    """

    def on_load_checkpoint(self, trainer: L.Trainer, pl_module: L.LightningModule, checkpoint: dict[str, Any]) -> None:
        """fit の再開なら止める。評価のための読み込みは通す。

        Args:
            trainer: checkpoint を読む Trainer。
            pl_module: 学習 module。
            checkpoint: 読み込む checkpoint。

        Returns:
            None

        Raises:
            ValueError: fit を checkpoint から再開しようとした場合。
        """
        if trainer.state.fn == TrainerFn.FITTING:
            raise ValueError("条件付き公平性 objective の途中再開は未対応")


class StepDualUpdate(L.Callback):
    """optimizer の更新の後に、forward で保留した双対変数の更新を確定する（EO rate・Minimax AUC・Population AUC）。

    目的関数が held-out を使い（`uses_held_out`）、DataModule が held-out を持つときは、確定の前に held-out の
    セル均等バッチを 1 つ測り直し、保留中の統計をその値で置き換える。学習バッチで測ると、少数セルを記憶して
    差が消えたように見え、重みが戻る（Cotter et al. 2019 の 2 データセット方式）。held-out は net を eval にして
    推論し、BatchNorm の統計を動かさない。
    held-out で測った診断値は `pl_module.held_out_diagnostics` に置き、目的関数の診断値は学習バッチのものに戻す。
    診断を記録する callback がその後の値を読めるよう、LitModule はこの callback を Trainer の callback より先に並べる。
    """

    def __init__(self) -> None:
        """held-out のバッチの流れを持たない状態で作る。

        Args:
            なし

        Returns:
            None
        """
        self._held_out: Iterator[Any] | None = None

    def setup(self, trainer: L.Trainer, pl_module: L.LightningModule, stage: str) -> None:
        """fit の前に、1 step 1 更新の前提を確かめる。

        Args:
            trainer: Trainer。
            pl_module: 学習 module。
            stage: Lightning の stage 名。

        Returns:
            None
        """
        if stage == "fit":
            require_single_device(trainer)

    def on_train_start(self, trainer: L.Trainer, pl_module: L.LightningModule) -> None:
        """目的関数が held-out を使い、DataModule が持つときだけ、held-out のバッチの流れを作る。

        Args:
            trainer: Trainer。
            pl_module: 学習 module。

        Returns:
            None
        """
        datamodule = trainer.datamodule
        uses_held_out = pl_module.training_objective.uses_held_out and getattr(datamodule, "data_holdout", None) is not None
        self._held_out = datamodule.held_out_batches() if uses_held_out else None

    def on_train_batch_end(self, trainer: L.Trainer, pl_module: L.LightningModule, outputs: Any, batch: Any, batch_idx: int) -> None:
        """held-out があれば測り直してから、保留中の双対の更新を確定する。

        Args:
            trainer: Trainer。
            pl_module: 学習 module。
            outputs: training_step の出力（使わない）。
            batch: 学習バッチ（使わない）。
            batch_idx: バッチ番号（使わない）。

        Returns:
            None
        """
        objective = pl_module.training_objective
        if self._held_out is not None:
            # held-out の forward は診断値も上書きするので、学習バッチの値と分けて残す。
            train_diagnostics = dict(objective.batch_diagnostics())
            image, attributes, target = pl_module.transfer_batch_to_device(next(self._held_out), pl_module.device, 0)
            with torch.no_grad():
                objective(ObjectiveInput(logits=pl_module.eval_logits(image, attributes), target=target.long(), attributes=attributes))
            pl_module.held_out_diagnostics = dict(objective.batch_diagnostics())
            objective._diagnostics = train_diagnostics
        objective.commit_update()


def _positive_step(value: float, name: str) -> float:
    if not math.isfinite(value) or value <= 0:
        raise ValueError(f"{name} は正の有限値が必要")
    return float(value)
