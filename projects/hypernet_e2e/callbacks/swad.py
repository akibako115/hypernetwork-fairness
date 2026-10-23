"""SWAD（Cha et al., NeurIPS 2021）：val 損失の谷にある重みを密に平均する。

公式実装（khanrc/swad の `LossValley`）を、評価の単位を epoch にして写したもの。
- 各 epoch の学習中、step ごとの重みを平均して「区間の平均」を作る
- val 損失が直近 `n_converge` 区間の最初で最小になったら収束とし、その区間から平均を始める。
  しきい値は、その `n_converge` 区間の val 損失の平均 × (1 + `tolerance_ratio`)
- 以後、直近 `n_tolerance` 区間の val 損失の最小がしきい値を超えたら谷の終わりとし、平均を打ち切る
- 学習の終わりに、平均した重みで train を 1 周流して BatchNorm の統計を取り直す（公式実装の `update_bn`）

平均の重みは区間の step 数に比例させ、各区間は 1 回だけ足す（公式実装では収束した区間が 2 回入る）。結果は `checkpoints/swad.ckpt`（`state_dict` の key は `net.` 始まり）と
`metrics/swad.json`（谷の始まりと終わりの epoch）に書く。
"""

from __future__ import annotations

import json
from collections import deque
from pathlib import Path

import lightning as L
import torch
from lightning.pytorch.callbacks import Callback


class _Segment:
    """1 区間（epoch）の重みの平均。"""

    def __init__(self, epoch: int, state: dict[str, torch.Tensor], steps: int, loss: float) -> None:
        self.epoch, self.state, self.steps, self.loss = epoch, state, steps, loss


class SWADCallback(Callback):
    """val 損失の谷の区間だけを重み付き平均し、学習後に BatchNorm を取り直して保存する。"""

    def __init__(self, n_converge: int = 3, n_tolerance: int = 6, tolerance_ratio: float = 0.3, monitor: str = "val/loss") -> None:
        """公式実装の既定値（n_converge 3、n_tolerance 6、tolerance_ratio 0.3）を使う。

        Args:
            n_converge: 収束の判定に使う区間数。
            n_tolerance: 谷の終わりの判定に使う区間数。
            tolerance_ratio: しきい値の余裕。
            monitor: val 損失の metric 名。

        Returns:
            None
        """
        super().__init__()
        self.n_converge, self.n_tolerance, self.tolerance_ratio, self.monitor = n_converge, n_tolerance, tolerance_ratio, monitor
        self.converge_q: deque[_Segment] = deque(maxlen=n_converge)
        self.smooth_q: deque[_Segment] = deque(maxlen=n_tolerance)
        self.total: dict[str, torch.Tensor] | None = None
        self.total_steps = 0
        self.averaged_epochs: list[int] = []
        self.converge_epoch: int | None = None
        self.threshold: float | None = None
        self.dead = False
        self._running: dict[str, torch.Tensor] | None = None
        self._steps = 0

    @staticmethod
    def _parameters(pl_module: L.LightningModule) -> dict[str, torch.Tensor]:
        return {name: p.detach() for name, p in pl_module.net.named_parameters()}

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx) -> None:
        params = self._parameters(pl_module)
        if self._running is None:
            self._running = {k: v.float().clone() for k, v in params.items()}
        else:
            for k, v in params.items():
                self._running[k].add_(v.float())
        self._steps += 1

    def _add(self, segment: _Segment) -> None:
        if self.total is None:
            self.total = {k: v * segment.steps for k, v in segment.state.items()}
        else:
            for k, v in segment.state.items():
                self.total[k].add_(v * segment.steps)
        self.total_steps += segment.steps
        self.averaged_epochs.append(segment.epoch)

    def update(self, segment: _Segment) -> None:
        """区間を 1 つ受け取り、公式実装の `LossValley.update_and_evaluate` と同じ規則で平均に足す。

        Args:
            segment: 直近の区間。

        Returns:
            None
        """
        if self.dead:
            return
        self.converge_q.append(segment)
        self.smooth_q.append(segment)
        if self.converge_epoch is None:
            if len(self.converge_q) < self.n_converge:
                return
            losses = [s.loss for s in self.converge_q]
            if min(range(len(losses)), key=losses.__getitem__) != 0:
                return
            self.converge_epoch = self.converge_q[0].epoch
            self.threshold = sum(losses) / len(losses) * (1 + self.tolerance_ratio)
            # 収束した区間より前でも、しきい値を下回り続けていた区間までさかのぼって平均に入れる（公式実装と同じ）
            queue = list(self.smooth_q)[: len(self.smooth_q) - self.n_converge + 1]
            start = 0
            for i in reversed(range(len(queue))):
                if queue[i].loss > self.threshold:
                    start = i + 1
                    break
            for s in queue[start:]:
                self._add(s)
            return
        if self.smooth_q[0].epoch <= self.converge_epoch:
            return
        if min(s.loss for s in self.smooth_q) > self.threshold:
            self.dead = True
            return
        self._add(self.smooth_q[0])

    def finalize(self) -> None:
        """学習の終わりに、谷が続いていれば残りの区間を公式実装の `get_final_model` と同じ規則で足す。

        Args:
            なし

        Returns:
            None
        """
        if self.converge_epoch is None or self.dead:
            return
        queue = [s for s in self.smooth_q if s.epoch > self.averaged_epochs[-1]]
        while queue:
            if min(s.loss for s in queue) > self.threshold:
                break
            self._add(queue.pop(0))

    def on_train_epoch_end(self, trainer, pl_module) -> None:
        if self._running is None or trainer.sanity_checking:
            return
        loss = trainer.callback_metrics.get(self.monitor)
        if loss is None:
            raise KeyError(f"SWAD は {self.monitor} を必要とする（val を毎 epoch 回す）")
        state = {k: (v / self._steps).cpu() for k, v in self._running.items()}
        self.update(_Segment(trainer.current_epoch, state, self._steps, float(loss)))
        self._running, self._steps = None, 0

    @torch.no_grad()
    def _update_bn(self, trainer, pl_module) -> None:
        """平均した重みで train を 1 周流し、BatchNorm の統計を累積平均で取り直す。"""
        net = pl_module.net
        norms = [m for m in net.modules() if isinstance(m, torch.nn.modules.batchnorm._BatchNorm)]
        for m in norms:
            m.reset_running_stats()
            m.momentum = None
        was_training = net.training
        net.train()
        for batch in trainer.datamodule.train_dataloader():
            image, attributes, _ = pl_module.transfer_batch_to_device(batch, pl_module.device, 0)
            pl_module(image, attributes)
        net.train(was_training)

    def on_fit_end(self, trainer, pl_module) -> None:
        self.finalize()
        record = {"converge_epoch": self.converge_epoch, "averaged_epochs": self.averaged_epochs, "dead_valley": self.dead, "threshold": self.threshold}
        if self.total is None:
            # 収束しなかったときは、公式実装と同じく最後の区間を使う
            last = self.converge_q[-1]
            self.total, self.total_steps = {k: v * last.steps for k, v in last.state.items()}, last.steps
            record["fallback_last_segment"] = last.epoch
        params = dict(pl_module.net.named_parameters())
        for k, v in self.total.items():
            params[k].data.copy_((v / self.total_steps).to(params[k].device, params[k].dtype))
        self._update_bn(trainer, pl_module)
        root = Path(trainer.default_root_dir)
        dirpath = Path(trainer.checkpoint_callback.dirpath) if trainer.checkpoint_callback else root / "checkpoints"
        dirpath.mkdir(parents=True, exist_ok=True)
        state = {f"net.{k}": v.detach().cpu() for k, v in pl_module.net.state_dict().items()}
        torch.save({"state_dict": state}, dirpath / "swad.ckpt")
        (root / "metrics").mkdir(parents=True, exist_ok=True)
        (root / "metrics" / "swad.json").write_text(json.dumps(record) + "\n")
