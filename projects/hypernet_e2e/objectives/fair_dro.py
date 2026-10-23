"""Jung et al. (ICLR 2023) Algorithm 1 の classwise FairDRO。"""

import json
import math
import warnings
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any

import lightning as L
import torch
from torch.nn import functional as F

from .conditional import ConditionalObjective, require_single_device
from .input import ObjectiveInput


def chi_square_best_response(losses: torch.Tensor, radius: float) -> torch.Tensor:
    """χ² 球 Q_rho 上の内側最大化の閉形式解を返す（Prop. 2, 式 8）。

    rho の単位は公式実装（sangwon-jung94/FairDRO の ``_update_mw_margin``）に合わせ、
    ``q* = 1/G + sqrt(2 rho / G) * 偏差 / ||偏差||`` とする。論文の式 8 は ``sqrt(rho / G)`` で、
    公式実装の rho は論文の rho の半分にあたる（φ(t) = (t-1)²/2 で χ² 球を測ったものと一致する）。
    公式 README の rho（UTKFace 0.3、CelebA 1.5 など）をそのまま使えるよう、こちらに揃える。

    Args:
        losses: [A] の群別損失。
        radius: rho（公式実装の単位）。

    Returns:
        Tensor: [A] の quasi-probability。偏差がなければ一様。
    """
    size = losses.numel()
    deviation = losses - losses.mean()
    norm = torch.linalg.vector_norm(deviation)
    uniform = torch.full_like(losses, 1 / size)
    if norm <= 0:
        return uniform
    return uniform + math.sqrt(2 * radius / size) * deviation / norm


def nonnegative_radius_limit(num_groups: int) -> float:
    """閉形式解 q* の全成分が正に保たれる rho の上限 1/(2(G-1)) を返す。

    偏差の単位ベクトルの最小成分は -sqrt((G-1)/G) なので、q* の最小値は (1 - sqrt(2 rho (G-1)))/G になる。
    rho がこの上限以上だと、誤り率のわずかな差でも q* の成分が 0 以下になる。G=2・rho=0.5（公式の既定値）では
    各クラスで片方の群の重みが 0 になり、一方の群だけを学ぶ解へ epoch ごとに振れた（conditional_fairness_one_stage）。
    負の重みは原著の定義域内なので禁止はしない。

    Args:
        num_groups: 群数 G（2 以上）。

    Returns:
        float: 1/(2(G-1))。
    """
    return 1 / (2 * (num_groups - 1))


class FairDROTaskLoss(ConditionalObjective):
    """クラスごとの群重み q^y で群別 CE を重み付けする。

    q は全件の 0-1 誤り率から epoch ごとに平滑化 IBR で更新する（式 10）。原著は train 全件で測る。
    DataModule が held-out を持つとき、`FairDROEpochUpdate` は held-out 全件の誤り率を渡す（train の記憶で少数セルの
    誤りが消え、q が戻るのを避けるため。EO・Minimax AUC の held-out と同じ扱い）。
    バッチはセル均等抽出（Alg. 1 l.6）なので母集団補正と class weight は使わない。
    """

    uses_held_out = True

    def __init__(
        self,
        num_groups: int,
        population_cell_counts: Sequence[Sequence[float]],
        group_key: str = "group_id",
        radius: float = 0.5,
        max_epochs: int = 30,
        class_weight: Sequence[float] | None = None,
    ) -> None:
        """一様な q と平滑化スケジュールを設定する。

        Args:
            num_groups: 群数。
            population_cell_counts: train の [G,2] 件数。セル存在の検証に使う。
            group_key: 群属性のキー。
            radius: χ² 球の半径 rho（公式実装の単位。既定 0.5 は公式の既定値）。
            max_epochs: eta_t = 1 - t/T の T。
            class_weight: 互換設定用。None のみ許可。

        Returns:
            None
        """
        if class_weight is not None:
            raise ValueError("FairDRO に class_weight は適用しない")
        super().__init__(num_groups, population_cell_counts, group_key)
        if not math.isfinite(radius) or radius <= 0:
            raise ValueError("radius は正の有限値が必要")
        if int(max_epochs) < 1:
            raise ValueError("max_epochs は 1 以上が必要")
        self.radius, self.max_epochs = float(radius), int(max_epochs)
        if self.radius >= nonnegative_radius_limit(num_groups):
            warnings.warn(
                f"FairDRO の radius={self.radius} は G={num_groups} の上限 {nonnegative_radius_limit(num_groups):.4g} 以上で、q* の成分が 0 以下になり得る（群の重みが端点へ振り切れる）",
                stacklevel=2,
            )
        self.register_buffer("q", torch.full((2, num_groups), 1 / num_groups, dtype=torch.float64))

    def forward(self, inputs: ObjectiveInput) -> torch.Tensor:
        """現在の q による classwise 群 CE を返す（式 9）。

        Args:
            inputs: 二値 logits と群・ラベル。

        Returns:
            Tensor: scalar loss。
        """
        _, cells = self._cells(inputs)
        ce = F.cross_entropy(inputs.logits.float(), inputs.target, reduction="none")
        risks = torch.stack([torch.stack([ce[m].mean() for m in row]) for row in cells])
        q = self.q.detach().float().T
        if self.training:
            score = inputs.logits[:, 1].float() - inputs.logits[:, 0].float()
            rates = torch.stack([torch.stack([(score[m] > 0).float().mean() for m in row]) for row in cells])
            # q は epoch 内で一定なので FairDROEpochUpdate が epoch 末に名前付きで記録する。
            self._diagnostics = {"risk": risks.detach(), "rates": rates.detach(), "cell_count": torch.stack([m.sum() for row in cells for m in row])}
        return (q * risks).sum() / 2

    def callbacks(self) -> list[L.Callback]:
        """途中再開の拒否に、epoch 末の q の更新を足して返す。

        Args:
            なし

        Returns:
            list[Callback]: `RejectFitResume` と `FairDROEpochUpdate`。
        """
        return [*super().callbacks(), FairDROEpochUpdate()]

    @torch.no_grad()
    def commit_update(self) -> None:
        """q は epoch 単位で更新するので step ごとには何もしない。

        `ConditionalFairnessRecorder` がこのメソッドの有無で双対を持つ目的関数を見分けるので残す。

        Args:
            なし

        Returns:
            None
        """
        self._pending = None

    @torch.no_grad()
    def update_from_errors(self, errors: torch.Tensor) -> dict[str, torch.Tensor]:
        """全件の群・クラス別 0-1 誤り率で q を平滑化 IBR 更新する。

        Args:
            errors: [G,2] の誤り率。train 全件、または held-out 全件で測ったもの。

        Returns:
            dict: q_star・eta・更新後 q・q の最小値 q_min（負なら成績の良いセルの CE に負の重みがかかる）。
        """
        errors = errors.to(self.q.device, torch.float64)
        if errors.shape != (self.num_groups, 2) or not torch.isfinite(errors).all():
            raise ValueError("errors は有限値の [G,2] が必要")
        step = int(self.update_count)
        if step >= self.max_epochs:
            raise RuntimeError("FairDRO の q 更新回数が max_epochs を超えた")
        eta = 1 - step / self.max_epochs
        q_star = torch.stack([chi_square_best_response(errors[:, c], self.radius) for c in range(2)])
        self.q.copy_((1 - eta) * self.q + eta * q_star)
        self.update_count.add_(1)
        return {"q_star": q_star, "eta": torch.tensor(eta, dtype=torch.float64), "q": self.q.clone(), "q_min": self.q.min().clone()}


class FairDROEpochUpdate(L.Callback):
    """epoch 末に全件の群・クラス別 0-1 誤り率を測り、FairDRO の q を更新する（Alg. 1 l.10）。

    評価 transform・sampler なしの split を、net を eval にして推論する。DataModule が held-out を持てば held-out 全件、
    持たなければ原著どおり train 全件で測る。q と誤り率は `train/fair_dro/*` に記録し、更新ごとの値を
    `metrics/fair_dro_q.jsonl` に 1 行ずつ残す。
    LitModule が目的関数の callback を先頭に並べるので、Trainer の callback の `on_train_epoch_end` は更新後の q を見る
    （checkpoint は最後に回るので、last・best にも更新後の q が入る）。今の callback で epoch 末に q を読むものは無い。
    """

    def setup(self, trainer: L.Trainer, pl_module: L.LightningModule, stage: str) -> None:
        """fit の前に、1 プロセスで q を進める前提を確かめる。

        Args:
            trainer: Trainer。
            pl_module: 学習 module。
            stage: Lightning の stage 名。

        Returns:
            None
        """
        if stage == "fit":
            require_single_device(trainer)

    def on_train_epoch_end(self, trainer: L.Trainer, pl_module: L.LightningModule) -> None:
        """全件の誤り率で q を更新し、記録する。

        Args:
            trainer: Trainer。
            pl_module: 学習 module。

        Returns:
            None
        """
        objective = pl_module.training_objective
        datamodule = trainer.datamodule
        if getattr(datamodule, "data_holdout", None) is not None:
            source, loader = "holdout", datamodule.held_out_evaluation_dataloader()
        else:
            source, loader = "train", datamodule.evaluation_dataloader("train", persistent_workers=False)
        errors, counts = _split_errors(pl_module, loader, objective.group_key, objective.num_groups)
        update = objective.update_from_errors(errors)
        for c in range(2):
            for g in range(objective.num_groups):
                pl_module.log(f"train/fair_dro/q/y{c}_g{g}", update["q"][c, g].float())
                pl_module.log(f"train/fair_dro/error/g{g}_y{c}", errors[g, c].float())
        pl_module.log("train/fair_dro/q_min", update["q_min"].float())
        row = {"epoch": pl_module.current_epoch, "update": int(objective.update_count), "eta": float(update["eta"]), "source": source, "q_min": float(update["q_min"])}
        row.update({key: update[key].tolist() for key in ("q_star", "q")}, error=errors.tolist(), count=counts.tolist())
        path = Path(trainer.default_root_dir) / "metrics" / "fair_dro_q.jsonl"
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a") as stream:
            stream.write(json.dumps(row) + "\n")


@torch.no_grad()
def _split_errors(pl_module: L.LightningModule, loader: Iterable[Any], group_key: str, num_groups: int) -> tuple[torch.Tensor, torch.Tensor]:
    """split 全件の群・クラス別 0-1 誤り率 `[G,2]` と件数を返す。"""
    wrong = torch.zeros((num_groups, 2), dtype=torch.float64, device=pl_module.device)
    counts = torch.zeros_like(wrong)
    for batch in loader:
        image, attributes, target = pl_module.transfer_batch_to_device(batch, pl_module.device, 0)
        logits = pl_module.eval_logits(image, attributes)
        index = attributes[group_key].long() * 2 + target.long()
        miss = ((logits[:, 1] - logits[:, 0] > 0).long() != target.long()).double()
        wrong.view(-1).index_add_(0, index, miss)
        counts.view(-1).index_add_(0, index, torch.ones_like(miss))
    if len(loader.dataset) != int(counts.sum()) or (counts == 0).any():
        raise ValueError(f"全件の誤り集計に空セルか欠損がある: {counts.tolist()}")
    return wrong / counts, counts
