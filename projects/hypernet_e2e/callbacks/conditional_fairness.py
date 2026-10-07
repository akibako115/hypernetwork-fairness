"""条件付き公平性の train 診断と、自然分布 validation 候補を記録する。"""

import hashlib
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import lightning as L
import numpy as np
import torch
from lightning.pytorch.trainer.states import TrainerFn
from sklearn.metrics import roc_auc_score

from projects.hypernet_e2e.objectives.equalized_odds import smooth_step

# EO の hard gap 規則で選ぶ手法。FairDRO は EO を狙う先行手法として同じ規則で比べる。
EO_SELECTION_METHODS = frozenset({"equalized_odds_rate", "fair_dro"})


def _best(rows: Sequence[dict[str, Any]], metrics: Sequence[tuple[str, int]]) -> dict[str, Any]:
    best = min(rows, key=lambda row: row["epoch"])
    for row in sorted(rows, key=lambda row: row["epoch"]):
        for metric, direction in metrics:
            difference = direction * (row[metric] - best[metric])
            if abs(difference) > 1e-8:
                if difference > 0:
                    best = row
                break
    return best


def select_candidate(candidates: Sequence[dict[str, Any]], parent: dict[str, Any], method: str) -> dict[str, Any]:
    """validation 指標だけで候補を選ぶ。EO 不成立時は明示する。

    Args:
        candidates: epoch ごとの validation 指標辞書。
        parent: 親 checkpoint の同じ validation 指標。
        method: minimax_auc または EO_SELECTION_METHODS の手法等。

    Returns:
        dict: status と選択された候補。親を含む。fair_dro は原著が報告する最終 epoch も併記する。
    """
    valid = [row for row in [parent, *candidates] if np.isfinite([row["auroc"], row["worst_age_auroc"], row["min_pair_auroc"], row["weighted_ce"], *row["tpr"], *row["tnr"]]).all()]
    eligible = [row for row in valid if row["auroc"] >= parent["auroc"] - 0.005 and row["worst_age_auroc"] >= parent["worst_age_auroc"] - 0.005]
    result = {"last_epoch": max(candidates, key=lambda row: row["epoch"])} if method == "fair_dro" and candidates else {}
    if method in EO_SELECTION_METHODS:

        def violation(row: Mapping[str, Any]) -> float:
            values = [np.ptp(row["tpr"]) - 0.1, np.ptp(row["tnr"]) - 0.1]
            values.extend(np.array(parent["tpr"]) - 0.02 - row["tpr"])
            values.extend(np.array(parent["tnr"]) - 0.02 - row["tnr"])
            return float(max(values))

        diagnostic = min(eligible, key=lambda row: (violation(row), -row["auroc"], row["epoch"])) if eligible else None
        eligible = [
            row
            for row in eligible
            if max(row["tpr"]) - min(row["tpr"]) <= 0.1 + 1e-6
            and max(row["tnr"]) - min(row["tnr"]) <= 0.1 + 1e-6
            and min(np.array(row["tpr"]) - parent["tpr"]) >= -0.02 - 1e-6
            and min(np.array(row["tnr"]) - parent["tnr"]) >= -0.02 - 1e-6
        ]
        if not eligible:
            return {"status": "infeasible", "selected": None, "diagnostic_minimum_violation": diagnostic, **result}
        selected = _best(eligible, [("weighted_ce", -1), ("auroc", 1)])
    else:
        selected = _best(eligible, [("min_pair_auroc", 1), ("worst_age_auroc", 1), ("auroc", 1)])
    return {"status": "selected", "selected": selected, **result}


class ConditionalFairnessRecorder(L.Callback):
    """学習診断、全 epoch 候補と親 validation 指標を記録する。

    sanity は全 validation を実行し、親 checkpoint の基準を確定する。
    test を読まず、自然分布のラベル条件付き率を標本数で集約する。
    1 stage の学習では sanity 時点のモデルが初期値で基準にならないので、`reference="deferred"`
    で選択を分析側（同じアーキテクチャ・seed の ERM run の validation を基準）に委ねる。
    """

    def __init__(self, method: str, group_key: str = "group_id", reference: str = "parent") -> None:
        """選択方法と群キーを受け取る。

        Args:
            method: 学習戦略名。
            group_key: 年齢群 ID の属性キー。
            reference: parent なら親 checkpoint 基準で選ぶ。deferred なら候補の記録だけにする。

        Returns:
            None
        """
        if reference not in {"parent", "deferred"}:
            raise ValueError("reference は parent/deferred")
        self.method, self.group_key, self.reference = method, group_key, reference
        self.candidates: list[dict[str, Any]] = []
        self.parent: dict[str, Any] | None = None

    def _write(self, trainer: L.Trainer, name: str, data: Any) -> None:
        path = Path(trainer.default_root_dir) / "metrics" / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) + "\n")

    def on_fit_start(self, trainer: L.Trainer, pl_module: L.LightningModule) -> None:
        """群・制約の定義を保存し、親評価の条件を確認する。

        Args:
            trainer: 実行中の Trainer。
            pl_module: 学習 module。

        Returns:
            None
        """
        if trainer.ckpt_path is not None:
            raise ValueError("ConditionalFairnessRecorder は途中再開に未対応。親 ERM から新規 run を開始する")
        if trainer.num_sanity_val_steps != float("inf") or trainer.world_size != 1:
            raise ValueError("親評価には全 validation の sanity と single device が必要")
        obj = pl_module.training_objective
        if int(trainer.datamodule.hparams.num_groups) != 2:
            raise ValueError("この候補 recorder は年齢2群専用")
        self._write(
            trainer,
            "fairness_definitions.json",
            {
                "groups": ["<65", "≥65"],
                "positive_label": "No Finding",
                "threshold": "z1-z0 > 0",
                "pairs": getattr(obj, "pairs", []),
                "constraints": getattr(obj, "constraints", []),
                "radius": getattr(obj, "radius", None),
                "q": "q[y][g]: FairDRO の classwise 群重み" if hasattr(obj, "radius") else None,
                "population_cell_counts": trainer.datamodule.data_train.df.groupby([self.group_key, "target"]).size().unstack().values.tolist(),
                "method": self.method,
                "reference": self.reference,
            },
        )

    def on_load_checkpoint(self, trainer: L.Trainer, pl_module: L.LightningModule, checkpoint: dict[str, Any]) -> None:
        """候補一覧を復元できないfit再開を明示的に拒否する。

        Args:
            trainer: checkpointを復元するTrainer。
            pl_module: 学習module。
            checkpoint: 読み込むcheckpoint。

        Returns:
            None
        """
        if trainer.state.fn == TrainerFn.FITTING:
            raise ValueError("ConditionalFairnessRecorder は途中再開に未対応。親 ERM から新規 run を開始する")

    def on_validation_epoch_start(self, trainer: L.Trainer, pl_module: L.LightningModule) -> None:
        """validation 統計のバッファを空にする。

        Args:
            trainer: Trainer。
            pl_module: 学習 module。

        Returns:
            None
        """
        self._validation: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = []

    def on_validation_batch_end(self, trainer: L.Trainer, pl_module: L.LightningModule, outputs: Mapping[str, Any], batch: Any, batch_idx: int, dataloader_idx: int = 0) -> None:
        """CSV 順の logits・target・群を集める。

        Args:
            trainer: Trainer。
            pl_module: 学習 module。
            outputs: validation_step 出力。
            batch: 入力。
            batch_idx: バッチ番号。
            dataloader_idx: loader 番号。

        Returns:
            None
        """
        self._validation.append((outputs["logits"].detach().float().cpu(), outputs["target"].cpu(), outputs["attributes"][self.group_key].cpu()))

    def _metrics(self, trainer: L.Trainer, pl_module: L.LightningModule) -> dict[str, Any]:
        logits, target, groups = [torch.cat(parts) for parts in zip(*self._validation, strict=True)]
        if len(target) != len(trainer.val_dataloaders.dataset):
            raise ValueError("公平性候補の評価には全 validation 行が必要")
        scores = (logits[:, 1] - logits[:, 0]).numpy()
        y, g = target.numpy(), groups.numpy()
        pair_auc = []
        for positive_group in range(2):
            for negative_group in range(2):
                mask = ((g == positive_group) & (y == 1)) | ((g == negative_group) & (y == 0))
                pair_auc.append(float(roc_auc_score(y[mask], scores[mask])))
        # 学習 objective が純 AUC でも選択用 CE は全条件で同じ母集団 class weight。
        train_frame = trainer.datamodule.data_train.df
        train_counts = torch.bincount(torch.tensor(train_frame.target.to_numpy()), minlength=2).double()
        weight = train_counts.reciprocal()
        weight /= weight.mean()
        weight = torch.tensor([round(float(value), 6) for value in weight], dtype=torch.float32)
        normalizer = (train_counts.float() / train_counts.sum().float() * weight).sum()
        ce = (torch.nn.functional.cross_entropy(logits, target, reduction="none") * weight[target]).mean() / normalizer
        tpr = [float((scores[(g == group) & (y == 1)] > 0).mean()) for group in range(2)]
        tnr = [float((scores[(g == group) & (y == 0)] <= 0).mean()) for group in range(2)]
        row = {
            "epoch": 0 if trainer.sanity_checking else trainer.current_epoch + 1,
            "auroc": float(roc_auc_score(y, scores)),
            "worst_age_auroc": min(pair_auc[0], pair_auc[3]),
            "min_pair_auroc": min(pair_auc),
            "pair_auroc": pair_auc,
            "weighted_ce": float(ce),
            "tpr": tpr,
            "tnr": tnr,
        }

        objective = pl_module.training_objective
        if hasattr(objective, "constraints"):
            score_tensor = torch.from_numpy(scores)
            upper, lower = smooth_step(1 + score_tensor / objective.tau), smooth_step(score_tensor / objective.tau)
            proxy, hard = [], []
            for index, (c, source, other) in enumerate(objective.constraints):
                epsilon = float(objective.epsilon[index])
                proxy.append(float(upper[(g == source) & (y == c)].mean() - lower[(g == other) & (y == c)].mean()) - epsilon)
                rates = tpr if c else [1 - value for value in tnr]
                hard.append(rates[source] - rates[other] - epsilon)
            row.update(proxy_constraints=proxy, hard_constraints=hard)
            if not trainer.sanity_checking and self._lambda_batches:
                row["lambda_task_mean"] = self._lambda_task_sum / self._lambda_batches
        return row

    def on_validation_epoch_end(self, trainer: L.Trainer, pl_module: L.LightningModule) -> None:
        """自然分布のペアAUC・率を全loggerへ送る。

        Args:
            trainer: Trainer。
            pl_module: 学習 module。

        Returns:
            None
        """
        self._validation_metrics = self._metrics(trainer, pl_module)
        if not trainer.sanity_checking:
            for name in ("min_pair_auroc", "worst_age_auroc", "weighted_ce"):
                pl_module.log(f"val/conditional/{name}", self._validation_metrics[name])
            for name in ("tpr", "tnr", "pair_auroc", "hard_constraints", "proxy_constraints"):
                for index, value in enumerate(self._validation_metrics.get(name, [])):
                    pl_module.log(f"val/conditional/{name}/{index}", value)

    def on_validation_end(self, trainer: L.Trainer, pl_module: L.LightningModule) -> None:
        """自然分布指標と epoch checkpoint の SHA を記録する。

        Args:
            trainer: Trainer。
            pl_module: 学習 module。

        Returns:
            None
        """
        if trainer.state.fn != TrainerFn.FITTING:
            return
        row = dict(self._validation_metrics)
        if trainer.sanity_checking:
            row["checkpoint"] = pl_module.hparams.backbone_checkpoint_path
            if row["checkpoint"]:
                with Path(row["checkpoint"]).open("rb") as stream:
                    row["sha256"] = hashlib.file_digest(stream, "sha256").hexdigest()
            self.parent = row
            self._write(trainer, "fairness_parent.json" if self.reference == "parent" else "fairness_initial.json", row)
            return
        path = Path(trainer.default_root_dir) / "checkpoints" / f"candidate_{row['epoch']:03d}.ckpt"
        path.parent.mkdir(parents=True, exist_ok=True)
        trainer.save_checkpoint(path, weights_only=True)
        with path.open("rb") as stream:
            digest = hashlib.file_digest(stream, "sha256").hexdigest()
        row.update(checkpoint=str(path.relative_to(trainer.default_root_dir)), sha256=digest)
        self.candidates.append(row)
        self._write(trainer, "fairness_candidates.json", self.candidates)

    def on_train_batch_end(self, trainer: L.Trainer, pl_module: L.LightningModule, outputs: Any, batch: Any, batch_idx: int) -> None:
        """forward が作った detached 診断値を CSV・W&B へ送る。

        Args:
            trainer: Trainer。
            pl_module: 学習 module。
            outputs: train 出力。
            batch: 入力。
            batch_idx: バッチ番号。

        Returns:
            None
        """
        objective = pl_module.training_objective
        if not hasattr(objective, "commit_update"):
            return
        diagnostics = objective.batch_diagnostics()
        if "lambda" in diagnostics:
            self._lambda_task_sum += float(diagnostics["lambda"][0])
            self._lambda_batches += 1
        for name, tensor in diagnostics.items():
            if name == "rates":
                counts = diagnostics["cell_count"].reshape(2, 2).double()
                self._rate_numerator += tensor.double() * counts
                self._rate_denominator += counts
                continue
            for index, value in enumerate(tensor.flatten()):
                pl_module.log(f"train/conditional/{name}/{index}", value, on_step=False, on_epoch=True, batch_size=1)

    def on_train_epoch_start(self, trainer: L.Trainer, pl_module: L.LightningModule) -> None:
        """率の分子・分母をリセットする。

        Args:
            trainer: Trainer。
            pl_module: 学習 module。

        Returns:
            None
        """
        self._rate_numerator = torch.zeros((2, 2), device=pl_module.device, dtype=torch.float64)
        self._rate_denominator = torch.zeros_like(self._rate_numerator)
        self._lambda_task_sum, self._lambda_batches = 0.0, 0

    def on_train_epoch_end(self, trainer: L.Trainer, pl_module: L.LightningModule) -> None:
        """率を全標本の分子・分母から算出する。

        Args:
            trainer: Trainer。
            pl_module: 学習 module。

        Returns:
            None
        """
        sampler = getattr(trainer.datamodule, "group_class_sampler", None)
        if sampler is not None and hasattr(sampler, "draw_counts"):
            for index, cell in enumerate(sampler.cells):
                draws = sampler.draw_counts[cell].sum()
                unique = (sampler.draw_counts[cell] > 0).sum()
                pl_module.log(f"train/conditional/sampler_unique/cell{index}", unique.float(), on_step=False, on_epoch=True)
                pl_module.log(f"train/conditional/sampler_duplicate_fraction/cell{index}", 1 - unique.float() / draws, on_step=False, on_epoch=True)
        if (self._rate_denominator > 0).all():
            rates = self._rate_numerator / self._rate_denominator
            for group in range(2):
                for c in range(2):
                    pl_module.log(f"train/conditional/rate/g{group}_y{c}", rates[group, c], on_step=False, on_epoch=True)
            for c in range(2):
                pl_module.log(f"train/conditional/rate_gap/y{c}", (rates[0, c] - rates[1, c]).abs(), on_step=False, on_epoch=True)

    def on_fit_end(self, trainer: L.Trainer, pl_module: L.LightningModule) -> None:
        """test を参照せず validation の選択結果を保存する。

        Args:
            trainer: Trainer。
            pl_module: 学習 module。

        Returns:
            None
        """
        if self.parent is None:
            raise RuntimeError("sanity の全件 validation 記録がない")
        if self.reference == "deferred":
            last = max(self.candidates, key=lambda row: row["epoch"]) if self.candidates else None
            self._write(trainer, "fairness_selection.json", {"status": "deferred", "selected": None, "last_epoch": last})
            return
        self._write(trainer, "fairness_selection.json", select_candidate(self.candidates, self.parent, self.method))
