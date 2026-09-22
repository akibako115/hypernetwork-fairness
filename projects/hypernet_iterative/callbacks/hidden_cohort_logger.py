"""固定 cohort の前提を検証し、cohort 別の指標を記録する callback 群。

`CohortSingleProcessCallback` は group ID を仮定できる実行構成かを、
`CohortValidityCallback` は sidecar の group ID が範囲内かを開始時に確かめる。
`HiddenCohortMetricsCallback` は cohort ごとの AUROC と worst-group 指標をログする。
"""

import lightning as L
import pandas as pd
import torch
import torch.nn.functional as F
from torchmetrics.functional.classification import binary_auroc


class CohortValidityCallback(L.Callback):
    """学習開始前にvalidation cohortがcheckpoint選択に使えることを検証する。"""

    def __init__(
        self,
        num_groups: int,
        group_key: str = "group_id",
        target_column: str = "target",
    ) -> None:
        """検証対象の group 数・列名を受け取り、不正な値なら即座に例外を送出する。"""
        if num_groups < 2:
            raise ValueError(f"num_groups must be at least 2, got {num_groups}")
        if not group_key:
            raise ValueError("group_key must not be empty")
        if not target_column:
            raise ValueError("target_column must not be empty")
        self.num_groups = num_groups
        self.group_key = group_key
        self.target_column = target_column

    def on_fit_start(self, trainer: L.Trainer, pl_module: L.LightningModule) -> None:
        """学習開始時に validation dataset の DataFrame を取得し、cohort の妥当性を検証する。

        Args:
            trainer: `datamodule.data_val` の取得元
            pl_module: 呼び出し元の LightningModule（検証には使わない）

        Returns:
            None

        Raises:
            TypeError: validation dataset が `.df` を持たない場合。
            ValueError: 必須列が無い、group ID が欠損・範囲外・未出現、
                またはいずれかの group が片方のクラスしか持たない場合。
        """
        dataset = getattr(trainer.datamodule, "data_val", None)
        frame = getattr(dataset, "df", None)
        if not isinstance(frame, pd.DataFrame):
            raise TypeError("validation dataset must expose its source DataFrame as .df")
        self._validate(frame)

    def _validate(self, frame: pd.DataFrame) -> None:
        """validation cohort の DataFrame が checkpoint 選択に使える形になっているか検証する。

        必須列の存在、group ID の欠損・範囲、全 group の網羅、
        各 group が両クラスを含むことを順に確認し、いずれかを満たさなければ例外を送出する。

        Args:
            frame: validation dataset が保持する `.df`（group_key・target_column を含む）。
        """
        # group_key / target_column が DataFrame に存在するか確認する
        required_columns = {self.group_key, self.target_column}
        missing_columns = required_columns - set(frame.columns)
        if missing_columns:
            raise ValueError(f"validation cohort is missing columns: {sorted(missing_columns)}")

        # group ID に欠損や [0, num_groups) の範囲外の値がないか確認する
        group_ids = frame[self.group_key]
        if group_ids.isna().any():
            raise ValueError("validation cohort contains missing group IDs")
        invalid_ids = sorted(set(group_ids) - set(range(self.num_groups)))
        if invalid_ids:
            raise ValueError(f"validation cohort has IDs outside [0, {self.num_groups - 1}]: {invalid_ids}")

        # num_groups で定義された group ID がすべて実データに出現するか確認する
        missing_groups = sorted(set(range(self.num_groups)) - set(group_ids))
        if missing_groups:
            raise ValueError(f"validation cohort is missing group IDs: {missing_groups}")

        # 各 group が正例・負例の両方を含むか確認する（AUROC 計算が可能であるための前提）
        invalid_groups = []
        for group_id in range(self.num_groups):
            labels = set(frame.loc[group_ids == group_id, self.target_column])
            if labels != {0, 1}:
                invalid_groups.append(group_id)
        if invalid_groups:
            raise ValueError(f"validation hidden cohorts must each contain both classes; invalid group IDs: {invalid_groups}")


class CohortSingleProcessCallback(L.Callback):
    """固定 cohort の学習を single-process に制限する。"""

    def on_fit_start(self, trainer: L.Trainer, pl_module: L.LightningModule) -> None:
        """GroupDRO の q や hidden-cohort 指標が distributed 未対応であるため、multi-process 実行を拒否する。

        Args:
            trainer: `world_size` の取得元
            pl_module: 呼び出し元の LightningModule（判定には使わない）

        Returns:
            None

        Raises:
            RuntimeError: `trainer.world_size` が 1 でない場合。
        """
        if trainer.world_size != 1:
            raise RuntimeError("fixed cohort experiments require trainer.world_size=1 because GroupDRO q and hidden-cohort metrics are not distributed")


class HiddenCohortMetricsCallback(L.Callback):
    """固定hidden cohortごとの検証・テスト指標をエポック末に記録する。"""

    def __init__(
        self,
        num_groups: int,
        group_key: str = "group_id",
        require_binary_val_groups: bool = False,
    ) -> None:
        """cohort 数・group ID の列名・validation での両クラス必須フラグを受け取り、バッファを初期化する。"""
        if num_groups < 2:
            raise ValueError(f"num_groups must be at least 2, got {num_groups}")
        if not group_key:
            raise ValueError("group_key must not be empty")
        self.num_groups = num_groups
        self.group_key = group_key
        self.require_binary_val_groups = require_binary_val_groups
        self._buffers: dict[str, list[dict]] = {"val": [], "test": []}

    def on_validation_epoch_start(self, trainer: L.Trainer, pl_module: L.LightningModule) -> None:
        """検証エポック開始時にバッファをクリアする。

        Args:
            trainer: 呼び出し元の Trainer
            pl_module: ログ先の LightningModule

        Returns:
            None
        """
        self._buffers["val"] = []

    def on_test_epoch_start(self, trainer: L.Trainer, pl_module: L.LightningModule) -> None:
        """テストエポック開始時にバッファをクリアする。

        Args:
            trainer: 呼び出し元の Trainer
            pl_module: ログ先の LightningModule

        Returns:
            None
        """
        self._buffers["test"] = []

    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx: int = 0) -> None:
        """検証バッチの step output をバッファへ追加する（属性データがないバッチは無視）。

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
            self._buffers["val"].append(outputs)

    def on_test_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx: int = 0) -> None:
        """テストバッチの step output をバッファへ追加する（属性データがないバッチは無視）。

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
            self._buffers["test"].append(outputs)

    def on_validation_epoch_end(self, trainer: L.Trainer, pl_module: L.LightningModule) -> None:
        """検証エポック末にバッファを集計し、cohort ごとの指標をログする。

        Args:
            trainer: sanity check 中かどうかの判定に使う
            pl_module: ログ先の LightningModule

        Returns:
            None

        Raises:
            ValueError: `require_binary_val_groups` が真で、sanity check 以外の検証 epoch に
                片方のクラスしか持たない cohort があった場合。
        """
        self._log_metrics(trainer, pl_module, "val")

    def on_test_epoch_end(self, trainer: L.Trainer, pl_module: L.LightningModule) -> None:
        """テストエポック末にバッファを集計し、cohort ごとの指標をログする。

        Args:
            trainer: 呼び出し元の Trainer
            pl_module: ログ先の LightningModule

        Returns:
            None
        """
        self._log_metrics(trainer, pl_module, "test")

    def _log_empty_metrics(self, pl_module: L.LightningModule, phase: str) -> None:
        """全バッチが空でもcohortごとの欠測指標を記録する。"""
        for group_id in range(self.num_groups):
            pl_module.log(f"{phase}/hidden_support_{group_id:02d}", 0.0)
            pl_module.log(f"{phase}/hidden_auroc_{group_id:02d}", float("nan"))
            pl_module.log(f"{phase}/hidden_loss_{group_id:02d}", float("nan"))
        pl_module.log(f"{phase}/hidden_valid_auroc_groups", 0.0)
        pl_module.log(f"{phase}/hidden_min_auroc", float("nan"))
        pl_module.log(f"{phase}/hidden_auroc_gap", float("nan"))
        pl_module.log(f"{phase}/hidden_max_loss", float("nan"))
        pl_module.log(f"{phase}/hidden_loss_gap", float("nan"))
        pl_module.log(f"{phase}/hidden_min_bacc", float("nan"))

    def _log_metrics(self, trainer: L.Trainer | None, pl_module: L.LightningModule, phase: str) -> None:
        """バッファ済みバッチを cohort（group_id）ごとに集計し、support・AUROC・loss・bacc をログする。

        group ごとの指標に加え、有効 group 数・最小 AUROC・AUROC gap・最大 loss・
        loss gap・最小 bacc という cohort 間の格差サマリも `<phase>/hidden_*` としてログする。
        `require_binary_val_groups=True` の場合、sanity check 以外で validation の
        いずれかの group が両クラスを含まなければ学習を止める。

        Args:
            trainer: `sanity_checking` 判定に使う Trainer（None 許容）。
            pl_module: ログ先の LightningModule。
            phase: "val" または "test"。
        """
        buffer = self._buffers[phase]
        if not buffer:
            self._log_empty_metrics(pl_module, phase)
            return
        # 全バッチの logits / target / group_id を結合し、エポック全体で1回だけ指標を計算する
        logits = torch.cat([item["logits"] for item in buffer])
        targets = torch.cat([item["target"] for item in buffer]).long()
        group_ids = torch.cat([item["attributes"][self.group_key] for item in buffer]).long()
        if logits.ndim != 2 or logits.shape[1] != 2:
            raise ValueError("HiddenCohortMetricsCallback requires binary logits with shape [batch_size, 2]")
        if (targets < 0).any() or (targets > 1).any():
            raise ValueError("HiddenCohortMetricsCallback requires binary targets in {0, 1}")
        if (group_ids < 0).any() or (group_ids >= self.num_groups).any():
            raise ValueError(f"{self.group_key!r} contains IDs outside [0, {self.num_groups - 1}]")

        # 正例確率・サンプルごとの loss・予測クラスを一括で算出しておき、group ごとの集計で使い回す
        probs = torch.softmax(logits, dim=1)[:, 1]
        losses = F.cross_entropy(logits, targets, reduction="none")
        preds = logits.argmax(dim=1)
        aurocs: list[float] = []
        baccs: list[float] = []
        mean_losses: list[float] = []
        invalid_val_groups: list[int] = []
        for group_id in range(self.num_groups):
            # group に属するサンプルを抽出し、support（サンプル数）を記録する
            mask = group_ids.eq(group_id)
            support = int(mask.sum())
            pl_module.log(f"{phase}/hidden_support_{group_id:02d}", float(support))
            if support == 0:
                pl_module.log(f"{phase}/hidden_auroc_{group_id:02d}", float("nan"))
                pl_module.log(f"{phase}/hidden_loss_{group_id:02d}", float("nan"))
                if phase == "val":
                    invalid_val_groups.append(group_id)
                continue
            group_targets = targets[mask]
            group_preds = preds[mask]
            mean_loss = float(losses[mask].mean())
            mean_losses.append(mean_loss)
            # 群ごとの loss を残す。max と gap だけでは、どの群が重く、その群が stage の中で
            # 改善したのかが後から追えない。GroupDRO の q は群 loss で動くので、q と対で要る。
            pl_module.log(f"{phase}/hidden_loss_{group_id:02d}", mean_loss)
            # クラスごとの recall（TPR/TNR）を求め、balanced accuracy として平均する
            recalls = []
            for label in (0, 1):
                label_mask = group_targets.eq(label)
                if label_mask.any():
                    recalls.append(float(group_preds[label_mask].eq(label).float().mean()))
            if group_targets.unique().numel() == 2:
                baccs.append(sum(recalls) / len(recalls))
            # 両クラスが揃っている group のみ AUROC を計算し、片方しかない group は集計から除外する
            if group_targets.unique().numel() == 2:
                auroc = float(binary_auroc(probs[mask], group_targets))
                aurocs.append(auroc)
                pl_module.log(f"{phase}/hidden_auroc_{group_id:02d}", auroc)
            else:
                pl_module.log(f"{phase}/hidden_auroc_{group_id:02d}", float("nan"))
                if phase == "val":
                    invalid_val_groups.append(group_id)

        # cohort 間の格差を要約する指標（最小/最大/gap）をまとめてログする
        pl_module.log(f"{phase}/hidden_valid_auroc_groups", float(len(aurocs)))
        pl_module.log(f"{phase}/hidden_min_auroc", min(aurocs) if aurocs else float("nan"))
        pl_module.log(f"{phase}/hidden_auroc_gap", max(aurocs) - min(aurocs) if len(aurocs) >= 2 else float("nan"))
        pl_module.log(f"{phase}/hidden_max_loss", max(mean_losses) if mean_losses else float("nan"))
        pl_module.log(f"{phase}/hidden_loss_gap", max(mean_losses) - min(mean_losses) if len(mean_losses) >= 2 else float("nan"))
        pl_module.log(f"{phase}/hidden_min_bacc", min(baccs) if baccs else float("nan"))
        # sanity check 中を除き、validation の cohort が両クラスを欠く場合は学習を止める
        if self.require_binary_val_groups and phase == "val" and invalid_val_groups and not getattr(trainer, "sanity_checking", False):
            raise ValueError(f"validation hidden cohorts must each contain both classes; invalid group IDs: {invalid_val_groups}")


class GroupDRODiagnosticsCallback(L.Callback):
    """GroupDROのadversarial weight推移をtrain epochごとに記録する。"""

    def on_train_epoch_end(self, trainer: L.Trainer, pl_module: L.LightningModule) -> None:
        """train epoch 末に GroupDRO の adversarial weight（`adv_probs`）を group ごとにログする。

        `training_objective` が GroupDRO でない、または `adv_probs` を持たない場合は何もしない。
        weight の最大値と、分布の偏りを示す entropy も併せて記録する。

        Args:
            trainer: 呼び出し元の Trainer（ログには使わない）
            pl_module: `training_objective` の取得元であり、ログ先

        Returns:
            None
        """
        objective = getattr(pl_module, "training_objective", None)
        probabilities = getattr(objective, "adv_probs", None)
        if probabilities is None:
            return
        probs = probabilities.detach()
        # group ごとの weight q_i と、その最大値をログする
        for group_id, value in enumerate(probs):
            pl_module.log(f"train/group_dro/q_{group_id:02d}", value)
        pl_module.log("train/group_dro/max_q", probs.max())
        # weight 分布の entropy（偏りの度合い）を計算してログする。log(0) を避けるため eps でクランプする
        entropy = -(probs * probs.clamp_min(torch.finfo(probs.dtype).eps).log()).sum()
        pl_module.log("train/group_dro/weight_entropy", entropy)
