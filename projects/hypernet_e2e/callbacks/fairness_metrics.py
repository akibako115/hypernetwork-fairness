"""属性別の公平性指標を epoch 単位で記録する callback。"""

from collections.abc import Mapping, Sequence

import lightning as L
import torch

from projects.hypernet_e2e.utils.metrics import build_eval_attributes, compute_fairness_metrics


class FairnessMetricsCallback(L.Callback):
    """validation / test の出力を epoch ごとに集約して属性別公平性指標を記録する。

    step output は `logits`、`target`、`attributes` を持つ dict とする。`attributes` に
    `evaluation_categorical` があればそれを優先し、なければ `categorical` を使う。二値分類では
    Eopp0 / Eopp1 / Eodds と group 性能由来の gap・worst-group 指標をログする。
    `intersectional_groups` は評価属性を指定順の mixed-radix ID にまとめる。
    学習用group IDを必要とせず、構成属性が欠損した行は交差群の評価から除外する。
    """

    def __init__(self, intersectional_groups: Mapping[str, Mapping[str, Sequence]] | None = None) -> None:
        """検証・テストの出力バッファと評価専用の交差群を設定する。

        Args:
            intersectional_groups: 群名から attributes（評価属性名の順序）と
                cardinalities（各属性のカテゴリ数）への対応。省略時は属性別のみ。

        Returns:
            None

        Raises:
            ValueError: 属性名や cardinality の定義が不正な場合。
        """
        super().__init__()
        self.intersectional_groups = {}
        for name, definition in (intersectional_groups or {}).items():
            names = list(definition["attributes"])
            cardinalities = list(definition["cardinalities"])
            if not names or len(set(names)) != len(names) or len(names) != len(cardinalities):
                raise ValueError(f"Invalid intersectional attributes: {name}")
            if any(not isinstance(size, int) or isinstance(size, bool) or size < 1 for size in cardinalities):
                raise ValueError(f"Invalid intersectional cardinalities: {name}")
            self.intersectional_groups[name] = (names, cardinalities)
        self._val_buffer: list[dict] = []
        self._test_buffer: list[dict] = []

    def on_validation_epoch_start(self, trainer: L.Trainer, pl_module: L.LightningModule) -> None:
        """検証 epoch 開始時に出力バッファをクリアする。

        Args:
            trainer: 呼び出し元の Trainer
            pl_module: ログ先の LightningModule

        Returns:
            None
        """
        self._val_buffer = []

    def on_test_epoch_start(self, trainer: L.Trainer, pl_module: L.LightningModule) -> None:
        """テスト epoch 開始時に出力バッファをクリアする。

        Args:
            trainer: 呼び出し元の Trainer
            pl_module: ログ先の LightningModule

        Returns:
            None
        """
        self._test_buffer = []

    def on_validation_batch_end(self, trainer: L.Trainer, pl_module: L.LightningModule, outputs, batch, batch_idx: int, dataloader_idx: int = 0) -> None:
        """検証 batch の step output を保持する。None は属性なしとして無視する。

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
            self._val_buffer.append(outputs)

    def on_test_batch_end(self, trainer: L.Trainer, pl_module: L.LightningModule, outputs, batch, batch_idx: int, dataloader_idx: int = 0) -> None:
        """テスト batch の step output を保持する。None は属性なしとして無視する。

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
            self._test_buffer.append(outputs)

    def on_validation_epoch_end(self, trainer: L.Trainer, pl_module: L.LightningModule) -> None:
        """検証 epoch の公平性指標を `val/<attribute>/<metric>` に記録する。

        Args:
            trainer: 呼び出し元の Trainer
            pl_module: ログ先の LightningModule

        Returns:
            None
        """
        self._log_attribute_metrics(pl_module, "val", self._val_buffer)

    def on_test_epoch_end(self, trainer: L.Trainer, pl_module: L.LightningModule) -> None:
        """テスト epoch の公平性指標を `test/<attribute>/<metric>` に記録する。

        Args:
            trainer: 呼び出し元の Trainer
            pl_module: ログ先の LightningModule

        Returns:
            None
        """
        self._log_attribute_metrics(pl_module, "test", self._test_buffer)

    def _log_attribute_metrics(self, pl_module: L.LightningModule, prefix: str, buffer: list[dict]) -> None:
        if not buffer:
            return
        logits = torch.cat([output["logits"] for output in buffer])
        targets = torch.cat([output["target"] for output in buffer])
        attr_key = "evaluation_categorical" if "evaluation_categorical" in buffer[0]["attributes"] else "categorical"
        categorical = torch.cat([output["attributes"][attr_key] for output in buffer])
        missing_key = f"{attr_key}_missing"
        categorical_missing = torch.cat([output["attributes"][missing_key] for output in buffer]) if missing_key in buffer[0]["attributes"] else None
        if attr_key == "evaluation_categorical":
            fairness_names = getattr(pl_module, "fairness_attribute_names", None)
            attr_names = fairness_names.get("categorical") if fairness_names is not None else (None if pl_module.attribute_names is None else pl_module.attribute_names.get(attr_key))
        else:
            attr_names = None if pl_module.attribute_names is None else pl_module.attribute_names.get("categorical")
        eval_attributes, eval_attribute_names = build_eval_attributes(categorical, attr_names, categorical_missing)
        self._append_intersectional_groups(eval_attributes, eval_attribute_names)
        for attribute_name, metrics in compute_fairness_metrics(logits, targets, eval_attributes, eval_attribute_names).items():
            for metric_name, value in metrics.items():
                pl_module.log(f"{prefix}/{attribute_name}/{metric_name}", value)

    def _append_intersectional_groups(self, attributes: dict, attribute_names: dict) -> None:
        values = attributes["categorical"]
        missing = attributes.get("categorical_missing", values < 0)
        names = attribute_names["categorical"]
        columns, masks = [values], [missing]
        for group_name, (components, cardinalities) in self.intersectional_groups.items():
            if group_name in names or any(name not in names for name in components):
                raise ValueError(f"Invalid intersectional evaluation attributes: {group_name}")
            group_id = torch.zeros(values.size(0), dtype=torch.long, device=values.device)
            group_missing = torch.zeros(values.size(0), dtype=torch.bool, device=values.device)
            for name, size in zip(components, cardinalities, strict=True):
                index = names.index(name)
                column, mask = values[:, index], missing[:, index]
                if ((~mask) & ((column < 0) | (column >= size))).any():
                    raise ValueError(f"Evaluation attribute out of range: {name}")
                group_id = group_id * size + column.masked_fill(mask, 0)
                group_missing |= mask
            columns.append(group_id[:, None])
            masks.append(group_missing[:, None])
        attributes["categorical"] = torch.cat(columns, dim=1)
        attributes["categorical_missing"] = torch.cat(masks, dim=1)
        attribute_names["categorical"] = names + list(self.intersectional_groups)
