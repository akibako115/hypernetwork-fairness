"""`compute.py` が書いた指標から、条件の平均と条件間の差（患者単位 bootstrap の 95% 区間）を作る。

条件は（学習手順・モデル・目的関数・checkpoint）。値は 3 seed の平均で、seed 間の SD を別に持つ。
差の区間は、全 run に共通の患者 resample の上で「seed 平均どうしの差」を取ったもの。学習の seed の
揺れは区間に入らない（seed の揺れは SD の列で見る）。
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

CONDITION = ["procedure", "model", "objective", "checkpoint"]
LABELS = {
    "one_stage": "1 stage (from ImageNet)",
    "two_stage": "2 stage (frozen ResNet ERM)",
    "resnet": "ResNet-50",
    "spatial_lora": "Spatial LoRA (stage3-4 + fc)",
    "spatial_lora_fc": "Spatial LoRA (fc only)",
    "erm": "ERM",
    "group_dro": "GroupDRO",
    "eo_rate": "EO rate",
    "best": "best val AUROC epoch",
    "last": "last epoch",
}


def condition_means(points: pd.DataFrame) -> pd.DataFrame:
    """run ごとの指標を、条件ごとの seed 平均と SD にまとめる。

    Args:
        points: `results/run_metrics_<split>.csv`

    Returns:
        pd.DataFrame: 条件 × 指標の平均（`<metric>`）と SD（`<metric>_sd`）、seed 数（`seeds`）
    """
    metrics = [c for c in points.columns if c not in {"run_id", "seed", "split", *CONDITION}]
    grouped = points.groupby(CONDITION)
    means = grouped[metrics].mean()
    sds = grouped[metrics].std().add_suffix("_sd")
    return means.join(sds).assign(seeds=grouped.size()).reset_index()


class Bootstrap:
    """`cache/bootstrap_<split>.npz` を、条件の seed 平均の resample として引けるようにする。"""

    def __init__(self, path: Path, points: pd.DataFrame) -> None:
        """bootstrap と、run と条件の対応を読む。

        Args:
            path: `compute.py` が書いた `.npz`
            points: 同じ split の `results/run_metrics_<split>.csv`

        Returns:
            None
        """
        with np.load(path, allow_pickle=False) as cached:
            self.metrics = [str(m) for m in cached["metrics"]]
            keys = [str(k) for k in cached["keys"]]
            values = cached["values"]
        self._values = dict(zip(keys, values, strict=True))
        self._points = points

    def condition(self, procedure: str, model: str, objective: str, checkpoint: str) -> np.ndarray:
        """条件の 3 seed を平均した指標の resample `[resample, 指標]` を返す。

        Args:
            procedure: `one_stage` / `two_stage`
            model: `resnet` / `spatial_lora` / `spatial_lora_fc`
            objective: `erm` / `group_dro` / `eo_rate`
            checkpoint: `best` / `last`

        Returns:
            np.ndarray: seed 平均の resample

        Raises:
            KeyError: 条件に当たる run が無い場合
        """
        rows = self._points[
            (self._points.procedure == procedure)
            & (self._points.model == model)
            & (self._points.objective == objective)
            & (self._points.checkpoint == checkpoint)
        ]
        if rows.empty:
            raise KeyError((procedure, model, objective, checkpoint))
        return np.mean([self._values[f"{r}|{checkpoint}"] for r in rows.run_id], axis=0)

    def difference(self, a: tuple[str, str, str, str], b: tuple[str, str, str, str]) -> pd.DataFrame:
        """条件 a − 条件 b の差の点推定（resample の平均ではなく、元データの seed 平均の差）と 95% 区間。

        Args:
            a: 比べる条件 `(procedure, model, objective, checkpoint)`
            b: 基準の条件

        Returns:
            pd.DataFrame: 指標ごとの `low`・`high`（2.5%・97.5%）
        """
        delta = self.condition(*a) - self.condition(*b)
        low, high = np.nanpercentile(delta, [2.5, 97.5], axis=0)
        return pd.DataFrame({"metric": self.metrics, "low": low, "high": high})


def compare(means: pd.DataFrame, bootstrap: Bootstrap, a: tuple, b: tuple) -> pd.DataFrame:
    """条件 a − b の差を、点推定と 95% 区間で返す。

    Args:
        means: `condition_means` の結果
        bootstrap: 同じ split の `Bootstrap`
        a: 比べる条件 `(procedure, model, objective, checkpoint)`
        b: 基準の条件

    Returns:
        pd.DataFrame: `metric`・`diff`・`low`・`high`
    """
    key = means.set_index(CONDITION)
    point = key.loc[a, bootstrap.metrics] - key.loc[b, bootstrap.metrics]
    interval = bootstrap.difference(a, b)
    return interval.assign(diff=point.to_numpy(dtype=float))[["metric", "diff", "low", "high"]]


def improved(diff: pd.DataFrame) -> dict[str, bool]:
    """「改善と呼ぶ条件」を満たすかを、差の区間で判定する。

    研究の前提の定義どおり、次の 3 つを同時に満たすときだけ改善と呼ぶ。CheXpert の害の大きい側は FPR。

    - 全体の性能を落とさない：全体 AUROC の差の区間の上端が 0 以上（有意に下がっていない）
    - 群のあいだの差を縮める：Eodds の差の区間の上端が 0 未満
    - 最悪群が上がる：最悪群の FPR の差の区間の上端が 0 未満

    Args:
        diff: `compare` の結果

    Returns:
        dict[str, bool]: 3 条件それぞれと、全部を満たすか（`improved`）
    """
    row = diff.set_index("metric")
    result = {
        "auroc_not_worse": bool(row.loc["auroc", "high"] >= 0),
        "eodds_smaller": bool(row.loc["eodds", "high"] < 0),
        "worst_fpr_lower": bool(row.loc["worst_fpr", "high"] < 0),
    }
    result["improved"] = all(result.values())
    return result
