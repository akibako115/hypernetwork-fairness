"""6 種類の図が共有する入力検証と集計。図と比較条件は requested_figures.ipynb に置く。

入力は package に取り込んだ run と best checkpoint の予測 cache。checkpoint 本体が remote に
ある場合も、run.json の SHA256 と cache の SHA256 を照合する。各 epoch の global TPR/FPR は
micro accuracy・macro recall (BACC)・固定 validation の陽性率から復元する。
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pandas as pd

from analysis.common.predictions import read_metadata
from analysis.common.run_artifacts import read_epoch_metrics, read_run_record
from analysis.hypernet_conditioning_grid.grid_metrics import (
    GROUP_LABELS,
    group_rates,
    load_split,
    positive_probability,
    summary,
)

KEYS = ["procedure", "model", "objective"]


def restore_rates(accuracy, bacc, prevalence: float):
    """accuracy と BACC から二値分類の TPR/FPR を復元する。

    Args:
        accuracy: 全 validation の micro accuracy。
        bacc: 全 validation のクラス別 recall の平均。
        prevalence: 同じ validation の陽性率。0.5 では復元不能。

    Returns:
        tuple: TPR, FPR。浮動小数点誤差を超えて範囲を外れたら例外。
    """
    if np.isclose(prevalence, 0.5):
        raise ValueError("陽性率 0.5 では accuracy と BACC が従属する")
    tpr = (accuracy - 2 * (1 - prevalence) * bacc) / (2 * prevalence - 1)
    fpr = 1 - 2 * bacc + tpr
    values = np.asarray([tpr, fpr])
    if not np.isfinite(values).all() or (values < -1e-5).any() or (values > 1 + 1e-5).any():
        raise ValueError("復元した率が [0,1] 外。split または集計定義を確認する")
    return np.clip(tpr, 0, 1), np.clip(fpr, 0, 1)


def best_reference(run_dir: Path) -> dict:
    """run.json に記録された best val AUROC checkpoint を返す。

    Args:
        run_dir: 取り込み済みの run。

    Returns:
        dict: path・sha256・score。
    """
    record = read_run_record(run_dir)
    if record["status"] != "succeeded":
        raise ValueError(f"未完了 run: {run_dir.name}")
    entries = [c["best"] for c in record["checkpoints"] if c["monitor"] == "val/auroc"]
    if len(entries) != 1:
        raise ValueError(f"best checkpoint が一意でない: {run_dir.name}")
    return entries[0]


def validated_cache(path: Path, run_dir: Path, split: str, split_sha: str) -> dict:
    """best の予測が manifest と現在の split に対応することを確認する。

    Args:
        path: 予測 cache。
        run_dir: 取り込み済み run。
        split: val / test。
        split_sha: 現在の split CSV の SHA256。

    Returns:
        dict: 検証した cache metadata。
    """
    metadata = read_metadata(path)
    best = best_reference(run_dir)
    expected = (run_dir.name, split, best["sha256"], split_sha)
    observed = (
        metadata["run_id"],
        metadata["split"],
        metadata["checkpoint_sha256"],
        metadata["split_sha256"],
    )
    if observed != expected:
        raise ValueError(f"cache 来歴の不一致: {path.name}")
    return metadata


def best_cache_files(table: pd.DataFrame, package: Path, split: str) -> dict:
    """ファイル名ではなくcheckpoint SHA256でbest予測を選ぶ。

    Args:
        table: 対象 run。
        package: study directory。
        split: val / test。

    Returns:
        dict: (run_id, best) → cache。bestとlastが同一バイト列ならlast名も採用。
    """
    expected = {r: best_reference(package / "runs" / r)["sha256"] for r in table.run_id}
    found = {}
    for path in sorted((package / "cache").glob(f"*_{split}.npz")):
        if path.name.startswith("bootstrap_"):
            continue
        metadata = read_metadata(path)
        run_id = metadata["run_id"]
        if run_id not in expected or metadata.get("checkpoint_sha256") != expected[run_id]:
            continue
        key = (run_id, "best")
        if key in found:
            raise ValueError(f"best cache が重複: {run_id}")
        found[key] = path
    return found


def prediction_tables(table: pd.DataFrame, package: Path, files: dict, split: str, split_path: Path):
    """best checkpoint の全体・8 群指標と来歴を計算する。

    Args:
        table: 対象 run と条件。
        package: study directory。
        files: (run_id, best|last) から cache path。
        split: val / test。
        split_path: 実際に使う split CSV。

    Returns:
        tuple: 全体、群別、来歴の DataFrame。
    """
    frame = load_split(split)
    target, group = frame.target.to_numpy(), frame.group.to_numpy()
    digest = hashlib.sha256(split_path.read_bytes()).hexdigest()
    points, groups, provenance = [], [], []
    for run in table.to_dict("records"):
        path = files[(run["run_id"], "best")]
        metadata = validated_cache(path, package / "runs" / run["run_id"], split, digest)
        probability = positive_probability(path, frame)
        overall = summary(probability, target, group)
        overall["bacc"] = (overall["tpr"] + 1 - overall["fpr"]) / 2
        rates = group_rates(probability, target, group)
        if not np.isfinite([rates.auroc, rates.tpr, rates.fpr]).all():
            raise ValueError("8 群のいずれかが評価不能")
        points.append({**run, "split": split, **overall})
        for g, label in enumerate(GROUP_LABELS):
            groups.append(
                {
                    **run,
                    "split": split,
                    "group": g,
                    "group_label": label,
                    "auroc": rates.auroc[g],
                    "bacc": (rates.tpr[g] + 1 - rates.fpr[g]) / 2,
                    "tpr": rates.tpr[g],
                    "fpr": rates.fpr[g],
                    "n_pos": int(rates.n_pos[g]),
                    "n_neg": int(rates.n_neg[g]),
                }
            )
        provenance.append(
            {
                **run,
                "split": split,
                "cache": str(path.relative_to(package)),
                "checkpoint": metadata["checkpoint"],
                "checkpoint_sha256": metadata["checkpoint_sha256"],
                "selected_checkpoint": best_reference(package / "runs" / run["run_id"])["path"],
                "split_sha256": digest,
            }
        )
    return pd.DataFrame(points), pd.DataFrame(groups), pd.DataFrame(provenance)


def learning_tables(table: pd.DataFrame, package: Path):
    """epoch ログを検証し、validation 推移・q 推移・best epoch を返す。

    Args:
        table: 対象 run と条件。
        package: study directory。

    Returns:
        tuple: 推移、q、best epoch の DataFrame。epoch は表示用に 1 始まり。
    """
    prevalence = load_split("val").target.mean()
    histories, weights, selections = [], [], []
    metrics = {
        "auroc": "val/auroc",
        "bacc": "val/bacc",
        "auroc_gap": "val/age_race_group/auroc_gap",
        "eopp0": "val/age_race_group/Eopp0",
        "worst_group_auroc": "val/age_race_group/worst_group_auroc",
    }
    for run in table.to_dict("records"):
        run_dir = package / "runs" / run["run_id"]
        best = best_reference(run_dir)
        raw = pd.DataFrame(read_epoch_metrics(run_dir / "metrics" / "metrics.csv"))
        if raw.epoch.tolist() != list(range(30)):
            raise ValueError(f"30 epoch がそろわない: {run['run_id']}")
        selected = int(Path(best["path"]).stem.rsplit("_", 1)[1])
        if not np.isclose(raw.loc[raw.epoch == selected, "val/auroc"].iloc[0], best["score"], atol=1e-6):
            raise ValueError("best checkpoint と validation score が不一致")
        if not np.isclose(raw["val/auroc"].max(), best["score"], atol=1e-6):
            raise ValueError("best は validation AUROC 最大ではない")
        history = raw[list(metrics.values())].rename(columns={v: k for k, v in metrics.items()})
        history["tpr"], history["fpr"] = restore_rates(raw["val/acc"], raw["val/bacc"], prevalence)
        history = history.assign(**run, epoch=raw.epoch.astype(int) + 1)
        histories.append(history)
        selections.append({**run, "best_epoch": selected + 1, "val_auroc": best["score"]})
        if run["objective"] == "group_dro":
            for kind in ("q_mean", "q_end"):
                cols = [f"train/group_dro/{kind}/g{g}" for g in range(8)]
                if not np.allclose(raw[cols].sum(axis=1), 1, atol=1e-5):
                    raise ValueError("q の和が 1 でない")
                for g, label in enumerate(GROUP_LABELS):
                    weights.append(
                        pd.DataFrame(
                            {
                                **run,
                                "epoch": raw.epoch.astype(int) + 1,
                                "group": g,
                                "group_label": label,
                                "kind": kind,
                                "q": raw[cols[g]],
                            }
                        )
                    )
    return pd.concat(histories, ignore_index=True), pd.concat(weights, ignore_index=True), pd.DataFrame(selections)
