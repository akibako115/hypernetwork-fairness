"""5つの主比較notebookで共有する入力検証・8群指標・q診断。

入力はpackageのruns/と予測cacheのみ。比較ペアと出力先はnotebookが指定する。
指標は学習実装をimportせず、goldenデータで検算する。
"""

from __future__ import annotations

import hashlib
import json
import re
import shlex
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from analysis.common.paths import local_data_path
from analysis.common.predictions import cache_path, load_cache, stale_reason
from analysis.common.run_artifacts import read_config, read_epoch_metrics, read_run_record, selected_checkpoint

GROUP_LABELS = [f"{age} × {race}" for age in ("<65", "≥65") for race in ("White", "Asian", "Black", "Others")]
METRICS = [
    "auroc",
    "worst_auroc",
    "best_auroc",
    "bacc",
    "worst_bacc",
    "auroc_gap",
    "bacc_gap",
    "Eopp0",
    "Eopp1",
    "Eodds",
]


def digest(path):
    """入力のSHA256を返す。

    Args:
        path: 検算するファイル。
    Returns:
        str: SHA256。
    """
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def validate_parent(record, config, candidates):
    """凍結2-stageの親が同seed ResNet ERMのbest checkpointかを検算する。

    Args:
        record: 子runの記録。
        config: 子runの設定。
        candidates: 取り込み済みrunのpreset/seed索引。
    Returns:
        None: 親を持たないImageNet初期化runでは何もしない。
    """
    parent = record.get("parent_run") or {}
    if parent.get("run_id") is None:
        return
    matches = candidates.get(("resnet-chexpert-race-known", record["seed"]), [])
    if len(matches) != 1 or not config["model"]["freeze_backbone"]:
        raise ValueError("2-stageの同seed親またはbackbone凍結が不正")
    path, parent_record, _ = matches[0]
    best = next(item["best"] for item in parent_record["checkpoints"] if item["monitor"] == "val/auroc")
    inherited = parent["backbone_checkpoint"]
    if parent_record["status"] != "succeeded" or parent["run_id"] != path.name:
        raise ValueError("2-stageの親が同seed ResNet ERMではない")
    if inherited["sha256"] != best["sha256"] or Path(inherited["path"]).name != Path(best["path"]).name:
        raise ValueError("2-stageの親checkpointがbest validationと異なる")


def select_runs(package, conditions, seeds):
    """preset名とseedで一意に選び、checkpoint・splitを記録と照合する。

    Args:
        package: 分析package。
        conditions: 条件IDからexperiment_nameへの対応。
        seeds: 固定したseedの列。
    Returns:
        tuple: 来歴表、test frame、(条件ID, seed)から入力directoryへの対応。
    """
    candidates = {}
    for path in sorted((package / "runs").iterdir()):
        if not (path / "run.json").is_file():
            continue
        record, config = read_run_record(path), read_config(path / "config.yaml")
        key = (config.get("experiment_name"), record["seed"])
        candidates.setdefault(key, []).append((path, record, config))
    rows, directories, split_hashes = [], {}, {}
    for condition, experiment in conditions.items():
        for seed in seeds:
            matches = candidates.get((experiment, seed), [])
            if len(matches) != 1:
                raise RuntimeError(f"{condition}/s{seed}: 成功runを一意に選べない（{len(matches)}件）")
            path, record, config = matches[0]
            if record["status"] != "succeeded":
                raise ValueError(f"{path.name}: 未完了run")
            validate_parent(record, config, candidates)
            split = local_data_path(config["data"]["cv_splits_dir"]) / "test.csv"
            if split.parent.name != "splits_race_known":
                raise ValueError(f"{path.name}: race-known以外のsplit")
            split_hash = split_hashes.setdefault(split, digest(split))
            manifest = json.loads((path / "data_manifest.json").read_text())
            for name, entry in manifest["splits"].items():
                recorded_split = split.parent / f"{name}.csv"
                if recorded_split not in split_hashes:
                    split_hashes[recorded_split] = digest(recorded_split)
                if split_hashes[recorded_split] != entry["sha256"]:
                    raise ValueError(f"{path.name}: {name} splitのSHA256不一致")
            checkpoint = selected_checkpoint(path)
            best = next(item["best"] for item in record["checkpoints"] if item["monitor"] == "val/auroc")
            if checkpoint.name != Path(best["path"]).name or digest(checkpoint) != best["sha256"]:
                raise ValueError(f"{path.name}: best validation checkpoint不一致")
            rows.append(
                {
                    "condition": condition,
                    "seed": seed,
                    "run_id": path.name,
                    "checkpoint": checkpoint.name,
                    "checkpoint_sha256": best["sha256"],
                    "split_sha256": split_hash,
                    "split_path": str(split),
                    "objective": config["training_strategy"]["name"],
                    "class_weight_scope": config["training_strategy"].get("class_weight_scope", "global"),
                    "freeze_backbone": config["model"]["freeze_backbone"],
                    "parent_run": (record.get("parent_run") or {}).get("run_id"),
                    "best_val_auroc": best["score"],
                }
            )
            directories[condition, seed] = path
    provenance = pd.DataFrame(rows)
    if provenance["split_sha256"].nunique() != 1:
        raise ValueError("比較対象のtest splitが異なる")
    frame = pd.read_csv(provenance.iloc[0]["split_path"])
    if frame["race_missing"].astype(bool).any() or frame["race_group_missing"].astype(bool).any():
        raise ValueError("race欠損を含むtest split")
    return provenance, frame, directories


def prediction_command(package, directories, device="cuda:0"):
    """欠損・古いcacheのみ生成するコマンドを提示する。推論は自動起動しない。

    Args:
        package: 分析package。
        directories: 選択済みdirectory。
        device: 利用者が明示する推論GPU。
    Returns:
        str: repo rootで実行するコマンド。全cacheが有効なら空文字。
    """
    missing = [
        path
        for path in directories.values()
        if stale_reason(cache_path(package / "cache", path.name, "test"), path) is not None
    ]
    if not missing:
        return ""
    parts = [
        "uv",
        "run",
        "python",
        "analysis/common/predictions.py",
        "--study",
        package.name,
        "--split",
        "test",
        "--device",
        device,
    ]
    for path in missing:
        parts.extend(["--run-dir", str(path)])
    return shlex.join(parts)


def read_predictions(package, directories, frame):
    """cacheのcheckpoint・行順・target・予測を検証して読む。

    Args:
        package: 分析package。
        directories: 選択済みdirectory。
        frame: 同じtest split。
    Returns:
        dict: (条件ID, seed)から予測arrayへの対応。
    """
    caches = {}
    for key, path in directories.items():
        cache = cache_path(package / "cache", path.name, "test")
        reason = stale_reason(cache, path)
        if reason is not None:
            raise RuntimeError(f"{key}: {reason}。直前のコマンドで予測cacheを生成してください。")
        cached = load_cache(cache, frame["image"].to_numpy(dtype=str))
        if cached["metadata"]["run_id"] != path.name or cached["metadata"]["split"] != "test":
            raise ValueError(f"{cache}: 別run/splitのcache")
        np.testing.assert_array_equal(cached["target"], frame["target"].to_numpy())
        np.testing.assert_array_equal(cached["predictions"], cached["logits"].argmax(axis=1))
        if not np.isfinite(cached["probabilities"]).all():
            raise ValueError(f"{cache}: 非有限確率")
        caches[key] = cached
    return caches


def group_ids(frame):
    """学習と同じ年齢2×race4の8群を定義する。欠損は-1で除外する。

    Args:
        frame: race-known test split。
    Returns:
        ndarray: age>=65の区分×4+race_group。metadata encoderのraceは変更しない。
    """
    race = frame["race_group"].to_numpy()
    valid = ~(frame["age_missing"].astype(bool) | frame["race_group_missing"].astype(bool))
    valid &= frame["age"].notna() & frame["race_group"].isin(range(4))
    return np.where(valid, (frame["age"].to_numpy() >= 65).astype(int) * 4 + race, -1).astype(int)


def performance(target, probability, prediction):
    """二値分類のAUROC・BACCと陽性/陰性性能を返す。

    Args:
        target: 正解ラベル。
        probability: class 1確率。
        prediction: 固定argmax予測。
    Returns:
        dict: 件数と指標。single-class群のAUROC/BACCはNaN。
    """
    n_pos, n_neg = int((target == 1).sum()), int((target == 0).sum())
    tpr = float((prediction[target == 1] == 1).mean()) if n_pos else np.nan
    tnr = float((prediction[target == 0] == 0).mean()) if n_neg else np.nan
    return {
        "n": len(target),
        "n_positive": n_pos,
        "n_negative": n_neg,
        "auroc": roc_auc_score(target, probability) if n_pos and n_neg else np.nan,
        "bacc": (tpr + tnr) / 2,
        "tpr": tpr,
        "tnr": tnr,
        "fpr": 1 - tnr,
    }


def spread(values, expected_groups):
    """全群で定義される場合だけworst・best・gapを返す。

    Args:
        values: 群ごとの指標。
        expected_groups: 比較対象群数。
    Returns:
        tuple: worst、best、gap、有効群数。
    """
    valid = np.asarray(values, dtype=float)
    valid = valid[np.isfinite(valid)]
    if len(valid) != expected_groups or expected_groups < 2:
        return np.nan, np.nan, np.nan, len(valid)
    return valid.min(), valid.max(), np.ptp(valid), len(valid)


def fairness_metrics(groups):
    """群間gapとworst/bestを返す。EoddsはTPR/FPR gapの平均。

    Args:
        groups: 群別performanceの表。欠損群も行として含める。
    Returns:
        dict: 群指標。群の欠損を無視してgapを縮めない。
    """
    row = {}
    for metric in ("auroc", "bacc", "tpr", "tnr", "fpr"):
        worst, best, gap, count = spread(groups[metric], len(groups))
        row.update(
            {f"worst_{metric}": worst, f"best_{metric}": best, f"{metric}_gap": gap, f"valid_groups_{metric}": count}
        )
    return {**row, "Eopp0": row["tnr_gap"], "Eopp1": row["tpr_gap"], "Eodds": (row["tpr_gap"] + row["fpr_gap"]) / 2}


def evaluate(caches, frame):
    """各seedを独立に評価し、overall・8群・公平性を返す。

    Args:
        caches: (条件ID, seed)から予測への対応。
        frame: 対応するtest split。
    Returns:
        tuple: seed別評価表、8群評価表。予測をseed間で混ぜない。
    """
    summary, subgroup = [], []
    ids = group_ids(frame)
    for (condition, seed), cached in caches.items():
        target, probability, prediction = cached["target"], cached["probabilities"][:, 1], cached["predictions"]
        rows = []
        for group, label in enumerate(GROUP_LABELS):
            mask = ids == group
            rows.append(
                {
                    "condition": condition,
                    "seed": seed,
                    "group_id": group,
                    "group": label,
                    **performance(target[mask], probability[mask], prediction[mask]),
                }
            )
        subgroup.extend(rows)
        summary.append(
            {
                "condition": condition,
                "seed": seed,
                **performance(target, probability, prediction),
                **fairness_metrics(pd.DataFrame(rows)),
            }
        )
    return pd.DataFrame(summary), pd.DataFrame(subgroup)


def paired_deltas(summary, pairs):
    """同seedの比較条件−baselineを算出する。seed不足は拒否する。

    Args:
        summary: seed別評価表。
        pairs: 固定した(baseline, 比較条件)の列。
    Returns:
        DataFrame: seed別の差。公平性gapは負が改善方向。
    """
    rows = []
    for baseline, treatment in pairs:
        left = summary[summary.condition == baseline].set_index("seed")
        right = summary[summary.condition == treatment].set_index("seed")
        if left.index.has_duplicates or right.index.has_duplicates or set(left.index) != set(right.index):
            raise ValueError(f"{baseline}→{treatment}: seedが一対一で対応しない")
        delta = right[METRICS] - left[METRICS]
        delta.insert(0, "pair", f"{baseline} → {treatment}")
        rows.append(delta.reset_index())
    return pd.concat(rows, ignore_index=True)


def plot_performance(summary, subgroup, labels):
    """主指標と8群AUROCの3seed平均・標準偏差を描く。

    Args:
        summary: seed別評価表。
        subgroup: 8群評価表。
        labels: 条件IDの表示名。
    Returns:
        Figure: 条件ランキングではなく固定順の比較図。
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 4), layout="constrained")
    for ax, metric in zip(axes[:2], ("auroc", "worst_auroc"), strict=True):
        stats = summary.groupby("condition")[metric].agg(["mean", "std"]).reindex(labels)
        ax.errorbar(range(len(labels)), stats["mean"], yerr=stats["std"], fmt="o", capsize=3)
        ax.set_xticks(range(len(labels)), labels.values(), rotation=20)
        ax.set_title(metric)
        ax.set_ylim(0.5, 1)
    for condition, label in labels.items():
        stats = subgroup[subgroup.condition == condition].groupby("group_id").auroc.agg(["mean", "std"])
        axes[2].errorbar(stats.index, stats["mean"], yerr=stats["std"], marker="o", capsize=2, label=label)
    axes[2].set_xticks(range(8), GROUP_LABELS, rotation=70)
    axes[2].set_ylim(0.5, 1)
    axes[2].set_title("8-group AUROC")
    axes[2].legend(fontsize=8)
    return fig


def read_q_history(directories, provenance):
    """DROのq_end/q_mean・weighted CE・countを全epoch検証する。

    Args:
        directories: 選択したrun。
        provenance: best checkpointを記録した表。
    Returns:
        DataFrame: condition/seed/epoch/group別の診断値。ERMは含めない。
    """
    rows = []
    for (condition, seed), path in directories.items():
        config = read_config(path / "config.yaml")
        if not config["training_strategy"]["name"].startswith("group_dro"):
            continue
        metadata = json.loads((path / "metrics/group_dro_groups.json").read_text())
        if metadata["group_attributes"] != ["age_group_65", "race_group"] or metadata["group_cardinalities"] != [2, 4]:
            raise ValueError("DRO群定義が異なる")
        if [entry["label"] for entry in metadata["groups"]] != GROUP_LABELS or [
            entry["id"] for entry in metadata["groups"]
        ] != list(range(8)):
            raise ValueError("DRO群の並びが異なる")
        checkpoint = provenance.query("condition == @condition and seed == @seed").iloc[0].checkpoint
        best_epoch = int(re.search(r"_(\d+)\.ckpt$", checkpoint).group(1))
        history = read_epoch_metrics(path / "metrics/metrics.csv")
        if [int(item["epoch"]) for item in history] != list(range(config["trainer"]["max_epochs"])):
            raise ValueError(f"{path.name}: epoch欠損")
        for epoch in history:
            for group, label in enumerate(GROUP_LABELS):
                values = {
                    metric: epoch[f"train/group_dro/{metric}/g{group}"]
                    for metric in ("q_end", "q_mean", "loss", "count")
                }
                if not all(np.isfinite(values[key]) for key in ("q_end", "q_mean", "count")):
                    raise ValueError(f"{path.name}: q/count欠損")
                if values["count"] > 0 and not np.isfinite(values["loss"]):
                    raise ValueError(f"{path.name}: 観測群のloss欠損")
                rows.append(
                    {
                        "condition": condition,
                        "seed": seed,
                        "epoch": int(epoch["epoch"]),
                        "group_id": group,
                        "group": label,
                        "best_epoch": best_epoch,
                        **values,
                    }
                )
    result = pd.DataFrame(rows)
    if not result.empty:
        sums = result.groupby(["condition", "seed", "epoch"])[["q_end", "q_mean"]].sum()
        if not np.allclose(sums, 1, atol=1e-5) or (result[["q_end", "q_mean"]] < 0).any().any():
            raise ValueError("qが確率分布でない")
    return result


def plot_q(history, labels):
    """seed別q_endの推移にbest checkpoint epochを重ねる。

    Args:
        history: 検証済みq診断表。
        labels: 条件表示名。
    Returns:
        Figure: 列は条件、行はseed。epoch表示は学習ログと同じ0始まり。
    """
    conditions = list(history.condition.unique())
    seeds = sorted(history.seed.unique())
    fig, axes = plt.subplots(
        len(seeds),
        len(conditions),
        figsize=(6 * len(conditions), 2.6 * len(seeds)),
        squeeze=False,
        layout="constrained",
        sharey=True,
    )
    colors = plt.get_cmap("tab10").colors
    for col, condition in enumerate(conditions):
        for row, seed in enumerate(seeds):
            ax = axes[row, col]
            selected = history[(history.condition == condition) & (history.seed == seed)]
            for group, data in selected.groupby("group_id", sort=True):
                ax.plot(data.epoch, data.q_end, label=GROUP_LABELS[group], color=colors[group])
            ax.axvline(selected.best_epoch.iloc[0], color="black", linestyle="--", linewidth=1)
            ax.set(title=f"{labels[condition]} / seed {seed}", xlabel="epoch (0-based)", ylabel="q_end", ylim=(0, 1))
    axes[0, -1].legend(fontsize=7, ncol=2)
    return fig
