"""HAM10000 の格子で、CheXpert と同じ 6 種類の図を作るための入力検証と集計。

図と比較条件は requested_figures_ham10000.ipynb に置く。CheXpert 側（`requested_analysis.py`）との
違いは評価の向きと群だけで、ここにはその判断を置く。

- 群は sex × 年齢 65 歳の 4 群。番号は `sex × 2 + (age ≥ 65)` で、学習側の GroupDRO の q と同じ並び。
- 陽性は悪性（mel / akiec）。害の大きい側は見落としなので、最悪群は TPR が最小の群（CheXpert は FPR 最大）。
- test の 294 件は sex か age を欠く。群を決められないので、全体指標も含めて属性の揃った 1,217 件で計算する。
  全行の AUROC は `auroc_all_rows` に別に残す。val に欠損は無い。

入力は package に取り込んだ run と、`analysis/common/predictions.py` が書いた best checkpoint の予測 cache。
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from analysis.common.paths import local_data_path
from analysis.common.predictions import cache_path, load_cache
from analysis.common.run_artifacts import read_config, read_epoch_metrics, read_run_record
from analysis.hypernet_conditioning_grid.requested_analysis import best_reference, restore_rates

KEYS = ["procedure", "model", "objective"]
GROUP_LABELS = ("female × <65", "female × ≥65", "male × <65", "male × ≥65")
SPLITS_DIR = "data/ham10000/splits"
# 率を出すセルの下限。data_pipeline/README.md の「群別指標の最小セル枚数」と同じ。
MIN_CELL = 20
THRESHOLD = 0.5


def condition(experiment: str) -> dict[str, str]:
    """preset 名から、学習手順・モデル・目的関数を読む。

    Args:
        experiment: config の `experiment_name`。例 `spatial-lora-ham10000-fc-from-resnet-sex-age-group-dro`

    Returns:
        dict[str, str]: `procedure`・`model`・`objective`
    """
    procedure = "two_stage" if "from-resnet" in experiment else "one_stage"
    if experiment.startswith("spatial-lora-ham10000-fc"):
        model = "spatial_lora_fc"
    elif experiment.startswith("spatial-lora"):
        model = "spatial_lora"
    else:
        model = "resnet"
    # Cotter et al. の形（各群の FPR を全体と比べる）で回し直した EO。全ペアの eo_rate とは別の目的関数として扱う。
    if "eo-fpr-overall" in experiment:
        objective = "eo_fpr_overall"
    elif "eo-rate" in experiment:
        objective = "eo_rate"
    elif "group-dro" in experiment:
        objective = "group_dro"
    else:
        objective = "erm"
    return {"procedure": procedure, "model": model, "objective": objective}


def runs(package: Path, objectives: tuple[str, ...]) -> pd.DataFrame:
    """package に取り込んだ HAM10000 の run のうち、指定した目的関数のものを条件の列つきで返す。

    `runs/` には CheXpert の run も並ぶので、experiment 名で HAM10000 の run だけを拾う。

    Args:
        package: study directory
        objectives: 読む目的関数。EO rate は回し直し中なので、いまは ERM と GroupDRO だけを渡す

    Returns:
        pd.DataFrame: `run_id`・`seed`・`experiment` と `KEYS`
    """
    rows = []
    for run_dir in sorted((package / "runs").iterdir()):
        experiment = read_config(run_dir / "config.yaml")["experiment_name"]
        if "-ham10000" not in experiment:
            continue
        rows.append(
            {
                "run_id": run_dir.name,
                "seed": int(read_run_record(run_dir)["seed"]),
                "experiment": experiment,
                **condition(experiment),
            }
        )
    table = pd.DataFrame(rows)
    return table.loc[table.objective.isin(objectives)].reset_index(drop=True)


def load_split(split: str) -> pd.DataFrame:
    """split CSV を読み、群番号を足す。sex か age が欠けた行の群は -1。

    Args:
        split: `val` または `test`

    Returns:
        pd.DataFrame: CSV の列に `group`（0〜3、欠損は -1）を足したもの
    """
    frame = pd.read_csv(local_data_path(f"{SPLITS_DIR}/{split}.csv"))
    complete = ~(frame["sex_missing"].astype(bool) | frame["age_missing"].astype(bool))
    # 学習側の age_group_65 と同じく、65 歳ちょうどは ≥65 に入る。
    group = frame["sex"].to_numpy(dtype=int) * 2 + (frame["age"].to_numpy(dtype=float) >= 65).astype(int)
    frame["group"] = np.where(complete, group, -1)
    return frame


def group_rates(probability: np.ndarray, target: np.ndarray, group: np.ndarray) -> pd.DataFrame:
    """4 群それぞれの AUROC・TPR・FPR・BACC（しきい値 0.5）を返す。下限に届かない率は NaN。

    Args:
        probability: 陽性確率
        target: 0/1 のラベル
        group: 群番号。-1 の行はどの群にも入らない

    Returns:
        pd.DataFrame: 1 行 = 1 群
    """
    predicted = probability >= THRESHOLD
    rows = []
    for g, label in enumerate(GROUP_LABELS):
        positive = (group == g) & (target == 1)
        negative = (group == g) & (target == 0)
        n_pos, n_neg = int(positive.sum()), int(negative.sum())
        tpr = predicted[positive].mean() if n_pos >= MIN_CELL else np.nan
        fpr = predicted[negative].mean() if n_neg >= MIN_CELL else np.nan
        in_group = positive | negative
        auroc = roc_auc_score(target[in_group], probability[in_group]) if min(n_pos, n_neg) >= MIN_CELL else np.nan
        rows.append(
            {
                "group": g,
                "group_label": label,
                "auroc": auroc,
                "bacc": (tpr + 1 - fpr) / 2,
                "tpr": tpr,
                "fpr": fpr,
                "n_pos": n_pos,
                "n_neg": n_neg,
            }
        )
    return pd.DataFrame(rows)


def summary(probability: np.ndarray, target: np.ndarray, group: np.ndarray) -> dict[str, float]:
    """全体と 4 群の差・最悪の一式を返す。全体指標も属性の揃った行だけで計算する。

    Eopp1 は群の TPR の最大 − 最小、Eopp0 は FPR の最大 − 最小、Eodds はその平均。min_tpr は TPR が
    最小の群の TPR（HAM10000 で害の大きい側）。gap は最良群が下がっても縮むので、図では min_tpr と並べる。

    Args:
        probability: 陽性確率（split の全行）
        target: 0/1 のラベル
        group: `load_split` の群番号

    Returns:
        dict[str, float]: 指標名 → 値
    """
    rates = group_rates(probability, target, group)
    if rates[["auroc", "tpr", "fpr"]].isna().any().any():
        raise ValueError("4 群のいずれかが評価不能")
    complete = group >= 0
    predicted = probability[complete] >= THRESHOLD
    positive, negative = target[complete] == 1, target[complete] == 0
    tpr, fpr = predicted[positive].mean(), predicted[negative].mean()
    eopp1 = rates.tpr.max() - rates.tpr.min()
    eopp0 = rates.fpr.max() - rates.fpr.min()
    return {
        "n_rows": int(complete.sum()),
        "auroc": roc_auc_score(target[complete], probability[complete]),
        "auroc_all_rows": roc_auc_score(target, probability),
        "bacc": (tpr + 1 - fpr) / 2,
        "tpr": tpr,
        "fpr": fpr,
        "auroc_gap": rates.auroc.max() - rates.auroc.min(),
        "worst_group_auroc": rates.auroc.min(),
        "eopp1": eopp1,
        "min_tpr": rates.tpr.min(),
        "eopp0": eopp0,
        "worst_fpr": rates.fpr.max(),
        "eodds": (eopp1 + eopp0) / 2,
    }


def validated_probability(package: Path, run_id: str, split: str, frame: pd.DataFrame) -> tuple[np.ndarray, dict]:
    """best checkpoint の予測 cache を、checkpoint・split の SHA256 と行対応を確かめてから読む。

    Args:
        package: study directory
        run_id: 取り込み済みの run
        split: `val` / `test`
        frame: `load_split` が返した同じ split

    Returns:
        tuple: 陽性確率、cache の metadata
    """
    path = cache_path(package / "cache", run_id, split)
    if not path.exists():
        raise FileNotFoundError(f"予測 cache が無い: {path.name}")
    cached = load_cache(path, frame["image"].to_numpy(dtype=str))
    metadata = cached["metadata"]
    best = best_reference(package / "runs" / run_id)
    split_sha = hashlib.sha256(local_data_path(f"{SPLITS_DIR}/{split}.csv").read_bytes()).hexdigest()
    observed = (metadata["run_id"], metadata["split"], metadata["checkpoint_sha256"], metadata["split_sha256"])
    if observed != (run_id, split, best["sha256"], split_sha):
        raise ValueError(f"cache 来歴の不一致: {path.name}")
    if not (cached["target"] == frame["target"].to_numpy()).all():
        raise ValueError(f"{path.name} の target が split CSV と一致しない")
    return cached["probabilities"][:, 1], {
        "cache": path.name,
        "selected_checkpoint": Path(best["path"]).name,
        **metadata,
    }


def prediction_tables(table: pd.DataFrame, package: Path, split: str) -> tuple[pd.DataFrame, ...]:
    """best checkpoint の全体指標・群別指標・来歴を run ごとに計算する。

    Args:
        table: `runs` が返した対象 run
        package: study directory
        split: `val` / `test`

    Returns:
        tuple: 全体、群別、来歴の DataFrame
    """
    frame = load_split(split)
    target, group = frame.target.to_numpy(), frame.group.to_numpy()
    points, groups, provenance = [], [], []
    for run in table.to_dict("records"):
        probability, metadata = validated_probability(package, run["run_id"], split, frame)
        points.append({**run, "split": split, **summary(probability, target, group)})
        groups.append(group_rates(probability, target, group).assign(**run, split=split))
        provenance.append({**run, "split": split, **metadata})
    return pd.DataFrame(points), pd.concat(groups, ignore_index=True), pd.DataFrame(provenance)


def _q_history(run: dict, raw: pd.DataFrame) -> list[pd.DataFrame]:
    """GroupDRO の q（epoch 内平均と epoch 末）を、和が 1 であることを確かめて縦持ちにする。"""
    frames = []
    for kind in ("q_mean", "q_end"):
        columns = [f"train/group_dro/{kind}/g{g}" for g in range(len(GROUP_LABELS))]
        if not np.allclose(raw[columns].sum(axis=1), 1, atol=1e-5):
            raise ValueError(f"q の和が 1 でない: {run['run_id']}")
        for g, label in enumerate(GROUP_LABELS):
            frames.append(
                pd.DataFrame(
                    {**run, "epoch": raw.epoch.astype(int) + 1, "group": g, "group_label": label, "kind": kind}
                ).assign(q=raw[columns[g]].to_numpy())
            )
    return frames


def learning_tables(table: pd.DataFrame, package: Path) -> tuple[pd.DataFrame, ...]:
    """epoch ログを検証し、validation の推移・q の推移・best epoch を返す。

    学習時の validation 群指標は sex × age の 4 群（`val/sex_age_group/*`）。最小 TPR の群は記録されて
    いないので推移に含まない。global TPR/FPR は accuracy・BACC・陽性率から復元する。

    Args:
        table: `runs` が返した対象 run
        package: study directory

    Returns:
        tuple: 推移、q、best epoch の DataFrame。epoch は表示用に 1 始まり
    """
    prevalence = load_split("val").target.mean()
    metrics = {
        "auroc": "val/auroc",
        "bacc": "val/bacc",
        "auroc_gap": "val/sex_age_group/auroc_gap",
        "eopp1": "val/sex_age_group/Eopp1",
        "worst_group_auroc": "val/sex_age_group/worst_group_auroc",
    }
    histories, weights, selections = [], [], []
    for run in table.to_dict("records"):
        best = best_reference(package / "runs" / run["run_id"])
        raw = pd.DataFrame(read_epoch_metrics(package / "runs" / run["run_id"] / "metrics" / "metrics.csv"))
        if raw.epoch.tolist() != list(range(30)):
            raise ValueError(f"30 epoch がそろわない: {run['run_id']}")
        selected = int(Path(best["path"]).stem.rsplit("_", 1)[1])
        if not np.isclose(raw.loc[raw.epoch == selected, "val/auroc"].iloc[0], best["score"], atol=1e-6):
            raise ValueError(f"best checkpoint と validation score が不一致: {run['run_id']}")
        if not np.isclose(raw["val/auroc"].max(), best["score"], atol=1e-6):
            raise ValueError(f"best は validation AUROC 最大ではない: {run['run_id']}")
        history = raw[list(metrics.values())].rename(columns={v: k for k, v in metrics.items()})
        history["tpr"], history["fpr"] = restore_rates(raw["val/acc"], raw["val/bacc"], prevalence)
        histories.append(history.assign(**run, epoch=raw.epoch.astype(int) + 1))
        selections.append({**run, "best_epoch": selected + 1, "val_auroc": best["score"]})
        if run["objective"] == "group_dro":
            weights.extend(_q_history(run, raw))
    return pd.concat(histories, ignore_index=True), pd.concat(weights, ignore_index=True), pd.DataFrame(selections)
