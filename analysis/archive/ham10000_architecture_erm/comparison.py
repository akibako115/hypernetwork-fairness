"""HAM10000のResNet・Spatial LoRA・HyperAdapt比較で使う入力検証・群別指標・paired bootstrap。

入力はpackageの`runs/`と予測cacheだけ。どの集合で比べるか、何を図にするかはnotebookが決める。
群の切り方と最小セル枚数はこのpackageの判断なので`analysis/common/`へ上げない。
"""

from __future__ import annotations

import hashlib
import json
import shlex
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from analysis.common.paths import local_data_path
from analysis.common.predictions import cache_path, load_cache, stale_reason
from analysis.common.run_artifacts import read_config, read_run_record, selected_checkpoint

MODELS = {"resnet": "ResNet-50", "spatial_lora": "Spatial LoRA", "hyperadapt": "HyperAdapt"}
EXPERIMENTS = {
    "resnet": "resnet-ham10000",
    "spatial_lora": "spatial-lora-ham10000",
    "hyperadapt": "hyperadapt-ham10000",
}
SEEDS = (42, 43, 44)
BASELINE = "resnet"
# rateを推定するセルの下限。data_pipeline/README.mdの「群別指標の最小セル枚数」と同じ規則。
MIN_CELL = 20
# 評価軸。値は(列, 表示名の列)。交差群は列の組で表す。
AXES = {
    "sex": (("sex",), ("female", "male")),
    "age_group_65": (("age_group_65",), ("<65", "≥65")),
    "sex × age": (("sex", "age_group_65"), ("female <65", "female ≥65", "male <65", "male ≥65")),
}


def digest(path: Path) -> str:
    """fileのSHA256を返す。"""
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def select_runs(package: Path) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """モデルとseedで成功runを一意に選び、checkpoint・splitを学習記録と照合する。

    Args:
        package: 分析package。
    Returns:
        tuple: 来歴表、test split、(モデル, seed)からrun directoryへの対応。
    Raises:
        RuntimeError: runが一意に決まらない場合。
        ValueError: 未完了run、checkpointやsplitのSHA256が記録と食い違う場合。
    """
    candidates: dict[tuple, list] = {}
    for path in sorted((package / "runs").iterdir()):
        record, config = read_run_record(path), read_config(path / "config.yaml")
        candidates.setdefault((config["experiment_name"], record["seed"]), []).append((path, record, config))
    rows, directories = [], {}
    for model, experiment in EXPERIMENTS.items():
        for seed in SEEDS:
            matches = candidates.get((experiment, seed), [])
            if len(matches) != 1:
                raise RuntimeError(f"{model}/s{seed}: runを一意に選べない（{len(matches)}件）")
            path, record, config = matches[0]
            if record["status"] != "succeeded":
                raise ValueError(f"{path.name}: 未完了run")
            splits_dir = local_data_path(config["data"]["cv_splits_dir"])
            manifest = json.loads((path / "data_manifest.json").read_text())
            for name, entry in manifest["splits"].items():
                if digest(splits_dir / f"{name}.csv") != entry["sha256"]:
                    raise ValueError(f"{path.name}: {name} splitのSHA256が学習時と違う")
            checkpoint = selected_checkpoint(path)
            best = next(item["best"] for item in record["checkpoints"] if item["monitor"] == "val/auroc")
            if checkpoint.name != Path(best["path"]).name or digest(checkpoint) != best["sha256"]:
                raise ValueError(f"{path.name}: best validation checkpointと一致しない")
            rows.append(
                {
                    "model": model,
                    "seed": seed,
                    "run_id": path.name,
                    "checkpoint": checkpoint.name,
                    "checkpoint_sha256": best["sha256"],
                    "best_val_auroc": best["score"],
                    "test_split": str(splits_dir / "test.csv"),
                    "test_sha256": digest(splits_dir / "test.csv"),
                }
            )
            directories[model, seed] = path
    provenance = pd.DataFrame(rows)
    if provenance["test_sha256"].nunique() != 1:
        raise ValueError("比較対象のtest splitが異なる")
    return provenance, pd.read_csv(provenance.loc[0, "test_split"]), directories


def prediction_command(package: Path, directories: dict, device: str = "cuda:0") -> str:
    """欠損・古いcacheだけを作るコマンドを返す。推論は自動で起動しない。"""
    missing = [p for p in directories.values() if stale_reason(cache_path(package / "cache", p.name, "test"), p)]
    if not missing:
        return ""
    parts = ["uv", "run", "python", "analysis/common/predictions.py", "--study", package.name]
    parts += ["--split", "test", "--device", device]
    for path in missing:
        parts += ["--run-dir", str(path)]
    return shlex.join(parts)


def read_scores(package: Path, directories: dict, frame: pd.DataFrame) -> dict:
    """cacheを読み、行順・targetを検算して、(モデル, seed)から陽性確率への対応を返す。

    Raises:
        RuntimeError: cacheが無いか古い場合。
        ValueError: 別runのcache、またはtargetがsplitと食い違う場合。
    """
    scores = {}
    for key, path in directories.items():
        cache = cache_path(package / "cache", path.name, "test")
        reason = stale_reason(cache, path)
        if reason is not None:
            raise RuntimeError(f"{key}: {reason}")
        cached = load_cache(cache, frame["image"].to_numpy(dtype=str))
        if cached["metadata"]["run_id"] != path.name:
            raise ValueError(f"{cache}: 別runのcache")
        np.testing.assert_array_equal(cached["target"], frame["target"].to_numpy())
        scores[key] = cached["probabilities"][:, 1]
    return scores


def group_ids(frame: pd.DataFrame) -> dict[str, np.ndarray]:
    """評価軸ごとの群番号を返す。sexかageが欠損する行は-1。

    age_group_65は学習中callbackと同じく65歳以上を1とする。HAM10000のageは5歳刻みなので、
    MEDFAIRの2区分とも同じ群になる。
    """
    complete = ~(frame["sex_missing"].astype(bool) | frame["age_missing"].astype(bool))
    columns = {"sex": frame["sex"].to_numpy(), "age_group_65": (frame["age"].to_numpy() >= 65).astype(int)}
    ids = {}
    for axis, (parts, _) in AXES.items():
        value = np.zeros(len(frame), dtype=int)
        for part in parts:
            value = value * 2 + columns[part]
        ids[axis] = np.where(complete, value, -1)
    return ids


def cell_metrics(target: np.ndarray, score: np.ndarray, min_cell: int = MIN_CELL) -> dict[str, float]:
    """閾値0.5での性能を返す。陽性・陰性が`min_cell`未満のセルの指標はNaNにする。

    TPRは陽性数、TNR・FPRは陰性数、AUROC・BACCは両方を見る。下限はtestの枚数だけで決まり、
    モデルの予測に依存しないので、モデル間の比較は常に同じ群の集合で行われる。
    """
    positive, negative = target == 1, target == 0
    n_pos, n_neg = int(positive.sum()), int(negative.sum())
    tpr = float((score[positive] >= 0.5).mean()) if n_pos >= min_cell else np.nan
    tnr = float((score[negative] < 0.5).mean()) if n_neg >= min_cell else np.nan
    both = n_pos >= min_cell and n_neg >= min_cell
    return {
        "n": len(target),
        "n_positive": n_pos,
        "auroc": float(roc_auc_score(target, score)) if both else np.nan,
        "bacc": (tpr + tnr) / 2,
        "tpr": tpr,
        "tnr": tnr,
        "fpr": 1 - tnr,
    }


def evaluate(target: np.ndarray, score: np.ndarray, groups: dict[str, np.ndarray]) -> tuple[dict, list[dict]]:
    """全体・群別・公平性指標を1つのモデル1seedについて返す。

    全体指標は`groups`の全軸で群が定義される行（欠損属性を除いた行）で計算する。
    worst・gapは評価可能な群が2つ以上ある場合だけ定義する。Eopp0はTNR gap、Eopp1はTPR gap、
    EoddsはTPR gapとFPR gapの平均。

    Returns:
        tuple: 1行分の要約dict（`overall/<指標>`・`<軸>/<指標>`）、群別の行のlist。
    """
    summary = {f"overall/{k}": v for k, v in cell_metrics(target, score).items() if k != "fpr"}
    rows = []
    for axis, ids in groups.items():
        labels = AXES[axis][1]
        per_group = []
        for index, label in enumerate(labels):
            mask = ids == index
            per_group.append({"axis": axis, "group": label, **cell_metrics(target[mask], score[mask])})
        rows.extend(per_group)
        table = pd.DataFrame(per_group)
        for metric in ("auroc", "bacc", "tpr", "tnr", "fpr"):
            values = table[metric].dropna()
            defined = len(values) >= 2
            summary[f"{axis}/worst_{metric}"] = values.min() if defined else np.nan
            summary[f"{axis}/{metric}_gap"] = np.ptp(values) if defined else np.nan
        summary[f"{axis}/Eopp0"] = summary[f"{axis}/tnr_gap"]
        summary[f"{axis}/Eopp1"] = summary[f"{axis}/tpr_gap"]
        summary[f"{axis}/Eodds"] = (summary[f"{axis}/tpr_gap"] + summary[f"{axis}/fpr_gap"]) / 2
        for metric in ("worst_tpr", "worst_tnr", "worst_fpr", "tnr_gap", "fpr_gap"):
            summary.pop(f"{axis}/{metric}")
    return summary, rows


def evaluate_all(scores: dict, frame: pd.DataFrame, mask: np.ndarray) -> tuple[pd.DataFrame, pd.DataFrame]:
    """各モデル・seedを独立に、`mask`で選んだ行の上で評価する。予測をseed間で混ぜない。

    Returns:
        tuple: seed別の要約表、群別の表。
    """
    target = frame["target"].to_numpy()[mask]
    groups = {axis: ids[mask] for axis, ids in group_ids(frame).items()}
    summaries, rows = [], []
    for (model, seed), score in scores.items():
        summary, group_rows = evaluate(target, score[mask], groups)
        summaries.append({"model": model, "seed": seed, **summary})
        rows.extend({"model": model, "seed": seed, **row} for row in group_rows)
    return pd.DataFrame(summaries), pd.DataFrame(rows)


def paired_bootstrap(
    scores: dict, frame: pd.DataFrame, mask: np.ndarray, clusters: np.ndarray, n_boot: int = 1000, seed: int = 0
) -> pd.DataFrame:
    """ResNetに対する差（比較モデル−ResNet、3 seed平均）のtest標本雑音を、cluster単位で見積もる。

    各反復で同じ再標本を全モデル・全seedに当て、seedごとの差を平均する。3 seedの標準偏差は
    学習のばらつきで、test標本の雑音を含まないので、両方を並べて読む。

    Args:
        scores: (モデル, seed)から陽性確率への対応。
        frame: test split。
        mask: 評価に使う行。
        clusters: 再標本化の単位（同じ病変の画像を同じclusterにする）。
        n_boot: 反復回数。
        seed: 乱数seed。
    Returns:
        DataFrame: 比較モデル・指標ごとの差の点推定、bootstrap SE、95%区間。
    """
    target = frame["target"].to_numpy()[mask]
    groups = {axis: ids[mask] for axis, ids in group_ids(frame).items()}
    # 以降の行番号はmask後の並びなので、予測も同じ行に絞ってから渡す。
    scores = {key: score[mask] for key, score in scores.items()}
    _, inverse = np.unique(clusters[mask], return_inverse=True)
    members = [np.flatnonzero(inverse == index) for index in range(inverse.max() + 1)]
    rng = np.random.default_rng(seed)
    point = _seed_mean_deltas(scores, target, groups, np.arange(len(target)))
    draws = []
    for _ in range(n_boot):
        index = np.concatenate([members[i] for i in rng.integers(0, len(members), len(members))])
        draws.append(_seed_mean_deltas(scores, target, groups, index))
    draws = pd.DataFrame(draws)
    rows = []
    for column, value in point.items():
        model, metric = column.split("|", 1)
        sample = draws[column].dropna()
        rows.append(
            {
                "model": model,
                "metric": metric,
                "delta": value,
                "boot_se": sample.std(),
                "ci_low": sample.quantile(0.025),
                "ci_high": sample.quantile(0.975),
                "valid_draws": len(sample),
            }
        )
    return pd.DataFrame(rows)


def _seed_mean_deltas(scores: dict, target: np.ndarray, groups: dict, index: np.ndarray) -> dict[str, float]:
    """`index`の行で各モデル・seedを評価し、同seedのResNetとの差をseed平均した値を返す。"""
    sub_groups = {axis: ids[index] for axis, ids in groups.items()}
    evaluated = {}
    for (model, seed), score in scores.items():
        summary, group_rows = evaluate(target[index], score[index], sub_groups)
        for row in group_rows:
            summary[f"{row['axis']}/{row['group']}/auroc"] = row["auroc"]
        evaluated[model, seed] = pd.Series(summary)
    deltas = {}
    for model in MODELS:
        if model == BASELINE:
            continue
        per_seed = pd.DataFrame([evaluated[model, s] - evaluated[BASELINE, s] for s in SEEDS])
        for metric, value in per_seed.mean(skipna=False).items():
            deltas[f"{model}|{metric}"] = value
    return deltas


COLORS = {"resnet": "tab:gray", "spatial_lora": "tab:blue", "hyperadapt": "tab:orange"}


def plot_seed_points(summary: pd.DataFrame, metrics: dict[str, str], ncols: int = 4) -> plt.Figure:
    """指標ごとに、モデル別のseed値（点）と3 seed平均（横棒）を描く。

    Args:
        summary: `evaluate_all`の要約表。
        metrics: 列名から表示名への対応。
        ncols: 1行に並べる指標数。
    Returns:
        Figure: モデルの並びは`MODELS`の固定順で、ランキングにしない。
    """
    nrows = -(-len(metrics) // ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.2 * ncols, 2.8 * nrows), layout="constrained", squeeze=False)
    for ax, (column, title) in zip(axes.flat, metrics.items(), strict=False):
        for position, model in enumerate(MODELS):
            values = summary.loc[summary["model"] == model, column].to_numpy(dtype=float)
            ax.scatter(np.full(len(values), position), values, color=COLORS[model], s=18, alpha=0.8)
            ax.hlines(np.nanmean(values), position - 0.25, position + 0.25, color=COLORS[model], lw=2)
        ax.set_xticks(range(len(MODELS)), MODELS.values(), rotation=20)
        ax.set_title(title, fontsize=10)
        ax.grid(axis="y", alpha=0.3)
    for ax in list(axes.flat)[len(metrics) :]:
        ax.set_visible(False)
    return fig


def plot_group_metric(groups: pd.DataFrame, metric: str) -> plt.Figure:
    """軸ごとに、群別の指標をモデル別の3 seed平均±標準偏差で描く。群名の下に陽性数/件数を添える。

    Args:
        groups: `evaluate_all`の群別表。
        metric: `auroc`・`tpr`・`tnr`など。最小セル未満の群は空欄になる。
    Returns:
        Figure: 軸ごとのsubplot。
    """
    fig, axes = plt.subplots(1, len(AXES), figsize=(3.4 * len(AXES), 3.2), layout="constrained", squeeze=False)
    width = 0.8 / len(MODELS)
    for ax, (axis, labels) in zip(axes[0], _axis_labels(), strict=True):
        part = groups[groups["axis"] == axis]
        for offset, model in enumerate(MODELS):
            stats = part[part["model"] == model].groupby("group", sort=False)[metric].agg(["mean", "std"])
            stats = stats.reindex(labels)
            x = np.arange(len(labels)) + (offset - (len(MODELS) - 1) / 2) * width
            ax.errorbar(
                x, stats["mean"], yerr=stats["std"], fmt="o", color=COLORS[model], ms=4, capsize=2, label=MODELS[model]
            )
        counts = part[part["model"] == BASELINE].groupby("group", sort=False)[["n_positive", "n"]].first()
        counts = counts.reindex(labels)
        ticks = [f"{label}\n{int(row.n_positive)}/{int(row.n)}" for label, row in counts.iterrows()]
        ax.set_xticks(range(len(labels)), ticks, fontsize=8)
        ax.set_title(axis, fontsize=10)
        ax.grid(axis="y", alpha=0.3)
    axes[0, 0].legend(fontsize=8)
    fig.suptitle(f"Per-group {metric.upper()} (mean ± SD over 3 seeds; positives/n under each group)", fontsize=11)
    return fig


def plot_paired_deltas(boot: pd.DataFrame, seed_deltas: pd.DataFrame, metrics: dict[str, str]) -> plt.Figure:
    """ResNetとの差（比較モデル−ResNet）を、bootstrap 95%区間と同seed差の点で描く。

    Args:
        boot: `paired_bootstrap`の表。
        seed_deltas: seed別の差（列にmodel・seedと指標）。
        metrics: 指標名から表示名への対応。
    Returns:
        Figure: 横棒がtest標本の95%区間、小点がseedごとの差。0の縦線が差なし。
    """
    fig, ax = plt.subplots(figsize=(7, 0.42 * len(metrics) * 2 + 1), layout="constrained")
    compared = [model for model in MODELS if model != BASELINE]
    for row, metric in enumerate(metrics):
        for offset, model in enumerate(compared):
            y = row * (len(compared) + 1) + offset
            entry = boot[(boot["model"] == model) & (boot["metric"] == metric)].iloc[0]
            ax.errorbar(
                entry["delta"],
                y,
                xerr=[[entry["delta"] - entry["ci_low"]], [entry["ci_high"] - entry["delta"]]],
                fmt="o",
                color=COLORS[model],
                capsize=3,
                label=MODELS[model] if row == 0 else None,
            )
            values = seed_deltas.loc[seed_deltas["model"] == model, metric]
            ax.scatter(values, np.full(len(values), y + 0.25), color=COLORS[model], s=8, alpha=0.6)
    ticks = [row * (len(compared) + 1) + (len(compared) - 1) / 2 for row in range(len(metrics))]
    ax.set_yticks(ticks, metrics.values(), fontsize=9)
    ax.invert_yaxis()
    ax.axvline(0, color="black", lw=0.8)
    ax.set_xlabel("Model − ResNet-50 (negative is better for gaps and Eopp/Eodds)")
    ax.legend(fontsize=8, loc="lower right")
    ax.grid(axis="x", alpha=0.3)
    return fig


def seed_deltas(summary: pd.DataFrame) -> pd.DataFrame:
    """同seedのResNetとの差を、比較モデル・seedごとに返す。"""
    baseline = summary[summary["model"] == BASELINE].set_index("seed").drop(columns="model")
    rows = []
    for model in MODELS:
        if model == BASELINE:
            continue
        compared = summary[summary["model"] == model].set_index("seed").drop(columns="model")
        rows.append((compared - baseline.loc[compared.index]).assign(model=model).reset_index())
    return pd.concat(rows, ignore_index=True)


def _axis_labels() -> list[tuple[str, tuple[str, ...]]]:
    """評価軸と群の表示名の組を、`AXES`の並びで返す。"""
    return [(axis, labels) for axis, (_, labels) in AXES.items()]
