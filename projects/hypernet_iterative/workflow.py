"""warmup、cohort 再生成、独立 stage process を反復する workflow。"""

from __future__ import annotations

import hashlib
import json
import re
import secrets
import subprocess
import sys
from collections import Counter
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
import torch
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf

from .cohorts.build import extract_embeddings, load_split_frames, save_artifact
from .run_logging import log_epoch_metrics, parent_wandb, read_epoch_metrics, text_log

_REPOSITORY_ROOT = Path(__file__).parents[2]

# class weight の有無。`inverse` の cohort stage は必ず group ごとの重みを使う。
_WEIGHTING_MODES = ("none", "inverse")


@dataclass(frozen=True)
class Stage:
    name: str
    kind: str


def _run_id_part(value: str) -> str:
    """experiment 名を run-id に使える slug へ落とす。

    Args:
        value: `experiment_name` などの表示用文字列

    Returns:
        str: 英数と `-` だけの slug。空になる場合は `"experiment"`
    """
    normalized = re.sub(r"[^a-z0-9]+", "-", value.lower()).strip("-")
    return normalized or "experiment"


def reserve_parent_run(config: DictConfig) -> Path:
    """全 stage と cohort artifact を所有する親 run directory を一意に予約する。

    `mkdir` の排他性で一意性を取るので、同時起動しても同じ directory を掴まない。

    Args:
        config: `paths.project_dir` と `seed` を持つ解決済み設定。予約時にそのまま保存する

    Returns:
        Path: `stages/` `artifacts/` `logs/` と `config.yaml` `run.json` を作った run directory

    Raises:
        RuntimeError: 100 回試しても一意な directory を取れなかった場合。
    """
    project_dir = Path(str(config.paths.project_dir))
    # directory 名だけで条件を読めるよう、hypernet_e2e と同じ `<時刻>-<experiment>-<seed>-<乱数>` に
    # 揃える。experiment 名に畳めない override は config.yaml が正本であり、名前へは入れない。
    experiment = _run_id_part(str(config.get("experiment_name", "iterative")))
    for _ in range(100):
        run_id = f"{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}-{experiment}-s{config.seed}-{secrets.token_hex(2)}"
        run_dir = project_dir / "runs" / run_id
        try:
            run_dir.mkdir(parents=True)
        except FileExistsError:
            continue
        (run_dir / "stages").mkdir()
        (run_dir / "artifacts").mkdir()
        (run_dir / "logs").mkdir()
        OmegaConf.save(config, run_dir / "config.yaml", resolve=True)
        _write_json(
            run_dir / "run.json",
            {
                "schema_version": 1,
                "run_id": run_id,
                "kind": "iterative_fit",
                "status": "running",
                "started_at": _timestamp(),
                "finished_at": None,
                "git_commit": _git_commit(),
                "seed": config.get("seed"),
                "stages": {},
                "selected_checkpoint": None,
                "wandb": None,
            },
        )
        return run_dir
    raise RuntimeError("一意な parent run directory を予約できない")


def _validate_plan(config: DictConfig) -> None:
    """run 全体の計画を、GPU 時間を使う前に検証する。

    `plan()` は `dry_run` からしか呼ばれないので、ここを `plan()` の中に置くと実 run では
    発火しない。`iteration.clusters` の打ち間違いは warmup fit を消費し切ったあと
    `cohorts.build.save_artifact` で初めて落ちる。`run_iterative()` も先頭でこれを呼ぶ。
    `weighting` をここで見るのは、dry-run でも打ち間違いが出るようにするためである。
    """
    iteration = config.iteration
    if min(int(iteration.warmup_epochs), int(iteration.stage_epochs), int(iteration.stages), int(iteration.clusters), int(iteration.n_init)) < 1:
        raise ValueError("iteration counts must be positive")
    weighting = str(config.get("weighting", "none"))
    if weighting not in _WEIGHTING_MODES:
        raise ValueError(f"unsupported weighting: {weighting}（{', '.join(_WEIGHTING_MODES)} のいずれか）")


def plan(config: DictConfig) -> list[Stage]:
    """設定から warmup と cohort stage の決定的な実行計画を作る。

    Args:
        config: `iteration.*` を持つ設定

    Returns:
        list[Stage]: warmup の fit に続き、cohort 生成と fit を `iteration.stages` 回並べたもの

    Raises:
        ValueError: `iteration.*` のいずれかが 1 未満の場合。
    """
    _validate_plan(config)
    iteration = config.iteration
    result = [Stage("warmup", "fit")]
    for number in range(1, int(iteration.stages) + 1):
        result.extend((Stage(f"cohort{number:02d}", "cohort"), Stage(f"stage{number:02d}", "fit")))
    return result


def run_stage(config: DictConfig, stage_dir: Path) -> dict[str, Any]:
    """解決 config を保存して子 process へ渡し、stage result を読む。

    stage ごとに process を分けることで、optimizer や GroupDRO の内部状態、GPU メモリが
    stage をまたがない。

    Args:
        config: この stage の解決済み設定。`stage_dir/config.yaml` に保存する
        stage_dir: この stage の出力先。まだ存在しないこと

    Returns:
        dict[str, Any]: 子 process の result.json。`metrics` と `checkpoints` を持つ

    Raises:
        subprocess.CalledProcessError: 子 process が非 0 で終了した場合。
    """
    stage_dir.mkdir(parents=True)
    (stage_dir / "checkpoints").mkdir()
    (stage_dir / "metrics").mkdir()
    config_path, result_path = stage_dir / "config.yaml", stage_dir / "result.json"
    OmegaConf.save(config, config_path, resolve=True)
    subprocess.run([sys.executable, "-m", "projects.hypernet_iterative.stage", "--config", str(config_path), "--result", str(result_path)], check=True)
    return json.loads(result_path.read_text())


def build_cohort(
    config: DictConfig,
    *,
    checkpoint_path: Path,
    reference_id: str,
    output_dir: Path,
) -> Path:
    """参照 checkpoint の metadata encoder で cohort sidecar を生成する。

    cohort の入力は画像 backbone の feature ではなく metadata embedding なので、親 process
    は checkpoint を ``net`` にだけ読み込み、CSV を画像 loader 経由にせず行順のまま処理する。
    これにより cohort artifact は data augmentation や sampler に依存しない。

    Args:
        config: model と data を構築できる解決済み設定。`iteration.clusters` / `n_init` /
            `seed` が KMeans の設定になる
        checkpoint_path: metadata encoder を取り出す参照 checkpoint
        reference_id: この cohort を生成した stage 名。sidecar に記録する
        output_dir: cohort artifact の出力先。既存 directory は上書きしない

    Returns:
        Path: 後続 stage の DataModule へ渡す `assignments.parquet` の path

    Raises:
        TypeError: model が warm-start に対応しない、または `net.metadata_encoder` を
            持たない場合。
    """
    model = instantiate(config.model)
    if not hasattr(model, "load_warm_start_checkpoint"):
        raise TypeError("cohort reference model must support load_warm_start_checkpoint")
    model.load_warm_start_checkpoint(str(checkpoint_path))
    encoder = getattr(getattr(model, "net", None), "metadata_encoder", None)
    if not isinstance(encoder, torch.nn.Module):
        raise TypeError("cohort reference model must expose net.metadata_encoder")

    datamodule = instantiate(config.data)
    raw, prepared = load_split_frames(datamodule, config.model.get("attribute_names"))
    embeddings = extract_embeddings(
        encoder,
        raw,
        prepared,
        attribute_names=config.model.get("attribute_names"),
        batch_size=int(config.data.batch_size),
        device=torch.device("cpu"),
    )
    return save_artifact(
        output_dir,
        embeddings,
        clusters=int(config.iteration.clusters),
        n_init=int(config.iteration.n_init),
        random_state=int(config.seed),
        reference_checkpoint=checkpoint_path,
        reference_id=reference_id,
    )


def cohort_stage_config(
    warmup_config: DictConfig,
    *,
    assignment_path: Path,
    checkpoint_path: Path,
    reference_id: str,
    class_weight: list[list[float]] | None = None,
) -> DictConfig:
    """warmup config から、固定 cohort を使う次 stage の config を作る。

    stage 間で引き継ぐのは ``net`` の tensor だけである。optimizer、scheduler、GroupDRO の
    adversarial weight は子 process ごとに新しく構築される。

    Args:
        warmup_config: warmup の解決済み設定
        assignment_path: この stage が使う固定 cohort の `assignments.parquet`
        checkpoint_path: warm-start 元の checkpoint。strategy が warm-start に対応しない
            場合は設定しない
        reference_id: この cohort を生成した stage 名
        class_weight: この cohort 向けに解いた `[clusters, num_classes]` の class weight。
            `weighting=inverse` のときだけ渡す。warmup の全体重みは引き継がない

    Returns:
        DictConfig: cohort DataModule、group 目的関数、hidden cohort callback、
            checkpoint 選択を差し替えた stage 設定

    Raises:
        ValueError: `iteration.cohort_training_strategy` または
            `iteration.cohort_checkpoint_selection` が未対応の値の場合。
    """
    result = OmegaConf.create(OmegaConf.to_container(warmup_config, resolve=True))
    clusters = int(result.iteration.clusters)
    group_key = "group_id"
    OmegaConf.update(result, "trainer.max_epochs", int(result.iteration.stage_epochs), merge=False)
    OmegaConf.update(result, "data._target_", "projects.hypernet_iterative.data.cohort_datamodule.CohortImageDataModule", merge=False)
    OmegaConf.update(result, "data.group_assignment_path", str(assignment_path), merge=False)
    OmegaConf.update(result, "data.num_groups", clusters, merge=False)
    OmegaConf.update(result, "data.group_key", group_key, merge=False)
    OmegaConf.update(
        result,
        "cohort",
        {"name": "metadata_kmeans", "reference_id": reference_id, "num_groups": clusters, "group_key": group_key},
        merge=False,
    )
    strategy_name = str(result.iteration.get("cohort_training_strategy", "group_dro"))
    strategy_target = {
        "group_dro": "projects.hypernet_iterative.loss.GroupDROTaskLoss",
        "uniform_group": "projects.hypernet_iterative.loss.UniformGroupTaskLoss",
        "uniform_group_iterative": "projects.hypernet_iterative.loss.UniformGroupTaskLoss",
    }.get(strategy_name)
    if strategy_target is None:
        raise ValueError(f"unsupported iteration.cohort_training_strategy: {strategy_name}")
    # `uniform_group` と `uniform_group_iterative` は同じ目的関数で、warm-start の可否だけが違う。
    # 反復条件で前者を選ぶと各 stage が ImageNet 初期化からやり直しになり、GroupDRO 条件と
    # 比較できる対照でなくなるため、名前を分けて warm-start の許可を明示する。
    supports_warm_start = strategy_name in {"group_dro", "uniform_group_iterative"}
    # 群内クラス均衡は目的関数ではなく class weight で表す。`weighting=inverse` が cohort
    # ごとに解いた `[clusters, num_classes]` を渡すと group loss が群内クラス平均になる。
    loss_config: dict[str, Any] = {
        "_target_": strategy_target,
        "num_groups": clusters,
        "class_weight": class_weight,
        "group_key": group_key,
    }
    if strategy_name == "group_dro":
        loss_config["step_size"] = float(result.iteration.group_dro_step_size)
    OmegaConf.update(
        result,
        "training_strategy",
        {"name": strategy_name, "uses_cohort_group_id": True, "supports_warm_start": supports_warm_start},
        merge=False,
    )
    OmegaConf.update(
        result,
        "model.loss_fn",
        loss_config,
        merge=False,
    )
    OmegaConf.update(result, "model.warm_start_checkpoint_path", str(checkpoint_path) if supports_warm_start else None, merge=False)
    for name, target, kwargs in (
        ("cohort_single_process", "projects.hypernet_iterative.callbacks.hidden_cohort_logger.CohortSingleProcessCallback", {}),
        ("cohort_validity", "projects.hypernet_iterative.callbacks.hidden_cohort_logger.CohortValidityCallback", {"num_groups": clusters, "group_key": group_key}),
        ("hidden_cohort_logger", "projects.hypernet_iterative.callbacks.hidden_cohort_logger.HiddenCohortMetricsCallback", {"num_groups": clusters, "group_key": group_key}),
    ):
        OmegaConf.update(result, f"callbacks.{name}", {"_target_": target, **kwargs}, merge=False)
    selection_name = str(result.iteration.get("cohort_checkpoint_selection", "global_auroc_bacc"))
    if selection_name == "hidden_min_auroc":
        OmegaConf.update(result, "checkpoint_selection", {"name": selection_name, "requires_hidden_cohort": True}, merge=False)
        # warmup は global_auroc_bacc なので、その config を写した時点で bacc_checkpoint が
        # 入っている。ここは差分を当てるだけなので、消さないと補助 checkpoint が両方残り、
        # 宣言側の checkpoint_selection/hidden_min_auroc.yaml と食い違う。
        result.callbacks.pop("bacc_checkpoint", None)
        OmegaConf.update(
            result,
            "callbacks.hidden_min_auroc_checkpoint",
            {
                "_target_": "lightning.pytorch.callbacks.ModelCheckpoint",
                "dirpath": None,
                "filename": "best_hidden_auroc_{epoch:03d}",
                "monitor": "val/hidden_min_auroc",
                "mode": "max",
                "save_top_k": 1,
                "save_last": False,
                "auto_insert_metric_name": False,
            },
            merge=False,
        )
    elif selection_name != "global_auroc_bacc":
        raise ValueError(f"unsupported iteration.cohort_checkpoint_selection: {selection_name}")
    return result


def write_data_manifest(run_dir: Path, config: DictConfig) -> None:
    """反復全体で共有する全 split の入力同一性を記録する。

    Args:
        run_dir: 書き込み先の親 run directory
        config: `data.data_dir` と `data.cv_splits_dir` を持つ設定

    Returns:
        None

    Raises:
        ValueError: いずれかの split CSV に `image` / `target` 列が無い場合。
    """
    data = config.data
    splits_dir = Path(str(data.cv_splits_dir))
    _write_json(
        run_dir / "data_manifest.json",
        {
            "schema_version": 1,
            "image_root": str(Path(str(data.data_dir)).resolve()),
            "splits": {split: _split_manifest(splits_dir / f"{split}.csv") for split in ("train", "val", "test")},
        },
    )


def write_preflight(run_dir: Path) -> None:
    """親 run の前に project-local golden preflight を実行・保存する。

    Args:
        run_dir: `preflight.json` の書き込み先

    Returns:
        None

    Raises:
        RuntimeError: preflight が非 0 で終了した場合。結果は先に保存する
    """
    golden_files = sorted((_REPOSITORY_ROOT / "tests" / "golden").glob("*.v*.json"))
    command = [sys.executable, "-m", "pytest", "projects/hypernet_iterative/tests", "-m", "preflight", "-q"]
    completed = subprocess.run(command, cwd=_REPOSITORY_ROOT, capture_output=True, text=True, check=False)
    _write_json(
        run_dir / "preflight.json",
        {
            "schema_version": 1,
            "git_commit": _git_commit(),
            "command": command,
            "goldens": [{"name": path.name, "sha256": _sha256_file(path)} for path in golden_files],
            "exit_code": completed.returncode,
            "stdout": completed.stdout,
            "stderr": completed.stderr,
        },
    )
    if completed.returncode:
        raise RuntimeError("golden preflight に失敗した")


def _resolve_inverse_class_weights(config: DictConfig) -> None:
    """`weighting=inverse` のとき warmup が使う全体 class weight を設定する。

    parent run の予約より前に解決する。ここで解くのは warmup の重みだけである。warmup に
    cohort は無いので全 train split の逆頻度を使い、cohort stage の重みは
    `resolve_group_class_weights` が cohort ごとに解き直す。cohort stage で全 group 共通の
    重みを使う条件は持たない（group loss に陽性率依存が残るため）。

    Args:
        config: 解決対象の設定。`model.loss_fn.class_weight` を書き換える

    Returns:
        None
    """
    if str(config.get("weighting", "none")) != "inverse":
        return
    frame = pd.read_csv(Path(str(config.data.cv_splits_dir)) / "train.csv")
    labels = [int(value) for value in frame["target"]]
    num_classes = int(config.data.num_classes)
    weights = _inverse_frequency_weights(labels, num_classes)
    OmegaConf.update(config, "model.loss_fn.class_weight", weights, merge=False)


def resolve_group_class_weights(config: DictConfig, assignment_path: Path) -> list[list[float]]:
    """固定 cohort の train 行から `w[g,c] = 1 / (C * f_{g,c})` を解く。

    この重みを group 目的関数へ渡すと group loss の期待値が群内クラス平均になり、group の
    陽性率に依存しなくなる。群ごとに重みを解くので cohort を作り直すたびに解き直す。

    正規化しない生の `1 / (C * f_{g,c})` を返す。group ごとにさらに正規化すると group loss
    の尺度が group ごとに変わり、adversarial weight が難しさではなく尺度を追う。全体に同じ
    定数を掛ける正規化は AdamW では効果が無いので入れない。

    Args:
        config: `data.cv_splits_dir`・`data.num_classes`・`iteration.clusters` を持つ設定
        assignment_path: この stage が使う cohort の `assignments.parquet`

    Returns:
        list[list[float]]: `[clusters, num_classes]` の class weight

    Raises:
        ValueError: train 行が cohort sidecar と1対1で対応しない場合、group ID や target が
            範囲外の場合、または空の (group, class) セルがある場合。空セルは重みが発散する
            ので、黙って落とさずここで止める。
    """
    clusters = int(config.iteration.clusters)
    num_classes = int(config.data.num_classes)
    assignments = pd.read_parquet(assignment_path)
    assignments = assignments[assignments["split"] == "train"]
    targets = pd.read_csv(Path(str(config.data.cv_splits_dir)) / "train.csv", usecols=["image", "target"])
    merged = targets.merge(assignments[["image", "group_id"]], on="image", how="inner")
    if len(merged) != len(targets):
        raise ValueError(f"train split の {len(targets)} 行に対し cohort sidecar と対応したのは {len(merged)} 行")

    counts = [[0] * num_classes for _ in range(clusters)]
    for (group_id, target), size in merged.groupby(["group_id", "target"]).size().items():
        if not 0 <= int(group_id) < clusters:
            raise ValueError(f"group_id は [0, {clusters - 1}] である必要があるが、{int(group_id)} が現れた")
        if not 0 <= int(target) < num_classes:
            raise ValueError(f"target は [0, {num_classes - 1}] である必要があるが、{int(target)} が現れた")
        counts[int(group_id)][int(target)] = int(size)

    weights = []
    for group_id, row in enumerate(counts):
        if min(row) == 0:
            raise ValueError(f"cohort group {group_id} に空の (group, class) セルがある: {row}")
        total = sum(row)
        weights.append([round(total / (num_classes * cell), 6) for cell in row])
    return weights


def _inverse_frequency_weights(labels: list[int], num_classes: int) -> list[float]:
    """平均が 1 になる逆頻度 class weight を返す。"""
    if not labels:
        raise ValueError("train.csv の target は空にできない")
    expected = list(range(num_classes))
    observed = sorted(set(labels))
    if observed != expected:
        raise ValueError(f"train.csv の target は {expected} である必要があるが、{observed} が指定された")
    counts = Counter(labels)
    sample_count = len(labels)
    raw_weights = [sample_count / (num_classes * counts[class_index]) for class_index in expected]
    mean_weight = sum(raw_weights) / len(raw_weights)
    return [round(weight / mean_weight, 6) for weight in raw_weights]


def _group_class_weight_for_stage(config: DictConfig, assignment_path: Path) -> list[list[float]] | None:
    """`weighting=inverse` のときだけ、この cohort 向けの class weight を解く。

    warmup の全体重みは cohort stage へ引き継がない。group 目的関数に共通の重みを渡すと
    group loss が陽性率に依存したままになるため、cohort があるなら必ず group ごとに解く。
    """
    if str(config.get("weighting", "none")) != "inverse":
        return None
    return resolve_group_class_weights(config, assignment_path)


def _log_stage_epochs(wandb_run: Any, stage_index: int, result: Mapping[str, Any], offset: int) -> int:
    """子が残した epoch metric を親の W&B run へ集約し、次の offset を返す。

    offset は config の epoch 数ではなく実際に記録された epoch 数で進める。`limit_*` を付けた
    run でも step がずれない。
    """
    csv_path = result.get("metrics_csv")
    if csv_path is None:
        raise ValueError("stage result に metrics_csv が無い。子が epoch metric を残していない")
    return log_epoch_metrics(wandb_run, stage_index, read_epoch_metrics(Path(csv_path)), offset)


def run_iterative(config: DictConfig) -> Path:
    """warmup → cohort 再生成 → warm-start stage を指定回数だけ実行する。

    次 stage へ渡す checkpoint と cohort の参照は各 stage の `last` で固定する。stage は
    GroupDRO の損失を下げており、global val AUROC の best を選ぶとその更新を stage 境界の
    たびに巻き戻すため、引き継ぎに best は使わない。`selected_checkpoint` に残るのは最後の
    stage の best val/auroc であって run 全体の best ではない。run をまたぐ選択は
    `stages.*.checkpoints` の score から分析側で決める。途中で失敗しても `run.json` には
    失敗として確定した状態が残る。

    Args:
        config: `iteration.*` を含む解決済み設定。`weighting=inverse` なら warmup の全体
            class weight をここで解き、cohort stage の重みは cohort を作り直すたびに group
            ごとに解き直す

    Returns:
        Path: 親 run directory

    Raises:
        ValueError: `iteration.*` のいずれかが 1 未満の場合。
        RuntimeError: golden preflight に失敗した場合。
    """
    _validate_plan(config)
    _resolve_inverse_class_weights(config)
    run_dir = reserve_parent_run(config)
    record_path = run_dir / "run.json"
    record = json.loads(record_path.read_text())
    try:
        with text_log(run_dir), parent_wandb(config, run_dir) as wandb_run:
            write_data_manifest(run_dir, config)
            write_preflight(run_dir)
            warmup = OmegaConf.create(OmegaConf.to_container(config, resolve=True))
            OmegaConf.update(warmup, "trainer.max_epochs", warmup.iteration.warmup_epochs, merge=False)
            result = run_stage(warmup, run_dir / "stages" / "warmup")
            record["stages"]["warmup"] = result
            checkpoint_path = Path(result["checkpoints"]["last"]["path"])
            selected_checkpoint = result["checkpoints"]["val/auroc"]
            epoch_offset = 0
            if wandb_run is not None:
                record["wandb"] = {"id": wandb_run.id, "url": wandb_run.url, "name": wandb_run.name}
                epoch_offset = _log_stage_epochs(wandb_run, 0, result, epoch_offset)
            for number in range(1, int(config.iteration.stages) + 1):
                cohort_name = f"cohort{number:02d}"
                artifact_dir = run_dir / "artifacts" / "cohorts" / cohort_name
                assignment_path = build_cohort(
                    warmup,
                    checkpoint_path=checkpoint_path,
                    reference_id="warmup" if number == 1 else f"stage{number - 1:02d}",
                    output_dir=artifact_dir,
                )
                # cohort は metric を持たないので W&B へは送らない。来歴は run.json と
                # cohort.json が持つ。
                record["stages"][cohort_name] = {
                    "artifact_dir": str(artifact_dir),
                    "assignment_path": str(assignment_path),
                    "reference_checkpoint": str(checkpoint_path),
                }

                stage_name = f"stage{number:02d}"
                stage_config = cohort_stage_config(
                    warmup,
                    assignment_path=assignment_path,
                    checkpoint_path=checkpoint_path,
                    reference_id=cohort_name,
                    class_weight=_group_class_weight_for_stage(warmup, assignment_path),
                )
                result = run_stage(stage_config, run_dir / "stages" / stage_name)
                record["stages"][stage_name] = result
                checkpoint_path = Path(result["checkpoints"]["last"]["path"])
                selected_checkpoint = result["checkpoints"]["val/auroc"]
                if wandb_run is not None:
                    epoch_offset = _log_stage_epochs(wandb_run, number, result, epoch_offset)
            record["selected_checkpoint"] = selected_checkpoint
            record["status"] = "succeeded"
    except BaseException as error:
        record.update(status="failed", error_type=type(error).__name__, error_message=str(error))
        raise
    finally:
        record["finished_at"] = _timestamp()
        _write_json(record_path, record)
    return run_dir


def _split_manifest(path: Path) -> dict[str, Any]:
    frame = pd.read_csv(path)
    if "image" not in frame or "target" not in frame:
        raise ValueError(f"split manifest には image と target 列が必要: {path}")
    return {
        "path": str(path.resolve()),
        "sha256": _sha256_file(path),
        "num_rows": len(frame),
        "image_set_sha256": _sha256_lines(frame["image"].astype(str).sort_values().tolist()),
        "target_distribution": {str(key): int(value) for key, value in frame["target"].value_counts(dropna=False).sort_index().items()},
    }


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256_lines(lines: list[str]) -> str:
    return hashlib.sha256("\n".join(lines).encode()).hexdigest()


def _git_commit() -> str | None:
    completed = subprocess.run(["git", "rev-parse", "HEAD"], cwd=_REPOSITORY_ROOT, capture_output=True, text=True, check=False)
    return completed.stdout.strip() if completed.returncode == 0 else None


def _timestamp() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
