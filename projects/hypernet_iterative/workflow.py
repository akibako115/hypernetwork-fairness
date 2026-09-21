"""warmup、cohort 再生成、独立 stage process を反復する workflow。"""

from __future__ import annotations

import hashlib
import json
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
from .run_logging import log_stage, parent_wandb, text_log

_REPOSITORY_ROOT = Path(__file__).parents[2]


@dataclass(frozen=True)
class Stage:
    name: str
    kind: str


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
    for _ in range(100):
        run_id = f"{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}-iterative-s{config.seed}-{secrets.token_hex(2)}"
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


def _validate_iteration(config: DictConfig) -> None:
    """反復計画の数値を、GPU 時間を使う前に検証する。

    `plan()` は `dry_run` からしか呼ばれないので、ここを `plan()` の中に置くと実 run では
    発火しない。`iteration.clusters` の打ち間違いは warmup fit を消費し切ったあと
    `cohorts.build.save_artifact` で初めて落ちる。`run_iterative()` も先頭でこれを呼ぶ。
    """
    iteration = config.iteration
    if min(int(iteration.warmup_epochs), int(iteration.stage_epochs), int(iteration.stages), int(iteration.clusters), int(iteration.n_init)) < 1:
        raise ValueError("iteration counts must be positive")


def plan(config: DictConfig) -> list[Stage]:
    """設定から warmup と cohort stage の決定的な実行計画を作る。

    Args:
        config: `iteration.*` を持つ設定

    Returns:
        list[Stage]: warmup の fit に続き、cohort 生成と fit を `iteration.stages` 回並べたもの

    Raises:
        ValueError: `iteration.*` のいずれかが 1 未満の場合。
    """
    _validate_iteration(config)
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
) -> DictConfig:
    """warmup config から、固定 cohort を使う次 stage の config を作る。

    stage 間で引き継ぐのは ``net`` の tensor だけである。optimizer、scheduler、GroupDRO の
    adversarial weight は子 process ごとに新しく構築される。

    Args:
        warmup_config: warmup の解決済み設定。class weight もここから引き継ぐ
        assignment_path: この stage が使う固定 cohort の `assignments.parquet`
        checkpoint_path: warm-start 元の checkpoint。strategy が warm-start に対応しない
            場合は設定しない
        reference_id: この cohort を生成した stage 名

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
        "group_dro_balanced": "projects.hypernet_iterative.loss.ClassBalancedGroupDROTaskLoss",
        "uniform_group": "projects.hypernet_iterative.loss.UniformGroupTaskLoss",
        "uniform_group_iterative": "projects.hypernet_iterative.loss.UniformGroupTaskLoss",
    }.get(strategy_name)
    if strategy_target is None:
        raise ValueError(f"unsupported iteration.cohort_training_strategy: {strategy_name}")
    # `uniform_group` と `uniform_group_iterative` は同じ目的関数で、warm-start の可否だけが違う。
    # 反復条件で前者を選ぶと各 stage が ImageNet 初期化からやり直しになり、GroupDRO 条件と
    # 比較できる対照でなくなるため、名前を分けて warm-start の許可を明示する。
    supports_warm_start = strategy_name in {"group_dro", "group_dro_balanced", "uniform_group_iterative"}
    loss_config: dict[str, Any] = {
        "_target_": strategy_target,
        "num_groups": clusters,
        "class_weight": result.model.loss_fn.get("class_weight"),
        "group_key": group_key,
    }
    if strategy_name == "group_dro":
        loss_config["step_size"] = 0.01
    elif strategy_name == "group_dro_balanced":
        loss_config.update(num_classes=int(result.data.num_classes), step_size=0.0001, loss_ema_momentum=0.01)
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
    """inverse weighting 時に train split から class weight を設定する。

    parent run の予約より前に解決する。`cohort_stage_config` は warmup config の
    `model.loss_fn.class_weight` をそのまま次 stage へ渡すので、解決先はここ1箇所に保つ。
    """
    if config.get("weighting", "none") != "inverse":
        return
    frame = pd.read_csv(Path(str(config.data.cv_splits_dir)) / "train.csv")
    labels = [int(value) for value in frame["target"]]
    num_classes = int(config.data.num_classes)
    weights = _inverse_frequency_weights(labels, num_classes)
    OmegaConf.update(config, "model.loss_fn.class_weight", weights, merge=False)


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


def run_iterative(config: DictConfig) -> Path:
    """warmup → cohort 再生成 → warm-start stage を指定回数だけ実行する。

    各 stage の参照 checkpoint は `val/auroc` の best で固定する。途中で失敗しても
    `run.json` には失敗として確定した状態が残る。

    Args:
        config: `iteration.*` を含む解決済み設定。`weighting=inverse` ならここで class
            weight を解決し、全 stage へ配る

    Returns:
        Path: 親 run directory

    Raises:
        ValueError: `iteration.*` のいずれかが 1 未満の場合。
        RuntimeError: golden preflight に失敗した場合。
    """
    _validate_iteration(config)
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
            checkpoint_path = Path(result["checkpoints"]["val/auroc"]["path"])
            selected_checkpoint = result["checkpoints"]["val/auroc"]
            if wandb_run is not None:
                record["wandb"] = {"id": wandb_run.id, "url": wandb_run.url, "name": wandb_run.name}
                log_stage(wandb_run, "warmup", result)
            for number in range(1, int(config.iteration.stages) + 1):
                cohort_name = f"cohort{number:02d}"
                artifact_dir = run_dir / "artifacts" / "cohorts" / cohort_name
                assignment_path = build_cohort(
                    warmup,
                    checkpoint_path=checkpoint_path,
                    reference_id="warmup" if number == 1 else f"stage{number - 1:02d}",
                    output_dir=artifact_dir,
                )
                cohort_result = {"artifact_dir": str(artifact_dir), "assignment_path": str(assignment_path), "reference_checkpoint": str(checkpoint_path)}
                record["stages"][cohort_name] = cohort_result
                if wandb_run is not None:
                    log_stage(wandb_run, cohort_name, cohort_result)

                stage_name = f"stage{number:02d}"
                stage_config = cohort_stage_config(
                    warmup,
                    assignment_path=assignment_path,
                    checkpoint_path=checkpoint_path,
                    reference_id=cohort_name,
                )
                result = run_stage(stage_config, run_dir / "stages" / stage_name)
                record["stages"][stage_name] = result
                checkpoint_path = Path(result["checkpoints"]["val/auroc"]["path"])
                selected_checkpoint = result["checkpoints"]["val/auroc"]
                if wandb_run is not None:
                    log_stage(wandb_run, stage_name, result)
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
