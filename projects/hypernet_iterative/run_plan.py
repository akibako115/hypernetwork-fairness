"""反復学習を始める前に、この設定で何が回るのかを 1 枚の表にする。"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, ListConfig, OmegaConf

_MISSING = object()

# 表に出す項目。`(見出し, ((ラベル, config path), ...))` の並び。ここに無い設定は親 run の
# `config.yaml` が持つので、ここは「起動前に人が判断する値」だけにする。
_SECTIONS: tuple[tuple[str, tuple[tuple[str, str], ...]], ...] = (
    (
        "帰属",
        (
            ("study", "study"),
            ("project", "project"),
            ("experiment", "experiment_name"),
            ("seed", "seed"),
        ),
    ),
    (
        "データ",
        (
            ("dataset", "dataset"),
            ("splits", "data.cv_splits_dir"),
            ("batch_size", "data.batch_size"),
        ),
    ),
    (
        "モデル",
        (
            ("net", "model.net._target_"),
            ("modulation_stages", "model.net.modulation_stages"),
            ("backbone_checkpoint", "model.backbone_checkpoint_path"),
            ("optimizer", "model.optimizer._target_"),
            ("lr", "model.optimizer.lr"),
        ),
    ),
    (
        "クラス重み",
        (
            ("weighting", "weighting"),
            ("class_weight (warmup)", "model.loss_fn.class_weight"),
        ),
    ),
    (
        "反復",
        (
            ("warmup_epochs", "iteration.warmup_epochs"),
            ("stages", "iteration.stages"),
            ("stage_epochs", "iteration.stage_epochs"),
            ("clusters", "iteration.clusters"),
            ("n_init", "iteration.n_init"),
            ("cohort_strategy", "iteration.cohort_training_strategy"),
            ("group_dro_step_size", "iteration.group_dro_step_size"),
            ("checkpoint_selection", "iteration.cohort_checkpoint_selection"),
        ),
    ),
    (
        "学習",
        (
            ("accelerator", "trainer.accelerator"),
            ("devices", "trainer.devices"),
            ("precision", "trainer.precision"),
            ("deterministic", "trainer.deterministic"),
        ),
    ),
    (
        "記録",
        (
            ("wandb project", "logger.wandb.project"),
            ("wandb group", "logger.wandb.group"),
            ("wandb job_type", "logger.wandb.job_type"),
            ("tags", "tags"),
        ),
    ),
)


def format_run_plan(config: DictConfig, stages: Sequence[Any]) -> str:
    """解決済み設定と stage 計画から、起動前に確認する条件の表を作る。

    `--cfg job` は合成直後の設定を出すので、実行時に決まる warmup の class weight は ``null`` の
    ままになる。この表は実行と同じ解決を通したあとの設定を読む。cohort stage の class weight は
    cohort を作り直すたびに群ごとに解き直すので、ここには出ない。

    Args:
        config: 検証と warmup class weight の解決まで済ませた設定
        stages: `plan()` が返した実行計画

    Returns:
        str: 表示用の複数行文字列。末尾に改行は付けない
    """
    sections = [(title, [(label, _value(config, path)) for label, path in rows if _select(config, path) is not _MISSING]) for title, rows in _SECTIONS]
    sections.insert(5, ("規模", _scale_rows(config, stages)))
    sections.append(("callbacks", [("callbacks", ", ".join(config.get("callbacks") or []) or "(なし)")]))
    sections.append(("明示指定", _override_rows()))

    width = max(len(label) for _, rows in sections for label, _ in rows)
    lines = ["=" * 78, "run plan — stage は実行していない"]
    for title, rows in sections:
        if not rows:
            # 該当する設定を持たない run では節ごと出さない（空の見出しは読む邪魔にしかならない）。
            continue
        lines.append(f"[{title}]")
        # ラベルが空の行は前の項目の続き（override や stage の 2 件目以降）なので、区切りを置かない。
        lines.extend(f"  {label.ljust(width)} {':' if label else ' '} {value}" for label, value in rows)
    lines.append("=" * 78)
    return "\n".join(lines)


def _scale_rows(config: DictConfig, stages: Sequence[Any]) -> list[tuple[str, str]]:
    """run 全体の大きさを、epoch 数と stage の並びで表す。

    epoch 総数は所要時間の当たりを付ける唯一の数字なので、`iteration.*` から引き算させない。
    """
    iteration = config.iteration
    total_epochs = int(iteration.warmup_epochs) + int(iteration.stages) * int(iteration.stage_epochs)
    order = [f"{stage.kind}: {stage.name}" for stage in stages]
    rows = [("total epochs", str(total_epochs))]
    rows.extend(("stage plan" if index == 0 else "", value) for index, value in enumerate(order))
    return rows


def _override_rows() -> list[tuple[str, str]]:
    """この起動で CLI から渡した override を 1 行ずつ返す。

    Hydra の外（test など）から呼ぶと取れないので、その場合は取れないことを出す。
    """
    try:
        overrides = list(HydraConfig.get().overrides.task)
    except ValueError:
        return [("overrides", "(Hydra 外からの呼び出しのため不明)")]
    if not overrides:
        return [("overrides", "(なし)")]
    return [("overrides" if index == 0 else "", value) for index, value in enumerate(overrides)]


def _select(config: DictConfig, path: str) -> Any:
    """config path の値を返す。無い key は ``_MISSING`` を返す。"""
    return OmegaConf.select(config, path, default=_MISSING)


def _value(config: DictConfig, path: str) -> str:
    """config path の値を、表に出す文字列にする。

    `_target_` は import path のままだと横に長く、読むのは末尾の class 名だけなので短縮する。
    """
    value = _select(config, path)
    if path.endswith("_target_") and isinstance(value, str):
        return value.rsplit(".", 1)[-1]
    if value is _MISSING or value is None:
        return "(なし)"
    if isinstance(value, (list, ListConfig)):
        return "[" + ", ".join(str(item) for item in value) + "]"
    return str(value)
