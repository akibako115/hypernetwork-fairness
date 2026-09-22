"""fit を始める前に、この設定で何が回るのかを 1 枚の表にする。"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, ListConfig, OmegaConf

_MISSING = object()

# 表に出す項目。`(見出し, ((ラベル, config path), ...))` の並び。ここに無い設定は
# run directory の `config.yaml` が持つので、ここは「起動前に人が判断する値」だけにする。
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
            ("train_sampling", "data.train_sampling"),
            ("num_groups", "data.num_groups"),
        ),
    ),
    (
        "モデル",
        (
            ("net", "model.net._target_"),
            ("modulation_stages", "model.net.modulation_stages"),
            ("backbone_checkpoint", "model.backbone_checkpoint_path"),
            ("freeze_backbone", "model.freeze_backbone"),
            ("optimizer", "model.optimizer._target_"),
            ("lr", "model.optimizer.lr"),
        ),
    ),
    (
        "学習",
        (
            ("max_epochs", "trainer.max_epochs"),
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


def format_run_plan(config: DictConfig, class_weight_path: str) -> str:
    """解決済み設定から、起動前に確認する条件の表を作る。

    `--cfg job` は合成直後の設定を出すので、実行時に決まる class weight は ``null`` のままになる。
    この表は fit と同じ解決を通したあとの設定を読むため、実際に使う重みが出る。CLI override を
    最後に並べるのは、「preset の既定」と「今回わざわざ振った値」を分けて読むためである。

    Args:
        config: path 正規化・検証・class weight 解決まで済ませた設定
        class_weight_path: 解決後の class weight が入っている config path

    Returns:
        str: 表示用の複数行文字列。末尾に改行は付けない
    """
    sections = [(title, [(label, _value(config, path)) for label, path in rows if _select(config, path) is not _MISSING]) for title, rows in _SECTIONS]
    sections.insert(3, ("クラス重み", [("weighting", _value(config, "weighting")), ("class_weight", _value(config, class_weight_path))]))
    sections.insert(4, ("目的関数", _objective_rows(config)))
    sections.append(("callbacks", [("callbacks", ", ".join(config.get("callbacks") or []) or "(なし)")]))
    sections.append(("明示指定", _override_rows()))

    width = max(len(label) for _, rows in sections for label, _ in rows)
    lines = ["=" * 78, "run plan — fit は実行していない"]
    for title, rows in sections:
        if not rows:
            # 該当する設定を持たない run では節ごと出さない（空の見出しは読む邪魔にしかならない）。
            continue
        lines.append(f"[{title}]")
        # ラベルが空の行は前の項目の続き（override の 2 件目以降）なので、区切りを置かない。
        lines.extend(f"  {label.ljust(width)} {':' if label else ' '} {value}" for label, value in rows)
    lines.append("=" * 78)
    return "\n".join(lines)


def _objective_rows(config: DictConfig) -> list[tuple[str, str]]:
    """目的関数の種別と、その数値設定を並べる。

    strategy ごとに持つ key が違う（Group DRO の `step_size` など）ので、設定を先に決め打ちせず
    `model.loss_fn` の scalar をそのまま出す。class weight は別の節が持つので除く。
    """
    rows = [("training_strategy", _value(config, "training_strategy.name")), ("loss_fn", _value(config, "model.loss_fn._target_"))]
    loss_fn = _select(config, "model.loss_fn")
    if isinstance(loss_fn, Mapping):
        rows.extend((f"loss_fn.{key}", _format(value)) for key, value in loss_fn.items() if key not in ("_target_", "class_weight") and isinstance(value, (int, float, str, bool)))
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
    return _format(value)


def _format(value: Any) -> str:
    """表に出す 1 つの値を、そのまま読める文字列にする。"""
    if value is _MISSING or value is None:
        return "(なし)"
    if isinstance(value, (list, ListConfig)):
        return "[" + ", ".join(str(item) for item in value) + "]"
    return str(value)
