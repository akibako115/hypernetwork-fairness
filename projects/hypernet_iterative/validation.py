"""iterative stage の cohort・引き継ぎ・checkpoint 設定を検証する。"""

from collections.abc import Sequence
from pathlib import Path

from omegaconf import DictConfig


def _is_group_class_weight(class_weight: object) -> bool:
    """class weight が group ごとの `[num_groups, num_classes]` かどうかを返す。"""
    if not isinstance(class_weight, Sequence) or isinstance(class_weight, (str, bytes)) or not class_weight:
        return False
    first = class_weight[0]
    return isinstance(first, Sequence) and not isinstance(first, (str, bytes))


def validate_training_config(config: DictConfig) -> None:
    """学習を始める前に cohort、warm-start、checkpoint 選択の組み合わせを検証する。

    Args:
        config: 検証する stage の解決済み設定

    Returns:
        None

    Raises:
        ValueError: cohort に必要な設定が欠けている場合、class weight の形が cohort と
            合わない場合、warm-start が許可されていない場合、
            または hidden cohort 選択に必要な callback が無い場合。
        FileNotFoundError: warm-start checkpoint が存在しない場合。
    """
    strategy = config.get("training_strategy")
    cohort = config.get("cohort")
    uses_cohort = cohort is not None
    if uses_cohort:
        assignment_path = config.data.get("group_assignment_path")
        if not assignment_path:
            raise ValueError("cohort_definition を使う場合は data.group_assignment_path が必要")
        devices = config.trainer.get("devices", 1)
        if not isinstance(devices, int) or isinstance(devices, bool) or devices != 1 or config.trainer.get("num_nodes", 1) != 1:
            raise ValueError("固定 cohort を使う実験は trainer.devices=1 かつ trainer.num_nodes=1 が必要")

    if strategy is not None and strategy.get("uses_cohort_group_id", False):
        if not uses_cohort or not config.data.get("group_assignment_path"):
            raise ValueError(f"training_strategy={strategy.name} には cohort_definition と data.group_assignment_path が必要")

    # cohort stage で class weight を使うなら group ごとに使う。全 group 共通の重みでは
    # group loss がその group の陽性率に依存したままで、adversarial weight が難しい group
    # ではなく陽性が多い group へ寄る。重みは cohort の構成から解くので、行数が合わない
    # stage config は別 cohort 向けに解いた重みを使っている。
    class_weight = (config.model.get("loss_fn") or {}).get("class_weight")
    if class_weight is not None and uses_cohort:
        if not _is_group_class_weight(class_weight):
            raise ValueError("cohort stage の class weight は group ごとの [num_groups, num_classes] である必要がある")
        if len(class_weight) != int(cohort.num_groups):
            raise ValueError(f"group ごとの class weight は {int(cohort.num_groups)} 行である必要があるが、{len(class_weight)} 行が指定された")
    if _is_group_class_weight(class_weight) and not uses_cohort:
        raise ValueError("group ごとの class weight は cohort を使う stage でのみ指定できる")

    warm_start_path = config.model.get("warm_start_checkpoint_path")
    if warm_start_path:
        if config.get("ckpt_path"):
            raise ValueError("model.warm_start_checkpoint_path は ckpt_path による resume と併用できない")
        if not uses_cohort or strategy is None or not strategy.get("supports_warm_start", False):
            raise ValueError("model.warm_start_checkpoint_path は supports_warm_start を宣言した cohort strategy でのみサポートされる")
        if not Path(warm_start_path).is_file():
            raise FileNotFoundError(f"warm-start checkpoint が見つからない: {warm_start_path}")

    selection = config.get("checkpoint_selection")
    if selection is None or not selection.get("requires_hidden_cohort", False):
        return
    callbacks = config.get("callbacks", {})
    required_callbacks = ("cohort_validity", "hidden_cohort_logger")
    missing_callbacks = [name for name in required_callbacks if name not in callbacks]
    if missing_callbacks:
        raise ValueError(f"checkpoint_selection=hidden_min_auroc には次の callback が必要: {missing_callbacks}")
    if not config.data.get("group_assignment_path"):
        raise ValueError("checkpoint_selection=hidden_min_auroc には data.group_assignment_path が必要")
