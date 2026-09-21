"""iterative stage の cohort・引き継ぎ・checkpoint 設定を検証する。"""

from pathlib import Path

from omegaconf import DictConfig


def validate_training_config(config: DictConfig) -> None:
    """学習を始める前に cohort、warm-start、checkpoint 選択の組み合わせを検証する。"""
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

    # ClassBalancedGroupDROTaskLoss は group loss を群内クラス平均へ置き換えて陽性率依存を
    # 取り除く目的関数である。そこへ共通の class weight を掛けると陽性側の寄与が押し戻され、
    # 置き換えの意味が消える。inverse weighting は experiment の既定値なので、組み合わせが
    # 通ってしまうと使えない run が最後まで走り切る。
    if strategy is not None and strategy.get("name") == "group_dro_balanced" and config.model.loss_fn.get("class_weight") is not None:
        raise ValueError("training_strategy=group_dro_balanced は class_weight と併用できない（weighting=none で実行する）")

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
