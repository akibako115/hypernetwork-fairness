import math

import pandas as pd
import pytest
import torch

from projects.hypernet_iterative.callbacks.hidden_cohort_logger import CohortSingleProcessCallback, CohortValidityCallback, GroupDRODiagnosticsCallback, HiddenCohortMetricsCallback


class _DummyModule:
    def __init__(self) -> None:
        self.logged: dict[str, float] = {}

    def log(self, name: str, value: float, **kwargs: object) -> None:
        self.logged[name] = float(value)


class _DummyTrainer:
    def __init__(self, frame: pd.DataFrame) -> None:
        self.datamodule = type("DataModule", (), {"data_val": type("Dataset", (), {"df": frame})()})()


class _SanityCheckingTrainer:
    sanity_checking = True


class _DistributedTrainer:
    world_size = 2


def test_hidden_cohort_logger_records_validation_metrics() -> None:
    callback = HiddenCohortMetricsCallback(num_groups=2, log_aggregate_metrics=True)
    module = _DummyModule()
    callback.on_validation_epoch_start(None, module)
    callback.on_validation_batch_end(
        None,
        module,
        {
            "logits": torch.tensor([[4.0, 0.0], [0.0, 4.0], [3.0, 0.0], [0.0, 3.0]]),
            "target": torch.tensor([0, 1, 1, 1]),
            "attributes": {"group_id": torch.tensor([0, 0, 1, 1])},
        },
        None,
        0,
    )

    callback.on_validation_epoch_end(None, module)

    assert module.logged["val/hidden_support_00"] == 2
    assert module.logged["val/hidden_support_01"] == 2
    assert module.logged["val/hidden_valid_auroc_groups"] == 1
    assert module.logged["val/hidden_min_auroc"] == pytest.approx(1.0)
    assert module.logged["val/hidden_max_loss"] > 0


def test_hidden_cohort_logger_rejects_single_class_validation_group() -> None:
    callback = HiddenCohortMetricsCallback(num_groups=2, require_binary_val_groups=True, log_aggregate_metrics=True)
    module = _DummyModule()
    callback.on_validation_epoch_start(None, module)
    callback.on_validation_batch_end(
        None,
        module,
        {
            "logits": torch.tensor([[4.0, 0.0], [0.0, 4.0]]),
            "target": torch.tensor([0, 1]),
            "attributes": {"group_id": torch.tensor([0, 1])},
        },
        None,
        0,
    )

    with pytest.raises(ValueError, match="both classes"):
        callback.on_validation_epoch_end(None, module)


def test_hidden_cohort_logger_rejects_empty_validation_group() -> None:
    callback = HiddenCohortMetricsCallback(num_groups=2, require_binary_val_groups=True)
    module = _DummyModule()
    callback.on_validation_epoch_start(None, module)
    callback.on_validation_batch_end(
        None,
        module,
        {
            "logits": torch.tensor([[4.0, 0.0], [0.0, 4.0]]),
            "target": torch.tensor([0, 1]),
            "attributes": {"group_id": torch.tensor([0, 0])},
        },
        None,
        0,
    )

    with pytest.raises(ValueError, match="both classes"):
        callback.on_validation_epoch_end(None, module)


def test_hidden_cohort_logger_does_not_reject_partial_sanity_validation() -> None:
    callback = HiddenCohortMetricsCallback(num_groups=2, require_binary_val_groups=True, log_aggregate_metrics=True)
    module = _DummyModule()
    callback.on_validation_epoch_start(_SanityCheckingTrainer(), module)
    callback.on_validation_batch_end(
        _SanityCheckingTrainer(),
        module,
        {
            "logits": torch.tensor([[4.0, 0.0]]),
            "target": torch.tensor([0]),
            "attributes": {"group_id": torch.tensor([0])},
        },
        None,
        0,
    )

    callback.on_validation_epoch_end(_SanityCheckingTrainer(), module)

    assert math.isnan(module.logged["val/hidden_min_auroc"])


@pytest.mark.parametrize(
    "frame",
    [
        pd.DataFrame({"group_id": [0, 0], "target": [0, 1]}),
        pd.DataFrame({"group_id": [0, 0, 1, 1], "target": [0, 1, 0, 0]}),
    ],
)
def test_cohort_validity_stops_before_training_for_invalid_validation_cohorts(frame: pd.DataFrame) -> None:
    callback = CohortValidityCallback(num_groups=2)

    with pytest.raises(ValueError):
        callback.on_fit_start(_DummyTrainer(frame), _DummyModule())


def test_cohort_single_process_callback_rejects_distributed_training() -> None:
    with pytest.raises(RuntimeError, match="world_size=1"):
        CohortSingleProcessCallback().on_fit_start(_DistributedTrainer(), _DummyModule())


def test_hidden_cohort_logger_records_nan_for_empty_and_single_class_test_groups() -> None:
    callback = HiddenCohortMetricsCallback(num_groups=2, log_aggregate_metrics=True)
    module = _DummyModule()
    callback.on_test_epoch_start(None, module)
    callback.on_test_batch_end(
        None,
        module,
        {
            "logits": torch.tensor([[4.0, 0.0]]),
            "target": torch.tensor([0]),
            "attributes": {"group_id": torch.tensor([0])},
        },
        None,
        0,
    )
    callback.on_test_epoch_end(None, module)

    assert module.logged["test/hidden_support_00"] == 1
    assert module.logged["test/hidden_support_01"] == 0
    assert math.isnan(module.logged["test/hidden_auroc_00"])
    assert math.isnan(module.logged["test/hidden_auroc_01"])
    assert math.isnan(module.logged["test/hidden_min_bacc"])


def test_hidden_cohort_logger_rejects_multiclass_logits_before_metric_computation() -> None:
    callback = HiddenCohortMetricsCallback(num_groups=2, log_aggregate_metrics=True)
    module = _DummyModule()
    callback.on_test_epoch_start(None, module)
    callback.on_test_batch_end(
        None,
        module,
        {
            "logits": torch.tensor([[3.0, 0.0, 0.0], [0.0, 3.0, 0.0]]),
            "target": torch.tensor([0, 1]),
            "attributes": {"group_id": torch.tensor([0, 1])},
        },
        None,
        0,
    )

    with pytest.raises(ValueError, match="binary logits"):
        callback.on_test_epoch_end(None, module)


def test_hidden_cohort_logger_records_all_groups_when_test_dataloader_is_empty() -> None:
    callback = HiddenCohortMetricsCallback(num_groups=2, log_aggregate_metrics=True)
    module = _DummyModule()
    callback.on_test_epoch_start(None, module)

    callback.on_test_epoch_end(None, module)

    assert module.logged["test/hidden_support_00"] == 0
    assert module.logged["test/hidden_support_01"] == 0
    assert math.isnan(module.logged["test/hidden_auroc_00"])
    assert math.isnan(module.logged["test/hidden_auroc_01"])
    assert module.logged["test/hidden_valid_auroc_groups"] == 0
    assert math.isnan(module.logged["test/hidden_min_auroc"])


def test_group_dro_diagnostics_logs_all_weights_and_summary() -> None:
    module = _DummyModule()
    module.training_objective = type("Objective", (), {"adv_probs": torch.tensor([0.25, 0.75])})()

    GroupDRODiagnosticsCallback().on_train_epoch_end(None, module)

    assert module.logged["train/group_dro/q_00"] == 0.25
    assert module.logged["train/group_dro/q_01"] == 0.75
    assert module.logged["train/group_dro/max_q"] == 0.75
    assert module.logged["train/group_dro/weight_entropy"] > 0


def test_group_loss_is_logged_for_every_cohort() -> None:
    callback = HiddenCohortMetricsCallback(num_groups=2, log_aggregate_metrics=True)
    module = _DummyModule()
    callback.on_validation_epoch_start(None, module)
    callback.on_validation_batch_end(
        None,
        module,
        {
            "logits": torch.tensor([[4.0, 0.0], [0.0, 4.0], [3.0, 0.0], [0.0, 3.0]]),
            "target": torch.tensor([0, 1, 1, 1]),
            "attributes": {"group_id": torch.tensor([0, 0, 1, 1])},
        },
        None,
        0,
    )

    callback.on_validation_epoch_end(None, module)

    # max と gap だけでは、どの群が重いのかも、その群が改善したのかも後から追えない。
    assert module.logged["val/hidden_loss_00"] > 0
    assert module.logged["val/hidden_loss_01"] > 0
    # bacc も同じ理由で群ごとに残す。両クラスが揃う群 00 だけが値を持ち、単一クラスの群 01 は nan。
    assert module.logged["val/hidden_bacc_00"] == pytest.approx(module.logged["val/hidden_min_bacc"])
    assert math.isnan(module.logged["val/hidden_bacc_01"])
    assert max(module.logged["val/hidden_loss_00"], module.logged["val/hidden_loss_01"]) == pytest.approx(module.logged["val/hidden_max_loss"])
