import torch

from projects.hypernet_two_stage.callbacks.fairness_metrics import FairnessMetricsCallback


class _Module:
    attribute_names = {"categorical": ["sex"]}
    fairness_attribute_names = {"categorical": ["sex", "age_group_65"]}

    def __init__(self) -> None:
        self.logged: dict[str, float] = {}

    def log(self, name: str, value: float) -> None:
        self.logged[name] = value


def test_callback_aggregates_batches_and_logs_all_fairness_metrics() -> None:
    callback = FairnessMetricsCallback()
    module = _Module()
    callback.on_validation_epoch_start(None, module)  # type: ignore[arg-type]
    for logits, target, categorical in [
        (torch.tensor([[4.0, 0.0], [0.0, 4.0]]), torch.tensor([0, 1]), torch.tensor([[0], [0]])),
        (torch.tensor([[0.0, 4.0], [4.0, 0.0]]), torch.tensor([0, 1]), torch.tensor([[1], [1]])),
    ]:
        callback.on_validation_batch_end(None, module, {"logits": logits, "target": target, "attributes": {"categorical": categorical}}, None, 0)  # type: ignore[arg-type]
    callback.on_validation_epoch_end(None, module)  # type: ignore[arg-type]

    assert set(module.logged) == {
        "val/sex/Eopp0",
        "val/sex/Eopp1",
        "val/sex/Eodds",
        "val/sex/auroc_gap",
        "val/sex/bacc_gap",
        "val/sex/worst_group_auroc",
        "val/sex/worst_group_bacc",
    }
    assert module.logged["val/sex/worst_group_auroc"] == 0.0
    assert module.logged["val/sex/worst_group_bacc"] == 0.0


def test_callback_uses_evaluation_attributes_instead_of_model_input_attributes() -> None:
    callback = FairnessMetricsCallback()
    module = _Module()
    callback.on_validation_epoch_start(None, module)  # type: ignore[arg-type]
    for logits, target, categorical in [
        (torch.tensor([[4.0, 0.0], [0.0, 4.0]]), torch.tensor([0, 1]), torch.tensor([[0, 0], [0, 0]])),
        (torch.tensor([[0.0, 4.0], [4.0, 0.0]]), torch.tensor([0, 1]), torch.tensor([[1, 1], [1, 1]])),
    ]:
        callback.on_validation_batch_end(
            None,
            module,
            {"logits": logits, "target": target, "attributes": {"evaluation_categorical": categorical, "evaluation_categorical_missing": torch.zeros_like(categorical, dtype=torch.bool)}},
            None,
            0,
        )  # type: ignore[arg-type]
    callback.on_validation_epoch_end(None, module)  # type: ignore[arg-type]

    assert "val/age_group_65/Eodds" in module.logged
    assert "val/frontal_lateral/Eodds" not in module.logged
