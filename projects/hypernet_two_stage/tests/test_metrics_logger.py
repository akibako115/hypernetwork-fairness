import pytest
import torch

from projects.hypernet_two_stage.callbacks.metrics_logger import MetricsLogger


class _DummyModule:
    def __init__(self) -> None:
        self.device = torch.device("cpu")
        self.logged: dict[str, object] = {}

    def log(self, name: str, value: object, **kwargs: object) -> None:
        self.logged[name] = value


@pytest.mark.parametrize("phase", ["train", "val", "test"])
def test_metrics_logger_logs_loss_accuracy_bacc_and_auroc(phase: str) -> None:
    callback = MetricsLogger(num_classes=2)
    module = _DummyModule()
    callback.setup(None, module, "fit")
    outputs = {
        "loss": torch.tensor(0.5),
        "logits": torch.tensor([[4.0, -1.0], [3.0, 0.0], [-2.0, 3.0], [-1.0, 4.0]]),
        "preds": torch.tensor([0, 0, 1, 1]),
        "target": torch.tensor([0, 0, 1, 1]),
    }

    callback._step(module, outputs, phase)

    assert set(module.logged) == {
        f"{phase}/loss",
        f"{phase}/acc",
        f"{phase}/bacc",
        f"{phase}/auroc",
    }
    assert callback._losses[phase].compute().item() == pytest.approx(0.5)
    assert callback._accuracies[phase].compute().item() == pytest.approx(1.0)
    assert callback._baccs[phase].compute().item() == pytest.approx(1.0)
    assert callback._aurocs[phase].compute().item() == pytest.approx(1.0)
