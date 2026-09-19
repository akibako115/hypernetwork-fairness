import base64
import hashlib
import json
import math
from io import BytesIO
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

from projects.hypernet_two_stage.callbacks.fairness_metrics import FairnessMetricsCallback
from projects.hypernet_two_stage.data.transform_utils import IMAGENET_FILL, IMAGENET_MEAN, IMAGENET_STD, train_transforms_chexpert, val_transforms_chexpert


def _golden() -> dict:
    golden_path = Path(__file__).parents[3] / "tests/golden/eval_transform.v1.json"
    return json.loads(golden_path.read_text())


@pytest.mark.preflight
def test_chexpert_eval_transform_reproduces_the_golden_cases() -> None:
    golden = _golden()
    transform = val_transforms_chexpert()

    assert IMAGENET_MEAN == golden["constants"]["IMAGENET_MEAN"]
    assert IMAGENET_STD == golden["constants"]["IMAGENET_STD"]
    assert list(IMAGENET_FILL) == golden["constants"]["IMAGENET_FILL"]

    for case in golden["cases"]:
        image_bytes = base64.b64decode(case["image"]["png_base64"])
        with Image.open(BytesIO(image_bytes)) as image:
            actual = transform(image.convert("RGB"))
        expected = case["expected"]

        assert list(actual.shape) == expected["shape"], case["name"]
        assert str(actual.numpy().dtype) == expected["dtype"], case["name"]
        assert hashlib.sha256(np.ascontiguousarray(actual.numpy()).tobytes()).hexdigest() == expected["sha256"], case["name"]
        for statistic in ("min", "max", "mean", "std"):
            assert np.isclose(getattr(actual, statistic)().item(), expected[statistic], atol=golden["tolerance"]), case["name"]


@pytest.mark.preflight
def test_chexpert_train_transform_composition_matches_the_golden_contract() -> None:
    golden = _golden()
    transform = train_transforms_chexpert()
    composition = [str(step) for step in transform.transforms]

    assert composition == golden["train_transform"]["composition"]


class _FairnessModule:
    def __init__(self, attribute_names: dict[str, list[str]] | None) -> None:
        self.attribute_names = attribute_names
        self.logged: dict[str, float] = {}

    def log(self, name: str, value: float) -> None:
        self.logged[name] = value


@pytest.mark.preflight
def test_fairness_callback_reproduces_the_golden_cases() -> None:
    golden_path = Path(__file__).parents[3] / "tests/golden/fairness_metrics.v1.json"
    golden = json.loads(golden_path.read_text())

    for case in golden["cases"]:
        module = _FairnessModule(case["attribute_names"])
        callback = FairnessMetricsCallback()
        callback.on_validation_epoch_start(None, module)  # type: ignore[arg-type]
        for batch_index, batch in enumerate(case["batches"]):
            callback.on_validation_batch_end(
                None,
                module,
                {
                    "logits": torch.tensor(batch["logits"]),
                    "target": torch.tensor(batch["target"]),
                    "attributes": {name: torch.tensor(values) for name, values in batch["attributes"].items()},
                },
                None,
                batch_index,
            )  # type: ignore[arg-type]
        callback.on_validation_epoch_end(None, module)  # type: ignore[arg-type]

        assert set(module.logged) == set(case["expected"]), case["name"]
        for name, expected in case["expected"].items():
            actual = module.logged[name]
            if expected is None:
                assert math.isnan(actual), f"{case['name']}: {name}"
            else:
                assert actual == pytest.approx(expected, abs=golden["tolerance"]), f"{case['name']}: {name}"
