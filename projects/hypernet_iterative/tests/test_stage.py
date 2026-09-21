"""子 process 側 stage runner の checkpoint 選択を検証する。"""

from types import SimpleNamespace

import pytest

from projects.hypernet_iterative.stage import _selected_checkpoint


def _checkpoint(monitor: str | None) -> SimpleNamespace:
    return SimpleNamespace(monitor=monitor, best_model_path="best.ckpt", last_model_path="last.ckpt")


def test_selection_ignores_callback_order_and_other_monitors() -> None:
    auroc = _checkpoint("val/auroc")

    selected = _selected_checkpoint([_checkpoint("val/bacc"), SimpleNamespace(), auroc, _checkpoint("val/hidden_min_auroc")])

    assert selected is auroc


@pytest.mark.parametrize(
    "callbacks",
    [
        [_checkpoint("val/bacc")],
        [_checkpoint("val/auroc"), _checkpoint("val/auroc")],
        [],
    ],
)
def test_selection_refuses_anything_but_exactly_one_val_auroc_checkpoint(callbacks: list) -> None:
    with pytest.raises(RuntimeError, match="val/auroc"):
        _selected_checkpoint(callbacks)
