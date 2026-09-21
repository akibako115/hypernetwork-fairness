"""TTY 非依存の進捗 callback の表示・hook 契約を検証する。"""

import sys

import torch

from projects.hypernet_iterative.callbacks.text_progress import TextProgressLogger


def test_text_progress_formats_scalar_tensors_to_four_decimal_places() -> None:
    assert TextProgressLogger._format_metric(torch.tensor(0.123456)) == "0.1235"


def test_text_progress_restores_the_previous_exception_hook() -> None:
    callback = TextProgressLogger()
    previous = sys.excepthook

    callback.setup(None, None, "fit")
    assert sys.excepthook == callback._excepthook
    callback.teardown(None, None, "fit")

    assert sys.excepthook == previous
