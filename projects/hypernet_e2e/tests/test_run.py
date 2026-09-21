"""e2e の単段 Lightning runner を検証する。"""

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from omegaconf import OmegaConf

from projects.hypernet_e2e import training


class _Recorder:
    def __init__(self, run_dir: Path) -> None:
        self.run_dir = run_dir
        self.completed: dict[str, float] | None = None
        self.failed: BaseException | None = None
        self.loggers: list[dict[str, str]] | None = None
        self.checkpoint_callbacks: list[object] | None = None

    def record_loggers(self, references: list[dict[str, str]]) -> None:
        self.loggers = list(references)

    def record_checkpoints(self, callbacks: list[object]) -> None:
        self.checkpoint_callbacks = list(callbacks)

    def succeed(self, metrics: dict[str, float]) -> None:
        self.completed = metrics

    def fail(self, error: BaseException) -> None:
        self.failed = error


class _Trainer:
    def __init__(self, *, should_fail: bool = False) -> None:
        self.should_fail = should_fail
        self.callback_metrics = {"val/auroc": torch.tensor(0.75), "val/loss": 0.5}
        self.fit_arguments: tuple[object, object] | None = None

    def fit(self, *, model: object, datamodule: object) -> None:
        self.fit_arguments = (model, datamodule)
        if self.should_fail:
            raise RuntimeError("fit failed")


def _config(tmp_path: Path) -> object:
    split_dir = tmp_path / "splits"
    split_dir.mkdir()
    (split_dir / "train.csv").write_text("target\n0\n1\n1\n1\n")
    return OmegaConf.create(
        {
            "seed": None,
            "weighting": "inverse",
            "data": {"cv_splits_dir": str(split_dir), "num_classes": 2},
            "model": {"loss_fn": {"class_weight": None}},
            "callbacks": {},
            "trainer": {},
        }
    )


def test_run_fit_resolves_weights_fits_once_and_records_scalar_metrics(tmp_path: Path, monkeypatch) -> None:
    run_dir = tmp_path / "run"
    (run_dir / "metrics").mkdir(parents=True)
    recorder = _Recorder(run_dir)
    trainer = _Trainer()
    data_module = object()
    model = object()

    monkeypatch.setattr(training.RunRecorder, "prepare_fit", lambda config: recorder)
    monkeypatch.setattr(training, "instantiate", lambda config, **_: data_module if config._get_node("cv_splits_dir") else model if config._get_node("loss_fn") else trainer)

    assert training.run_fit(_config(tmp_path)) == run_dir
    assert trainer.fit_arguments == (model, data_module)
    assert recorder.completed == {"val/auroc": 0.75, "val/loss": 0.5}
    assert recorder.failed is None
    assert json.loads((run_dir / "metrics" / "fit.json").read_text()) == {"val/auroc": 0.75, "val/loss": 0.5}
    assert recorder.loggers == []
    assert recorder.checkpoint_callbacks == []
    assert (run_dir / "logs" / "train.log").is_file()


def test_run_fit_marks_the_reserved_run_failed_when_fit_raises(tmp_path: Path, monkeypatch) -> None:
    run_dir = tmp_path / "run"
    (run_dir / "metrics").mkdir(parents=True)
    recorder = _Recorder(run_dir)
    trainer = _Trainer(should_fail=True)

    monkeypatch.setattr(training.RunRecorder, "prepare_fit", lambda config: recorder)
    monkeypatch.setattr(training, "instantiate", lambda config, **_: SimpleNamespace() if config._get_node("cv_splits_dir") or config._get_node("loss_fn") else trainer)

    with pytest.raises(RuntimeError, match="fit failed"):
        training.run_fit(_config(tmp_path))

    assert isinstance(recorder.failed, RuntimeError)


def test_inverse_frequency_weights_reject_missing_class_and_matches_old_normalization() -> None:
    assert training._inverse_frequency_weights([0, 1, 1, 1], 2) == [1.5, 0.5]
    with pytest.raises(ValueError, match="必要"):
        training._inverse_frequency_weights([0, 0], 2)


def test_scalar_metrics_writes_non_finite_values_as_json_null(tmp_path: Path) -> None:
    metrics = training._scalar_metrics({"finite": torch.tensor(0.75), "nan": torch.tensor(float("nan")), "infinity": float("inf")})
    path = tmp_path / "metrics.json"
    training._write_metrics(path, metrics)

    assert json.loads(path.read_text()) == {"finite": 0.75, "infinity": None, "nan": None}
