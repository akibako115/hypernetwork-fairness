"""単段 run record が入力・preflight・終了状態を保存することを検証する。"""

import hashlib
import json
import subprocess
from pathlib import Path

import pytest
from omegaconf import OmegaConf

from projects.hypernet_e2e.run_record import RunRecorder


def _config(tmp_path: Path) -> object:
    image_dir = tmp_path / "images"
    split_dir = tmp_path / "splits"
    image_dir.mkdir(exist_ok=True)
    split_dir.mkdir(exist_ok=True)
    (split_dir / "train.csv").write_text("image,target\na.png,0\nb.png,1\n")
    (split_dir / "val.csv").write_text("image,target\nc.png,1\n")
    return OmegaConf.create(
        {
            "paths": {"project_dir": str(tmp_path / "project")},
            "data": {"data_dir": str(image_dir), "cv_splits_dir": str(split_dir)},
            "callbacks": {"model_checkpoint": {"dirpath": None}},
            "experiment_name": "Spatial LoRA / ERM",
            "seed": 42,
        }
    )


def test_prepare_fit_records_config_manifest_and_passing_preflight(tmp_path: Path, monkeypatch) -> None:
    commands: list[list[str]] = []

    def fake_run(command: list[str], **_: object) -> subprocess.CompletedProcess[str]:
        commands.append(command)
        if command[0] == "git":
            return subprocess.CompletedProcess(command, 0, stdout="deadbeef\n", stderr="")
        return subprocess.CompletedProcess(command, 0, stdout="3 passed\n", stderr="")

    monkeypatch.setattr("projects.hypernet_e2e.run_record.subprocess.run", fake_run)
    recorder = RunRecorder.prepare_fit(_config(tmp_path), project_dir=tmp_path / "project")

    assert recorder.run_dir.parent == tmp_path / "project" / "runs"
    assert "-spatial-lora-erm-s42-" in recorder.run_dir.name
    assert {path.name for path in recorder.run_dir.iterdir()} == {
        "artifacts",
        "checkpoints",
        "config.yaml",
        "data_manifest.json",
        "logs",
        "metrics",
        "preflight.json",
        "run.json",
    }
    manifest = json.loads((recorder.run_dir / "data_manifest.json").read_text())
    assert manifest["splits"]["train"]["num_rows"] == 2
    assert manifest["splits"]["train"]["target_distribution"] == {"0": 1, "1": 1}
    assert f"run_dir: {recorder.run_dir}" in (recorder.run_dir / "config.yaml").read_text()
    assert f"dirpath: {recorder.run_dir / 'checkpoints'}" in (recorder.run_dir / "config.yaml").read_text()
    assert json.loads((recorder.run_dir / "preflight.json").read_text())["exit_code"] == 0
    assert any(command[:3] == ["git", "rev-parse", "HEAD"] for command in commands)

    recorder.succeed({"val/auroc": 0.75})
    run_record = json.loads((recorder.run_dir / "run.json").read_text())
    assert run_record["status"] == "succeeded"
    assert run_record["result_summary"] == {"val/auroc": 0.75}


def test_preflight_failure_keeps_failed_run(tmp_path: Path, monkeypatch) -> None:
    def fake_run(command: list[str], **_: object) -> subprocess.CompletedProcess[str]:
        if command[0] == "git":
            return subprocess.CompletedProcess(command, 0, stdout="deadbeef\n", stderr="")
        return subprocess.CompletedProcess(command, 1, stdout="", stderr="failed")

    monkeypatch.setattr("projects.hypernet_e2e.run_record.subprocess.run", fake_run)

    try:
        RunRecorder.prepare_fit(_config(tmp_path), project_dir=tmp_path / "project")
    except RuntimeError as error:
        assert str(error) == "golden preflight に失敗した"
    else:
        raise AssertionError("preflight failure must stop the fit")

    run_record = next((tmp_path / "project" / "runs").glob("*/run.json"))
    assert json.loads(run_record.read_text())["status"] == "failed"


def test_prepare_fit_points_the_experiment_logger_at_the_reserved_run(tmp_path: Path, monkeypatch) -> None:
    def fake_run(command: list[str], **_: object) -> subprocess.CompletedProcess[str]:
        if command[0] == "git":
            return subprocess.CompletedProcess(command, 0, stdout="deadbeef\n", stderr="")
        return subprocess.CompletedProcess(command, 0, stdout="1 passed\n", stderr="")

    monkeypatch.setattr("projects.hypernet_e2e.run_record.subprocess.run", fake_run)
    config = _config(tmp_path)
    OmegaConf.update(config, "logger", {"wandb": {"save_dir": None, "name": None}}, merge=False)

    recorder = RunRecorder.prepare_fit(config, project_dir=tmp_path / "project")

    assert config.logger.wandb.save_dir == str(recorder.run_dir)
    assert config.logger.wandb.name == recorder.run_dir.name
    assert f"save_dir: {recorder.run_dir}" in (recorder.run_dir / "config.yaml").read_text()
    assert json.loads((recorder.run_dir / "run.json").read_text())["loggers"] == []

    reference = {"logger": "WandbLogger", "id": "abc123", "name": "run-name", "url": "https://wandb.ai/run"}
    recorder.record_loggers([reference])

    assert json.loads((recorder.run_dir / "run.json").read_text())["loggers"] == [reference]


def _passing_preflight(monkeypatch) -> None:
    def fake_run(command: list[str], **_: object) -> subprocess.CompletedProcess[str]:
        if command[0] == "git":
            return subprocess.CompletedProcess(command, 0, stdout="deadbeef\n", stderr="")
        return subprocess.CompletedProcess(command, 0, stdout="1 passed\n", stderr="")

    monkeypatch.setattr("projects.hypernet_e2e.run_record.subprocess.run", fake_run)


class _CheckpointCallback:
    def __init__(self, best_model_path: str, last_model_path: str = "") -> None:
        self.monitor = "val/auroc"
        self.mode = "max"
        self.best_model_path = best_model_path
        self.last_model_path = last_model_path
        self.best_model_score = 0.75


def _stage1_run(tmp_path: Path, monkeypatch) -> tuple[Path, Path]:
    """best checkpoint を1つ記録した 1 段目 run を作り、その run directory と checkpoint を返す。"""
    _passing_preflight(monkeypatch)
    recorder = RunRecorder.prepare_fit(_config(tmp_path), project_dir=tmp_path / "project")
    checkpoint = recorder.run_dir / "checkpoints" / "best.ckpt"
    checkpoint.write_bytes(b"stage1 weights")
    recorder.record_checkpoints([_CheckpointCallback(str(checkpoint))])
    recorder.succeed({"val/auroc": 0.75})
    return recorder.run_dir, checkpoint


def test_record_checkpoints_keeps_the_selection_criterion_next_to_the_digest(tmp_path: Path, monkeypatch) -> None:
    run_dir, checkpoint = _stage1_run(tmp_path, monkeypatch)

    (entry,) = json.loads((run_dir / "run.json").read_text())["checkpoints"]
    assert entry["monitor"] == "val/auroc"
    assert entry["mode"] == "max"
    assert entry["best"]["path"] == str(checkpoint)
    assert entry["best"]["score"] == 0.75
    assert entry["best"]["sha256"] == hashlib.sha256(b"stage1 weights").hexdigest()
    assert entry["last"] is None


def test_record_checkpoints_skips_callbacks_that_produced_nothing(tmp_path: Path, monkeypatch) -> None:
    _passing_preflight(monkeypatch)
    recorder = RunRecorder.prepare_fit(_config(tmp_path), project_dir=tmp_path / "project")

    recorder.record_checkpoints([object(), _CheckpointCallback("")])

    assert json.loads((recorder.run_dir / "run.json").read_text())["checkpoints"] == []


def test_second_stage_run_records_which_run_produced_its_backbone(tmp_path: Path, monkeypatch) -> None:
    stage1_dir, checkpoint = _stage1_run(tmp_path, monkeypatch)
    config = _config(tmp_path)
    OmegaConf.update(config, "model", {"backbone_checkpoint_path": str(checkpoint)}, merge=False)

    recorder = RunRecorder.prepare_fit(config, project_dir=tmp_path / "project")

    parent = json.loads((recorder.run_dir / "run.json").read_text())["parent_run"]
    assert parent["run_id"] == stage1_dir.name
    assert parent["backbone_checkpoint"]["sha256"] == hashlib.sha256(b"stage1 weights").hexdigest()


def test_second_stage_run_refuses_a_backbone_that_changed_since_stage_one_recorded_it(tmp_path: Path, monkeypatch) -> None:
    _, checkpoint = _stage1_run(tmp_path, monkeypatch)
    checkpoint.write_bytes(b"tampered weights")
    config = _config(tmp_path)
    OmegaConf.update(config, "model", {"backbone_checkpoint_path": str(checkpoint)}, merge=False)

    with pytest.raises(RuntimeError, match="SHA-256"):
        RunRecorder.prepare_fit(config, project_dir=tmp_path / "project")


def test_single_stage_run_records_no_parent(tmp_path: Path, monkeypatch) -> None:
    _passing_preflight(monkeypatch)

    recorder = RunRecorder.prepare_fit(_config(tmp_path), project_dir=tmp_path / "project")

    assert json.loads((recorder.run_dir / "run.json").read_text())["parent_run"] is None
