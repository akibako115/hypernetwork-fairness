from pathlib import Path
from types import SimpleNamespace

import pytest
from omegaconf import OmegaConf

from projects.hypernet_two_stage import workflow


def test_workflow_passes_stage1_best_checkpoint_to_frozen_stage2(tmp_path: Path, monkeypatch) -> None:
    config = OmegaConf.create(
        {
            "paths": {"project_dir": str(tmp_path)},
            "seed": 7,
            "weighting": "none",
            "model": {"loss_fn": {"class_weight": [1.0, 2.0]}},
        }
    )
    observed = []

    def fake_run_stage(name, stage_config, stage_dir, run_id):
        observed.append((name, stage_config, stage_dir, run_id))
        return {"checkpoints": {"best_val_auroc": {"path": str(tmp_path / f"{name}.ckpt"), "sha256": name}, "last": None}, "metrics": {}}

    monkeypatch.setattr(workflow, "_run_stage", fake_run_stage)
    monkeypatch.setattr(workflow, "_write_data_manifest", lambda *_: None)
    monkeypatch.setattr(workflow, "_write_preflight", lambda *_: None)
    run_dir = workflow.run_two_stage(config)

    assert [name for name, _, _, _ in observed] == ["stage1", "stage2"]
    assert [run_id for _, _, _, run_id in observed] == [run_dir.name, run_dir.name]
    stage2 = observed[1][1]
    assert stage2.model.backbone_checkpoint_path == str(tmp_path / "stage1.ckpt")
    assert stage2.model.freeze_backbone is True
    assert stage2.model.loss_fn.class_weight == [1.0, 2.0]
    assert (run_dir / "run.json").is_file()
    assert (run_dir / "logs" / "train.log").is_file()


def test_best_auroc_checkpoint_rejects_a_callback_monitoring_something_else() -> None:
    """monitor を差し替えた run の checkpoint を best_val_auroc として記録させない。"""
    auroc = SimpleNamespace(monitor="val/auroc", best_model_path="/tmp/best.ckpt")
    loss = SimpleNamespace(monitor="val/loss", best_model_path="/tmp/loss.ckpt")

    assert workflow._best_auroc_checkpoint([loss, auroc], "stage1") is auroc

    with pytest.raises(RuntimeError, match="val/auroc を monitor"):
        workflow._best_auroc_checkpoint([loss], "stage1")
    with pytest.raises(RuntimeError, match="val/auroc を monitor"):
        workflow._best_auroc_checkpoint([auroc, auroc], "stage1")


def test_best_auroc_checkpoint_requires_a_written_checkpoint() -> None:
    empty = SimpleNamespace(monitor="val/auroc", best_model_path="")

    with pytest.raises(RuntimeError, match="best val/auroc checkpoint を出力"):
        workflow._best_auroc_checkpoint([empty], "stage2")
