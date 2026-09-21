import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from omegaconf import OmegaConf

from projects.hypernet_two_stage import workflow


def _config(tmp_path: Path) -> OmegaConf:
    """stage 固有設定を両 stage に持つ、workflow が受け取る形の最小 config。"""
    return OmegaConf.create(
        {
            "paths": {"project_dir": str(tmp_path)},
            "seed": 7,
            "weighting": "none",
            "class_weight": [1.0, 2.0],
            "trainer": {"max_epochs": 30},
            "stage1": {
                "model": {"net": {"_target_": "resnet"}, "loss_fn": {"class_weight": "${class_weight}"}, "freeze_backbone": False, "backbone_checkpoint_path": None},
                "trainer": {"max_epochs": None},
            },
            "stage2": {
                "model": {"net": {"_target_": "spatial_lora"}, "loss_fn": {"class_weight": "${class_weight}"}, "freeze_backbone": True, "backbone_checkpoint_path": None},
                "trainer": {"max_epochs": 10},
            },
        }
    )


def _fake_stage(tmp_path: Path, observed: list):
    def run_stage(name, stage_config, stage_dir, run_id):
        observed.append((name, stage_config, stage_dir, run_id))
        checkpoint = tmp_path / f"{name}.ckpt"
        checkpoint.write_bytes(name.encode())
        return {
            "checkpoints": {
                "best_val_auroc": {
                    "path": str(checkpoint),
                    "role": "best_val_auroc",
                    "score": 0.5,
                    "sha256": hashlib.sha256(name.encode()).hexdigest(),
                },
                "last": None,
            },
            "metrics": {},
        }

    return run_stage


def test_workflow_passes_stage1_best_checkpoint_to_frozen_stage2(tmp_path: Path, monkeypatch) -> None:
    observed: list = []
    monkeypatch.setattr(workflow, "_run_stage", _fake_stage(tmp_path, observed))
    monkeypatch.setattr(workflow, "_write_data_manifest", lambda *_: None)
    monkeypatch.setattr(workflow, "_write_preflight", lambda *_: None)

    run_dir = workflow.run_two_stage(_config(tmp_path))

    assert [name for name, _, _, _ in observed] == ["stage1", "stage2"]
    assert [run_id for _, _, _, run_id in observed] == [run_dir.name, run_dir.name]
    stage2 = observed[1][1]
    assert stage2.model.backbone_checkpoint_path == str(tmp_path / "stage1.ckpt")
    assert stage2.model.freeze_backbone is True
    assert stage2.model.loss_fn.class_weight == [1.0, 2.0]
    assert (run_dir / "run.json").is_file()
    assert (run_dir / "logs" / "train.log").is_file()


def test_workflow_fails_when_the_stage1_checkpoint_changed_after_it_was_recorded(tmp_path: Path, monkeypatch) -> None:
    """stage 間で渡す checkpoint は、記録した hash と一致する実体でなければならない。"""
    observed: list = []
    inner = _fake_stage(tmp_path, observed)

    def run_stage(name, stage_config, stage_dir, run_id):
        result = inner(name, stage_config, stage_dir, run_id)
        (tmp_path / f"{name}.ckpt").write_bytes(b"tampered")
        return result

    monkeypatch.setattr(workflow, "_run_stage", run_stage)
    monkeypatch.setattr(workflow, "_write_data_manifest", lambda *_: None)
    monkeypatch.setattr(workflow, "_write_preflight", lambda *_: None)

    with pytest.raises(RuntimeError, match="SHA-256 が記録と一致しない"):
        workflow.run_two_stage(_config(tmp_path))


def test_stage_config_routes_each_stage_to_its_own_model_and_trainer(tmp_path: Path) -> None:
    config = _config(tmp_path)

    stage1 = workflow._stage_config(config, "stage1")
    stage2 = workflow._stage_config(config, "stage2")

    assert stage1.model.net._target_ == "resnet"
    assert stage2.model.net._target_ == "spatial_lora"
    # stage の trainer は差分。null は共有 trainer の値をそのまま使う。
    assert stage1.trainer.max_epochs == 30
    assert stage2.trainer.max_epochs == 10
    # 実行した stage の条件だけを stage の config.yaml に残す。
    assert "stage1" not in stage1 and "stage2" not in stage1
    assert "stage1" not in stage2 and "stage2" not in stage2


def test_stage_config_requires_the_stage_section() -> None:
    with pytest.raises(ValueError, match="stage2.model が必要"):
        workflow._stage_config(OmegaConf.create({"trainer": {}, "stage1": {"model": {}}}), "stage2")


def test_inverse_weighting_resolves_one_shared_class_weight(tmp_path: Path) -> None:
    """両 stage の loss_fn が同じ ${class_weight} を参照するので、解決先は1箇所に保つ。"""
    splits = tmp_path / "splits"
    splits.mkdir()
    (splits / "train.csv").write_text("target\n" + "0\n" * 3 + "1\n")
    config = _config(tmp_path)
    config.weighting = "inverse"
    config.class_weight = None
    config.data = {"cv_splits_dir": str(splits), "num_classes": 2}

    workflow._resolve_inverse_class_weights(config)

    assert config.class_weight == [0.5, 1.5]
    assert workflow._stage_config(config, "stage1").model.loss_fn.class_weight == [0.5, 1.5]
    assert workflow._stage_config(config, "stage2").model.loss_fn.class_weight == [0.5, 1.5]


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


def test_checkpoint_reference_records_role_score_and_a_streamed_digest(tmp_path: Path) -> None:
    path = tmp_path / "best.ckpt"
    path.write_bytes(b"x" * (1024 * 1024 + 7))

    reference = workflow._checkpoint_reference(path, "best_val_auroc", 0.812)

    assert reference["role"] == "best_val_auroc"
    assert reference["score"] == pytest.approx(0.812)
    assert reference["sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()


def test_run_json_records_what_is_needed_to_reproduce_the_parent_run(tmp_path: Path, monkeypatch) -> None:
    """e2e と two_stage の結果を並べられるよう、再現情報の水準を揃える。"""
    observed: list = []
    monkeypatch.setattr(workflow, "_run_stage", _fake_stage(tmp_path, observed))
    monkeypatch.setattr(workflow, "_write_data_manifest", lambda *_: None)
    monkeypatch.setattr(workflow, "_write_preflight", lambda *_: None)

    run_dir = workflow.run_two_stage(_config(tmp_path))
    record = json.loads((run_dir / "run.json").read_text())

    assert record["run_id"] == run_dir.name
    assert record["kind"] == "two_stage"
    assert record["seed"] == 7
    assert record["status"] == "succeeded"
    assert record["finished_at"] is not None
    assert "git_commit" in record
    assert record["selected_checkpoint"]["role"] == "best_val_auroc"


def test_reserved_run_id_follows_the_shared_path_convention(tmp_path: Path) -> None:
    config = _config(tmp_path)
    config.experiment_name = "Two Stage CheXpert ERM"

    with_seed = workflow._reserve_run_dir(config).name
    config.seed = None
    without_seed = workflow._reserve_run_dir(config).name

    assert "-two-stage-chexpert-erm-s7-" in with_seed
    assert "-two-stage-chexpert-erm-snone-" in without_seed


def test_data_manifest_records_the_image_set_hash(tmp_path: Path) -> None:
    splits = tmp_path / "splits"
    splits.mkdir()
    for name in ("train", "val"):
        (splits / f"{name}.csv").write_text(f"image,target\nb_{name}.png,0\na_{name}.png,1\n")
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    config = OmegaConf.create({"data": {"cv_splits_dir": str(splits), "data_dir": str(tmp_path / "images")}})

    workflow._write_data_manifest(run_dir, config)
    manifest = json.loads((run_dir / "data_manifest.json").read_text())

    expected = hashlib.sha256(b"a_train.png\nb_train.png").hexdigest()
    assert manifest["splits"]["train"]["image_set_sha256"] == expected
    assert manifest["splits"]["train"]["num_rows"] == 2
