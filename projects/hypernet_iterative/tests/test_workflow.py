"""parent iterative workflow の stage 接続を実データなしで検証する。"""

from pathlib import Path

import pytest
from omegaconf import OmegaConf

from projects.hypernet_iterative import workflow


def _config(tmp_path: Path):
    return OmegaConf.create(
        {
            "paths": {"project_dir": str(tmp_path)},
            "seed": 7,
            "project": "hypernet_iterative",
            "logger": {},
            "data": {"_target_": "unused", "batch_size": 2, "num_classes": 2},
            "model": {"loss_fn": {"_target_": "unused"}, "attribute_names": {}},
            "callbacks": {"model_checkpoint": {}},
            "trainer": {"max_epochs": 1},
            "iteration": {"warmup_epochs": 1, "stage_epochs": 2, "stages": 2, "clusters": 2, "n_init": 1},
        }
    )


def test_workflow_builds_a_cohort_then_warm_starts_each_stage(tmp_path: Path, monkeypatch) -> None:
    config = _config(tmp_path)
    observed_stages = []
    observed_cohorts = []

    def fake_run_stage(stage_config, stage_dir):
        observed_stages.append((stage_config, stage_dir))
        checkpoint = tmp_path / f"{stage_dir.name}.ckpt"
        checkpoint.touch()
        return {"metrics": {}, "checkpoints": {"val/auroc": {"path": str(checkpoint)}}}

    def fake_build_cohort(stage_config, *, checkpoint_path, reference_id, output_dir):
        observed_cohorts.append((stage_config, checkpoint_path, reference_id, output_dir))
        output_dir.mkdir(parents=True)
        assignment = output_dir / "assignments.parquet"
        assignment.touch()
        return assignment

    monkeypatch.setattr(workflow, "run_stage", fake_run_stage)
    monkeypatch.setattr(workflow, "build_cohort", fake_build_cohort)
    monkeypatch.setattr(workflow, "write_data_manifest", lambda *_: None)
    monkeypatch.setattr(workflow, "write_preflight", lambda *_: None)

    run_dir = workflow.run_iterative(config)

    assert [directory.name for _, directory in observed_stages] == ["warmup", "stage01", "stage02"]
    assert [reference for _, _, reference, _ in observed_cohorts] == ["warmup", "stage01"]
    assert observed_stages[1][0].model.warm_start_checkpoint_path.endswith("warmup.ckpt")
    assert observed_stages[2][0].model.warm_start_checkpoint_path.endswith("stage01.ckpt")
    assert observed_stages[1][0].data.group_assignment_path.endswith("cohort01/assignments.parquet")
    assert observed_stages[1][0].trainer.max_epochs == 2
    record = __import__("json").loads((run_dir / "run.json").read_text())
    assert record["status"] == "succeeded"
    assert set(record["stages"]) == {"warmup", "cohort01", "stage01", "cohort02", "stage02"}


def test_uniform_group_iterative_warm_starts_each_stage(tmp_path: Path) -> None:
    config = _config(tmp_path)
    config.iteration.cohort_training_strategy = "uniform_group_iterative"

    stage = workflow.cohort_stage_config(
        config,
        assignment_path=tmp_path / "assignments.parquet",
        checkpoint_path=tmp_path / "reference.ckpt",
        reference_id="warmup",
    )

    assert stage.model.loss_fn._target_.endswith("UniformGroupTaskLoss")
    assert stage.training_strategy.supports_warm_start is True
    assert stage.model.warm_start_checkpoint_path.endswith("reference.ckpt")


def test_uniform_group_restarts_each_stage_from_the_backbone_initialization(tmp_path: Path) -> None:
    config = _config(tmp_path)
    config.iteration.cohort_training_strategy = "uniform_group"

    stage = workflow.cohort_stage_config(
        config,
        assignment_path=tmp_path / "assignments.parquet",
        checkpoint_path=tmp_path / "reference.ckpt",
        reference_id="warmup",
    )

    assert stage.training_strategy.supports_warm_start is False
    assert stage.model.warm_start_checkpoint_path is None


@pytest.mark.parametrize("key", ["warmup_epochs", "stage_epochs", "stages", "clusters", "n_init"])
def test_run_iterative_rejects_non_positive_iteration_counts_before_reserving_a_run(tmp_path: Path, key: str) -> None:
    """GPU 時間を使う前に落ちること。`plan()` は dry-run からしか呼ばれない。"""
    config = _config(tmp_path)
    config.iteration[key] = 0

    with pytest.raises(ValueError, match="iteration counts must be positive"):
        workflow.run_iterative(config)

    assert not (tmp_path / "runs").exists()


def test_data_manifest_records_all_splits_and_input_identity(tmp_path: Path) -> None:
    split_dir = tmp_path / "splits"
    split_dir.mkdir()
    for name in ("train", "val", "test"):
        (split_dir / f"{name}.csv").write_text("image,target\na.png,0\nb.png,1\n")
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    config = OmegaConf.create({"data": {"data_dir": str(tmp_path / "images"), "cv_splits_dir": str(split_dir)}})

    workflow.write_data_manifest(run_dir, config)

    manifest = __import__("json").loads((run_dir / "data_manifest.json").read_text())
    assert set(manifest["splits"]) == {"train", "val", "test"}
    assert manifest["splits"]["train"]["num_rows"] == 2
    assert len(manifest["splits"]["train"]["sha256"]) == 64


def test_cohort_stage_config_supports_the_declared_strategy_and_selection(tmp_path: Path) -> None:
    config = _config(tmp_path)
    config.iteration.cohort_training_strategy = "group_dro_balanced"
    config.iteration.cohort_checkpoint_selection = "hidden_min_auroc"

    stage = workflow.cohort_stage_config(
        config,
        assignment_path=tmp_path / "assignments.parquet",
        checkpoint_path=tmp_path / "reference.ckpt",
        reference_id="warmup",
    )

    assert stage.model.loss_fn._target_.endswith("ClassBalancedGroupDROTaskLoss")
    assert stage.model.loss_fn.num_classes == 2
    assert stage.checkpoint_selection.name == "hidden_min_auroc"
    assert stage.callbacks.hidden_min_auroc_checkpoint.monitor == "val/hidden_min_auroc"


def test_inverse_weighting_resolves_the_class_weight_once_and_carries_it_into_every_stage(tmp_path: Path, monkeypatch) -> None:
    split_dir = tmp_path / "splits"
    split_dir.mkdir()
    (split_dir / "train.csv").write_text("image,target\na.png,0\nb.png,1\nc.png,1\nd.png,1\n")
    config = _config(tmp_path)
    config.weighting = "inverse"
    config.data.cv_splits_dir = str(split_dir)
    observed_stages = []

    def fake_run_stage(stage_config, stage_dir):
        observed_stages.append(stage_config)
        checkpoint = tmp_path / f"{stage_dir.name}.ckpt"
        checkpoint.touch()
        return {"metrics": {}, "checkpoints": {"val/auroc": {"path": str(checkpoint)}}}

    def fake_build_cohort(stage_config, *, checkpoint_path, reference_id, output_dir):
        output_dir.mkdir(parents=True)
        assignment = output_dir / "assignments.parquet"
        assignment.touch()
        return assignment

    monkeypatch.setattr(workflow, "run_stage", fake_run_stage)
    monkeypatch.setattr(workflow, "build_cohort", fake_build_cohort)
    monkeypatch.setattr(workflow, "write_data_manifest", lambda *_: None)
    monkeypatch.setattr(workflow, "write_preflight", lambda *_: None)

    run_dir = workflow.run_iterative(config)

    assert [list(stage.model.loss_fn.class_weight) for stage in observed_stages] == [[1.5, 0.5]] * 3
    # 親 run の config.yaml も解決済みの重みを持つ。後から run を読む側が再計算しなくてよい。
    assert list(OmegaConf.load(run_dir / "config.yaml").model.loss_fn.class_weight) == [1.5, 0.5]


def test_inverse_frequency_weights_reject_missing_class_and_matches_old_normalization() -> None:
    assert workflow._inverse_frequency_weights([0, 1, 1, 1], 2) == [1.5, 0.5]
    with pytest.raises(ValueError, match="必要"):
        workflow._inverse_frequency_weights([0, 0], 2)


def test_weighting_none_leaves_the_class_weight_untouched(tmp_path: Path) -> None:
    config = _config(tmp_path)

    workflow._resolve_inverse_class_weights(config)

    assert "class_weight" not in config.model.loss_fn
