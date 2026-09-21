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
            "iteration": {"warmup_epochs": 1, "stage_epochs": 2, "stages": 2, "clusters": 2, "n_init": 1, "group_dro_step_size": 0.01},
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
    config.iteration.cohort_training_strategy = "group_dro"
    config.iteration.cohort_checkpoint_selection = "hidden_min_auroc"

    stage = workflow.cohort_stage_config(
        config,
        assignment_path=tmp_path / "assignments.parquet",
        checkpoint_path=tmp_path / "reference.ckpt",
        reference_id="warmup",
    )

    assert stage.model.loss_fn._target_.endswith("GroupDROTaskLoss")
    assert stage.model.loss_fn.step_size == config.iteration.group_dro_step_size
    assert stage.checkpoint_selection.name == "hidden_min_auroc"
    assert stage.callbacks.hidden_min_auroc_checkpoint.monitor == "val/hidden_min_auroc"


def test_cohort_stage_does_not_inherit_the_warmup_class_weight(tmp_path: Path) -> None:
    """warmup の全体重みは cohort stage へ渡らない。渡ると陽性率依存が残る。"""
    config = _config(tmp_path)
    config.model.loss_fn.class_weight = [0.201358, 1.798642]

    inherited = workflow.cohort_stage_config(
        config,
        assignment_path=tmp_path / "assignments.parquet",
        checkpoint_path=tmp_path / "reference.ckpt",
        reference_id="warmup",
    )
    solved = workflow.cohort_stage_config(
        config,
        assignment_path=tmp_path / "assignments.parquet",
        checkpoint_path=tmp_path / "reference.ckpt",
        reference_id="warmup",
        class_weight=[[0.6, 3.0], [0.55, 5.5]],
    )

    assert inherited.model.loss_fn.class_weight is None
    assert OmegaConf.to_container(solved.model.loss_fn.class_weight) == [[0.6, 3.0], [0.55, 5.5]]


def test_inverse_frequency_weights_reject_missing_class_and_matches_old_normalization() -> None:
    assert workflow._inverse_frequency_weights([0, 1, 1, 1], 2) == [1.5, 0.5]
    with pytest.raises(ValueError, match="必要"):
        workflow._inverse_frequency_weights([0, 0], 2)


def test_weighting_none_leaves_the_class_weight_untouched(tmp_path: Path) -> None:
    config = _config(tmp_path)

    workflow._resolve_inverse_class_weights(config)

    assert "class_weight" not in config.model.loss_fn


def _cohort_split(tmp_path: Path, *, targets: list[int], groups: list[int]) -> tuple[Path, Path]:
    """train.csv と同じ image 列を持つ cohort sidecar を書き出す。"""
    import pandas as pd

    split_dir = tmp_path / "splits"
    split_dir.mkdir(exist_ok=True)
    images = [f"{index}.png" for index in range(len(targets))]
    pd.DataFrame({"image": images, "target": targets}).to_csv(split_dir / "train.csv", index=False)
    assignment_path = tmp_path / "assignments.parquet"
    pd.DataFrame({"split": ["train"] * len(images), "image": images, "group_id": groups}).to_parquet(assignment_path)
    return split_dir, assignment_path


def test_group_class_weights_are_the_inverse_of_each_cohort_cell_frequency(tmp_path: Path) -> None:
    """w[g, c] = 1 / (C * f_{g,c})。group 内で標本平均が 1 になるので group 間の尺度が揃う。"""
    split_dir, assignment_path = _cohort_split(
        tmp_path,
        targets=[0, 0, 1, 1, 0, 0, 0, 1],
        groups=[0, 0, 0, 0, 1, 1, 1, 1],
    )
    config = _config(tmp_path)
    config.data.cv_splits_dir = str(split_dir)

    weights = workflow.resolve_group_class_weights(config, assignment_path)

    # group 0 は 2陰性 + 2陽性、group 1 は 3陰性 + 1陽性。
    assert weights == [[1.0, 1.0], [0.666667, 2.0]]


def test_group_class_weights_reject_an_empty_cohort_cell(tmp_path: Path) -> None:
    """空セルは重みが発散する。黙って落とさず、学習を始める前に止める。"""
    split_dir, assignment_path = _cohort_split(
        tmp_path,
        targets=[0, 0, 1, 1, 0, 0, 0, 0],
        groups=[0, 0, 0, 0, 1, 1, 1, 1],
    )
    config = _config(tmp_path)
    config.data.cv_splits_dir = str(split_dir)

    with pytest.raises(ValueError, match="空の \\(group, class\\) セル"):
        workflow.resolve_group_class_weights(config, assignment_path)


def test_inverse_weighting_resolves_a_group_weight_for_every_cohort(tmp_path: Path, monkeypatch) -> None:
    """cohort を作り直すたびに group ごとの重みを解き直し、warmup だけ全体重みを使う。"""
    split_dir, assignment_path = _cohort_split(
        tmp_path,
        targets=[0, 0, 1, 1, 0, 0, 0, 1],
        groups=[0, 0, 0, 0, 1, 1, 1, 1],
    )
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
        return assignment_path

    monkeypatch.setattr(workflow, "run_stage", fake_run_stage)
    monkeypatch.setattr(workflow, "build_cohort", fake_build_cohort)
    monkeypatch.setattr(workflow, "write_data_manifest", lambda *_: None)
    monkeypatch.setattr(workflow, "write_preflight", lambda *_: None)

    run_dir = workflow.run_iterative(config)

    warmup, *stages = [OmegaConf.to_container(stage.model.loss_fn.class_weight) for stage in observed_stages]
    assert warmup == [0.75, 1.25]
    assert stages == [[[1.0, 1.0], [0.666667, 2.0]]] * 2
    # 親 run の config.yaml も解決済みの重みを持つ。後から run を読む側が再計算しなくてよい。
    assert list(OmegaConf.load(run_dir / "config.yaml").model.loss_fn.class_weight) == [0.75, 1.25]


@pytest.mark.parametrize("entry_point", [workflow.plan, workflow.run_iterative])
def test_unsupported_weighting_is_rejected_by_the_dry_run_and_the_real_run(tmp_path: Path, entry_point) -> None:
    """dry-run でも打ち間違いが出ること。実 run では run directory を作る前に落ちること。"""
    config = _config(tmp_path)
    config.weighting = "balanced"

    with pytest.raises(ValueError, match="unsupported weighting"):
        entry_point(config)

    assert not (tmp_path / "runs").exists()
