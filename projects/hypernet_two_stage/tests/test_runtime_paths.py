"""実行時 path が current working directory に依存しないことを検証する。"""

from pathlib import Path

from omegaconf import OmegaConf

from projects.hypernet_two_stage.runtime_paths import normalize_runtime_paths, repository_root


def test_normalize_runtime_paths_resolves_relative_paths_from_repository_root(tmp_path: Path, monkeypatch) -> None:
    config = OmegaConf.create(
        {
            "paths": {"project_dir": "projects/hypernet_two_stage", "data_dir": "data"},
            "data": {"data_dir": "data/chexpert/images", "cv_splits_dir": "data/chexpert/splits"},
            "stage1": {"model": {"backbone_checkpoint_path": "artifacts/backbone.ckpt"}},
            "stage2": {"model": {"backbone_checkpoint_path": None}},
        }
    )
    monkeypatch.chdir(tmp_path)

    normalize_runtime_paths(config)

    root = repository_root()
    assert config.paths.project_dir == str(root / "projects/hypernet_two_stage")
    assert config.paths.data_dir == str(root / "data")
    assert config.data.data_dir == str(root / "data/chexpert/images")
    assert config.data.cv_splits_dir == str(root / "data/chexpert/splits")
    # stage ごとに model を持つので、checkpoint path も stage ごとに正規化する。
    assert config.stage1.model.backbone_checkpoint_path == str(root / "artifacts/backbone.ckpt")
    assert config.stage2.model.backbone_checkpoint_path is None


def test_normalize_runtime_paths_keeps_absolute_paths_and_null_inputs(tmp_path: Path) -> None:
    config = OmegaConf.create(
        {
            "paths": {"project_dir": str(tmp_path)},
            "data": {"data_dir": str(tmp_path / "images")},
            "stage2": {"model": {"backbone_checkpoint_path": None}},
        }
    )

    normalize_runtime_paths(config)

    assert config.paths.project_dir == str(tmp_path)
    assert config.data.data_dir == str(tmp_path / "images")
    assert config.stage2.model.backbone_checkpoint_path is None


def test_repository_root_is_stable_under_a_symlinked_entry_point() -> None:
    root = repository_root()

    assert root.is_absolute()
    assert root == root.resolve()
    assert (root / "projects" / "hypernet_two_stage").is_dir()
