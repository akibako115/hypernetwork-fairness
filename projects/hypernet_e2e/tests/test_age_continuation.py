"""実験名を変えても継続学習の親検証を回避できないことを固定する。"""

from pathlib import Path

import pytest
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

from projects.hypernet_e2e.training import _prepare_age_continuation


def _config(method="minimax_auc"):
    with initialize_config_dir(version_base="1.3", config_dir=str(Path(__file__).parents[1] / "configs")):
        return compose(config_name="train", overrides=[f"experiment=resnet_chexpert_from_resnet_age_{method}_race_known", "study=minimax_auc_vs_eo", "experiment_name=custom-name"])


def _parent(tmp_path, seed=42, strategy="erm"):
    path = tmp_path / "parent" / "checkpoints" / "best.ckpt"
    path.parent.mkdir(parents=True)
    path.touch()
    OmegaConf.save(OmegaConf.create({"seed": seed, "training_strategy": {"name": strategy}}), path.parent.parent / "config.yaml")
    return path


def test_custom_name_still_requires_parent_and_checks_precision():
    config = _config()
    with pytest.raises(ValueError, match="親 ERM checkpoint が必要"):
        _prepare_age_continuation(config)
    _prepare_age_continuation(config, dry_run=True)
    config.trainer.precision = "16-mixed"
    with pytest.raises(ValueError, match="bf16/float32"):
        _prepare_age_continuation(config, dry_run=True)


@pytest.mark.parametrize("invalid", ["missing_config", "seed", "missing_seed", "strategy", "missing_checkpoint"])
def test_parent_provenance_must_be_verifiable(tmp_path, invalid):
    config = _config()
    path = _parent(tmp_path)
    config.model.backbone_checkpoint_path = str(path)
    parent_config = path.parent.parent / "config.yaml"
    if invalid == "missing_config":
        parent_config.unlink()
    elif invalid == "missing_checkpoint":
        path.unlink()
    else:
        parent = OmegaConf.load(parent_config)
        if invalid == "seed":
            parent.seed = 43
        elif invalid == "missing_seed":
            del parent.seed
        else:
            parent.training_strategy.name = "minimax_auc"
        OmegaConf.save(parent, parent_config)
    with pytest.raises(ValueError):
        _prepare_age_continuation(config, dry_run=True)


def test_verified_parent_and_logger_group_survive_renaming(tmp_path):
    config = _config()
    config.model.backbone_checkpoint_path = str(_parent(tmp_path))
    _prepare_age_continuation(config)
    assert config.logger.wandb.group == "maeo_age_minimax_auc_eta0.01_tau0"


def test_fair_dro_logger_group_names_radius(tmp_path):
    config = _config("fair_dro")
    config.model.backbone_checkpoint_path = str(_parent(tmp_path))
    config.model.loss_fn.radius = 0.3
    _prepare_age_continuation(config)
    assert config.logger.wandb.group == "maeo_age_fair_dro_rho0.3"
