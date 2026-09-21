"""two-stage の Hydra preset が experiment logger と実行ログの保存先を正しく指すことを検証する。"""

from pathlib import Path

from hydra import compose, initialize_config_dir


def _compose(**kwargs: object):
    config_dir = Path(__file__).parent.parent / "configs"
    with initialize_config_dir(version_base="1.3", config_dir=str(config_dir)):
        return compose(config_name="train", **kwargs)


def test_default_logger_is_wandb_with_its_output_left_to_the_workflow() -> None:
    cfg = _compose()

    assert set(cfg.logger) == {"wandb"}
    assert cfg.logger.wandb._target_ == "lightning.pytorch.loggers.wandb.WandbLogger"
    assert cfg.logger.wandb.project == "fairness_hypernet_two_stage"
    assert cfg.logger.wandb.log_model is False
    assert cfg.logger.wandb.save_dir is None
    assert cfg.logger.wandb.name is None


def test_hydra_job_logging_keeps_no_shared_log_file_outside_the_run_directory() -> None:
    cfg = _compose(return_hydra_config=True)

    assert set(cfg.hydra.job_logging.handlers) == {"console"}
    assert list(cfg.hydra.job_logging.root.handlers) == ["console"]
