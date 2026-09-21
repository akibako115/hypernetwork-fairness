"""run 単位の text log と experiment logger 接続を検証する。"""

import logging
from pathlib import Path
from types import SimpleNamespace

from omegaconf import OmegaConf

from projects.hypernet_two_stage import run_logging


class WandbLogger:
    """型名だけで参照を取り出す経路を確認するための代役。"""

    def __init__(self) -> None:
        self.experiment = SimpleNamespace(name="run-name", id="abc123", url="https://wandb.ai/run")


def test_text_log_writes_into_the_run_directory_and_detaches_afterwards(tmp_path: Path) -> None:
    logger = logging.getLogger("projects.hypernet_two_stage.tests.text_log")
    logger.setLevel(logging.INFO)
    root = logging.getLogger()
    previous_level = root.level
    root.setLevel(logging.INFO)
    handlers_before = list(root.handlers)
    try:
        with run_logging.text_log(tmp_path / "logs") as path:
            logger.info("inside the run")
        logger.info("after the run")
    finally:
        root.setLevel(previous_level)

    assert path == tmp_path / "logs" / "train.log"
    contents = path.read_text()
    assert "inside the run" in contents
    assert "after the run" not in contents
    assert root.handlers == handlers_before


def test_text_log_captures_lightning_records_that_do_not_propagate_to_root(tmp_path: Path) -> None:
    lightning = logging.getLogger("lightning.pytorch")
    previous_propagate = lightning.propagate
    previous_level = lightning.level
    handlers_before = list(lightning.handlers)
    lightning.propagate = False
    lightning.setLevel(logging.INFO)
    try:
        with run_logging.text_log(tmp_path / "logs") as path:
            lightning.info("`Trainer.fit` stopped")
        lightning.info("after the run")
    finally:
        lightning.propagate = previous_propagate
        lightning.setLevel(previous_level)

    contents = path.read_text()
    assert "`Trainer.fit` stopped" in contents
    assert "after the run" not in contents
    assert lightning.handlers == handlers_before


def test_inject_logger_outputs_sets_the_reserved_directory_and_keeps_explicit_names() -> None:
    config = OmegaConf.create(
        {
            "logger": {
                "wandb": {"save_dir": None, "name": None},
                "csv": {"save_dir": None, "name": "csv/"},
                "plain": {"project": "no-save-dir"},
            }
        }
    )

    run_logging.inject_logger_outputs(config, Path("/runs/run-1"), "run-1")

    assert config.logger.wandb.save_dir == "/runs/run-1"
    assert config.logger.wandb.name == "run-1"
    assert config.logger.csv.save_dir == "/runs/run-1"
    assert config.logger.csv.name == "csv/"
    assert "save_dir" not in config.logger.plain


def test_inject_logger_outputs_accepts_a_config_without_a_logger_group() -> None:
    config = OmegaConf.create({"logger": {}})

    run_logging.inject_logger_outputs(config, Path("/runs/run-1"), "run-1")

    assert config.logger == {}


def test_experiment_loggers_yields_nothing_for_an_empty_group() -> None:
    with run_logging.experiment_loggers(OmegaConf.create({"logger": {}})) as loggers:
        assert loggers == []

    assert run_logging.as_trainer_loggers([]) is False


def test_experiment_loggers_instantiates_the_group_and_closes_wandb(monkeypatch) -> None:
    created = []
    finished = []
    monkeypatch.setattr(run_logging, "instantiate", lambda value: created.append(value) or WandbLogger())
    monkeypatch.setattr(run_logging, "_finish_wandb", lambda: finished.append(True))
    config = OmegaConf.create({"logger": {"wandb": {"_target_": "unused"}}})

    with run_logging.experiment_loggers(config) as loggers:
        assert len(loggers) == 1
        assert run_logging.as_trainer_loggers(loggers) == loggers

    assert len(created) == 1
    assert finished == [True]


def test_experiment_loggers_closes_wandb_when_the_body_raises(monkeypatch) -> None:
    finished = []
    monkeypatch.setattr(run_logging, "_finish_wandb", lambda: finished.append(True))

    try:
        with run_logging.experiment_loggers(OmegaConf.create({"logger": {}})):
            raise RuntimeError("fit failed")
    except RuntimeError:
        pass

    assert finished == [True]


def test_logger_references_keeps_only_wandb_runs() -> None:
    references = run_logging.logger_references([WandbLogger(), SimpleNamespace(experiment=None)])

    assert references == [{"logger": "WandbLogger", "name": "run-name", "id": "abc123", "url": "https://wandb.ai/run"}]
