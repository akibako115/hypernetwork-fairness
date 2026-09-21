"""run directory 配下に text log と experiment logger を接続する。"""

from __future__ import annotations

import logging
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from importlib.util import find_spec
from pathlib import Path
from typing import Any

from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf

_FILE_LOG_FORMAT = "[%(asctime)s][%(name)s][%(levelname)s] - %(message)s"
_DETACHED_LOGGERS = ("lightning.pytorch", "lightning.fabric")
_MISSING = object()


def inject_logger_outputs(config: DictConfig, output_dir: Path, run_name: str) -> None:
    """experiment logger の保存先と run 名を、予約済みの出力先に合わせる。

    ``save_dir`` を持つ logger は出力先を run directory に向ける。``name`` が ``null`` の logger
    だけ ``run_name`` を入れ、config が明示した名前は保つ。設定の書き換えは config 保存より前に
    行い、``config.yaml`` から実際の保存先を読めるようにする。

    Args:
        config: 書き換え対象の設定。`logger` group を in-place で更新する
        output_dir: `save_dir` を持つ logger に入れる出力先
        run_name: `name` が null の logger にだけ入れる run 名

    Returns:
        None
    """
    loggers = OmegaConf.select(config, "logger", default=_MISSING)
    if loggers is _MISSING or not loggers:
        return
    for key in loggers:
        if OmegaConf.select(config, f"logger.{key}.save_dir", default=_MISSING) is not _MISSING:
            OmegaConf.update(config, f"logger.{key}.save_dir", str(output_dir), merge=False)
        if OmegaConf.select(config, f"logger.{key}.name", default=_MISSING) is None:
            OmegaConf.update(config, f"logger.{key}.name", run_name, merge=False)


@contextmanager
def text_log(log_dir: Path) -> Iterator[Path]:
    """実行ログを ``log_dir/train.log`` に残す。

    Hydra 自身の file handler は repository root へ書くため、run ごとの実行ログはこの handler が
    所有する。handler は文脈を抜けるときに必ず外し、後続の run のログを混ぜない。

    Args:
        log_dir: `train.log` を作る directory。無ければ作る

    Yields:
        Path: 書き込み先の log file path
    """
    log_dir.mkdir(parents=True, exist_ok=True)
    path = log_dir / "train.log"
    handler = logging.FileHandler(path, encoding="utf-8")
    handler.setFormatter(logging.Formatter(_FILE_LOG_FORMAT))
    targets = [logging.getLogger(), *_detached_loggers()]
    for target in targets:
        target.addHandler(handler)
    try:
        yield path
    finally:
        for target in targets:
            target.removeHandler(handler)
        handler.close()


def _detached_loggers() -> list[logging.Logger]:
    """root へ伝播しない Lightning の logger を返す。

    Lightning は root に handler が無い状態で import されると自分の logger の伝播を切る。Hydra は
    job 開始時に handler を設定するので、import が先行するこの構成では trainer の記録が root に
    届かない。train.log から fit の経過が消えないよう、その logger にも直接 handler を足す。
    """
    loggers = [logging.getLogger(name) for name in _DETACHED_LOGGERS]
    return [logger for logger in loggers if not logger.propagate]


@contextmanager
def experiment_loggers(config: DictConfig) -> Iterator[list[Any]]:
    """`logger` group から Lightning logger を作り、終了時に wandb run を閉じる。

    group が空（``logger=none``）なら空 list を返す。例外で終わった fit でも wandb run を
    開いたままにしない。

    Args:
        config: `logger` group を持ちうる解決済み設定

    Yields:
        list[Any]: 構築した Lightning logger。group が空なら空 list
    """
    group = config.get("logger")
    loggers = [instantiate(value) for value in group.values()] if group else []
    try:
        yield loggers
    finally:
        _finish_wandb()


def logger_references(loggers: Sequence[Any]) -> list[dict[str, str]]:
    """wandb run を後から辿るための name・id・url を返す。

    wandb を import せずに済ませるため型名で判定する。参照の取得は experiment を初期化するので、
    fit が失敗した run でも dashboard への link が run.json に残る。

    Args:
        loggers: `experiment_loggers` が返した logger

    Returns:
        list[dict[str, str]]: WandbLogger ごとの `logger` / `name` / `id` / `url`。
            wandb を使わない run では空 list
    """
    references = []
    for logger in loggers:
        if type(logger).__name__ != "WandbLogger":
            continue
        experiment = logger.experiment
        references.append({"logger": "WandbLogger", "name": experiment.name, "id": experiment.id, "url": experiment.url})
    return references


def _finish_wandb() -> None:
    """開いている wandb run を閉じる。wandb が無い環境では何もしない。"""
    if find_spec("wandb") is None:
        return
    import wandb

    if wandb.run is not None:
        wandb.finish()


def as_trainer_loggers(loggers: Sequence[Any]) -> list[Any] | bool:
    """Lightning Trainer の ``logger`` 引数に渡す値を返す。空なら logging を無効にする。

    Args:
        loggers: `experiment_loggers` が返した logger

    Returns:
        list[Any] | bool: logger の list。空の場合は logging を切る False
    """
    return list(loggers) if loggers else False
