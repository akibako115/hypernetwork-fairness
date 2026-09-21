"""親 iterative run のローカル text log と単一 W&B run を管理する。"""

from __future__ import annotations

import logging
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from omegaconf import DictConfig


@contextmanager
def parent_wandb(config: DictConfig, run_dir: Path) -> Iterator[Any | None]:
    """親 workflow だけが W&B run を開始・終了し、失敗は呼び出し元へ送出する。

    子 stage process は logger を持たない。run を1本に保つため、開始と終了をここが所有する。

    Args:
        config: `logger.wandb` を持ちうる解決済み設定。無ければ W&B を使わない
        run_dir: 親 run directory。run 名と出力先に使う

    Yields:
        Any | None: 開始した W&B run。`logger.wandb` が無い場合は None
    """
    settings = config.get("logger", {}).get("wandb") if config.get("logger") else None
    if settings is None:
        yield None
        return
    import wandb

    run = wandb.init(
        project=settings.project,
        name=run_dir.name,
        dir=str(run_dir),
        tags=list(settings.get("tags", [])),
        mode="offline" if settings.get("offline", False) else "online",
        config={"run_dir": str(run_dir), "project": config.project, "iteration": dict(config.iteration)},
    )
    try:
        yield run
    except BaseException:
        run.finish(exit_code=1)
        raise
    else:
        run.finish(exit_code=0)


def log_stage(run: Any, stage: str, result: Mapping[str, Any]) -> None:
    """子 process が残した scalar metrics を parent W&B run へ集約する。

    Args:
        run: 集約先の W&B run
        stage: stage 名。`stage/<name>/<metric>` の接頭辞になる
        result: 子 process の result.json。`metrics` の None は送らない

    Returns:
        None
    """
    metrics = result.get("metrics", {})
    run.log({f"stage/{stage}/{name}": value for name, value in metrics.items() if value is not None})


@contextmanager
def text_log(run_dir: Path) -> Iterator[Path]:
    """親 workflow の標準 logging を run-local `logs/train.log` に複製する。

    Args:
        run_dir: 親 run directory。`logs/train.log` をこの下に作る

    Yields:
        Path: 書き込み先の log file path
    """
    log_dir = run_dir / "logs"
    log_dir.mkdir(exist_ok=True)
    path = log_dir / "train.log"
    handler = logging.FileHandler(path, encoding="utf-8")
    handler.setFormatter(logging.Formatter("[%(asctime)s][%(name)s][%(levelname)s] %(message)s"))
    root = logging.getLogger()
    root.addHandler(handler)
    try:
        yield path
    finally:
        root.removeHandler(handler)
        handler.close()
