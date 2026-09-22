"""親 iterative run のローカル text log と単一 W&B run を管理する。"""

from __future__ import annotations

import csv
import logging
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from omegaconf import DictConfig, OmegaConf


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
        # group は study（この run が属する仮説）、job_type は code project。W&B 上ではこの 2 つで
        # 「どの仮説の、どちら側の run か」を絞る。config 側の値をそのまま渡す。
        group=settings.get("group"),
        job_type=settings.get("job_type"),
        name=run_dir.name,
        dir=str(run_dir),
        tags=list(settings.get("tags", [])),
        mode="offline" if settings.get("offline", False) else "online",
        # 解決済み設定をそのまま載せる。ここに入れた値だけが W&B の絞り込み列になるので、
        # 拾う key を選ぶと「その条件では並べられない run」が後から出る。
        config={**OmegaConf.to_container(config, resolve=True), "run_dir": str(run_dir)},
    )
    try:
        yield run
    except BaseException:
        run.finish(exit_code=1)
        raise
    else:
        run.finish(exit_code=0)


def read_epoch_metrics(csv_path: Path) -> list[dict[str, float]]:
    """子 process の `metrics.csv` を epoch ごとの1件へまとめる。

    `CSVLogger` は `log_metrics` 呼び出しごとに1行書くため、同じ epoch の train と val が
    別行に分かれる。空セルはその行が書かなかった指標を表すので落とし、epoch 単位で1つに
    まとめ直す。`step` 列は optimizer step であり、親が振る通し epoch と混ざるので除く。

    Args:
        csv_path: `stages/<name>/metrics/metrics.csv`

    Returns:
        list[dict[str, float]]: epoch 昇順の metric。`epoch` キーは stage 内の epoch 番号

    Raises:
        FileNotFoundError: CSV が無い場合。子が metric を残さずに終えたことを意味する。
    """
    if not csv_path.is_file():
        raise FileNotFoundError(f"stage の epoch metric が見つからない: {csv_path}")
    by_epoch: dict[int, dict[str, float]] = {}
    with csv_path.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            epoch = int(float(row["epoch"]))
            merged = by_epoch.setdefault(epoch, {"epoch": float(epoch)})
            for name, value in row.items():
                if name in ("epoch", "step") or value in (None, ""):
                    continue
                merged[name] = float(value)
    return [by_epoch[epoch] for epoch in sorted(by_epoch)]


def log_epoch_metrics(run: Any, stage_index: int, rows: Sequence[Mapping[str, float]], offset: int) -> int:
    """stage 1本分の epoch metric を、run 全体で通し番号の step として W&B へ送る。

    stage ごとに接頭辞を付けず素のキーで送るので、`val/auroc` などは run 全体で1本の曲線に
    なる。stage の境目は `stage_index` で読む。

    Args:
        run: 集約先の W&B run
        stage_index: warmup を 0、以降の cohort stage を 1, 2, ... とした通し番号
        rows: `read_epoch_metrics` の戻り値
        offset: この stage の最初の epoch に割り当てる通し step

    Returns:
        int: 次の stage へ渡す offset（`offset + len(rows)`）
    """
    for index, row in enumerate(rows):
        payload = {name: value for name, value in row.items() if name != "epoch"}
        payload["stage_index"] = float(stage_index)
        payload["stage_epoch"] = row["epoch"]
        run.log(payload, step=offset + index)
    return offset + len(rows)


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
