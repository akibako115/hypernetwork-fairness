"""run が残した artifact の読み方を 1 箇所に集める。

`run.json` / `config.yaml` / `stages/*/metrics/metrics.csv` / W&B の transaction log は
repo が定めた run の形であり、どの仮説から読んでも同じ意味を持つ。ここで共有するのは
その読み方までで、**群の切り方・指標の定義・図は package に閉じる**。仮説ごとの判断を
ここへ上げると、別の仮説がその判断を暗黙に引き継いでしまう。
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import yaml


def read_run_record(run_dir: Path) -> dict[str, Any]:
    """`run.json` を読む。

    Args:
        run_dir: `projects/<project>/runs/<run-id>`

    Returns:
        dict[str, Any]: run record 全体
    """
    return json.loads((run_dir / "run.json").read_text(encoding="utf-8"))


def read_config(config_path: Path) -> dict[str, Any]:
    """解決済み `config.yaml` を読む。

    run directory にも stage directory にも同じ名前で置かれているので、path を受け取る。

    Args:
        config_path: `config.yaml`

    Returns:
        dict[str, Any]: 解決済み設定
    """
    return yaml.safe_load(config_path.read_text(encoding="utf-8"))


def read_epoch_metrics(csv_path: Path) -> list[dict[str, float]]:
    """`CSVLogger` の出力を epoch ごとの 1 行へまとめる。

    train と val は別行に書かれるので epoch で束ね直す。空セルはその行が記録しなかった
    指標を表すので落とす。`step` は optimizer step であり epoch と混ざるため除く。

    Args:
        csv_path: `stages/<name>/metrics/metrics.csv`

    Returns:
        list[dict[str, float]]: epoch 昇順の metric
    """
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


def read_wandb_history(run_dir: Path) -> list[dict[str, float]]:
    """`WandbLogger` の transaction log を epoch ごとの 1 行へまとめる。

    CSVLogger を付けずに回した run では、epoch 推移がローカルに残る場所が
    `wandb/<run>/run-*.wandb` しかない。この file は leveldb 形式の record 列で、
    `history` record 1 つが `log()` 1 回に対応する。Lightning は train と val を別の
    record に書くので、`read_epoch_metrics` と同じく epoch で束ね直す。

    system stats や telemetry など history 以外の record も同じ file に混ざるので、
    history だけを拾う。`scan_record` ではなく `scan_data` を使う。block 境界をまたいだ
    record は fragment に分かれて書かれており、`scan_record` はそれを繋がないまま返すため
    parse に失敗する。

    Args:
        run_dir: `projects/<project>/runs/<run-id>`

    Returns:
        list[dict[str, float]]: epoch 昇順の metric
    """
    # wandb の import は重い。transaction log を読む run だけが払えばよい。
    from wandb.proto import wandb_internal_pb2
    from wandb.sdk.internal import datastore

    store = datastore.DataStore()
    store.open_for_scan(str(next((run_dir / "wandb").glob("run-*/run-*.wandb"))))
    by_epoch: dict[int, dict[str, float]] = {}
    while (scanned := store.scan_data()) is not None:
        record = wandb_internal_pb2.Record()
        record.ParseFromString(scanned)
        if record.WhichOneof("record_type") != "history":
            continue
        item = {}
        for entry in record.history.item:
            name = "/".join(entry.nested_key) if entry.nested_key else entry.key
            item[name] = json.loads(entry.value_json)
        merged = by_epoch.setdefault(int(item["epoch"]), {"epoch": float(item["epoch"])})
        # `_step` や `trainer/global_step` は optimizer step 側の軸なので epoch の表に混ぜない。
        metrics = {name: float(value) for name, value in item.items() if name.startswith(("train/", "val/", "test/"))}
        merged.update(metrics)
    return [by_epoch[epoch] for epoch in sorted(by_epoch)]


def selected_checkpoint(run_dir: Path) -> Path:
    """その run が「これが成果物」と記録した checkpoint を返す。

    iterative は `run.json` の `selected_checkpoint` が最終 stage の best val AUROC を指す。
    記録された path は実行環境（container）のものなので、run directory からの相対位置だけを
    使う。単段 run は stage を持たないので `checkpoints/best_val_auroc_*.ckpt` を取る。
    どちらも val AUROC で選んだ checkpoint であり、選び方は揃っている。

    Args:
        run_dir: `projects/<project>/runs/<run-id>`

    Returns:
        Path: 評価する checkpoint

    Raises:
        FileNotFoundError: checkpoint がちょうど 1 つに定まらない場合
    """
    record = read_run_record(run_dir)
    if "selected_checkpoint" in record:
        parts = Path(record["selected_checkpoint"]["path"]).parts
        return run_dir.joinpath(*parts[parts.index(run_dir.name) + 1 :])
    checkpoints = sorted((run_dir / "checkpoints").glob("best_val_auroc_*.ckpt"))
    if len(checkpoints) != 1:
        found = len(checkpoints)
        raise FileNotFoundError(f"{run_dir} の best_val_auroc checkpoint は 1 つ必要だが、{found} 件見つかった")
    return checkpoints[0]


def write_csv(path: Path, rows: list[dict[str, Any]], columns: list[str]) -> None:
    """指定した列だけを CSV へ書き出す。欠けている列は空欄にする。

    改行は LF に固定する。csv の既定は CRLF で、Git は commit 時に LF へ正規化するため、
    既定のままだと書き出すたびに working tree と index が食い違う。

    Args:
        path: 出力先
        rows: 書き出す行
        columns: 出力する列の順序

    Returns:
        None
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore", lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
