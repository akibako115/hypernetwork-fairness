"""run が残した artifact の読み方を 1 箇所に集める。

`run.json` / `config.yaml` / `metrics/metrics.csv` または `stages/*/metrics/metrics.csv` / W&B の transaction log は
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
        run_dir: `analysis/<study>/runs/<run-id>`

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
        csv_path: `metrics/metrics.csv` または `stages/<name>/metrics/metrics.csv`

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

    CSVLogger を付けずに回した過去の run では、epoch 推移がローカルに残る場所が
    `wandb/<run>/run-*.wandb` しかない。この file は leveldb 形式の record 列で、
    `history` record 1 つが `log()` 1 回に対応する。Lightning は train と val を別の
    record に書くので、`read_epoch_metrics` と同じく epoch で束ね直す。

    system stats や telemetry など history 以外の record も同じ file に混ざるので、
    history だけを拾う。`scan_record` ではなく `scan_data` を使う。block 境界をまたいだ
    record は fragment に分かれて書かれており、`scan_record` はそれを繋がないまま返すため
    parse に失敗する。

    Args:
        run_dir: `analysis/<study>/runs/<run-id>`

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


_PROJECT_BY_KIND = {"fit": "hypernet_e2e", "iterative_fit": "hypernet_iterative"}


def run_project(run_dir: Path) -> str:
    """その run を回した project を返す。

    package の `runs/` には両 project の run が並ぶので、置き場所の path からは project を
    読めない。`run.json` の `kind` は project ごとに固定の値なので、そこから引く。

    Args:
        run_dir: `analysis/<study>/runs/<run-id>` または `projects/<project>/runs/<run-id>`

    Returns:
        str: `hypernet_e2e` / `hypernet_iterative`

    Raises:
        ValueError: `kind` が既知の値でない場合
    """
    kind = read_run_record(run_dir).get("kind")
    if kind not in _PROJECT_BY_KIND:
        raise ValueError(f"{run_dir} の run.json の kind {kind!r} から project を決められない")
    return _PROJECT_BY_KIND[kind]


DEFAULT_CHECKPOINT = "best"


def _within_run(run_dir: Path, recorded: str) -> Path:
    # 記録された path は実行環境（container）のものなので、run directory からの相対位置だけを使う。
    parts = Path(recorded).parts
    return run_dir.joinpath(*parts[parts.index(run_dir.name) + 1 :])


def _best_val_auroc_checkpoint(run_dir: Path) -> Path:
    record = read_run_record(run_dir)
    if "selected_checkpoint" in record:
        return _within_run(run_dir, record["selected_checkpoint"]["path"])
    for entry in record.get("checkpoints") or []:
        if entry.get("monitor") == "val/auroc" and (entry.get("best") or {}).get("path"):
            return _within_run(run_dir, entry["best"]["path"])
    # checkpoint の記録を持たない run は、ファイル名から一意に決まるときだけ受け付ける。
    checkpoints = sorted((run_dir / "checkpoints").glob("best_val_auroc_*.ckpt"))
    if len(checkpoints) != 1:
        found = len(checkpoints)
        raise FileNotFoundError(f"{run_dir} の best_val_auroc checkpoint は 1 つ必要だが、{found} 件見つかった")
    return checkpoints[0]


def selected_checkpoint(run_dir: Path, name: str = DEFAULT_CHECKPOINT) -> Path:
    """評価する checkpoint を名前で選ぶ。

    `"best"`（既定）は、その run が val AUROC で選んだ checkpoint。iterative は `run.json` の
    `selected_checkpoint`（最終 stage の best）、単段 run は `run.json` の `checkpoints` のうち monitor が
    `val/auroc` のものの best。全 epoch を残した run では `best_val_auroc_*.ckpt` が epoch の数だけあり、
    ファイル名では best を決められないので、記録を正とする。

    それ以外の名前 `xxx` は、best と同じ directory の `xxx.ckpt` を指す。例：`swad`（SWAD の平均重み）、
    `last`、`best_val_worst_sexage_012`、`best_val_auroc_009`（全 epoch を残した run の特定の epoch）。

    Args:
        run_dir: `analysis/<study>/runs/<run-id>`
        name: `"best"` か、拡張子を除いた checkpoint のファイル名。

    Returns:
        Path: 評価する checkpoint

    Raises:
        FileNotFoundError: best が 1 つに定まらない場合、または `xxx.ckpt` が無い場合
        ValueError: name が path を含む場合
    """
    best = _best_val_auroc_checkpoint(run_dir)
    if name == DEFAULT_CHECKPOINT:
        return best
    if not name or Path(name).name != name or name.endswith(".ckpt"):
        raise ValueError(f"checkpoint はファイル名から .ckpt を除いた名前で指定する: {name!r}")
    path = best.parent / f"{name}.ckpt"
    if not path.is_file():
        available = sorted(candidate.stem for candidate in best.parent.glob("*.ckpt"))
        raise FileNotFoundError(f"{run_dir} に {path.relative_to(run_dir)} が無い。選べる名前: {available}")
    return path


def write_csv(path: Path, rows: list[dict[str, Any]], columns: list[str]) -> None:
    """指定した列だけを CSV へ書き出す。欠けている列は空欄にする。

    改行は LF に固定する。csv の既定は CRLF で、`results/` は Git 追跡外なので index とは
    食い違わないが、同じ表を別の環境で作り直したときに全行が差分になるのを避ける。

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
