"""予測 cache から、run × checkpoint × split ごとの指標と、患者単位 bootstrap の指標を書き出す。

入力は `runs.md` に載せた run の予測 cache（`cache/<run-id>_<ckpt hash>_<split hash>_<split>.npz`）。
bootstrap は患者の resample を全 run で共通にして、条件間の差の区間を notebook で取れるようにする。
重い（AUROC を resample ごとに 9 回ずつ計算する）ので notebook から外した。

出力:
- `results/run_metrics.csv`：1 行 = 1 run × checkpoint（best / last）× split。指標は `grid_metrics.summary`
- `cache/bootstrap_<split>.npz`：`keys`（`<run-id>|<checkpoint>`）、`metrics`（指標名）、
  `values`（`[run × checkpoint, resample, 指標]`）

使い方:
    uv run python analysis/hypernet_conditioning_grid/compute.py --split test --resamples 200
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd
import rootutils

ROOT = rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from analysis.common.predictions import read_metadata  # noqa: E402
from analysis.common.run_artifacts import read_run_record, write_csv  # noqa: E402
from analysis.hypernet_conditioning_grid.grid_metrics import (  # noqa: E402
    load_split,
    patient_bootstrap_weights,
    positive_probability,
    summary,
)

PACKAGE = ROOT / "analysis" / "hypernet_conditioning_grid"
RUN_ID = re.compile(r"`(\d{8}T\d{6}Z-([^`]+?)-s(4[234])-[0-9a-f]{4})`")


def condition(experiment: str) -> dict[str, str]:
    """preset 名（run ID の中ほど）から、学習手順・モデル・目的関数を読む。

    Args:
        experiment: 例 `spatial-lora-chexpert-fc-from-resnet-age-race-eo-rate-dual-holdout`

    Returns:
        dict[str, str]: `procedure`・`model`・`objective`
    """
    procedure = "two_stage" if "from-resnet" in experiment else "one_stage"
    if experiment.startswith("spatial-lora-chexpert-fc"):
        model = "spatial_lora_fc"
    elif experiment.startswith("spatial-lora"):
        model = "spatial_lora"
    else:
        model = "resnet"
    # Cotter et al. の形（各群の FPR を全体と比べる）で回し直した EO。全ペアの eo_rate とは別の目的関数として扱う。
    if "eo-fpr-overall" in experiment:
        objective = "eo_fpr_overall"
    elif "eo-rate" in experiment:
        objective = "eo_rate"
    elif "group-dro" in experiment:
        objective = "group_dro"
    else:
        objective = "erm"
    return {"procedure": procedure, "model": model, "objective": objective}


def runs() -> pd.DataFrame:
    """`runs.md` に載せた CheXpert の run を、条件の列つきで返す。

    `runs.md` には HAM10000 の run も載るので、run ID の experiment 名で CheXpert の run だけを拾う。

    Args:
        なし

    Returns:
        pd.DataFrame: `run_id`・`seed`・`procedure`・`model`・`objective`
    """
    text = (PACKAGE / "runs.md").read_text()
    rows = [
        {"run_id": run_id, "seed": int(seed), **condition(experiment)}
        for run_id, experiment, seed in RUN_ID.findall(text)
        if "-chexpert-" in experiment
    ]
    return pd.DataFrame(rows).drop_duplicates("run_id")


def cache_files(split: str) -> dict[tuple[str, str], Path]:
    """split の予測 cache を `(run_id, best|last)` で引けるようにする。

    Args:
        split: `val` または `test`

    Returns:
        dict[tuple[str, str], Path]: `(run_id, checkpoint)` → cache
    """
    found: dict[tuple[str, str], Path] = {}
    for path in sorted((PACKAGE / "cache").glob(f"*_{split}.npz")):
        if path.name.startswith("bootstrap_"):
            continue
        metadata = read_metadata(path)
        name = Path(metadata["checkpoint"]).name
        label = "last" if name == "last.ckpt" else "best"
        key = (metadata["run_id"], label)
        if key in found:
            raise ValueError(f"{key} の cache が 2 つある: {found[key].name}, {path.name}")
        found[key] = path
    return found


def main() -> None:
    """全 run の指標と bootstrap を書き出す。

    Args:
        なし

    Returns:
        None
    """
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--split", default="test", choices=("val", "test"))
    parser.add_argument("--resamples", type=int, default=200)
    args = parser.parse_args()

    frame = load_split(args.split)
    target, group = frame["target"].to_numpy(), frame["group"].to_numpy()
    weights = patient_bootstrap_weights(frame["patient"], args.resamples)
    table = runs()
    files = cache_files(args.split)
    # best と last が同じ checkpoint の run は、cache 名（checkpoint の SHA-256 で決まる）も同じになり 1 つしか無い。
    for run_id in table["run_id"]:
        recorded = read_run_record(PACKAGE / "runs" / run_id)["checkpoints"][0]
        if (run_id, "best") not in files and recorded["best"]["sha256"] == recorded["last"]["sha256"]:
            files[(run_id, "best")] = files[(run_id, "last")]
    missing = [(r, c) for r in table["run_id"] for c in ("best", "last") if (r, c) not in files]
    if missing:
        raise FileNotFoundError(f"cache が足りない（{len(missing)} 件）: {missing[:3]} ...")

    rows, keys, values = [], [], []
    for run in table.itertuples(index=False):
        for checkpoint in ("best", "last"):
            probability = positive_probability(files[(run.run_id, checkpoint)], frame)
            point = summary(probability, target, group)
            rows.append({**run._asdict(), "checkpoint": checkpoint, "split": args.split, **point})
            keys.append(f"{run.run_id}|{checkpoint}")
            values.append([list(summary(probability, target, group, w).values()) for w in weights])
            print(f"{run.run_id} {checkpoint}: auroc {point['auroc']:.3f} eodds {point['eodds']:.3f}")

    metrics = list(point)
    results = PACKAGE / "results"
    results.mkdir(exist_ok=True)
    columns = ["run_id", "seed", "procedure", "model", "objective", "checkpoint", "split", *metrics]
    write_csv(results / f"run_metrics_{args.split}.csv", rows, columns)
    np.savez_compressed(
        PACKAGE / "cache" / f"bootstrap_{args.split}.npz",
        keys=np.array(keys),
        metrics=np.array(metrics),
        values=np.array(values),
    )


if __name__ == "__main__":
    main()
