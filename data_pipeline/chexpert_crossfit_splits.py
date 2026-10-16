"""race-known split の train を患者単位で A / B に二分した cross-fit 用 split を作る。

1 段目のモデルを A で学習し、2 段目（adapter・潜在コード）を 1 段目が見ていない B で学習するために使う。
val / test は `splits_race_known` と同じ行をそのまま複製し、既存 run と同じ集合で評価できるようにする。

出力: `data/chexpert/splits_crossfit/{train_a,train_b,val,test}.csv` と `manifest.json`

Usage:
    uv run python -m data_pipeline.chexpert_crossfit_splits
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "data" / "chexpert" / "splits_race_known"
OUTPUT = ROOT / "data" / "chexpert" / "splits_crossfit"


def patient_of(image: pd.Series) -> pd.Series:
    """`train/patient00024/study2/view1_frontal.jpg` から `patient00024` を取り出す。"""
    return image.str.split("/").str[1]


def halve_by_patient(frame: pd.DataFrame, seed: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    """患者を無作為に半分ずつ A / B へ割り当てる。同じ患者の画像は必ず同じ側に入る。"""
    patients = np.sort(patient_of(frame.image).unique())
    rng = np.random.default_rng(seed)
    in_a = set(rng.permutation(patients)[: len(patients) // 2])
    mask = patient_of(frame.image).isin(in_a).to_numpy()
    return frame[mask], frame[~mask]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def summary(frame: pd.DataFrame) -> dict:
    return {
        "rows": len(frame),
        "patients": int(patient_of(frame.image).nunique()),
        "positive_rate": round(float(frame.target.mean()), 4),
        "age_ge65_rate": round(float((frame.age >= 65).mean()), 4),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    OUTPUT.mkdir(parents=True, exist_ok=True)
    train = pd.read_csv(SOURCE / "train.csv")
    train_a, train_b = halve_by_patient(train, args.seed)
    outputs = {"train_a": train_a, "train_b": train_b}
    for split in ("val", "test"):
        outputs[split] = pd.read_csv(SOURCE / f"{split}.csv")
    for name, frame in outputs.items():
        frame.to_csv(OUTPUT / f"{name}.csv", index=False)

    manifest = {
        "source_dir": str(SOURCE.relative_to(ROOT)),
        "source_train_sha256": sha256(SOURCE / "train.csv"),
        "seed": args.seed,
        "rule": "train の患者を seed 付き permutation で半分ずつ A / B に割り当てる。val / test は複製",
        "splits": {name: summary(frame) for name, frame in outputs.items()},
    }
    (OUTPUT / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(manifest["splits"], indent=1))


if __name__ == "__main__":
    main()
