"""HAM10000 の train から、公平性損失の群の重みを測るための held-out を病変単位で切り出す。

CheXpert の `chexpert_dual_holdout_splits` と同じ方式。群の成績を学習に使っている画像で測ると、少数セルの記憶で
差が消えたように見えて群の重みが戻るので、train の病変 20% を held-out に分ける。モデルの重みは残り 80% だけで学習し、
群の重みは held-out で更新する（Cotter et al. 2019 の 2 データセット方式）。ERM の対照も同じ 80% で学習する。
同じ病変の画像は必ず同じ側に入る。val / test は `splits` と同じ行をそのまま複製する。

出力: `data/ham10000/splits_dual_holdout/{train,holdout,val,test}.csv` と `manifest.json`

Usage:
    uv run python -m data_pipeline.ham10000_dual_holdout_splits
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "data" / "ham10000" / "splits"
OUTPUT = ROOT / "data" / "ham10000" / "splits_dual_holdout"
HOLDOUT_FRACTION = 0.2


def split_off_holdout(frame: pd.DataFrame, seed: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    """病変の 20% を held-out へ割り当てる。同じ病変の画像は必ず同じ側に入る。"""
    lesions = np.sort(frame.lesion_id.unique())
    rng = np.random.default_rng(seed)
    held = set(rng.permutation(lesions)[: int(len(lesions) * HOLDOUT_FRACTION)])
    mask = frame.lesion_id.isin(held).to_numpy()
    return frame[~mask], frame[mask]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def summary(frame: pd.DataFrame) -> dict:
    """行数・病変数・陽性率と、性別 × 年齢（65 歳）の 4 セルの陽性数。"""
    known = frame[(frame.sex_missing == 0) & (frame.age_missing == 0)]
    cells = known.groupby([known.sex, known.age >= 65]).target.agg(["size", "sum"])
    return {
        "rows": len(frame),
        "lesions": int(frame.lesion_id.nunique()),
        "positive_rate": round(float(frame.target.mean()), 4),
        "sex_age65_cells": {f"sex{sex}_age{'>=65' if old else '<65'}": {"rows": int(r["size"]), "positives": int(r["sum"])} for (sex, old), r in cells.iterrows()},
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    OUTPUT.mkdir(parents=True, exist_ok=True)
    train, holdout = split_off_holdout(pd.read_csv(SOURCE / "train.csv"), args.seed)
    outputs = {"train": train, "holdout": holdout}
    for split in ("val", "test"):
        outputs[split] = pd.read_csv(SOURCE / f"{split}.csv")
    for name, frame in outputs.items():
        frame.to_csv(OUTPUT / f"{name}.csv", index=False)

    manifest = {
        "source_dir": str(SOURCE.relative_to(ROOT)),
        "source_train_sha256": sha256(SOURCE / "train.csv"),
        "seed": args.seed,
        "rule": "train の病変（lesion_id）を seed 付き permutation で 20% を held-out、残りを train に割り当てる。val / test は複製",
        "splits": {name: summary(frame) for name, frame in outputs.items()},
    }
    (OUTPUT / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(manifest["splits"], indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
