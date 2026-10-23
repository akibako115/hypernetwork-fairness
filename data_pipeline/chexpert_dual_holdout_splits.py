"""race-known split の train から、公平性損失の群の重みを測るための held-out を患者単位で切り出す。

公平性損失（EO rate 制約・minimax AUC）は、群の成績を測って悪い群の重みを上げる。その成績を学習に使っている
画像で測ると、少数セルを記憶して差が消えたように見え、重みが戻ってしまう（`analysis/archive/conditional_fairness_one_stage`。
2026-10-17 に消し、snapshot/agent-era-20261016 ブランチに残る）。
そこで train の患者 20% を held-out に分け、モデルの重みは残り 80% だけで学習し、群の重みは held-out で更新する
（Cotter et al. 2019 の 2 データセット方式）。ERM の対照も同じ 80% で学習し、データの量をそろえる。
val / test は `splits_race_known` と同じ行をそのまま複製する。

出力: `data/chexpert/splits_race_known_dual_holdout/{train,holdout,val,test}.csv` と `manifest.json`

Usage:
    uv run python -m data_pipeline.chexpert_dual_holdout_splits
"""

from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd

from data_pipeline.chexpert_crossfit_splits import ROOT, SOURCE, patient_of, sha256, summary

OUTPUT = ROOT / "data" / "chexpert" / "splits_race_known_dual_holdout"
HOLDOUT_FRACTION = 0.2


def split_off_holdout(frame: pd.DataFrame, seed: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    """患者の 20% を held-out へ割り当てる。同じ患者の画像は必ず同じ側に入る。"""
    patients = np.sort(patient_of(frame.image).unique())
    rng = np.random.default_rng(seed)
    held = set(rng.permutation(patients)[: int(len(patients) * HOLDOUT_FRACTION)])
    mask = patient_of(frame.image).isin(held).to_numpy()
    return frame[~mask], frame[mask]


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
        "rule": "train の患者を seed 付き permutation で 20% を held-out、残りを train に割り当てる。val / test は複製",
        "splits": {name: summary(frame) for name, frame in outputs.items()},
    }
    (OUTPUT / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(manifest["splits"], indent=1))


if __name__ == "__main__":
    main()
