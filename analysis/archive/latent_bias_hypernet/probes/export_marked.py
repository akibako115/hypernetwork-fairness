"""マーカー植え込みの 1 段目で、train_b の全画像に印を付けた版と付けない版の表現を書き出す。

監査用データ（B）にも偏りが残る条件を作るため。画像ごとに 2 版のどちらかを選べば、任意の印の付け方の
h(x)・logit を再現できる（印の有無以外は同じ画像なので、eval の前処理は決定的）。

出力: `cache/stage1/<stage1>/train_b_mark{all,none}.npz`（train_stage1.py の `<split>.npz` と同じ形式）

    uv run python analysis/archive/latent_bias_hypernet/probes/export_marked.py --stage1 a_plant_s42
"""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd
import rootutils
import torch

ROOT = rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from analysis.archive.latent_bias_hypernet.probes.train_stage1 import (  # noqa: E402
    PACKAGE,
    SPLITS,
    eval_loader,
    forward_split,
    imagenet_resnet50,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage1", default="a_plant_s42")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    out = PACKAGE / "cache" / "stage1" / args.stage1
    network = imagenet_resnet50(args.device)
    network.load_state_dict(torch.load(out / "best.pt", map_location=args.device))
    frame = pd.read_csv(SPLITS / "train_b.csv")
    for tag, value in (("all", True), ("none", False)):
        marks = np.full(len(frame), value)
        logits, features = forward_split(network, eval_loader(frame, marks), args.device)
        np.savez(
            out / f"train_b_mark{tag}.npz",
            image=frame.image.to_numpy(dtype=str),
            target=frame.target.to_numpy(),
            logits=logits,
            features=features,
            marked=marks,
        )
        print(tag, flush=True)


if __name__ == "__main__":
    main()
