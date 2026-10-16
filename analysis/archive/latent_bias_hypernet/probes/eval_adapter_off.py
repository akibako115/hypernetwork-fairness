"""2 段目を学習したあとの重みから Spatial LoRA を外し、backbone と共有 classifier だけで test を評価する。

凍結した 2 段目なら、外すと 1 段目そのものに戻る（補正はすべて hypernet 側にある）。
同時学習なら、backbone が動いた分だけ補正が hypernet の外に漏れる。その量を比べる。

出力: `analysis/archive/latent_bias_hypernet/cache/adapter/<name>/off_test.npy`

    uv run python analysis/archive/latent_bias_hypernet/probes/eval_adapter_off.py --name jt_plant_joint_meta --plant
"""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd
import rootutils
import torch

ROOT = rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from analysis.archive.latent_bias_hypernet.planting import marker_mask  # noqa: E402
from analysis.archive.latent_bias_hypernet.probes.train_stage1 import eval_loader, forward_split  # noqa: E402
from projects.hypernet_e2e.models.resnet.network import ResNet  # noqa: E402

PACKAGE = ROOT / "analysis" / "archive" / "latent_bias_hypernet"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", required=True)
    parser.add_argument("--plant", action="store_true")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    out = PACKAGE / "cache" / "adapter" / args.name
    state = torch.load(out / "final.pt", map_location="cpu")
    plain = {k: v for k, v in state.items() if k.startswith("backbone.")}
    plain["fc.weight"] = state["fc.base_linear.weight"]
    plain["fc.bias"] = state["fc.base_linear.bias"]
    network = ResNet(layers=[3, 4, 6, 3], block="Bottleneck", num_classes=2)
    network.load_state_dict(plain)
    network.to(args.device)
    frame = pd.read_csv(ROOT / "data" / "chexpert" / "splits_crossfit" / "test.csv")
    marks = marker_mask(frame, "test") if args.plant else np.zeros(len(frame), dtype=bool)
    logits, _ = forward_split(network, eval_loader(frame, marks), args.device)
    np.save(out / "off_test.npy", logits[:, 1] - logits[:, 0])
    print("saved", out / "off_test.npy")


if __name__ == "__main__":
    main()
