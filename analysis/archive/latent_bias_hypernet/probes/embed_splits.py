"""HAM10000・FairVLMed の split（`data/<dataset>/splits`）の画像を DINOv2 で埋め込む（e(x)）。

embed_isic.py の HAM・FairVLMed 版。facebook/dinov2-base をローカルの HF cache から読み、224 で CLS と patch 平均を
連結した 1536 次元を float16 で保存する。あわせて split CSV を `cache/context/<prefix>_<split>.csv` に写す。

出力: `cache/embed/dinov2/<prefix>_<split>.npz`（image、embedding、marked）、`cache/context/<prefix>_<split>.csv`
（prefix は ham10000 なら `ham`、fairvlmed なら `fvl`）

    HF_HUB_OFFLINE=1 uv run python analysis/archive/latent_bias_hypernet/probes/embed_splits.py [--dataset fairvlmed]
"""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd
import rootutils
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModel

ROOT = rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from analysis.archive.latent_bias_hypernet.probes.embed import MODELS, build_transform  # noqa: E402

PACKAGE = ROOT / "analysis" / "archive" / "latent_bias_hypernet"
SPLITS = ("train", "val", "test")
PREFIX = {"ham10000": "ham", "fairvlmed": "fvl"}


class Images(Dataset):
    def __init__(self, frame: pd.DataFrame, root, transform):
        self.images = frame.image.to_numpy(dtype=str)
        self.root = root
        self.transform = transform

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, i: int):
        with Image.open(self.root / "images" / self.images[i]) as img:
            return self.transform(img.convert("RGB"))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="ham10000", choices=tuple(PREFIX))
    args = parser.parse_args()
    root, prefix = ROOT / "data" / args.dataset, PREFIX[args.dataset]
    out = PACKAGE / "cache" / "embed" / "dinov2"
    out.mkdir(parents=True, exist_ok=True)
    model = AutoModel.from_pretrained(MODELS["dinov2"][0]).to("cuda").eval()
    transform = build_transform("dinov2", 224)
    for split in SPLITS:
        frame = pd.read_csv(root / "splits" / f"{split}.csv")
        frame.to_csv(PACKAGE / "cache" / "context" / f"{prefix}_{split}.csv", index=False)
        dl = DataLoader(Images(frame, root, transform), batch_size=128, num_workers=8, prefetch_factor=2)
        parts = []
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            for image in dl:
                hidden = model(pixel_values=image.to("cuda", non_blocking=True)).last_hidden_state
                pooled = torch.cat([hidden[:, 0], hidden[:, 1:].mean(dim=1)], dim=1)
                parts.append(pooled.float().cpu().numpy().astype(np.float16))
        np.savez(
            out / f"{prefix}_{split}.npz",
            image=frame.image.to_numpy(dtype=str),
            embedding=np.concatenate(parts),
            marked=np.zeros(len(frame), dtype=bool),
        )
        print(split, len(frame), flush=True)


if __name__ == "__main__":
    main()
