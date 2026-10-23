"""群探しの空間にする、No Finding の学習と無関係な画像表現 e(x) を書き出す。

- `raddino`：microsoft/rad-dino（胸部 X 線で自己教師あり学習した ViT-B/14、ラベルは使っていない）
- `dinov2`：facebook/dinov2-base（自然画像で自己教師あり学習）

どちらもローカルの HF cache から読む（取得はしない）。CLS token と patch token 平均を連結した 1536 次元を
float16 で保存する。val・test は race-known と同じ行（`subgroups.load_split` が行順を検算する）。
2026-10-17 に `analysis/archive/latent_bias_hypernet/probes/embed.py` から移し、マーカーを描く植え込み用の
選択肢を外した（移す前の版は snapshot/agent-era-20261016 ブランチにある）。

出力: `analysis/latent_subgroup_spaces/cache/<model>/<split>.npz`（image、embedding）

    HF_HUB_OFFLINE=1 uv run python analysis/latent_subgroup_spaces/embed.py --model raddino
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import rootutils
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from transformers import AutoModel

ROOT = rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

PACKAGE = ROOT / "analysis" / "latent_subgroup_spaces"
SPLITS = ROOT / "data" / "chexpert" / "splits_crossfit"
IMAGES = ROOT / "data" / "chexpert" / "images"
HUB = Path.home() / ".cache" / "huggingface" / "hub"
MODELS = {
    "raddino": (HUB / "models--microsoft--rad-dino/snapshots/110cbc18d5133582e320b43d53bf5c44e410c936", 0.5307, 0.2583),
    "dinov2": (HUB / "models--facebook--dinov2-base/snapshots/f9e44c814b77203eaa57a6bdbbd535f21ede1415", None, None),
}
IMAGENET_MEAN, IMAGENET_STD = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]


class Images(Dataset):
    def __init__(self, frame: pd.DataFrame, transform):
        self.images = frame.image.to_numpy(dtype=str)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, i: int):
        with Image.open(IMAGES / self.images[i]) as img:
            image = img.convert("RGB")
        return self.transform(image)


def build_transform(name: str, size: int) -> transforms.Compose:
    _, mean, std = MODELS[name]
    mean3 = [mean] * 3 if mean is not None else IMAGENET_MEAN
    std3 = [std] * 3 if std is not None else IMAGENET_STD
    return transforms.Compose(
        [
            transforms.Resize(size, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.CenterCrop(size),
            transforms.ToTensor(),
            transforms.Normalize(mean3, std3),
        ]
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=tuple(MODELS), default="raddino")
    parser.add_argument("--size", type=int, default=518)
    parser.add_argument("--splits", nargs="+", default=["val", "test"])
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    out = PACKAGE / "cache" / args.model
    out.mkdir(parents=True, exist_ok=True)
    model = AutoModel.from_pretrained(MODELS[args.model][0]).to(args.device).eval()
    transform = build_transform(args.model, args.size)
    for split in args.splits:
        frame = pd.read_csv(SPLITS / f"{split}.csv")
        loader = DataLoader(Images(frame, transform), batch_size=128, num_workers=8, prefetch_factor=2)
        parts = []
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            for image in loader:
                hidden = model(pixel_values=image.to(args.device, non_blocking=True)).last_hidden_state
                pooled = torch.cat([hidden[:, 0], hidden[:, 1:].mean(dim=1)], dim=1)
                parts.append(pooled.float().cpu().numpy().astype(np.float16))
        np.savez(out / f"{split}.npz", image=frame.image.to_numpy(dtype=str), embedding=np.concatenate(parts))
        print(split, len(frame), flush=True)


if __name__ == "__main__":
    main()
