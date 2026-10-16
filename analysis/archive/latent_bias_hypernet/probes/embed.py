"""振り分け係の入力にする、No Finding の学習と無関係な画像表現 e(x) を書き出す。

- `raddino`：microsoft/rad-dino（胸部 X 線で自己教師あり学習した ViT-B/14、ラベルは使っていない）
- `dinov2`：facebook/dinov2-base（自然画像で自己教師あり学習）

どちらもローカルの HF cache から読む（取得はしない）。CLS token と patch token 平均を連結した 1536 次元を
float16 で保存する。`--plant` では planting.py の規則でマーカーを描いた画像を、`--mark-all` では全画像に
マーカーを描いた画像を埋め込む。

出力: `analysis/archive/latent_bias_hypernet/cache/embed/<model>[_plant]/<split>.npz`（image、embedding、marked）

    HF_HUB_OFFLINE=1 uv run python analysis/archive/latent_bias_hypernet/probes/embed.py --model raddino
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

from analysis.archive.latent_bias_hypernet.planting import marker_mask, stamp  # noqa: E402

PACKAGE = ROOT / "analysis" / "archive" / "latent_bias_hypernet"
SPLITS = ROOT / "data" / "chexpert" / "splits_crossfit"
IMAGES = ROOT / "data" / "chexpert" / "images"
HUB = Path.home() / ".cache" / "huggingface" / "hub"
MODELS = {
    "raddino": (HUB / "models--microsoft--rad-dino/snapshots/110cbc18d5133582e320b43d53bf5c44e410c936", 0.5307, 0.2583),
    "dinov2": (HUB / "models--facebook--dinov2-base/snapshots/f9e44c814b77203eaa57a6bdbbd535f21ede1415", None, None),
}
IMAGENET_MEAN, IMAGENET_STD = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]


class Images(Dataset):
    def __init__(self, frame: pd.DataFrame, transform, marked: np.ndarray):
        self.images = frame.image.to_numpy(dtype=str)
        self.transform = transform
        self.marked = marked

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, i: int):
        with Image.open(IMAGES / self.images[i]) as img:
            image = img.convert("RGB")
        if self.marked[i]:
            image = stamp(image)
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
    parser.add_argument("--plant", action="store_true")
    parser.add_argument("--mark-all", action="store_true", help="全画像に印を付ける（監査用データの偏りを作る素材）")
    parser.add_argument("--splits", nargs="+", default=["train_a", "train_b", "val", "test"])
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    suffix = "_markall" if args.mark_all else ("_plant" if args.plant else "")
    out = PACKAGE / "cache" / "embed" / (args.model + suffix)
    out.mkdir(parents=True, exist_ok=True)
    model = AutoModel.from_pretrained(MODELS[args.model][0]).to(args.device).eval()
    transform = build_transform(args.model, args.size)
    for split in args.splits:
        frame = pd.read_csv(SPLITS / f"{split}.csv")
        if args.mark_all:
            marked = np.ones(len(frame), dtype=bool)
        else:
            marked = marker_mask(frame, split) if args.plant else np.zeros(len(frame), dtype=bool)
        loader = DataLoader(Images(frame, transform, marked), batch_size=128, num_workers=8, prefetch_factor=2)
        parts = []
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            for image in loader:
                hidden = model(pixel_values=image.to(args.device, non_blocking=True)).last_hidden_state
                pooled = torch.cat([hidden[:, 0], hidden[:, 1:].mean(dim=1)], dim=1)
                parts.append(pooled.float().cpu().numpy().astype(np.float16))
        np.savez(
            out / f"{split}.npz", image=frame.image.to_numpy(dtype=str), embedding=np.concatenate(parts), marked=marked
        )
        print(split, len(frame), flush=True)


if __name__ == "__main__":
    main()
