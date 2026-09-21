"""CheXpert 画像から凍結 encoder の特徴を抽出する。

実行例:
    DATA_FOLDER=/data uv run python -m projects.foundation_linear_probe.extract \
      --encoder dinov2_base --splits train val test
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import models
from torchvision.transforms import InterpolationMode
from torchvision.transforms import v2 as transforms
from transformers import AutoImageProcessor, AutoModel

from projects.foundation_linear_probe.chexpert import load_split, split_path
from projects.foundation_linear_probe.encoders import DinoCLS, DinoProcessor


@dataclass
class Encoder:
    """凍結済みモデルと画像変換、出力次元をまとめる。"""

    model: nn.Module
    transform: Callable[[Image.Image], torch.Tensor]
    embed_dim: int
    metadata: dict[str, str]


class ImagePaths(Dataset):
    """画像パスを encoder 入力テンソルへ変換する Dataset。"""

    def __init__(self, paths: Sequence[str], transform: Callable[[Image.Image], torch.Tensor]) -> None:
        """画像パスとモデル固有の変換を保持する。"""
        self.paths = list(paths)
        self.transform = transform

    def __len__(self) -> int:
        """画像数を返す。"""
        return len(self.paths)

    def __getitem__(self, index: int) -> torch.Tensor:
        """指定画像を RGB にして変換する。"""
        with Image.open(self.paths[index]) as image:
            return self.transform(image.convert("RGB"))


def load_encoder(name: str, device: torch.device) -> Encoder:
    """比較対象として固定した encoder を読み込む。"""
    if name == "resnet50_imagenet":
        weights = models.ResNet50_Weights.IMAGENET1K_V1
        model = models.resnet50(weights=weights)
        model.fc = nn.Identity()
        transform = transforms.Compose(
            [
                transforms.Resize(256, interpolation=InterpolationMode.BILINEAR, antialias=True),
                transforms.CenterCrop((224, 224)),
                transforms.ToImage(),
                transforms.ToDtype(torch.float32, scale=True),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )
        return Encoder(
            model.eval().requires_grad_(False).to(device),
            transform,
            2048,
            {"model_id": "resnet50.IMAGENET1K_V1", "revision": weights.url},
        )

    model_id, revision = {
        "dinov2_base": ("facebook/dinov2-base", "f9e44c814b77203eaa57a6bdbbd535f21ede1415"),
        "rad_dino": ("microsoft/rad-dino", "110cbc18d5133582e320b43d53bf5c44e410c936"),
    }[name]
    backbone = AutoModel.from_pretrained(model_id, revision=revision)
    processor = AutoImageProcessor.from_pretrained(model_id, revision=revision)
    return Encoder(
        DinoCLS(backbone).eval().requires_grad_(False).to(device),
        DinoProcessor(processor),
        int(backbone.config.hidden_size),
        {"model_id": model_id, "revision": revision},
    )


def file_sha256(path: Path) -> str:
    """split CSV の内容を識別する SHA256 を返す。"""
    return hashlib.sha256(path.read_bytes()).hexdigest()


@torch.inference_mode()
def extract_embeddings(
    encoder: Encoder,
    paths: Sequence[str],
    *,
    device: torch.device,
    batch_size: int,
    num_workers: int,
) -> np.ndarray:
    """入力順を保って、画像ごとの特徴行列 `(N, embed_dim)` を返す。"""
    loader = DataLoader(
        ImagePaths(paths, encoder.transform),
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=False,
        pin_memory=device.type == "cuda",
    )
    batches = []
    for pixels in loader:
        embeddings = encoder.model(pixels.to(device, non_blocking=True))
        if embeddings.shape != (len(pixels), encoder.embed_dim):
            raise RuntimeError(f"unexpected embedding shape: {tuple(embeddings.shape)}")
        batches.append(embeddings.float().cpu().numpy())
    return np.concatenate(batches)


def main() -> None:
    """指定した encoder と split の特徴を NPZ へ保存する。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--encoder", choices=["resnet50_imagenet", "dinov2_base", "rad_dino"], required=True)
    parser.add_argument("--splits", nargs="+", choices=["train", "val", "test"], default=["train", "val", "test"])
    parser.add_argument("--output-dir", type=Path, default=Path("projects/foundation_linear_probe/outputs/features"))
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--limit", type=int, help="動作確認用に各 split の先頭 N 件だけ抽出する。")
    args = parser.parse_args()
    if args.batch_size <= 0 or args.num_workers < 0 or (args.limit is not None and args.limit <= 0):
        parser.error("batch-size / limit must be positive; num-workers must be nonnegative")

    device = torch.device(args.device)
    encoder = load_encoder(args.encoder, device)
    metadata = {"encoder": args.encoder, **encoder.metadata, "splits": {}}
    for split in args.splits:
        frame = load_split(split)
        if args.limit is not None:
            frame = frame.head(args.limit)
        embeddings = extract_embeddings(
            encoder,
            frame["image_path"].tolist(),
            device=device,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
        )
        destination = args.output_dir / args.encoder / f"{split}.npz"
        destination.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            destination,
            image=frame["image"].to_numpy(dtype=str),
            target=frame["target"].to_numpy(dtype=np.int64),
            embedding=embeddings.astype(np.float16),
        )
        metadata["splits"][split] = {
            "n_images": len(frame),
            "split_csv_sha256": file_sha256(split_path(split)),
        }
        print(f"saved {len(frame):,} embeddings to {destination}")
    metadata_path = args.output_dir / args.encoder / "metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n")


if __name__ == "__main__":
    main()
