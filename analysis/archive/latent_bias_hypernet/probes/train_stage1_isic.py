"""ISIC 2019 の施設間 split（isic_sites.py）で、BCN20000 だけを使って 1 段目の ResNet-50 を学習する。

train_stage1.py の ISIC 版。ImageNet 初期化、AdamW（lr 1e-4、wd 1e-2）、bf16、fit split の陽性率から作る
global inverse class weight、ISIC 2019 の既存 transform（色を変える augmentation なし）。`isic_val_a` の AUROC が
最大の epoch の重みで、全 split の分類 head 直前の表現と logit を書き出す。

出力: `cache/stage1/<name>/isic_<split>.npz`（train_stage1.py と同じ形式）

    uv run python analysis/archive/latent_bias_hypernet/probes/train_stage1_isic.py --name isic_bcn_s42
"""

from __future__ import annotations

import argparse
import json
import time

import numpy as np
import pandas as pd
import rootutils
import torch
from PIL import Image
from sklearn.metrics import roc_auc_score
from torch import nn
from torch.utils.data import DataLoader, Dataset

ROOT = rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from analysis.archive.latent_bias_hypernet.probes.train_stage1 import PACKAGE, imagenet_resnet50  # noqa: E402
from projects.hypernet_e2e.data.transform_utils import (  # noqa: E402
    train_transforms_isic2019,
    val_transforms_isic2019,
)

IMAGES = ROOT / "data" / "isic2019" / "images_cropped_no_shades_of_gray" / "ISIC_2019_Training_Input"
SPLITS = ("isic_train_a", "isic_val_a", "isic_audit", "isic_audit_v", "isic_test")


class Images(Dataset):
    def __init__(self, frame: pd.DataFrame, transform):
        self.images = frame.image.to_numpy(dtype=str)
        self.target = frame.target.to_numpy()
        self.transform = transform

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, i: int):
        with Image.open(IMAGES / self.images[i]) as img:
            image = img.convert("RGB")
        return self.transform(image), int(self.target[i]), i


def loader(frame: pd.DataFrame, train: bool, batch: int = 128) -> DataLoader:
    transform = train_transforms_isic2019() if train else val_transforms_isic2019()
    return DataLoader(
        Images(frame, transform),
        batch_size=batch if train else 256,
        shuffle=train,
        drop_last=train,
        num_workers=8,
        pin_memory=True,
        prefetch_factor=2,
        persistent_workers=train,
    )


def forward(network: nn.Module, dl: DataLoader, device: str) -> tuple[np.ndarray, np.ndarray]:
    network.eval()
    logits_parts, feature_parts = [], []
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        for image, _y, _i in dl:
            logits, features = network.forward_with_features(image.to(device, non_blocking=True))
            logits_parts.append(logits.float().cpu().numpy())
            feature_parts.append(features.to(torch.float16).cpu().numpy())
    network.train()
    return np.concatenate(logits_parts), np.concatenate(feature_parts)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    out = PACKAGE / "cache" / "stage1" / args.name
    out.mkdir(parents=True, exist_ok=True)
    frames = {s: pd.read_csv(PACKAGE / "cache" / "context" / f"{s}.csv") for s in SPLITS}
    fit = frames["isic_train_a"]
    pos = fit.target.mean()
    cw = torch.tensor([0.5 / (1 - pos), 0.5 / pos], device=args.device, dtype=torch.float32)
    network = imagenet_resnet50(args.device).train()
    optimizer = torch.optim.AdamW(network.parameters(), lr=args.lr, weight_decay=1e-2)
    train_dl, val_dl = loader(fit, True), loader(frames["isic_val_a"], False)
    log = {"args": vars(args), "epochs": []}
    best = -1.0
    for epoch in range(args.epochs):
        start, losses = time.time(), []
        for image, y, _i in train_dl:
            image, y = image.to(args.device, non_blocking=True), y.to(args.device)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                logits = network(image).float()
            loss = nn.functional.cross_entropy(logits, y, weight=cw)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
        logits, _ = forward(network, val_dl, args.device)
        auroc = roc_auc_score(frames["isic_val_a"].target, logits[:, 1] - logits[:, 0])
        record = {"epoch": epoch, "train_loss": float(np.mean(losses)), "val_auroc": auroc, "sec": time.time() - start}
        log["epochs"].append(record)
        print(json.dumps(record), flush=True)
        if auroc > best:
            best, log["best_epoch"] = auroc, epoch
            torch.save(network.state_dict(), out / "best.pt")
    (out / "log.json").write_text(json.dumps(log, indent=1))
    network.load_state_dict(torch.load(out / "best.pt", map_location=args.device))
    for split, frame in frames.items():
        logits, features = forward(network, loader(frame, False), args.device)
        np.savez(
            out / f"{split}.npz",
            image=frame.image.to_numpy(dtype=str),
            target=frame.target.to_numpy(),
            logits=logits,
            features=features,
            marked=np.zeros(len(frame), dtype=bool),
        )
        print(split, "AUROC", round(roc_auc_score(frame.target, logits[:, 1] - logits[:, 0]), 4), flush=True)


if __name__ == "__main__":
    main()
