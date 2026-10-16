"""FairVLMed で「属性ごとの処理に伸びしろがあるか」の関門診断を行う（CheXpert との対照）。

1. ImageNet ResNet-50 を FairVLMed train で ERM 学習（run 契約の外、W&B なし）。val AUROC 最大の epoch を採用。
2. 採用した重みで train / val / test の 2048 次元表現を取る。
3. heads.py と同じ head 群（共有・群別切片・群別Δw・独立・hypernet・MLP）を表現の上で学習し、群別 AUROC を比べる。

群は race 3 区分（Asian / Black / White）× 年齢 <65 / ≥65 の 6 群。
出力: `results/fairvlmed/heads_s<seed>.csv`、表現は `cache/fairvlmed/s<seed>_<split>.npz`

    uv run python analysis/archive/fairness_mechanism_probe/probes/fairvlmed_headroom.py --seed 42
"""

from __future__ import annotations

import argparse
import sys

import numpy as np
import pandas as pd
import rootutils
import torch
from PIL import Image
from sklearn.metrics import roc_auc_score
from torch import nn
from torch.utils.data import DataLoader, Dataset

ROOT = rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)
PACKAGE = ROOT / "analysis" / "archive" / "fairness_mechanism_probe"
sys.path.insert(0, str(PACKAGE))

import heads  # noqa: E402

from projects.hypernet_e2e.data.transform_utils import (  # noqa: E402
    train_transforms_fairvlmed,
    val_transforms_fairvlmed,
)
from projects.hypernet_e2e.models.resnet.network import ResNet  # noqa: E402
from projects.hypernet_e2e.models.utils import load_compatible_state_dict  # noqa: E402

SPLITS = ROOT / "data" / "fairvlmed" / "splits"
IMAGES = ROOT / "data" / "fairvlmed" / "images"
RACES = ("Asian", "Black", "White")
GROUP_NAMES = tuple(f"{a}×{r}" for a in ("<65", "≥65") for r in RACES)
CATEGORICAL = (("gender", 2), ("race", 3), ("ethnicity", 2), ("language", 3), ("maritalstatus", 5))


def group_ids(frame: pd.DataFrame) -> np.ndarray:
    return ((frame.age.to_numpy() >= 65).astype(np.int64) * 3 + frame.race.to_numpy()).astype(np.int64)


class Frame(Dataset):
    def __init__(self, frame, transform):
        self.frame, self.transform = frame.reset_index(drop=True), transform

    def __len__(self):
        return len(self.frame)

    def __getitem__(self, i):
        with Image.open(IMAGES / self.frame.image[i]) as img:
            return self.transform(img.convert("RGB")), int(self.frame.target[i])


def build_network(device):
    network = ResNet(layers=[3, 4, 6, 3], block="Bottleneck", num_classes=2)
    state = torch.load(ROOT / "checkpoints" / "resnet50_imagenet.pth", map_location="cpu", weights_only=False)
    state = state.get("state_dict", state)
    result = load_compatible_state_dict(
        network.backbone, {k.removeprefix("backbone."): v for k, v in state.items()}, load_fc=False
    )
    if result["missing_keys"]:
        raise RuntimeError("ImageNet 重みが backbone を埋めていない")
    return network.to(device)


def forward(network, loader, device, features=False):
    network.eval()
    scores, feats = [], []
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        for image, _ in loader:
            logits, feat = network.forward_with_features(image.to(device))
            scores.append((logits[:, 1] - logits[:, 0]).float().cpu().numpy())
            if features:
                feats.append(feat.float().cpu().numpy())
    network.train()
    return np.concatenate(scores), (np.concatenate(feats) if features else None)


def train_erm(seed, frames, device, epochs=20):
    torch.manual_seed(seed)
    kw = dict(num_workers=6, pin_memory=True)  # 共有メモリを抑える（2026-10-06 の OOM 対策）
    train_loader = DataLoader(
        Frame(frames["train"], train_transforms_fairvlmed()), batch_size=64, shuffle=True, drop_last=True, **kw
    )
    eval_loaders = {
        s: DataLoader(Frame(f, val_transforms_fairvlmed()), batch_size=256, **kw) for s, f in frames.items()
    }
    network = build_network(device)
    optimizer = torch.optim.AdamW(network.parameters(), lr=1e-4, weight_decay=1e-2)
    best, best_state = -1.0, None
    for epoch in range(epochs):
        for image, y in train_loader:
            with torch.autocast("cuda", dtype=torch.bfloat16):
                loss = nn.functional.cross_entropy(network(image.to(device)).float(), y.to(device))
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
        val_score, _ = forward(network, eval_loaders["val"], device)
        auc = roc_auc_score(frames["val"].target, val_score)
        print(f"seed {seed} epoch {epoch} val auroc {auc:.4f}", flush=True)
        if auc > best:
            best, best_state = auc, {k: v.detach().clone() for k, v in network.state_dict().items()}
    network.load_state_dict(best_state)
    return network, eval_loaders


def metadata(frame, age_mean, age_std):
    parts = [np.eye(c, dtype=np.float32)[frame[n].clip(lower=0).to_numpy()] for n, c in CATEGORICAL]
    parts.append(((frame.age.to_numpy() - age_mean) / age_std).astype(np.float32)[:, None])
    return np.concatenate(parts, 1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    frames = {s: pd.read_csv(SPLITS / f"{s}.csv") for s in ("train", "val", "test")}
    network, loaders = train_erm(args.seed, frames, args.device)
    cache = PACKAGE / "cache" / "fairvlmed"
    cache.mkdir(parents=True, exist_ok=True)
    splits = {}
    for s, loader in loaders.items():
        score, feat = forward(network, loader, args.device, features=True)
        np.savez(cache / f"s{args.seed}_{s}.npz", score=score, features=feat.astype(np.float16))
        splits[s] = heads.Split(
            feat,
            frames[s].target.to_numpy(),
            group_ids(frames[s]),
            frames[s],
            np.stack([np.zeros_like(score), score], 1),
        )
    heads.GROUP_NAMES = GROUP_NAMES  # group_aurocs が読む群名を差し替える
    train = splits["train"]
    mean, std = train.features.mean(0), train.features.std(0) + 1e-6
    a_mean, a_std = train.frame.age.mean(), train.frame.age.std()
    data = {s: heads.to_device(v, mean, std, metadata(v.frame, a_mean, a_std), args.device) for s, v in splits.items()}
    rows = [
        {
            "seed": args.seed,
            "head": "original_fc",
            **heads.group_aurocs(splits["test"].logits[:, 1], splits["test"].target, splits["test"].group),
        }
    ]
    for label, (kind, penalty) in heads.HEADS.items():
        torch.manual_seed(args.seed)
        head = heads.Head(kind, train.features.shape[1], data["train"]["m"].shape[1]).to(args.device)
        head = heads.fit_head(head, data["train"], data["val"], splits["val"], penalty, epochs=40)
        score = heads.predict(head, data["test"])
        rows.append(
            {"seed": args.seed, "head": label, **heads.group_aurocs(score, splits["test"].target, splits["test"].group)}
        )
        print(label, round(rows[-1]["overall"], 4), round(rows[-1]["worst"], 4), flush=True)
    out = PACKAGE / "results" / "fairvlmed"
    out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out / f"heads_s{args.seed}.csv", index=False)


if __name__ == "__main__":
    main()
