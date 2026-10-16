"""cross-fit の 1 段目：ImageNet 初期化の ResNet-50 を train_a（または train_b）だけで ERM 学習する。

run 契約（projects/*/runs）の外で回す探索用の学習。毎 epoch に val AUROC を測り、最大の epoch の重みで
全 split（train_a / train_b / val / test）の分類 head 直前の表現と logit を書き出す。

出力: `analysis/archive/latent_bias_hypernet/cache/stage1/<name>/`
    - `best.pt`（best val AUROC の state_dict）、`log.json`
    - `<split>.npz`（image、target、logits [n,2]、features [n,2048] float16、marked [n] bool）

    uv run python analysis/archive/latent_bias_hypernet/probes/train_stage1.py --name a_s42 --fit-split train_a --seed 42
    uv run python analysis/archive/latent_bias_hypernet/probes/train_stage1.py --name a_plant_s42 --fit-split train_a --plant
    uv run python analysis/archive/latent_bias_hypernet/probes/train_stage1.py --name a_yfdev_s42 --epochs 6 \
        --subgroup young_female_device --undersample-pos 0.1

`--subgroup <rule> --undersample-pos p` は、学習 split のうち `planting.HIDDEN_SUBGROUPS[rule]` に入る陽性を
割合 p だけ残す（`--undersample-pacer-pos p` は `--subgroup pacer` の旧名）。`--subgroup-mode flip` では、
残さない陽性を除く代わりに陰性ラベルへ反転する。`--random-drop-pos N --drop-seed s` は対照で、要因と無関係に
陽性を N 枚除く。
メタデータに無い部分集団で陽性を過少に学習させ、展開先でその部分集団の予測が系統的に低くなる状況を作る。
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

from analysis.archive.latent_bias_hypernet.planting import (  # noqa: E402
    HIDDEN_SUBGROUPS,
    hidden_subgroup,
    marker_mask,
    stamp,
)
from projects.hypernet_e2e.data.transform_utils import (  # noqa: E402
    train_transforms_chexpert,
    val_transforms_chexpert,
)
from projects.hypernet_e2e.models.resnet.network import ResNet  # noqa: E402
from projects.hypernet_e2e.models.utils import load_compatible_state_dict  # noqa: E402

PACKAGE = ROOT / "analysis" / "archive" / "latent_bias_hypernet"
SPLITS = ROOT / "data" / "chexpert" / "splits_crossfit"
IMAGES = ROOT / "data" / "chexpert" / "images"
ALL_SPLITS = ("train_a", "train_b", "val", "test")


class Images(Dataset):
    """split CSV の行を (image, target, index) にする。`marked` の行にはマーカーを描く。"""

    def __init__(self, frame: pd.DataFrame, transform, marked: np.ndarray):
        self.images = frame.image.to_numpy(dtype=str)
        self.target = frame.target.to_numpy()
        self.transform = transform
        self.marked = marked

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, i: int):
        with Image.open(IMAGES / self.images[i]) as img:
            image = img.convert("RGB")
        if self.marked[i]:
            image = stamp(image)
        return self.transform(image), int(self.target[i]), i


def imagenet_resnet50(device: str) -> nn.Module:
    network = ResNet(layers=[3, 4, 6, 3], block="Bottleneck", num_classes=2)
    state = torch.load(ROOT / "checkpoints" / "resnet50_imagenet.pth", map_location="cpu", weights_only=False)
    state = state.get("state_dict", state)
    state = {k.removeprefix("net.").removeprefix("backbone."): v for k, v in state.items()}
    result = load_compatible_state_dict(network.backbone, state, load_fc=False)
    if result["missing_keys"]:
        raise RuntimeError("ImageNet 重みが backbone を埋めていない")
    return network.to(device)


def forward_split(network: nn.Module, loader: DataLoader, device: str) -> tuple[np.ndarray, np.ndarray]:
    """loader を 1 周して logits と表現を返す。"""
    network.eval()
    logits_parts, feature_parts = [], []
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        for image, _y, _i in loader:
            logits, features = network.forward_with_features(image.to(device, non_blocking=True))
            logits_parts.append(logits.float().cpu().numpy())
            feature_parts.append(features.to(torch.float16).cpu().numpy())
    network.train()
    return np.concatenate(logits_parts), np.concatenate(feature_parts)


def eval_loader(frame: pd.DataFrame, marked: np.ndarray) -> DataLoader:
    # 評価 loader は persistent にしない（2026-10-06 の OOM、fairness_mechanism_probe/runs.md）。
    dataset = Images(frame, val_transforms_chexpert(), marked)
    return DataLoader(dataset, batch_size=256, num_workers=8, pin_memory=True, prefetch_factor=2)


def train(args: argparse.Namespace, frames: dict[str, pd.DataFrame], marks: dict[str, np.ndarray]) -> dict:
    out = PACKAGE / "cache" / "stage1" / args.name
    fit = frames[args.fit_split]
    fit_marks = marks[args.fit_split]
    if args.random_drop_pos:
        # 対照：要因と無関係に、同じ枚数の陽性をランダムに除く（植え込みの損害と学習の揺れを分けるため）
        pos_idx = np.flatnonzero(fit.target.to_numpy() == 1)
        drop_idx = np.random.default_rng(args.drop_seed).choice(pos_idx, args.random_drop_pos, replace=False)
        keep = np.ones(len(fit), dtype=bool)
        keep[drop_idx] = False
        fit, fit_marks = fit[keep].reset_index(drop=True), fit_marks[keep]
        print(f"randomly dropped positives: {len(pos_idx)} -> {len(pos_idx) - args.random_drop_pos}", flush=True)
    if args.undersample_pacer_pos < 1:
        args.undersample_pos, args.subgroup = args.undersample_pacer_pos, "pacer"
    if args.undersample_pos < 1:
        context = pd.read_csv(PACKAGE / "cache" / "context" / f"{args.fit_split}.csv")
        if not (context.image.to_numpy() == fit.image.to_numpy()).all():
            raise ValueError("context と split の行順が一致しない")
        hit = hidden_subgroup(context, args.subgroup) & (context.target == 1).to_numpy()
        drop = hit & (np.random.default_rng(0).random(len(fit)) >= args.undersample_pos)
        if args.subgroup_mode == "flip":
            # 間引く代わりに陰性ラベルへ反転する（部分集団で系統的に誤ったラベル付け）
            fit = fit.copy()
            fit.loc[drop, "target"] = 0
            print(f"flipped {args.subgroup} positives:", int(drop.sum()), "of", int(hit.sum()), flush=True)
        else:
            fit, fit_marks = fit[~drop].reset_index(drop=True), fit_marks[~drop]
            print(
                f"undersampled {args.subgroup} positives:", int(hit.sum()), "->", int((hit & ~drop).sum()), flush=True
            )
    train_loader = DataLoader(
        Images(fit, train_transforms_chexpert(), fit_marks),
        batch_size=args.batch,
        shuffle=True,
        drop_last=True,
        num_workers=8,
        pin_memory=True,
        prefetch_factor=2,
        persistent_workers=True,
    )
    val_loader = eval_loader(frames["val"], marks["val"])
    # M01 と同じ global inverse class weight。fit split の陽性率から作る。
    pos = fit.target.mean()
    cw = torch.tensor([0.5 / (1 - pos), 0.5 / pos], device=args.device, dtype=torch.float32)
    network = imagenet_resnet50(args.device).train()
    optimizer = torch.optim.AdamW(network.parameters(), lr=args.lr, weight_decay=1e-2)
    log = {"args": vars(args), "n_fit": len(fit), "epochs": []}
    best = -1.0
    for epoch in range(args.epochs):
        start = time.time()
        losses = []
        for image, y, _i in train_loader:
            image, y = image.to(args.device, non_blocking=True), y.to(args.device)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                logits = network(image).float()
            loss = nn.functional.cross_entropy(logits, y, weight=cw)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
        logits, _ = forward_split(network, val_loader, args.device)
        auroc = roc_auc_score(frames["val"].target, logits[:, 1] - logits[:, 0])
        record = {"epoch": epoch, "train_loss": float(np.mean(losses)), "val_auroc": auroc, "sec": time.time() - start}
        log["epochs"].append(record)
        print(json.dumps(record), flush=True)
        if auroc > best:
            best = auroc
            log["best_epoch"] = epoch
            torch.save(network.state_dict(), out / "best.pt")
        (out / "log.json").write_text(json.dumps(log, indent=1))
    return log


def export(args: argparse.Namespace, frames: dict[str, pd.DataFrame], marks: dict[str, np.ndarray]) -> None:
    out = PACKAGE / "cache" / "stage1" / args.name
    network = imagenet_resnet50(args.device)
    network.load_state_dict(torch.load(out / "best.pt", map_location=args.device))
    for split, frame in frames.items():
        logits, features = forward_split(network, eval_loader(frame, marks[split]), args.device)
        np.savez(
            out / f"{split}.npz",
            image=frame.image.to_numpy(dtype=str),
            target=frame.target.to_numpy(),
            logits=logits,
            features=features,
            marked=marks[split],
        )
        print(split, "AUROC", round(roc_auc_score(frame.target, logits[:, 1] - logits[:, 0]), 4), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", required=True)
    parser.add_argument("--fit-split", choices=("train_a", "train_b"), default="train_a")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--batch", type=int, default=128)
    parser.add_argument("--plant", action="store_true", help="planting.py のマーカーを描く")
    parser.add_argument("--undersample-pacer-pos", type=float, default=1.0, help="`--subgroup pacer` の旧名")
    parser.add_argument("--subgroup", choices=tuple(HIDDEN_SUBGROUPS), default="pacer")
    parser.add_argument("--undersample-pos", type=float, default=1.0, help="部分集団の陽性を残す割合")
    parser.add_argument(
        "--subgroup-mode", choices=("drop", "flip"), default="drop", help="残さない陽性を除くか、陰性にするか"
    )
    parser.add_argument("--random-drop-pos", type=int, default=0, help="要因と無関係に除く陽性の枚数（対照）")
    parser.add_argument("--drop-seed", type=int, default=0)
    parser.add_argument("--export-only", action="store_true")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    (PACKAGE / "cache" / "stage1" / args.name).mkdir(parents=True, exist_ok=True)
    frames = {s: pd.read_csv(SPLITS / f"{s}.csv") for s in ALL_SPLITS}
    marks = {s: (marker_mask(f, s) if args.plant else np.zeros(len(f), dtype=bool)) for s, f in frames.items()}
    if not args.export_only:
        train(args, frames, marks)
    export(args, frames, marks)


if __name__ == "__main__":
    main()
