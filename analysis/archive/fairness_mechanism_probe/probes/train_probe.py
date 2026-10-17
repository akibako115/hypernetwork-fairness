"""探索用の短い fine-tune probe。run 契約（projects/*/runs）には載せない当たり付けの学習。

M01（ResNet ERM）の best checkpoint、または ImageNet 重みから始め、train の部分集合・目的関数を変えて
数 epoch だけ学習する。毎 epoch に val / test の score を保存し、選択規則は分析側で決める。

出力: `analysis/archive/fairness_mechanism_probe/cache/probes/<name>/epoch_XX_{val,test}.npy` と `log.json`

    uv run python analysis/archive/fairness_mechanism_probe/probes/train_probe.py --name spec_old_s42 \
        --init m01 --seed 42 --subset old --epochs 5
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

from analysis.common.predictions import load_network  # noqa: E402
from projects.hypernet_e2e.data.transform_utils import (  # noqa: E402
    train_transforms_medfair,
    val_transforms_medfair,
)
from projects.hypernet_e2e.models.resnet.network import ResNet  # noqa: E402
from projects.hypernet_e2e.models.utils import load_compatible_state_dict  # noqa: E402

PACKAGE = ROOT / "analysis" / "archive" / "fairness_mechanism_probe"
SPLITS = ROOT / "data" / "chexpert" / "splits_race_known"
IMAGES = ROOT / "data" / "chexpert" / "images"
M01 = {
    42: "20261002T124957Z-resnet-chexpert-race-known-s42-d6d3",
    43: "20261002T142200Z-resnet-chexpert-race-known-s43-bbfc",
    44: "20261002T155222Z-resnet-chexpert-race-known-s44-3d73",
}
CATEGORICAL = (("sex", 2), ("race", 6), ("ethnicity", 2), ("frontal_lateral", 2), ("ap_pa", 2))


class Frame(Dataset):
    """split CSV の行を (image, target, group, meta, index) にする。"""

    def __init__(self, frame: pd.DataFrame, transform, age_mean: float, age_std: float):
        self.frame = frame.reset_index(drop=True)
        self.transform = transform
        self.group = ((self.frame.age >= 65).astype(int) * 4 + self.frame.race_group).to_numpy()
        parts = [np.eye(c, dtype=np.float32)[self.frame[n].to_numpy()] for n, c in CATEGORICAL]
        parts.append(((self.frame.age.to_numpy() - age_mean) / age_std).astype(np.float32)[:, None])
        self.meta = np.concatenate(parts, axis=1)

    def __len__(self) -> int:
        return len(self.frame)

    def __getitem__(self, i: int):
        with Image.open(IMAGES / self.frame.image[i]) as img:
            image = self.transform(img.convert("RGB"))
        return image, int(self.frame.target[i]), int(self.group[i]), self.meta[i], i


def subset(frame: pd.DataFrame, name: str) -> pd.DataFrame:
    if name == "all":
        return frame
    if name == "old":
        return frame[frame.age >= 65]
    if name == "young":
        return frame[frame.age < 65]
    # 高齢群と同じ件数（70,322）にそろえた対照。データ量と群の違いを分ける。
    n_old = int((frame.age >= 65).sum())
    if name == "random_nold":
        return frame.sample(n_old, random_state=0)
    if name == "young_nold":
        return frame[frame.age < 65].sample(n_old, random_state=0)
    raise ValueError(name)


def build_network(init: str, seed: int, device: str) -> nn.Module:
    if init == "m01":
        return load_network(PACKAGE / "runs" / M01[seed], torch.device(device)).train()
    network = ResNet(layers=[3, 4, 6, 3], block="Bottleneck", num_classes=2)
    # 学習側（LitModule.load_backbone_checkpoint）と同じ部分ロード。fc は新規初期化。
    state = torch.load(ROOT / "checkpoints" / "resnet50_imagenet.pth", map_location="cpu", weights_only=False)
    state = state.get("state_dict", state)
    state = {k.removeprefix("net.").removeprefix("backbone."): v for k, v in state.items()}
    result = load_compatible_state_dict(network.backbone, state, load_fc=False)
    print("imagenet load:", len(result["loaded_keys"]), "loaded, missing", result["missing_keys"][:5])
    if result["missing_keys"]:
        raise RuntimeError("ImageNet 重みが backbone を埋めていない")
    return network.to(device).train()


def objective_loss(name: str, logits: torch.Tensor, y: torch.Tensor, g: torch.Tensor, cw: torch.Tensor) -> torch.Tensor:
    """群・ラベルを使う目的関数。`erm` は global inverse class weight の CE。"""
    ce = nn.functional.cross_entropy(logits, y, weight=cw)
    if name == "erm":
        return ce
    raise ValueError(name)


def evaluate(network: nn.Module, loader: DataLoader, device: str) -> np.ndarray:
    network.eval()
    scores = []
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        for image, *_ in loader:
            logits = network(image.to(device, non_blocking=True)).float()
            scores.append((logits[:, 1] - logits[:, 0]).cpu().numpy())
    network.train()
    return np.concatenate(scores)


def group_report(score: np.ndarray, frame: pd.DataFrame) -> dict[str, float]:
    group = ((frame.age >= 65).astype(int) * 4 + frame.race_group).to_numpy()
    y = frame.target.to_numpy()
    out = {"overall": roc_auc_score(y, score)}
    for g in range(8):
        m = group == g
        out[f"g{g}"] = roc_auc_score(y[m], score[m])
    old = frame.age.to_numpy() >= 65
    out["old"] = roc_auc_score(y[old], score[old])
    out["young"] = roc_auc_score(y[~old], score[~old])
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", required=True)
    parser.add_argument("--init", choices=("m01", "imagenet"), default="m01")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--subset", default="all")
    parser.add_argument("--objective", default="erm")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--batch", type=int, default=128)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max-steps-per-epoch", type=int, default=0, help="0 なら subset を 1 周")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    out = PACKAGE / "cache" / "probes" / args.name
    out.mkdir(parents=True, exist_ok=True)
    train_full = pd.read_csv(SPLITS / "train.csv")
    age_mean, age_std = train_full.age.mean(), train_full.age.std()
    train = subset(train_full, args.subset)
    frames = {s: pd.read_csv(SPLITS / f"{s}.csv") for s in ("val", "test")}
    # 共有メモリを抑える。16 worker・prefetch 4・persistent の評価 loader で 1 プロセス 25GB に達し、
    # 2026-10-06 に OOM killer が他プロセスまで止めた。評価 loader は persistent にしない。
    loader_kw = dict(num_workers=8, pin_memory=True, prefetch_factor=2)
    train_loader = DataLoader(
        Frame(train, train_transforms_medfair(), age_mean, age_std),
        batch_size=args.batch,
        shuffle=True,
        drop_last=True,
        persistent_workers=True,
        **loader_kw,
    )
    eval_loaders = {
        s: DataLoader(Frame(f, val_transforms_medfair(), age_mean, age_std), batch_size=256, **loader_kw)
        for s, f in frames.items()
    }
    # class weight は全 train の頻度から（M01 と同じ 0.206 / 1.794）。subset でも変えない。
    pos = train_full.target.mean()
    cw = torch.tensor([0.5 / (1 - pos), 0.5 / pos], device=args.device, dtype=torch.float32)
    network = build_network(args.init, args.seed, args.device)
    optimizer = torch.optim.AdamW(network.parameters(), lr=args.lr, weight_decay=1e-2)
    log = {"args": vars(args), "n_train": len(train), "epochs": []}
    for epoch in range(args.epochs):
        start = time.time()
        losses = []
        for step, (image, y, g, _meta, _) in enumerate(train_loader):
            if args.max_steps_per_epoch and step >= args.max_steps_per_epoch:
                break
            image, y, g = image.to(args.device, non_blocking=True), y.to(args.device), g.to(args.device)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                logits = network(image).float()
            loss = objective_loss(args.objective, logits, y, g, cw)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
        record = {"epoch": epoch, "train_loss": float(np.mean(losses)), "seconds": time.time() - start}
        for s, loader in eval_loaders.items():
            score = evaluate(network, loader, args.device)
            np.save(out / f"epoch_{epoch:02d}_{s}.npy", score.astype(np.float32))
            record[s] = group_report(score, frames[s])
        log["epochs"].append(record)
        print(
            json.dumps({k: (round(v, 4) if isinstance(v, float) else v) for k, v in record.items() if k != "test"}),
            flush=True,
        )
        (out / "log.json").write_text(json.dumps(log, indent=1))


if __name__ == "__main__":
    main()
