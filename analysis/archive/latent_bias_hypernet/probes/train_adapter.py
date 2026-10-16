"""backbone 側の 2 段目：凍結した 1 段目に Spatial LoRA（stage3・stage4・fc）を付けて学習する。

hypernet_e2e の `spatial_lora_chexpert_from_resnet` と同じ SpatialLoRAResNet を組み、1 段目の重みを
`load_base_state_dict` で読んで `freeze_base_model` で固定する。条件 c の作り方だけを差し替える。

- `meta`：hypernet_e2e と同じ MetadataEncoder（sex・race・ethnicity・撮影条件・age）
- `none`：全員に同じ学習可能な定数ベクトル（条件付けない残差 adapter）
- `router`：e(x) を受ける振り分け係が K 個のコードから 1 つ選び、そのコードブックのベクトルを c にする

`--fit-split` を 1 段目と同じ split（in-sample）にするか、見ていない split（cross-fit）にするかが主な比較軸。
run 契約の外で回す探索用の学習。毎 epoch の val / test score を保存し、選択規則は分析側で決める。

出力: `analysis/archive/latent_bias_hypernet/cache/adapter/<name>/`（`epoch_XX_{val,test}.npy`、`log.json`）

    uv run python analysis/archive/latent_bias_hypernet/probes/train_adapter.py --name ad_a_fitb_meta \
        --stage1 a_s42 --fit-split train_b --condition meta
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

from analysis.archive.latent_bias_hypernet.planting import marker_mask, stamp  # noqa: E402
from projects.hypernet_e2e.data.transform_utils import (  # noqa: E402
    train_transforms_chexpert,
    val_transforms_chexpert,
)
from projects.hypernet_e2e.models.resnet.network import ResNetBackbone  # noqa: E402
from projects.hypernet_e2e.models.spatial_hypernet.metadata_encoder import MetadataEncoder  # noqa: E402
from projects.hypernet_e2e.models.spatial_hypernet.network import SpatialLoRAResNet  # noqa: E402

PACKAGE = ROOT / "analysis" / "archive" / "latent_bias_hypernet"
SPLITS = ROOT / "data" / "chexpert" / "splits_crossfit"
IMAGES = ROOT / "data" / "chexpert" / "images"
CATEGORICAL = ("sex", "race", "ethnicity", "frontal_lateral", "ap_pa")
CARDINALITIES = (2, 6, 2, 2, 2)


class ConstantEncoder(nn.Module):
    """全員に同じ c を返す。条件付けない残差 adapter にするための encoder。"""

    def __init__(self, output_dim: int = 16):
        super().__init__()
        self.output_dim = output_dim
        self.code = nn.Parameter(torch.randn(output_dim))

    def forward(self, attributes):
        return self.code.expand(attributes["categorical"].shape[0], -1)


class RouterEncoder(nn.Module):
    """e(x) から K 個のコードの 1 つを選び、そのベクトルを c にする。直近 batch の振り分け確率を残す。"""

    def __init__(self, route_dim: int, K: int, output_dim: int = 16, hidden: int = 256):
        super().__init__()
        self.output_dim = output_dim
        self.router = nn.Sequential(nn.Linear(route_dim, hidden), nn.ReLU(), nn.Linear(hidden, K))
        self.codebook = nn.Parameter(torch.randn(K, output_dim))
        self.tau = 1.0
        self.last_p = None
        self.last_assign = None

    def forward(self, attributes):
        logits = self.router(attributes["route"])
        self.last_p = logits.softmax(-1)
        if self.training:
            q = nn.functional.gumbel_softmax(logits, tau=self.tau, hard=True)
        else:
            q = nn.functional.one_hot(logits.argmax(-1), logits.shape[-1]).float()
        self.last_assign = q.argmax(-1)
        return q @ self.codebook


class Rows(Dataset):
    def __init__(self, frame: pd.DataFrame, attrs: dict[str, np.ndarray], transform, marked: np.ndarray):
        self.images = frame.image.to_numpy(dtype=str)
        self.target = frame.target.to_numpy()
        self.attrs = attrs
        self.transform = transform
        self.marked = marked

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, i: int):
        with Image.open(IMAGES / self.images[i]) as img:
            image = img.convert("RGB")
        if self.marked[i]:
            image = stamp(image)
        return self.transform(image), int(self.target[i]), {k: v[i] for k, v in self.attrs.items()}


def attribute_arrays(frame: pd.DataFrame, age_mu: float, age_sd: float, route: np.ndarray | None) -> dict:
    """MetadataEncoder の入力辞書（と、router 用の e(x)）を作る。"""
    attrs = {
        "categorical": frame[list(CATEGORICAL)].to_numpy().astype(np.int64),
        "categorical_missing": frame[[f"{c}_missing" for c in CATEGORICAL]].to_numpy().astype(np.float32),
        "continuous": ((frame[["age"]].to_numpy() - age_mu) / age_sd).astype(np.float32),
        "continuous_missing": frame[["age_missing"]].to_numpy().astype(np.float32),
    }
    if route is not None:
        attrs["route"] = route
    return attrs


def build(args, route_dim: int) -> SpatialLoRAResNet:
    if args.condition == "meta":
        encoder = MetadataEncoder(list(CARDINALITIES), 1, embedding_dim=4, hidden_dim=128, output_dim=16)
    elif args.condition == "none":
        encoder = ConstantEncoder()
    else:
        encoder = RouterEncoder(route_dim, args.K)
    net = SpatialLoRAResNet(
        num_classes=2,
        backbone=ResNetBackbone([3, 4, 6, 3], "Bottleneck"),
        metadata_encoder=encoder,
        rank=4,
        modulation_stages=args.stages,
        adapted_convs=["conv2"],
    )
    state = torch.load(PACKAGE / "cache" / "stage1" / args.stage1 / "best.pt", map_location="cpu")
    result = net.load_base_state_dict(state)
    if result["missing_keys"] and any(k.startswith("backbone.") for k in result["missing_keys"]):
        raise RuntimeError(f"1 段目の重みが backbone を埋めていない: {result['missing_keys'][:5]}")
    net.freeze_base_model()
    return net


def to_device(attrs: dict, device: str) -> dict:
    return {k: v.to(device, non_blocking=True) for k, v in attrs.items()}


def evaluate(net, loader, device) -> tuple[np.ndarray, np.ndarray | None]:
    net.eval()
    scores, assigns = [], []
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        for image, _y, attrs in loader:
            logits = net(image.to(device, non_blocking=True), to_device(attrs, device)).float()
            scores.append((logits[:, 1] - logits[:, 0]).cpu().numpy())
            if isinstance(net.metadata_encoder, RouterEncoder):
                assigns.append(net.metadata_encoder.last_assign.cpu().numpy())
    net.train()
    net.backbone.eval()  # 凍結した BatchNorm の統計を動かさない
    return np.concatenate(scores), (np.concatenate(assigns) if assigns else None)


def worst8(score: np.ndarray, frame: pd.DataFrame) -> float:
    group = (frame.age.to_numpy() >= 65).astype(int) * 4 + frame.race_group.to_numpy()
    y = frame.target.to_numpy()
    return float(min(roc_auc_score(y[group == g], score[group == g]) for g in range(8)))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", required=True)
    parser.add_argument("--stage1", default="a_s42")
    parser.add_argument("--fit-split", choices=("train_a", "train_b"), required=True)
    parser.add_argument("--condition", choices=("meta", "none", "router"), default="meta")
    parser.add_argument("--route", default="raddino")
    parser.add_argument("--K", type=int, default=16)
    parser.add_argument("--beta", type=float, default=0.05)
    parser.add_argument("--stages", nargs="+", default=["stage3", "stage4", "fc"])
    parser.add_argument("--plant", action="store_true")
    parser.add_argument("--unfreeze-backbone", action="store_true", help="同時学習の対照（backbone も更新）")
    parser.add_argument("--epochs", type=int, default=6)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--batch", type=int, default=128)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    out = PACKAGE / "cache" / "adapter" / args.name
    out.mkdir(parents=True, exist_ok=True)
    names = (args.fit_split, "val", "test")
    frames = {s: pd.read_csv(SPLITS / f"{s}.csv") for s in names}
    marks = {s: (marker_mask(f, s) if args.plant else np.zeros(len(f), dtype=bool)) for s, f in frames.items()}
    fit = frames[args.fit_split]
    age_mu, age_sd = fit.age.mean(), fit.age.std()
    routes = {s: None for s in names}
    if args.condition == "router":
        emb_dir = PACKAGE / "cache" / "embed" / (args.route + ("_plant" if args.plant else ""))
        raw = {s: np.load(emb_dir / f"{s}.npz")["embedding"].astype(np.float32) for s in names}
        mu, sd = raw[args.fit_split].mean(0), raw[args.fit_split].std(0) + 1e-6
        routes = {s: (r - mu) / sd for s, r in raw.items()}
    attrs = {s: attribute_arrays(f, age_mu, age_sd, routes[s]) for s, f in frames.items()}

    net = build(args, routes[args.fit_split].shape[1] if args.condition == "router" else 0).to(args.device)
    if args.unfreeze_backbone:
        # 同時学習の対照：backbone と共有 classifier も更新する（BatchNorm の統計は凍結したまま）
        for parameter in net.parameters():
            parameter.requires_grad = True
    net.initialize_with_attributes({k: torch.as_tensor(v) for k, v in attrs[args.fit_split].items()})
    params = [p for p in net.parameters() if p.requires_grad]
    print("trainable params", sum(p.numel() for p in params), flush=True)
    optimizer = torch.optim.AdamW(params, lr=args.lr, weight_decay=1e-2)
    pos = fit.target.mean()
    cw = torch.tensor([0.5 / (1 - pos), 0.5 / pos], device=args.device, dtype=torch.float32)
    kw = dict(num_workers=8, pin_memory=True, prefetch_factor=2)
    train_loader = DataLoader(
        Rows(fit, attrs[args.fit_split], train_transforms_chexpert(), marks[args.fit_split]),
        batch_size=args.batch,
        shuffle=True,
        drop_last=True,
        persistent_workers=True,
        **kw,
    )
    eval_loaders = {
        s: DataLoader(Rows(frames[s], attrs[s], val_transforms_chexpert(), marks[s]), batch_size=256, **kw)
        for s in ("val", "test")
    }
    log = {"args": vars(args), "epochs": []}
    for split in ("val", "test"):  # epoch -1 = 補正前（LoRA の差分は初期値 0）
        score, _ = evaluate(net, eval_loaders[split], args.device)
        np.save(out / f"epoch_-1_{split}.npy", score)
    net.train()
    net.backbone.eval()
    for epoch in range(args.epochs):
        start = time.time()
        losses = []
        if isinstance(net.metadata_encoder, RouterEncoder):
            net.metadata_encoder.tau = max(0.3, 0.8**epoch)
        for image, y, a in train_loader:
            image, y, a = image.to(args.device, non_blocking=True), y.to(args.device), to_device(a, args.device)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                logits = net(image, a).float()
            loss = nn.functional.cross_entropy(logits, y, weight=cw)
            if isinstance(net.metadata_encoder, RouterEncoder):
                p_bar = net.metadata_encoder.last_p.float().mean(0)
                loss = loss + args.beta * (p_bar * torch.log(p_bar + 1e-9)).sum()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
        record = {"epoch": epoch, "train_loss": float(np.mean(losses)), "sec": time.time() - start}
        for split in ("val", "test"):
            score, assign = evaluate(net, eval_loaders[split], args.device)
            np.save(out / f"epoch_{epoch:02d}_{split}.npy", score)
            if assign is not None:
                np.save(out / f"epoch_{epoch:02d}_{split}_assign.npy", assign)
            record[f"{split}_auroc"] = roc_auc_score(frames[split].target, score)
            record[f"{split}_worst8"] = worst8(score, frames[split])
        log["epochs"].append(record)
        print(json.dumps(record), flush=True)
        (out / "log.json").write_text(json.dumps(log, indent=1))
    torch.save(net.state_dict(), out / "final.pt")


if __name__ == "__main__":
    main()
