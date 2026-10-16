"""凍結 M01 表現の上で、属性による条件付けの形だけを変えた head を学習し、群別 AUROC を比べる。

問い：群ごとに異なる決定境界を許すと、群内の順位（AUROC）は改善するか。改善するなら
どの形の条件付け（切片だけ／群別の重み／連続年齢との交互作用／非線形）が要るか。

入力は `features.load_split`（M01 seed 42–44 の train / val / test 表現）。
train で学習し、val の全体 AUROC で epoch を選び、test を報告する。
出力は `results/heads/heads_by_seed.csv`（seed × head × 群）。

    uv run python analysis/archive/fairness_mechanism_probe/heads.py --seeds 42 43 44
"""

from __future__ import annotations

import argparse
from collections.abc import Callable

import numpy as np
import pandas as pd
import torch
from features import GROUP_NAMES, PACKAGE, Split, load_split
from sklearn.metrics import roc_auc_score
from torch import nn

N_GROUPS = len(GROUP_NAMES)
CATEGORICAL = (("sex", 2), ("race", 6), ("ethnicity", 2), ("frontal_lateral", 2), ("ap_pa", 2))


def metadata_matrix(frame: pd.DataFrame, age_mean: float, age_std: float) -> np.ndarray:
    """モデル入力と同じ属性（カテゴリの one-hot と標準化 age）を並べる。"""
    parts = [np.eye(card, dtype=np.float32)[frame[name].to_numpy()] for name, card in CATEGORICAL]
    parts.append(((frame["age"].to_numpy() - age_mean) / age_std).astype(np.float32)[:, None])
    return np.concatenate(parts, axis=1)


class Head(nn.Module):
    """s(x, a) を返す head。`kind` で条件付けの形を切り替える。"""

    def __init__(self, kind: str, dim: int, meta_dim: int, rank: int = 4, hidden: int = 256):
        super().__init__()
        self.kind = kind
        self.shared = nn.Linear(dim, 1)
        if kind in ("group_bias", "group_delta", "group_full"):
            self.group_bias = nn.Parameter(torch.zeros(N_GROUPS))
        if kind in ("group_delta", "group_full"):
            self.group_weight = nn.Parameter(torch.zeros(N_GROUPS, dim))
        if kind == "age_delta":
            self.age_weight = nn.Linear(dim, 1)
        if kind == "hyper":
            self.encoder = nn.Sequential(nn.Linear(meta_dim, 32), nn.ReLU(), nn.Linear(32, rank + 1))
            self.basis = nn.Parameter(torch.zeros(dim, rank))
        if kind in ("mlp", "mlp_meta"):
            in_dim = dim + (meta_dim if kind == "mlp_meta" else 0)
            self.mlp = nn.Sequential(nn.Linear(in_dim, hidden), nn.ReLU(), nn.Dropout(0.2), nn.Linear(hidden, 1))

    def forward(self, x: torch.Tensor, group: torch.Tensor, meta: torch.Tensor) -> torch.Tensor:
        if self.kind == "mlp":
            return self.mlp(x).squeeze(1)
        if self.kind == "mlp_meta":
            return self.mlp(torch.cat([x, meta], dim=1)).squeeze(1)
        score = self.shared(x).squeeze(1)
        if self.kind == "group_full":
            # 共有重みを使わず、群ごとに独立した線形 head（群の外の標本は勾配に寄与しない）
            return (x * self.group_weight[group]).sum(1) + self.group_bias[group]
        if self.kind == "group_bias":
            return score + self.group_bias[group]
        if self.kind == "group_delta":
            return score + (x * self.group_weight[group]).sum(1) + self.group_bias[group]
        if self.kind == "age_delta":
            return score + meta[:, -1] * self.age_weight(x).squeeze(1)
        if self.kind == "hyper":
            code = self.encoder(meta)
            return score + ((x @ self.basis) * code[:, :-1]).sum(1) + code[:, -1]
        return score

    def penalty(self) -> torch.Tensor:
        """属性依存部分の L2。共有部分は AdamW の weight decay に任せる。"""
        if self.kind == "group_delta":
            return self.group_weight.pow(2).sum(1).mean()
        if self.kind == "group_full":
            return self.group_weight.pow(2).sum(1).mean()
        if self.kind == "age_delta":
            return self.age_weight.weight.pow(2).sum()
        if self.kind == "hyper":
            return self.basis.pow(2).sum()
        return torch.zeros((), device=self.shared.weight.device)


def group_aurocs(score: np.ndarray, target: np.ndarray, group: np.ndarray) -> dict[str, float]:
    """全体・8 群・worst・群平均の AUROC。"""
    result = {"overall": roc_auc_score(target, score)}
    per_group = []
    for g, name in enumerate(GROUP_NAMES):
        mask = group == g
        per_group.append(roc_auc_score(target[mask], score[mask]))
        result[name] = per_group[-1]
    result["worst"] = min(per_group)
    result["mean_group"] = float(np.mean(per_group))
    return result


def to_device(split: Split, mean: np.ndarray, std: np.ndarray, meta: np.ndarray, device: str) -> dict:
    """標準化した表現と属性を GPU に載せる。"""
    return {
        "x": torch.from_numpy((split.features - mean) / std).to(device),
        "y": torch.from_numpy(split.target.astype(np.float32)).to(device),
        "g": torch.from_numpy(split.group).to(device),
        "m": torch.from_numpy(meta).to(device),
    }


def predict(head: Head, data: dict) -> np.ndarray:
    head.eval()
    with torch.no_grad():
        out = torch.cat(
            [
                head(x, g, m)
                for x, g, m in zip(data["x"].split(8192), data["g"].split(8192), data["m"].split(8192), strict=True)
            ]
        )
    head.train()
    return out.float().cpu().numpy()


def fit_head(head: Head, train: dict, val: dict, val_split: Split, penalty: float, epochs: int = 15) -> Head:
    """global inverse class weight の BCE で学習し、val 全体 AUROC 最大の epoch の重みを返す。"""
    pos_rate = train["y"].mean()
    weight = torch.where(train["y"] > 0, 0.5 / pos_rate, 0.5 / (1 - pos_rate))
    optimizer = torch.optim.AdamW(head.parameters(), lr=1e-3, weight_decay=1e-4)
    n = len(train["y"])
    best, best_state = -1.0, None
    for _ in range(epochs):
        order = torch.randperm(n, device=train["y"].device)
        for batch in order.split(1024):
            logit = head(train["x"][batch], train["g"][batch], train["m"][batch])
            loss = nn.functional.binary_cross_entropy_with_logits(logit, train["y"][batch], weight=weight[batch])
            loss = loss + penalty * head.penalty()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        auc = roc_auc_score(val_split.target, predict(head, val))
        if auc > best:
            best, best_state = auc, {k: v.detach().clone() for k, v in head.state_dict().items()}
    head.load_state_dict(best_state)
    return head


HEADS: dict[str, tuple[str, float]] = {
    "shared": ("shared", 0.0),
    "group_bias": ("group_bias", 0.0),
    "group_delta_l2=1e-1": ("group_delta", 1e-1),
    "group_delta_l2=1e-2": ("group_delta", 1e-2),
    "group_delta_l2=1e-3": ("group_delta", 1e-3),
    "group_full": ("group_full", 1e-3),
    "age_delta": ("age_delta", 1e-3),
    "hyper_rank4": ("hyper", 1e-3),
    "mlp": ("mlp", 0.0),
    "mlp_meta": ("mlp_meta", 0.0),
}


def run_seed(seed: int, device: str, log: Callable[[str], None]) -> list[dict]:
    splits = {name: load_split(seed, name) for name in ("train", "val", "test")}
    train = splits["train"]
    mean, std = train.features.mean(0), train.features.std(0) + 1e-6
    age_mean, age_std = train.frame["age"].mean(), train.frame["age"].std()
    data = {
        name: to_device(split, mean, std, metadata_matrix(split.frame, age_mean, age_std), device)
        for name, split in splits.items()
    }
    rows = []
    original = splits["test"].logits[:, 1] - splits["test"].logits[:, 0]
    rows.append(
        {"seed": seed, "head": "original_fc", **group_aurocs(original, splits["test"].target, splits["test"].group)}
    )
    for label, (kind, penalty) in HEADS.items():
        torch.manual_seed(seed)
        head = Head(kind, train.features.shape[1], data["train"]["m"].shape[1]).to(device)
        head = fit_head(head, data["train"], data["val"], splits["val"], penalty)
        score = predict(head, data["test"])
        rows.append({"seed": seed, "head": label, **group_aurocs(score, splits["test"].target, splits["test"].group)})
        log(f"seed {seed} {label}: overall {rows[-1]['overall']:.4f} worst {rows[-1]['worst']:.4f}")
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    rows = []
    for seed in args.seeds:
        rows.extend(run_seed(seed, args.device, print))
    out = PACKAGE / "results" / "heads"
    out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out / "heads_by_seed.csv", index=False)


if __name__ == "__main__":
    main()
