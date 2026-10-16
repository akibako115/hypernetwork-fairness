"""凍結 M01 表現と共有線形 head の上で、目的関数だけを変えて群別 AUROC を比べる。

問い：明示的な公平性損失は、表現を固定したとき群内の順位をどこまで動かせるか。
heads.py が「条件付けの形」を振るのに対し、ここは「目的関数」を振る（head は共有 linear で固定）。

- `erm`：global inverse class weight の BCE
- `group_balanced`：群ごとの平均損失を等重みで平均
- `group_dro`：群損失の指数勾配で q を更新（step 0.01、学習側と同じ形）
- `rank_worst_l<λ>`：BCE + λ·softmax 重み付きの群内 pairwise logistic 順位損失
  （群 × クラスで 32 件ずつ別に抽出した順位用 batch で計算。CE 側の分布は変えない）

出力: `results/objectives_frozen/objectives_s<seed>.csv`

    uv run python analysis/archive/fairness_mechanism_probe/objectives_frozen.py --seed 42
"""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd
import torch
from features import PACKAGE, load_split
from heads import N_GROUPS, group_aurocs
from torch import nn

PER_CELL = 32


def cell_indices(target: np.ndarray, group: np.ndarray) -> list[list[torch.Tensor]]:
    """群 g・クラス c の train index。"""
    return [[torch.from_numpy(np.flatnonzero((group == g) & (target == c))) for c in (0, 1)] for g in range(N_GROUPS)]


def rank_losses(head: nn.Linear, x: torch.Tensor, cells, device: str) -> torch.Tensor:
    """各群で陽性・陰性 32 件ずつ抽出し、pairwise logistic 順位損失を返す（[8]）。"""
    out = []
    for neg, pos in cells:
        i_neg = neg[torch.randint(len(neg), (PER_CELL,))].to(device)
        i_pos = pos[torch.randint(len(pos), (PER_CELL,))].to(device)
        s_neg, s_pos = head(x[i_neg]).squeeze(1), head(x[i_pos]).squeeze(1)
        out.append(nn.functional.softplus(-(s_pos[:, None] - s_neg[None, :])).mean())
    return torch.stack(out)


def train(objective: str, lam: float, data: dict, cells, val, device: str, epochs: int = 15) -> nn.Linear:
    torch.manual_seed(0)
    x, y, g = data["x"], data["y"], data["g"]
    head = nn.Linear(x.shape[1], 1).to(device)
    optimizer = torch.optim.AdamW(head.parameters(), lr=1e-3, weight_decay=1e-4)
    pos_rate = y.mean()
    weight = torch.where(y > 0, 0.5 / pos_rate, 0.5 / (1 - pos_rate))
    q = torch.full((N_GROUPS,), 1.0 / N_GROUPS, device=device)
    best, best_state = -1.0, None
    for _ in range(epochs):
        for batch in torch.randperm(len(y), device=device).split(1024):
            loss_i = nn.functional.binary_cross_entropy_with_logits(
                head(x[batch]).squeeze(1), y[batch], weight=weight[batch], reduction="none"
            )
            if objective in ("erm", "rank_worst"):
                loss = loss_i.mean()
            else:
                counts = torch.bincount(g[batch], minlength=N_GROUPS).float()
                group_loss = torch.zeros(N_GROUPS, device=device).index_add_(0, g[batch], loss_i) / counts.clamp(min=1)
                present = counts > 0
                if objective == "group_balanced":
                    loss = group_loss[present].mean()
                else:
                    q = q * torch.exp(0.01 * group_loss.detach() * present)
                    q = q / q.sum()
                    loss = (q * group_loss).sum()
            if objective == "rank_worst":
                ranks = rank_losses(head, x, cells, device)
                loss = loss + lam * (torch.softmax(ranks.detach() / 0.05, 0) * ranks).sum()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        with torch.no_grad():
            score = head(val["x"]).squeeze(1).cpu().numpy()
        auc = group_aurocs(score, val["target"], val["group"])["overall"]
        if auc > best:
            best, best_state = auc, {k: v.clone() for k, v in head.state_dict().items()}
    head.load_state_dict(best_state)
    return head


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    splits = {s: load_split(args.seed, s) for s in ("train", "val", "test")}
    mean, std = splits["train"].features.mean(0), splits["train"].features.std(0) + 1e-6

    def prep(split):
        return {
            "x": torch.from_numpy((split.features - mean) / std).to(args.device),
            "y": torch.from_numpy(split.target.astype(np.float32)).to(args.device),
            "g": torch.from_numpy(split.group).to(args.device),
            "target": split.target,
            "group": split.group,
        }

    data = {s: prep(split) for s, split in splits.items()}
    cells = cell_indices(splits["train"].target, splits["train"].group)
    configs = [
        ("erm", 0.0),
        ("group_balanced", 0.0),
        ("group_dro", 0.0),
        ("rank_worst", 0.3),
        ("rank_worst", 1.0),
        ("rank_worst", 3.0),
    ]
    rows = []
    for objective, lam in configs:
        head = train(objective, lam, data["train"], cells, data["val"], args.device)
        with torch.no_grad():
            score = head(data["test"]["x"]).squeeze(1).cpu().numpy()
        old = splits["test"].frame.age.to_numpy() >= 65
        result = group_aurocs(score, data["test"]["target"], data["test"]["group"])
        from sklearn.metrics import roc_auc_score

        result["old"] = roc_auc_score(data["test"]["target"][old], score[old])
        result["young"] = roc_auc_score(data["test"]["target"][~old], score[~old])
        label = objective if not lam else f"{objective}_l{lam}"
        rows.append({"seed": args.seed, "objective": label, **result})
        print(
            label,
            {k: round(v, 4) for k, v in result.items() if k in ("overall", "worst", "mean_group", "old", "young")},
            flush=True,
        )
    out = PACKAGE / "results" / "objectives_frozen"
    out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out / f"objectives_s{args.seed}.csv", index=False)


if __name__ == "__main__":
    main()
