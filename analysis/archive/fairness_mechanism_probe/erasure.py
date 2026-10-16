"""凍結 M01 表現から属性（年齢・race）の線形情報を消し、共有 head を学習し直して群別性能を比べる。

問い：属性を表現から「消す」方向（不変化）と、属性で「条件付ける」方向（heads.py）は、
群内の順位・群間の動作点にそれぞれどう効くか。

- 消去は LEACE（Belrose et al., 2023）の閉形式：白色化空間で属性との相互共分散の列空間を射影で除く。
  年齢は連続値 1 次元と 10 歳刻み one-hot、race は 4 区分 one-hot の 3 通り。
- head は global inverse class weight の logistic 回帰（共有、属性入力なし）。
- 消去後に属性がどれだけ残るかを、非線形 probe（MLP）でも確かめる。

出力: `results/erasure/erasure_s<seed>.csv`

    uv run python analysis/archive/fairness_mechanism_probe/erasure.py --seed 42
"""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd
import torch
from features import PACKAGE, load_split
from heads import group_aurocs
from sklearn.metrics import r2_score, roc_auc_score
from torch import nn


def leace(x: torch.Tensor, z: torch.Tensor) -> callable:
    """x [n,d]・z [n,k] から、z と線形無相関になる消去関数を返す（中心化は内部で行う）。"""
    mu_x, mu_z = x.mean(0), z.mean(0)
    xc, zc = x - mu_x, z - mu_z
    cov = xc.T @ xc / len(x) + 1e-4 * torch.eye(x.shape[1], device=x.device)
    eigval, eigvec = torch.linalg.eigh(cov)
    whiten = eigvec @ torch.diag(eigval.rsqrt()) @ eigvec.T
    unwhiten = eigvec @ torch.diag(eigval.sqrt()) @ eigvec.T
    cross = whiten @ (xc.T @ zc / len(x))  # [d,k]
    u, s, _ = torch.linalg.svd(cross, full_matrices=False)
    u = u[:, s > 1e-6 * s.max()]
    projection = unwhiten @ u @ u.T @ whiten  # 消す成分

    def erase(v: torch.Tensor) -> torch.Tensor:
        return v - (v - mu_x) @ projection.T

    return erase


def fit_logistic(x, y, val_x, val_y, epochs=15, seed=0):
    torch.manual_seed(seed)
    head = nn.Linear(x.shape[1], 1).to(x.device)
    opt = torch.optim.AdamW(head.parameters(), lr=1e-3, weight_decay=1e-4)
    pos = y.mean()
    w = torch.where(y > 0, 0.5 / pos, 0.5 / (1 - pos))
    best, state = -1, None
    for _ in range(epochs):
        for b in torch.randperm(len(y), device=x.device).split(1024):
            loss = nn.functional.binary_cross_entropy_with_logits(head(x[b]).squeeze(1), y[b], weight=w[b])
            opt.zero_grad()
            loss.backward()
            opt.step()
        with torch.no_grad():
            auc = roc_auc_score(val_y.cpu().numpy(), head(val_x).squeeze(1).cpu().numpy())
        if auc > best:
            best, state = auc, {k: v.clone() for k, v in head.state_dict().items()}
    head.load_state_dict(state)
    return head


def mlp_probe(x, z, val_x, val_z, regression: bool, epochs=10):
    """消去後に残る属性情報を非線形 probe で測る（train で学習、val で評価）。"""
    torch.manual_seed(0)
    out_dim = 1 if regression else int(z.max().item()) + 1
    net = nn.Sequential(nn.Linear(x.shape[1], 256), nn.ReLU(), nn.Linear(256, out_dim)).to(x.device)
    opt = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=1e-4)
    for _ in range(epochs):
        for b in torch.randperm(len(x), device=x.device).split(1024):
            pred = net(x[b])
            loss = (
                nn.functional.mse_loss(pred.squeeze(1), z[b]) if regression else nn.functional.cross_entropy(pred, z[b])
            )
            opt.zero_grad()
            loss.backward()
            opt.step()
    with torch.no_grad():
        pred = net(val_x)
    if regression:
        return r2_score(val_z.cpu().numpy(), pred.squeeze(1).cpu().numpy())
    prob = torch.softmax(pred, 1).cpu().numpy()
    return roc_auc_score(val_z.cpu().numpy(), prob, multi_class="ovr")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    dev = args.device
    sp = {s: load_split(args.seed, s) for s in ("train", "val", "test")}
    mean, std = sp["train"].features.mean(0), sp["train"].features.std(0) + 1e-6
    x = {s: torch.from_numpy((v.features - mean) / std).to(dev) for s, v in sp.items()}
    y = {s: torch.from_numpy(v.target.astype(np.float32)).to(dev) for s, v in sp.items()}
    age = {s: torch.from_numpy(((v.frame.age.to_numpy() - 60) / 18).astype(np.float32)).to(dev) for s, v in sp.items()}
    race = {s: torch.from_numpy(v.frame.race_group.to_numpy()).to(dev) for s, v in sp.items()}
    decade = sp["train"].frame.age.floordiv(10).clip(1, 9).to_numpy().astype(int) - 1
    concepts = {
        "none": None,
        "age_linear": age["train"][:, None],
        "age_decade_onehot": torch.from_numpy(np.eye(9, dtype=np.float32)[decade]).to(dev),
        "race_onehot": nn.functional.one_hot(race["train"], 4).float(),
        "age_decade+race": torch.cat(
            [
                torch.from_numpy(np.eye(9, dtype=np.float32)[decade]).to(dev),
                nn.functional.one_hot(race["train"], 4).float(),
            ],
            1,
        ),
    }
    rows = []
    test = sp["test"]
    old = test.frame.age.to_numpy() >= 65
    for name, z in concepts.items():
        erase = (lambda v: v) if z is None else leace(x["train"], z)
        xe = {s: erase(v) for s, v in x.items()}
        head = fit_logistic(xe["train"], y["train"], xe["val"], y["val"])
        with torch.no_grad():
            score = head(xe["test"]).squeeze(1).cpu().numpy()
        r = {"seed": args.seed, "erased": name, **group_aurocs(score, test.target, test.group)}
        r["old"] = roc_auc_score(test.target[old], score[old])
        r["young"] = roc_auc_score(test.target[~old], score[~old])
        pos = test.target == 1
        r["tpr_old"], r["tpr_young"] = (score[pos & old] > 0).mean(), (score[pos & ~old] > 0).mean()
        neg = ~pos
        r["tnr_old"], r["tnr_young"] = (score[neg & old] <= 0).mean(), (score[neg & ~old] <= 0).mean()
        r["mlp_age_r2"] = mlp_probe(xe["train"], age["train"], xe["test"], age["test"], True)
        r["mlp_race_auroc"] = mlp_probe(xe["train"], race["train"], xe["test"], race["test"], False)
        rows.append(r)
        print({k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items()}, flush=True)
    out = PACKAGE / "results" / "erasure"
    out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out / f"erasure_s{args.seed}.csv", index=False)


if __name__ == "__main__":
    main()
