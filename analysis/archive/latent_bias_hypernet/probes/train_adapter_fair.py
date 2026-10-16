"""Q2：2 段目の hypernet に公平性損失をかけ、属性の群の性能差が縮むかを見る。

凍結した 1 段目に Spatial LoRA（stage3・stage4・fc）を付けて学習する（`train_adapter.py` と同じ構成）。
主に回すのは `meta`（属性で条件付けた hypernet）。`none`（全員に同じ定数）は
容量をそろえた対照で、hypernet の利得が出たときにだけ要る。
損失は 3 つ。群は年齢（65 歳で 2 区分）× race の 8 群。

- `erm`：class weight 付きの交差エントロピー。ふつうの抽出
- `eo`：EO rate 制約（Cotter et al. JMLR 2019 の proxy-Lagrangian、`EqualizedOddsRateTaskLoss`）。
  8 群の全有向ペアの TPR 差・FPR 差 ≤ 0.1（112 本）。task 側へ寄せて始める（初期 task 重み 0.9。
  一様だと task の重みが 1/113 になり、全予測を片側へ寄せる自明解に落ちる。TFCO の既定も task 寄り）
- `minimax_auc`：Yang et al. AAAI 2023 の群ペア AUC の minimax（`MinimaxAUCTaskLoss`、全 64 ペア）

公平性損失は群 × クラスの 16 セルから 8 件ずつ抽出したバッチで学習する（全セルが無いと損失を定義できない）。
**群の重み（双対変数）は、学習に使わない held-out で測った値で更新する。**
2 段目の学習データの患者 20% を held-out にし、
残り 80% で重みを学習する。学習に使った画像で測ると、少数セルを記憶して差が消えたように見え、重みが戻ってしまう
（`conditional_fairness_one_stage`、2026-10-08）。Cotter et al. の 2 データセット方式にあたる。
ERM も同じ 80% で学習し、データの量をそろえる。

`--procedure` で 2 段目の学習の仕方を選ぶ。どちらも、1 段目と 2 段目を合わせて train 全体を使う。

- `crossfit`：1 段目は train_a で学習した `a_s42`。2 段目は 1 段目が見ていない train_b の 80% で学習し、
  20% を held-out にする
- `insample`：1 段目は train 全体で学習した M01（`probes/export_m01_stage1.py`）。2 段目も同じ train の 80% で学習し、
  20% を held-out にする（1 段の run と同じ分け方、`data/chexpert/splits_race_known_dual_holdout/`）。
  held-out も 1 段目が学習した画像なので、群の成績は 1 段目が記憶した画像の上で測ることになる。
  cross-fit の動機（学習済みのデータでは残差が小さく、補正に勾配が通らない）を確かめるための条件

出力: `analysis/archive/latent_bias_hypernet/cache/adapter_fair/<procedure>/<loss>_<condition>_s<seed>/`
（`epoch_XX_{val,test}.npy`、`log.json`。log には epoch ごとの held-out の群別の率と、群の重みを残す）

    uv run python analysis/archive/latent_bias_hypernet/probes/train_adapter_fair.py \
        --procedure crossfit --loss eo --condition meta --seed 42
"""

from __future__ import annotations

import argparse
import json
import time

import numpy as np
import pandas as pd
import rootutils
import torch
from sklearn.metrics import roc_auc_score
from torch import nn
from torch.utils.data import DataLoader

ROOT = rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from analysis.archive.latent_bias_hypernet.probes.train_adapter import (  # noqa: E402
    PACKAGE,
    SPLITS,
    Rows,
    attribute_arrays,
    build,
    evaluate,
    to_device,
    worst8,
)
from projects.hypernet_e2e.data.transform_utils import (  # noqa: E402
    train_transforms_chexpert,
    val_transforms_chexpert,
)
from projects.hypernet_e2e.objectives.equalized_odds import EqualizedOddsRateTaskLoss  # noqa: E402
from projects.hypernet_e2e.objectives.input import ObjectiveInput  # noqa: E402
from projects.hypernet_e2e.objectives.minimax_auc import MinimaxAUCTaskLoss  # noqa: E402

G = 8
PER_CELL = 8  # 16 セル × 8 件 = 128 件。ERM のバッチと同じ大きさ
HOLDOUT = 0.2
BATCH = 128
STAGE1 = {"crossfit": "a_s42", "insample": "m01_s42"}
INSAMPLE_SPLITS = ROOT / "data" / "chexpert" / "splits_race_known_dual_holdout"


def demo8(frame: pd.DataFrame) -> np.ndarray:
    """年齢（65 歳で 2 区分）× race 4 区分の 8 群。q1_definitions の属性群と同じ番号。"""
    return (frame.age.to_numpy() >= 65).astype(int) * 4 + frame.race_group.to_numpy().astype(int)


def split_holdout(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """患者単位で held-out を切り出す。seed によらず同じ分け方にする（runs の間で比べるため）。"""
    patient = frame.image.str.split("/").str[1]
    units = patient.unique()
    held = set(np.random.default_rng(0).choice(units, int(len(units) * HOLDOUT), replace=False))
    mask = patient.isin(held).to_numpy()
    return frame[~mask].reset_index(drop=True), frame[mask].reset_index(drop=True)


def cell_batches(group: np.ndarray, y: np.ndarray, steps: int, rng: np.random.Generator) -> list[np.ndarray]:
    """16 セルから PER_CELL 件ずつ復元抽出したバッチの行番号を steps 本作る。"""
    cells = [np.flatnonzero((group == g) & (y == c)) for g in range(G) for c in (0, 1)]
    return [np.concatenate([rng.choice(cell, PER_CELL) for cell in cells]) for _ in range(steps)]


def make_objective(loss: str, group: np.ndarray, y: np.ndarray, class_weight: list[float]) -> nn.Module:
    """学習データ（80%）の群 × クラスの件数から公平性損失を作る。ステップ幅などは既存実装の論文準拠の既定値。"""
    counts = [[float(((group == g) & (y == c)).sum()) for c in (0, 1)] for g in range(G)]
    if loss == "eo":
        return EqualizedOddsRateTaskLoss(G, counts, class_weight=class_weight, initial_task_weight=0.9)
    return MinimaxAUCTaskLoss(G, counts, pair_scope="all")


def held_out_dual_step(net, objective, batch, device) -> dict:
    """held-out のバッチで群の成績を測り、双対変数だけを更新する。モデルの勾配は作らない。"""
    image, y, attrs = batch
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        logits = net(image.to(device, non_blocking=True), to_device(attrs, device)).float()
    attrs = to_device(attrs, device)
    objective(ObjectiveInput(logits, y.to(device), {"group_id": attrs["group_id"]}))
    objective.commit_update()
    diag = objective.batch_diagnostics()
    if isinstance(objective, EqualizedOddsRateTaskLoss):
        return {"rates": diag["rates"].cpu().numpy()}  # [G, 2]：陰性の FPR、陽性の TPR
    return {"risk": diag["risk"].cpu().numpy()}  # 64 ペアの順位損失


def dual_summary(objective) -> dict:
    """epoch 末の群の重み。EO は λ（先頭が task）、minimax AUC は q を陽性群 × 陰性群の 8×8 に並べたもの。"""
    if isinstance(objective, EqualizedOddsRateTaskLoss):
        lam, _ = objective._stationary()
        return {"lambda": lam.cpu().numpy().tolist()}
    return {"q": objective.log_q.exp().reshape(G, G).cpu().numpy().tolist()}


def train_epoch(net, objective, optimizer, loaders, cw, args) -> dict:
    """1 epoch 学習する。公平性損失では、モデルを 1 歩更新するたびに held-out で双対変数を 1 歩進める。"""
    losses, held = [], []
    held_iter = iter(loaders["holdout"]) if objective is not None else None
    for image, y, a in loaders["fit"]:
        image, y, a = image.to(args.device, non_blocking=True), y.to(args.device), to_device(a, args.device)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            logits = net(image, a).float()
        if objective is None:
            loss = nn.functional.cross_entropy(logits, y, weight=cw)
        else:
            loss = objective(ObjectiveInput(logits, y, {"group_id": a["group_id"]}))
            objective._pending = None  # 学習バッチの統計は双対の更新に使わない
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
        if objective is not None:
            held.append(held_out_dual_step(net, objective, next(held_iter), args.device))
    record = {"train_loss": float(np.mean(losses))}
    if held:
        key = "rates" if "rates" in held[0] else "risk"
        record[f"holdout_{key}"] = np.mean([h[key] for h in held], axis=0).tolist()
        record |= dual_summary(objective)
    return record


def loaders_for(args, frames, attrs, group) -> dict:
    """学習・held-out・評価の DataLoader。公平性損失の 2 つはセル均等抽出、ERM はふつうの抽出。"""
    kw = dict(num_workers=args.workers, pin_memory=True, prefetch_factor=2)
    zeros = {s: np.zeros(len(f), dtype=bool) for s, f in frames.items()}
    fit_rows = Rows(frames["fit"], attrs["fit"], train_transforms_chexpert(), zeros["fit"])
    steps = len(frames["fit"]) // BATCH
    rng = np.random.default_rng(args.seed)
    if args.loss == "erm":
        fit = DataLoader(fit_rows, batch_size=BATCH, shuffle=True, drop_last=True, **kw)
    else:
        batches = cell_batches(group["fit"], frames["fit"].target.to_numpy(), steps * args.epochs, rng)
        fit = [batches[e * steps : (e + 1) * steps] for e in range(args.epochs)]
    out = {"fit": fit, "fit_rows": fit_rows, "kw": kw, "steps": steps, "rng": rng}
    out["holdout_rows"] = Rows(frames["holdout"], attrs["holdout"], val_transforms_chexpert(), zeros["holdout"])
    for s in ("val", "test"):
        out[s] = DataLoader(Rows(frames[s], attrs[s], val_transforms_chexpert(), zeros[s]), batch_size=256, **kw)
    return out


def epoch_loaders(prep: dict, epoch: int, group: dict, frames: dict, loss: str) -> dict:
    """その epoch の学習・held-out の DataLoader。セル均等抽出のバッチは前もって決めてある。"""
    if loss == "erm":
        return {"fit": prep["fit"]}
    fit = DataLoader(prep["fit_rows"], batch_sampler=prep["fit"][epoch], **prep["kw"])
    held_batches = cell_batches(group["holdout"], frames["holdout"].target.to_numpy(), prep["steps"], prep["rng"])
    return {"fit": fit, "holdout": DataLoader(prep["holdout_rows"], batch_sampler=held_batches, **prep["kw"])}


def setup(args):
    """split・属性・群・モデル・損失を用意する。"""
    if args.procedure == "crossfit":
        fit, holdout = split_holdout(pd.read_csv(SPLITS / "train_b.csv"))
    else:
        fit, holdout = (pd.read_csv(INSAMPLE_SPLITS / f"{s}.csv") for s in ("train", "holdout"))
    frames = {"fit": fit, "holdout": holdout} | {s: pd.read_csv(SPLITS / f"{s}.csv") for s in ("val", "test")}
    age_mu, age_sd = fit.age.mean(), fit.age.std()
    group = {s: demo8(f) for s, f in frames.items()}
    attrs = {s: attribute_arrays(f, age_mu, age_sd, None) | {"group_id": group[s]} for s, f in frames.items()}
    stage1 = STAGE1[args.procedure]
    net = build(argparse.Namespace(condition=args.condition, stages=args.stages, stage1=stage1, K=0), 0)
    net = net.to(args.device)
    net.initialize_with_attributes({k: torch.as_tensor(v) for k, v in attrs["fit"].items() if k != "group_id"})
    pos = fit.target.mean()
    cw_list = [0.5 / (1 - pos), 0.5 / pos]
    objective = None
    if args.loss != "erm":
        objective = make_objective(args.loss, group["fit"], fit.target.to_numpy(), cw_list).to(args.device).train()
    cw = torch.tensor(cw_list, device=args.device, dtype=torch.float32)
    return frames, attrs, group, net, objective, cw


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--loss", choices=("erm", "eo", "minimax_auc"), required=True)
    parser.add_argument("--condition", choices=("meta", "none"), required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--procedure", choices=("crossfit", "insample"), required=True)
    parser.add_argument("--stages", nargs="+", default=["stage3", "stage4", "fc"])
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    out = PACKAGE / "cache" / "adapter_fair" / args.procedure / f"{args.loss}_{args.condition}_s{args.seed}"
    out.mkdir(parents=True, exist_ok=True)
    frames, attrs, group, net, objective, cw = setup(args)
    prep = loaders_for(args, frames, attrs, group)
    optimizer = torch.optim.AdamW([p for p in net.parameters() if p.requires_grad], lr=args.lr, weight_decay=1e-2)
    log = {"args": vars(args), "n": {s: len(f) for s, f in frames.items()}, "epochs": []}
    for split in ("val", "test"):  # epoch -1 = 補正前（LoRA の差分は初期値 0）
        np.save(out / f"epoch_-1_{split}.npy", evaluate(net, prep[split], args.device)[0])
    net.train()
    net.backbone.eval()  # 凍結した BatchNorm の統計を動かさない
    for epoch in range(args.epochs):
        start = time.time()
        record = {"epoch": epoch} | train_epoch(
            net, objective, optimizer, epoch_loaders(prep, epoch, group, frames, args.loss), cw, args
        )
        record["sec"] = time.time() - start
        for split in ("val", "test"):
            score, _ = evaluate(net, prep[split], args.device)
            np.save(out / f"epoch_{epoch:02d}_{split}.npy", score)
            record[f"{split}_auroc"] = roc_auc_score(frames[split].target, score)
            record[f"{split}_worst8"] = worst8(score, frames[split])
        log["epochs"].append(record)
        print(json.dumps({k: v for k, v in record.items() if not isinstance(v, list)}), flush=True)
        (out / "log.json").write_text(json.dumps(log, indent=1))


if __name__ == "__main__":
    main()
