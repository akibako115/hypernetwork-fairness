"""ラベル生成過程で条件付けた head：学習時だけレポートの文脈 c で条件付け、推論時は c の項を外す。

問い：「No Finding」ラベルの意味がレポートの書かれ方（特定の問いに答えるだけの追跡撮影か）で変わるなら、
その文脈を学習時に条件として吸収させると、画像の正常さを表す共有部分 w·x は改善するか。
特に高齢群（追跡撮影型の陽性が多い）で効くか。hypernetwork を「属性ごとの最適な処理」ではなく
「ラベル生成過程のモデル」として使う案の、凍結特徴での当たり付け。

- c：impression が FOCUSED を含み AFFIRMED を含まない（label_construct.py と同じ正規表現）。全画像で計算。
- head：s(x, c) = w·x + b + [c]·(Δw·x + Δb)。`bias` は Δb だけ、`delta` は Δw も（L2 1e-1）。
- 推論：c の項を外した w·x + b（文脈を知らない読み）と、実際の c を入れた値の両方を評価。
- 年齢との交互作用も見る：c を 年齢 2 区分 × focus 2 区分 の 4 セルにした版。

出力: `results/label_process/label_process_heads.csv`

    uv run python analysis/archive/fairness_mechanism_probe/label_process_heads.py
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import torch
from features import PACKAGE, REPOSITORY_ROOT, load_split
from label_construct import report_type
from sklearn.metrics import roc_auc_score
from torch import nn


def focus_flag(frame: pd.DataFrame, impressions: pd.DataFrame) -> np.ndarray:
    merged = frame[["image"]].merge(impressions, on="image", how="left", validate="one_to_one")
    return (report_type(merged.section_impression) == "focused_followup").to_numpy().astype(np.int64)


class ContextHead(nn.Module):
    def __init__(self, dim: int, n_context: int, with_delta: bool):
        super().__init__()
        self.shared = nn.Linear(dim, 1)
        self.bias = nn.Parameter(torch.zeros(n_context))
        self.delta = nn.Parameter(torch.zeros(n_context, dim)) if with_delta else None

    def forward(self, x, c, use_context: bool = True):
        s = self.shared(x).squeeze(1)
        if not use_context:
            return s
        s = s + self.bias[c]
        if self.delta is not None:
            s = s + (x * self.delta[c]).sum(1)
        return s


def fit(head, x, y, c, val_x, val_y, val_c, l2=1e-1, epochs=15):
    opt = torch.optim.AdamW(head.parameters(), lr=1e-3, weight_decay=1e-4)
    pos = y.mean()
    w = torch.where(y > 0, 0.5 / pos, 0.5 / (1 - pos))
    best, state = -1, None
    for _ in range(epochs):
        for b in torch.randperm(len(y), device=x.device).split(1024):
            loss = nn.functional.binary_cross_entropy_with_logits(head(x[b], c[b]), y[b], weight=w[b])
            if head.delta is not None:
                loss = loss + l2 * head.delta.pow(2).sum(1).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
        with torch.no_grad():
            # epoch 選択は「文脈を外した読み」の val AUROC で行う（推論時に使う形）
            auc = roc_auc_score(val_y.cpu().numpy(), head(val_x, val_c, use_context=False).cpu().numpy())
        if auc > best:
            best, state = auc, {k: v.clone() for k, v in head.state_dict().items()}
    head.load_state_dict(state)
    return head


def evaluate(score, frame, rtype) -> dict:
    y = frame.target.to_numpy()
    pos = y == 1
    old = frame.age.to_numpy() >= 65
    out = {"overall": roc_auc_score(y, score)}
    for kind in ("all", "affirmed_normal", "focused_followup"):
        keep = pos if kind == "all" else pos & (rtype == kind)
        for flag, label in ((False, "young"), (True, "old")):
            m = (old == flag) & (keep | ~pos)
            out[f"{kind}:{label}"] = roc_auc_score(y[m], score[m])
    return out


def main(device: str = "cuda") -> None:
    plus = pd.read_csv(
        REPOSITORY_ROOT / "data/chexpert/df_chexpert_plus_240401.csv", usecols=["path_to_image", "section_impression"]
    )
    plus = plus.rename(columns={"path_to_image": "image"})
    rows = []
    for seed in (42, 43, 44):
        sp = {s: load_split(seed, s) for s in ("train", "val", "test")}
        mean, std = sp["train"].features.mean(0), sp["train"].features.std(0) + 1e-6
        x = {s: torch.from_numpy((v.features - mean) / std).to(device) for s, v in sp.items()}
        y = {s: torch.from_numpy(v.target.astype(np.float32)).to(device) for s, v in sp.items()}
        focus = {s: focus_flag(v.frame, plus) for s, v in sp.items()}
        old = {s: (v.frame.age.to_numpy() >= 65).astype(np.int64) for s, v in sp.items()}
        test_rtype = report_type(
            sp["test"].frame[["image"]].merge(plus, on="image", how="left").section_impression
        ).to_numpy()
        contexts = {
            "none": {s: np.zeros_like(focus[s]) for s in sp},
            "focus": focus,
            "age": old,
            "age×focus": {s: old[s] * 2 + focus[s] for s in sp},
        }
        for cname, ctx in contexts.items():
            for with_delta in (False, True):
                if cname == "none" and with_delta:
                    continue
                torch.manual_seed(seed)
                c = {s: torch.from_numpy(v).to(device) for s, v in ctx.items()}
                head = ContextHead(x["train"].shape[1], 4, with_delta).to(device)
                head = fit(head, x["train"], y["train"], c["train"], x["val"], y["val"], c["val"])
                with torch.no_grad():
                    for use_context in (False, True):
                        if cname == "none" and use_context:
                            continue
                        score = head(x["test"], c["test"], use_context=use_context).cpu().numpy()
                        row = {
                            "seed": seed,
                            "context": cname,
                            "delta": with_delta,
                            "inference": "with c" if use_context else "shared only",
                        }
                        row.update(evaluate(score, sp["test"].frame, test_rtype))
                        rows.append(row)
                        print({k: (round(v, 4) if isinstance(v, float) else v) for k, v in row.items()}, flush=True)
    table = pd.DataFrame(rows)
    out = PACKAGE / "results" / "label_process"
    out.mkdir(parents=True, exist_ok=True)
    table.to_csv(out / "label_process_heads.csv", index=False)
    pd.set_option("display.width", 250)
    print(
        table.groupby(["context", "delta", "inference"], sort=False)
        .mean(numeric_only=True)
        .drop(columns="seed")
        .round(4)
        .to_string()
    )


if __name__ == "__main__":
    main()
