"""撮影文脈（過去検査の有無・検査順）で条件付けた head と、人口統計で条件付けた head を比べる。

問い：hypernetwork の条件として、人口統計（年齢・race）より撮影文脈のほうが群内順位の伸びしろを持つか。
文脈は CheXpert Plus の `patient_report_date_order`（患者内の検査順、推論時に病院情報から得られる）。
評価セルは 年齢 2 区分 × 初回/追跡 の 4 セル。heads.py の Head を、群定義と metadata だけ差し替えて使う。

出力: `results/context_heads/context_heads.csv`

    uv run python analysis/archive/fairness_mechanism_probe/context_heads.py
"""

from __future__ import annotations

import heads
import numpy as np
import pandas as pd
import torch
from features import PACKAGE, REPOSITORY_ROOT, load_split
from sklearn.metrics import roc_auc_score

CELLS = ("<65 first", "<65 follow-up", "≥65 first", "≥65 follow-up")


def add_context(split):
    plus = pd.read_csv(
        REPOSITORY_ROOT / "data/chexpert/df_chexpert_plus_240401.csv",
        usecols=["path_to_image", "patient_report_date_order"],
    )
    order = split.frame[["image"]].merge(
        plus.rename(columns={"path_to_image": "image"}), on="image", how="left", validate="one_to_one"
    )
    order = order.patient_report_date_order.to_numpy()
    followup = (order > 1).astype(np.int64)
    old = (split.frame.age.to_numpy() >= 65).astype(np.int64)
    return old * 2 + followup, np.log(order).astype(np.float32), followup


def run(seed: int, device="cuda") -> list[dict]:
    sp = {s: load_split(seed, s) for s in ("train", "val", "test")}
    ctx = {s: add_context(v) for s, v in sp.items()}
    mean, std = sp["train"].features.mean(0), sp["train"].features.std(0) + 1e-6
    a_mean, a_std = sp["train"].frame.age.mean(), sp["train"].frame.age.std()
    rows = []
    variants = {
        # (群の定義, metadata に文脈を入れるか)
        "demographic": ("demo", False),
        "context": ("ctx", True),
    }
    for label, (grouping, with_ctx) in variants.items():
        data = {}
        for s, v in sp.items():
            meta = heads.metadata_matrix(v.frame, a_mean, a_std)
            if with_ctx:
                meta = np.concatenate([meta, ctx[s][2][:, None].astype(np.float32), ctx[s][1][:, None]], 1)
            d = heads.to_device(v, mean, std, meta, device)
            if grouping == "ctx":
                # 4 セルを 8 群の枠に入れる（使わない ID は空のまま）
                d["g"] = torch.from_numpy(ctx[s][0]).to(device)
            data[s] = d
        for kind, pen in (
            ("shared", 0.0),
            ("group_bias", 0.0),
            ("group_delta", 1e-1),
            ("hyper", 1e-3),
            ("mlp_meta", 0.0),
        ):
            if label == "context" and kind == "shared":
                continue
            torch.manual_seed(seed)
            head = heads.Head(kind, sp["train"].features.shape[1], data["train"]["m"].shape[1]).to(device)
            head = heads.fit_head(head, data["train"], data["val"], sp["val"], pen)
            score = heads.predict(head, data["test"])
            y = sp["test"].target
            row = {"seed": seed, "condition_on": label, "head": kind, "overall": roc_auc_score(y, score)}
            for c, name in enumerate(CELLS):
                m = ctx["test"][0] == c
                row[name] = roc_auc_score(y[m], score[m])
            rows.append(row)
            print(row, flush=True)
    return rows


def main() -> None:
    rows = []
    for seed in (42, 43, 44):
        rows.extend(run(seed))
    out = PACKAGE / "results" / "context_heads"
    out.mkdir(parents=True, exist_ok=True)
    table = pd.DataFrame(rows)
    table.to_csv(out / "context_heads.csv", index=False)
    print(
        table.groupby(["condition_on", "head"], sort=False)
        .mean(numeric_only=True)
        .drop(columns="seed")
        .round(4)
        .to_string()
    )


if __name__ == "__main__":
    main()
