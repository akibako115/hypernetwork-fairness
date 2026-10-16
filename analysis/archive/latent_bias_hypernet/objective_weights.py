"""公平性手法の群の重みが、学習の終わりに弱い群へ寄ったかを run の記録から読む（HAM10000）。

手法ごとに、群ごとの重みを 1 本のベクトルにまとめ、基準（学習を始めたときの重み）との比を取る。

- GroupDRO：`metrics.csv` の `q_end/g*`。基準は一様
- minimax AUC：`objective_diagnostics.jsonl` のペア（陽性群 g, 陰性群 h）の q を、群ごとの周辺（陽性側と陰性側の平均）に
  まとめる。基準は train の群 × クラスの件数から作る初期の q（陽性群の割合 × 陰性群の割合）
- FairDRO：`fair_dro_q.jsonl` の q（クラスごとの群の分布）をクラスで平均する。基準は一様
- EO 制約：双対 lambda は群の対ごとの制約なので群に割り振らない。最大値の一様に対する比と、制約が効いていた割合だけ返す

重みが最大の群が、同じ seed の ERM で val の AUROC が最も低い群と一致するかも返す。
`fairness_eval.py` から呼ぶ。
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

GROUP_COLUMNS = {"sex-age": "sexage", "gmm8": "gmm8"}


def last_jsonl(path) -> dict:
    with path.open() as handle:
        lines = [line for line in handle if line.strip()]
    return json.loads(lines[-1])


def train_counts(train: pd.DataFrame, groups: np.ndarray) -> np.ndarray:
    """train の [群, クラス] 件数。"""
    k = int(groups.max()) + 1
    out = np.zeros((k, 2))
    np.add.at(out, (groups, train.target.to_numpy()), 1)
    return out


def minimax_weights(run_dir, counts: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    q = np.asarray(last_jsonl(run_dir / "metrics" / "objective_diagnostics.jsonl")["q_after"]).reshape(len(counts), -1)
    pos, neg = counts[:, 1] / counts[:, 1].sum(), counts[:, 0] / counts[:, 0].sum()
    prior = np.outer(pos, neg)
    prior /= prior.sum()
    return (q.sum(1) + q.sum(0)) / 2, (prior.sum(1) + prior.sum(0)) / 2


def group_weights(run_dir, method: str, counts: np.ndarray) -> tuple[np.ndarray, np.ndarray] | None:
    """(学習の終わりの群の重み, 基準の重み)。群に割り振れない手法は None。"""
    k = len(counts)
    if method == "gdro":
        frame = pd.read_csv(run_dir / "metrics" / "metrics.csv")
        columns = [f"train/group_dro/q_end/g{g}" for g in range(k)]
        return frame.dropna(subset=columns)[columns].iloc[-1].to_numpy(float), np.full(k, 1 / k)
    if method == "minimaxauc":
        return minimax_weights(run_dir, counts)
    if method == "fairdro":
        q = np.asarray(last_jsonl(run_dir / "metrics" / "fair_dro_q.jsonl")["q"])
        return q.mean(0), np.full(k, 1 / k)
    return None


def eo_summary(run_dir) -> dict:
    with (run_dir / "metrics" / "objective_diagnostics.jsonl").open() as handle:
        rows = [json.loads(line) for line in handle if line.strip()]
    lam = np.asarray(rows[-1]["lambda"])
    active = np.mean([r["active_fraction"] for r in rows])
    return {"w_max_over_ref": float(lam.max() * len(lam)), "eo_active_fraction_mean": float(active)}


def describe(run_dir, exp: str, train: pd.DataFrame, groups: dict, worst: dict) -> dict:
    """1 本の run の重みの偏り。`worst` は {群の定義: ERM の val 最悪群}。"""
    method, _, group = exp.removeprefix("resnet-ham10000-fg-").partition("-")
    if method == "eo":
        return eo_summary(run_dir)
    if group not in GROUP_COLUMNS:
        return {}
    name = GROUP_COLUMNS[group]
    counts = train_counts(train, groups[name])
    pair = group_weights(run_dir, method, counts)
    if pair is None:
        return {}
    w, ref = pair
    ratio = w / ref
    top = int(np.argmax(ratio))
    return {
        "w_max_over_ref": float(ratio.max()),
        "w_tv_from_ref": float(np.abs(w / w.sum() - ref / ref.sum()).sum() / 2),
        "top_group": top,
        "erm_val_worst_group": worst[name],
        "top_is_val_worst": top == worst[name],
        "worst_weight_over_ref": float(ratio[worst[name]]),
    }
