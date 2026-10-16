"""群別閾値のあとに残る TPR 差：雑音か本当の差かを分け、属性から動作点を決める方式と比べる。

問い
1. 群別閾値（val で決めて test で評価）のあとに残る TPR 差 0.22〜0.31 は、閾値推定と評価の標本雑音で
   説明できるか、それとも val と test の間に系統的な違いがあるか。
2. 閾値を群ごとに独立に決める代わりに、属性から閾値を決めて群の間で情報を共有すると、TPR 差は縮むか。
   その代償として TNR はどれだけ動くか。

方法
- スコアは M01（ResNet ERM）の logit 差。モデルは変えず、閾値の決め方だけを比べる。
- 目標 TPR は「閾値を決める側の全体 TPR（閾値0）」。どの方式も全群をこの TPR にそろえようとする。
- val ∪ test を患者単位で無作為に半分ずつ分け直し（校正側・評価側）、各方式を校正側で決めて評価側で測る。
  これを繰り返した分布と、元の val→test の値を比べる。元の値が分布の中なら、残差は分け方の偶然で説明できる。
- 雑音の下限：全群の真の TPR が等しいと仮定し、評価側の陽性数で二項乱数を引いたときに出る TPR 差。

閾値の決め方
- `shared0`：全員 0（argmax）。
- `group`：群ごとに、校正側の陽性スコアの分位点（8群独立、群をまたいで情報を共有しない）。
- `shrunk`：`group` を `additive` へ、校正側の陽性数に応じて縮める（n/(n+k)、k=100）。
- `additive`：陽性スコアの線形分位点回帰。入力は年齢2区分＋race 4区分の加法モデル（群間で情報を共有）。
- `age_race_smooth`：同じく線形分位点回帰で、入力は連続年齢（2次まで）＋race。閾値は個人ごとに決まる。
- `mlp_meta`：属性（年齢・race・sex・撮影方向・AP/PA）→小さな MLP→閾値。pinball 損失で学習（hypernet 型の
  条件付き切片に相当）。

出力: `results/residual_tpr_gap/{resplits,original_split}.csv`

    uv run python analysis/archive/fairness_mechanism_probe/residual_tpr_gap.py --repeats 100
"""

from __future__ import annotations

import argparse
import warnings

import numpy as np
import pandas as pd
import torch
from features import CACHE, M01_RUNS, PACKAGE, SPLITS, group_ids
from sklearn.linear_model import QuantileRegressor
from torch import nn

from analysis.common.predictions import cache_path, load_cache

N_GROUPS = 8
SHRINK_K = 100.0


def load_scores(seed: int) -> pd.DataFrame:
    """val と test を 1 つの frame にまとめ、M01 のスコア・群・患者 ID を付ける。"""
    parts = []
    for split in ("val", "test"):
        frame = pd.read_csv(SPLITS / f"{split}.csv")
        cached = load_cache(cache_path(CACHE, M01_RUNS[seed], split), frame.image.to_numpy(dtype=str))
        frame["score"] = cached["logits"][:, 1] - cached["logits"][:, 0]
        frame["split"] = split
        parts.append(frame)
    frame = pd.concat(parts, ignore_index=True)
    frame["group"] = group_ids(frame)
    frame["patient"] = frame.image.str.split("/").str[1]
    return frame


def design(frame: pd.DataFrame, kind: str) -> np.ndarray:
    """分位点回帰の説明変数。"""
    race = np.eye(4)[frame.race_group.to_numpy()]
    if kind == "additive":
        return np.column_stack([(frame.age >= 65).astype(float), race[:, 1:]])
    age = (frame.age.to_numpy() - 60) / 18
    return np.column_stack([age, age**2, race[:, 1:]])


def meta(frame: pd.DataFrame) -> np.ndarray:
    race = np.eye(4)[frame.race_group.to_numpy()]
    age = ((frame.age.to_numpy() - 60) / 18)[:, None]
    other = frame[["sex", "frontal_lateral", "ap_pa"]].to_numpy().astype(float)
    return np.column_stack([age, race, other]).astype(np.float32)


def fit_quantile(x: np.ndarray, y: np.ndarray, q: float):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return QuantileRegressor(quantile=q, alpha=0.0, solver="highs").fit(x, y)


def fit_mlp_quantile(x: np.ndarray, y: np.ndarray, q: float, seed: int) -> nn.Module:
    """属性→閾値の小さな MLP を pinball 損失で学習する。出力の初期値は全体分位点。"""
    torch.manual_seed(seed)
    net = nn.Sequential(nn.Linear(x.shape[1], 16), nn.ReLU(), nn.Linear(16, 1))
    nn.init.zeros_(net[2].weight)
    nn.init.constant_(net[2].bias, float(np.quantile(y, q)))
    xt, yt = torch.from_numpy(x), torch.from_numpy(y.astype(np.float32))
    opt = torch.optim.AdamW(net.parameters(), lr=1e-2, weight_decay=1e-2)
    for _ in range(400):
        diff = yt - net(xt).squeeze(1)
        loss = torch.maximum(q * diff, (q - 1) * diff).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
    return net


def thresholds(cal: pd.DataFrame, ev: pd.DataFrame, seed: int) -> dict[str, np.ndarray]:
    """校正側 cal で各方式の閾値を決め、評価側 ev の各行に対する閾値を返す。"""
    pos = cal[cal.target == 1]
    target_tpr = (pos.score > 0).mean()
    q = 1 - target_tpr  # 陽性スコアのこの分位点を閾値にすると TPR が target_tpr になる
    out = {"shared0": np.zeros(len(ev))}
    group_q = pos.groupby("group").score.quantile(q).reindex(range(N_GROUPS)).to_numpy()
    out["group"] = group_q[ev.group.to_numpy()]
    additive = fit_quantile(design(pos, "additive"), pos.score.to_numpy(), q)
    out["additive"] = additive.predict(design(ev, "additive"))
    group_frame = pd.DataFrame(
        {"age": np.where(np.arange(N_GROUPS) >= 4, 70, 50), "race_group": np.arange(N_GROUPS) % 4}
    )
    additive_by_group = additive.predict(design(group_frame, "additive"))
    n_pos = pos.groupby("group").size().reindex(range(N_GROUPS)).to_numpy()
    weight = n_pos / (n_pos + SHRINK_K)
    out["shrunk"] = (weight * group_q + (1 - weight) * additive_by_group)[ev.group.to_numpy()]
    smooth = fit_quantile(design(pos, "smooth"), pos.score.to_numpy(), q)
    out["age_race_smooth"] = smooth.predict(design(ev, "smooth"))
    net = fit_mlp_quantile(meta(pos), pos.score.to_numpy(), q, seed)
    with torch.no_grad():
        out["mlp_meta"] = net(torch.from_numpy(meta(ev))).squeeze(1).numpy()
    return out


def rates(ev: pd.DataFrame, threshold: np.ndarray) -> dict[str, float]:
    predicted = ev.score.to_numpy() > threshold
    y, g = ev.target.to_numpy(), ev.group.to_numpy()
    tpr = np.array([predicted[(g == k) & (y == 1)].mean() for k in range(N_GROUPS)])
    tnr = np.array([(~predicted)[(g == k) & (y == 0)].mean() for k in range(N_GROUPS)])
    result = {"tpr_gap": np.ptp(tpr), "worst_tpr": tpr.min(), "tnr_gap": np.ptp(tnr), "worst_tnr": tnr.min()}
    result["overall_tpr"] = predicted[y == 1].mean()
    result["overall_tnr"] = (~predicted)[y == 0].mean()
    result.update({f"tpr_g{k}": tpr[k] for k in range(N_GROUPS)})
    result.update({f"tnr_g{k}": tnr[k] for k in range(N_GROUPS)})
    return result


def noise_floor(ev: pd.DataFrame, tpr: float, rng: np.random.Generator, draws: int = 2000) -> float:
    """全群の真の TPR が tpr で等しいとき、評価側の陽性数だけから出る TPR 差の平均。"""
    n_pos = ev[ev.target == 1].groupby("group").size().reindex(range(N_GROUPS)).to_numpy()
    samples = rng.binomial(n_pos, tpr, size=(draws, N_GROUPS)) / n_pos
    return float(np.ptp(samples, axis=1).mean())


def evaluate(cal, ev, seed, split_id, rng) -> list[dict]:
    rows = []
    target_tpr = (cal[cal.target == 1].score > 0).mean()
    floor = noise_floor(ev, target_tpr, rng)
    for method, threshold in thresholds(cal, ev, seed).items():
        rows.append({"seed": seed, "split": split_id, "method": method, "noise_floor": floor, **rates(ev, threshold)})
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repeats", type=int, default=100)
    args = parser.parse_args()
    out = PACKAGE / "results" / "residual_tpr_gap"
    out.mkdir(parents=True, exist_ok=True)
    original, resplits = [], []
    for seed in M01_RUNS:
        rng = np.random.default_rng(seed)
        frame = load_scores(seed)
        original += evaluate(frame[frame.split == "val"], frame[frame.split == "test"], seed, "original", rng)
        patients = frame.patient.unique()
        for r in range(args.repeats):
            cal_patients = set(rng.choice(patients, size=len(patients) // 2, replace=False))
            in_cal = frame.patient.isin(cal_patients)
            resplits += evaluate(frame[in_cal], frame[~in_cal], seed, r, rng)
        print(f"seed {seed} done", flush=True)
    original_table, resplit_table = pd.DataFrame(original), pd.DataFrame(resplits)
    original_table.to_csv(out / "original_split.csv", index=False)
    resplit_table.to_csv(out / "resplits.csv", index=False)
    cols = ["tpr_gap", "worst_tpr", "tnr_gap", "worst_tnr", "overall_tpr", "overall_tnr", "noise_floor"]
    pd.set_option("display.width", 250)
    print("元の val→test（3seed 平均）")
    print(original_table.groupby("method", sort=False)[cols].mean().round(3).to_string())
    print("分け直し（3seed × repeats の平均）")
    print(resplit_table.groupby("method", sort=False)[cols].mean().round(3).to_string())


if __name__ == "__main__":
    main()
