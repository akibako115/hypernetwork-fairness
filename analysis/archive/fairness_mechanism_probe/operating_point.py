"""動作点：群別閾値の後処理と、全体再較正後の群内較正を M01 3seed で調べる。

問い：共通閾値での EO の差は、モデルの順位の問題か、基底率の差と閾値の問題か。
閾値は val で決め、test で評価する。出力: `results/operating_point/{thresholds,calibration}.csv`

    uv run python analysis/archive/fairness_mechanism_probe/operating_point.py
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from features import GROUP_NAMES, M01_RUNS, PACKAGE, load_split
from sklearn.linear_model import LogisticRegression


def thresholds(policy: str, score: np.ndarray, y: np.ndarray, g: np.ndarray) -> np.ndarray:
    th = np.zeros(8)
    overall_tpr = (score[y == 1] > 0).mean()
    for k in range(8):
        s, t = score[g == k], y[g == k]
        if policy == "group_youden":
            grid = np.quantile(s, np.linspace(0.01, 0.99, 400))
            th[k] = grid[int(np.argmax([(s[t == 1] > c).mean() + (s[t == 0] <= c).mean() for c in grid]))]
        elif policy == "group_tpr_match":
            th[k] = np.quantile(s[t == 1], 1 - overall_tpr)
    return th


def main() -> None:
    out = PACKAGE / "results" / "operating_point"
    out.mkdir(parents=True, exist_ok=True)
    rows, calib = [], []
    for seed in M01_RUNS:
        val, test = load_split(seed, "val"), load_split(seed, "test")
        sv, st = val.logits[:, 1] - val.logits[:, 0], test.logits[:, 1] - test.logits[:, 0]
        for policy in ("shared0", "group_youden", "group_tpr_match"):
            th = thresholds(policy, sv, val.target, val.group)
            tpr = np.array([(st[(test.group == k) & (test.target == 1)] > th[k]).mean() for k in range(8)])
            tnr = np.array([(st[(test.group == k) & (test.target == 0)] <= th[k]).mean() for k in range(8)])
            bacc = (tpr + tnr) / 2
            rows.append(
                dict(
                    seed=seed,
                    policy=policy,
                    Eopp1=np.ptp(tpr),
                    Eopp0=np.ptp(tnr),
                    worst_bacc=bacc.min(),
                    mean_bacc=bacc.mean(),
                )
            )
        platt = LogisticRegression(C=1e6).fit(sv[:, None], val.target)
        p = platt.predict_proba(st[:, None])[:, 1]
        for k, name in enumerate(GROUP_NAMES):
            m = test.group == k
            calib.append(dict(seed=seed, group=name, observed=test.target[m].mean(), mean_predicted=p[m].mean()))
    table = pd.DataFrame(rows)
    table.to_csv(out / "thresholds.csv", index=False)
    print(table.groupby("policy", sort=False).mean(numeric_only=True).drop(columns="seed").round(3))
    cal = pd.DataFrame(calib)
    cal.to_csv(out / "calibration.csv", index=False)
    print(cal.groupby("group", sort=False).mean(numeric_only=True).drop(columns="seed").round(3))


if __name__ == "__main__":
    main()
