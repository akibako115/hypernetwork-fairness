"""Q2 の補足：公平性損失の効果は、ERM の出力に群別のしきい値（群ごとの平行移動）を当てるだけで代わりが利くか。

Q2（`q2_fairness_loss.py`）では、公平性損失は最悪群 AUROC・AUROC gap を動かさず、Eopp1・Eopp0 だけを下げた。
群内の順位が変わらないなら、損失がしたことは群ごとのスコアの平行移動（= 群別しきい値）に近いはずで、
それなら学習し直さなくても、ERM に val で決めたしきい値を当てれば同じ効果になる。これを 2 つの向きで確かめる。

1. しきい値の側から：各手順の ERM に、val で決めた群別しきい値を当てた Eopp1・Eopp0 と、損失のモデルを比べる。
   - `group_youden`：群ごとに TPR − FPR が最大の点（Q2 の参照と同じ規則。群の間で率をそろえることは狙わない）
   - `group_match`：群ごとに、全体の動作点（全体 Youden のしきい値での TPR・FPR）に最も近い点
     （|TPR − TPR*| と |FPR − FPR*| の大きいほうが最小）。率をそろえる規則で、全体の動作点は保つ
2. 損失の側から：val で、損失のモデルのスコアを「ERM のスコアの定数倍 + 群ごとの定数」で回帰し、その近似を
   test に当てる（`erm_offset`）。損失のモデルの全体 AUROC の低下と Eopp が再現できれば、損失の効果は
   群ごとの平行移動でほぼ説明できる。近似の R²（val）も残す

しきい値・回帰はすべて val で決め、test では動かさない。指標は Q2 と同じ規則（`q1_definitions.group_rows`）で、
seed ごとに求めて 3 seed で平均する。区間は患者単位の bootstrap 95%（Q2 と同じ引き直し）。

入力: `cache/adapter_fair/<procedure>/<loss>_meta_s<seed>/epoch_XX_{val,test}.npy`（epoch は val の全体 AUROC で選ぶ）
出力: `results/q2_threshold_equivalence.csv`

    uv run python -m analysis.archive.latent_bias_hypernet.q2_threshold_equivalence
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import roc_auc_score

from analysis.archive.latent_bias_hypernet.data import PACKAGE
from analysis.archive.latent_bias_hypernet.q1_definitions import (
    MIN_NEG,
    MIN_POS,
    bootstrap_index,
    youden,
)
from analysis.archive.latent_bias_hypernet.q2_fairness_loss import CONDITION, RUNS, SEEDS, SPLITS, demo8, select_epoch

PROCEDURES = ("onestage", "insample", "crossfit")
LOSSES = ("eo", "minimax_auc")
KEYS = ("overall_auroc", "tpr", "fpr", "eopp1", "eopp0")


def scores(procedure: str, loss: str, seed: int, val: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """val の全体 AUROC で選んだ epoch の (val, test) のスコア。"""
    run = RUNS / procedure / f"{loss}_{CONDITION}_s{seed}"
    epoch = select_epoch(run, loss, "auroc", val)
    return tuple(np.load(run / f"epoch_{epoch:02d}_{split}.npy") for split in ("val", "test"))


def group_youden(s_val: np.ndarray, y: np.ndarray, g: np.ndarray) -> np.ndarray:
    return np.array([youden(s_val[g == k], y[g == k]) for k in range(8)])


def group_match(s_val: np.ndarray, y: np.ndarray, g: np.ndarray) -> np.ndarray:
    """群ごとに、全体 Youden の動作点（TPR*, FPR*）に最も近い点のしきい値。

    1 つのしきい値では群の ROC 曲線上の 1 点しか選べず、群の AUROC が違えば TPR と FPR を同時には合わせられない。
    そこで 2 つのずれの大きいほうを最小にする。全体の動作点に合わせるのは、全員陽性のような退化した解
    （率の差は 0 になる）を避け、損失のモデルと同じく全体 Youden の付近で比べるため。
    """
    pred = s_val >= youden(s_val, y)
    target_tpr, target_fpr = pred[y == 1].mean(), pred[y == 0].mean()
    out = np.empty(8)
    for k in range(8):
        sk, yk = s_val[g == k], y[g == k]
        candidates = np.unique(sk)
        tpr = (sk[yk == 1][None, :] >= candidates[:, None]).mean(1)
        fpr = (sk[yk == 0][None, :] >= candidates[:, None]).mean(1)
        out[k] = candidates[np.argmin(np.maximum(abs(tpr - target_tpr), abs(fpr - target_fpr)))]
    return out


def offset_fit(erm_val, loss_val, g_val, erm_test, g_test) -> tuple[np.ndarray, np.ndarray, float]:
    """損失のスコアを ERM のスコアの定数倍 + 群ごとの定数で近似する（val で回帰、test に当てる）。"""

    def design(s, g):
        return np.c_[s, np.eye(8)[g]]

    model = LinearRegression(fit_intercept=False).fit(design(erm_val, g_val), loss_val)
    r2 = model.score(design(erm_val, g_val), loss_val)
    return model.predict(design(erm_val, g_val)), model.predict(design(erm_test, g_test)), r2


def metrics(score: np.ndarray, threshold, y: np.ndarray, g: np.ndarray) -> dict[str, float]:
    """全体 AUROC、全体の TPR・FPR、Eopp1・Eopp0。threshold は数値か、群ごとの配列。"""
    cut = threshold[g] if np.ndim(threshold) else threshold
    pred = (score >= cut).astype(float)
    # Q2（q1_definitions.group_rows）と同じく、陽性・陰性とも MIN_POS・MIN_NEG 件以上の群だけで率の差を取る。
    # bootstrap のたびに群別 AUROC まで求めると遅いので、率だけを bincount で数える。
    pos, neg = np.bincount(g[y == 1], minlength=8), np.bincount(g[y == 0], minlength=8)
    keep = (pos >= MIN_POS) & (neg >= MIN_NEG)
    tpr = np.bincount(g[y == 1], pred[y == 1], minlength=8)[keep] / pos[keep]
    fpr = np.bincount(g[y == 0], pred[y == 0], minlength=8)[keep] / neg[keep]
    return {
        "overall_auroc": roc_auc_score(y, score),
        "tpr": pred[y == 1].mean(),
        "fpr": pred[y == 0].mean(),
        "eopp1": np.ptp(tpr),
        "eopp0": np.ptp(fpr),
    }


def seed_rules(procedure: str, loss: str, seed: int, val: pd.DataFrame, g_test: np.ndarray) -> dict[str, tuple]:
    """1 seed 分の、規則ごとの (test のスコア, しきい値)。ERM の 3 規則と、損失のモデル、その平行移動の近似。"""
    yv, gv = val.target.to_numpy(), demo8(val)
    erm_val, erm_test = scores(procedure, "erm", seed, val)
    loss_val, loss_test = scores(procedure, loss, seed, val)
    fit_val, fit_test, r2 = offset_fit(erm_val, loss_val, gv, erm_test, g_test)
    return {
        "erm": (erm_test, youden(erm_val, yv)),
        "erm_group_youden": (erm_test, group_youden(erm_val, yv, gv)),
        "erm_group_match": (erm_test, group_match(erm_val, yv, gv)),
        f"{loss}": (loss_test, youden(loss_val, yv)),
        f"{loss}_as_erm_offset": (fit_test, youden(fit_val, yv)),
        "r2": r2,
    }


def evaluate(runs: list[tuple], y: np.ndarray, g: np.ndarray, index: np.ndarray) -> dict[str, float]:
    per_seed = [metrics(s[index], th, y[index], g[index]) for s, th in runs]
    return {k: float(np.mean([m[k] for m in per_seed])) for k in KEYS}


def procedure_rows(procedure: str, val, test, draws) -> list[dict]:
    """1 手順分の行。ERM の規則は損失によらないので、最初の損失の分だけ残す。"""
    y, g = test.target.to_numpy(), demo8(test)
    rows, seen = [], set()
    for loss in LOSSES:
        per_seed = [seed_rules(procedure, loss, seed, val, g) for seed in SEEDS]
        for rule in [r for r in per_seed[0] if r != "r2"]:
            if rule in seen:
                continue
            seen.add(rule)
            runs = [p[rule] for p in per_seed]
            point = evaluate(runs, y, g, np.arange(len(y)))
            boot = pd.DataFrame([evaluate(runs, y, g, i) for i in draws])
            row = {"procedure": procedure, "rule": rule}
            for k in KEYS:
                lo, hi = boot[k].quantile([0.025, 0.975])
                row |= {k: point[k], f"{k}_lo": lo, f"{k}_hi": hi}
            if rule.endswith("_as_erm_offset"):
                row["offset_r2_val"] = float(np.mean([p["r2"] for p in per_seed]))
            rows.append(row)
            print(procedure, rule, flush=True)
    return rows


def main() -> None:
    val, test = pd.read_csv(SPLITS / "val.csv"), pd.read_csv(SPLITS / "test.csv")
    units = test.image.str.split("/").str[1].to_numpy()
    draws = bootstrap_index(units, np.random.default_rng(0))[:200]
    rows = [row for procedure in PROCEDURES for row in procedure_rows(procedure, val, test, draws)]
    table = pd.DataFrame(rows)
    table.to_csv(PACKAGE / "results" / "q2_threshold_equivalence.csv", index=False)
    cols = ["procedure", "rule", *KEYS, "offset_r2_val"]
    with pd.option_context("display.width", 250):
        print(table[cols].round(3).to_string(index=False))


if __name__ == "__main__":
    main()
