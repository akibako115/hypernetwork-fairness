"""Q2 の報告用の表：hypernet に公平性損失をかけると、属性の群の性能差が縮むか。学習の手順 3 つで比べる。

属性で条件付けた hypernet（Spatial LoRA）に、損失 3 つ（ERM・EO rate・minimax AUC）× seed 3 つを回した run を読み、
Q1 と同じ物差しで測る。手順は次の 3 つで、どれも 1 段目と 2 段目を合わせて train 全体を使う。
- `onestage`：ImageNet から全層を学習（hypernet_e2e の `*_dual_holdout` preset。予測は `export_onestage.py`）
- `insample`：train 全体で学習した M01 を凍結し、同じ train で 2 段目を学習（`probes/train_adapter_fair.py`）
- `crossfit`：train_a で学習した a_s42 を凍結し、train_b で 2 段目を学習（同上）

比べるのは各手順の中での「公平性損失 − ERM」で、手順の間で絶対値は比べない（1 段は 2 段の倍のデータで学習する）。
群は年齢（65 歳で 2 区分）× race の 8 群、指標は最悪群 AUROC・AUROC gap・Eopp1・Eopp0・全体 BAcc・BAcc gap。
BAcc は (TPR + TNR) / 2 で、全体 BAcc は全体のしきい値での値、BAcc gap は群の BAcc の最大 − 最小（Eopp と同じ群）。
しきい値は各 run の val の Youden。区間は患者単位の bootstrap 95% で、seed 3 本の平均に対して求める。
差（同じ手順の ERM との差。2 段の ERM は補正前との差）は同じ bootstrap 標本の上で求める（対応のある比較）。

epoch の選び方は 2 通りで、どちらも val だけを使う。
- `auroc`：val の全体 AUROC が最大の epoch（属性を使わない。全損失で共通）
- `target`：損失自身の狙いが val で最もよい epoch。EO は 8 群の TPR 差と FPR 差の大きいほうが最小、
  minimax AUC は群ペアの AUROC の最小値が最大。ERM は `auroc` と同じ

参照として、2 段の補正前の 1 段目に val で群ごとの Youden しきい値を当てた場合（群別しきい値）を置く。
AUROC は変わらず、Eopp1・Eopp0 だけが動く。

損失が働いていたかの確認として、学習の終わりの群の重みが、held-out で最も悪い群に向いていたかも出す。

出力: `results/q2_fairness_loss.csv`（指標）、`results/q2_fairness_weights.csv`（群の重み）

    uv run python -m analysis.archive.latent_bias_hypernet.q2_fairness_loss
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from analysis.archive.latent_bias_hypernet.data import PACKAGE, ROOT
from analysis.archive.latent_bias_hypernet.q1_definitions import METRICS, bootstrap_index, group_rows, summarize, youden

RUNS = PACKAGE / "cache" / "adapter_fair"
SPLITS = ROOT / "data" / "chexpert" / "splits_crossfit"
LOSSES = ("erm", "eo", "minimax_auc")
CONDITION = "meta"
PROCEDURES = ("onestage", "insample", "crossfit")
SEEDS = (42, 43, 44)
Q2_METRICS = (*METRICS, "overall_bacc", "bacc_gap")


def demo8(frame: pd.DataFrame) -> np.ndarray:
    return (frame.age.to_numpy() >= 65).astype(int) * 4 + frame.race_group.to_numpy().astype(int)


def rates(score: np.ndarray, y: np.ndarray, g: np.ndarray, threshold: float) -> tuple[np.ndarray, np.ndarray]:
    pred = score >= threshold
    tpr = np.array([pred[(g == k) & (y == 1)].mean() for k in range(8)])
    fpr = np.array([pred[(g == k) & (y == 0)].mean() for k in range(8)])
    return tpr, fpr


def pair_auroc_min(score: np.ndarray, y: np.ndarray, g: np.ndarray) -> float:
    """陽性群 a と陰性群 b の全 64 ペアの AUROC の最小値（minimax AUC が上げようとする量）。"""
    pos = [score[(g == a) & (y == 1)] for a in range(8)]
    neg = [score[(g == b) & (y == 0)] for b in range(8)]
    return min((p[:, None] > n[None, :]).mean() + 0.5 * (p[:, None] == n[None, :]).mean() for p in pos for n in neg)


def select_epoch(run, loss: str, rule: str, val: pd.DataFrame) -> int:
    """val だけで epoch を選ぶ。規則は module docstring のとおり。run は run の directory。"""
    epochs = sorted(int(p.stem.split("_")[1]) for p in run.glob("epoch_*_val.npy") if "_-1_" not in p.name)
    y, g = val.target.to_numpy(), demo8(val)
    best, chosen = -np.inf, None
    for e in epochs:
        s = np.load(run / f"epoch_{e:02d}_val.npy")
        if rule == "auroc" or loss == "erm":
            value = roc_auc_score(y, s)
        elif loss == "eo":
            tpr, fpr = rates(s, y, g, youden(s, y))
            value = -max(np.ptp(tpr), np.ptp(fpr))
        else:
            value = pair_auroc_min(s, y, g)
        if value > best:
            best, chosen = value, e
    return chosen


def load_condition(procedure: str, loss: str, rule: str, val) -> list[tuple[np.ndarray, float]]:
    """seed ごとの (test のスコア, val の Youden しきい値)。"""
    out = []
    for seed in SEEDS:
        run = RUNS / procedure / f"{loss}_{CONDITION}_s{seed}"
        e = select_epoch(run, loss, rule, val)
        s_val = np.load(run / f"epoch_{e:02d}_val.npy")
        out.append((np.load(run / f"epoch_{e:02d}_test.npy"), youden(s_val, val.target.to_numpy())))
    return out


def seed_mean_metrics(runs, y, g, idx=None) -> dict[str, float]:
    """seed ごとに Q2_METRICS を求めて平均する。idx は bootstrap の行番号。"""
    idx = np.arange(len(y)) if idx is None else idx
    per_seed = [seed_metrics(s[idx], th, y[idx], g[idx]) for s, th in runs]
    return {k: float(np.mean([m[k] for m in per_seed])) for k in Q2_METRICS}


def seed_metrics(score: np.ndarray, threshold: float, y: np.ndarray, g: np.ndarray) -> dict[str, float]:
    t = group_rows(g, y, score, threshold, False)
    pred = score >= threshold
    bacc = (t.tpr + 1 - t.fpr) / 2
    overall = (pred[y == 1].mean() + 1 - pred[y == 0].mean()) / 2
    return summarize(t) | {"overall_bacc": overall, "bacc_gap": bacc.max() - bacc.min()}


def group_threshold_reference(base_val, base_test, val, test, draws) -> dict:
    """補正前の 1 段目に、val で群ごとの Youden しきい値を当てた場合の Eopp1・Eopp0。"""
    yv, gv, yt, gt = val.target.to_numpy(), demo8(val), test.target.to_numpy(), demo8(test)
    th = np.array([youden(base_val[gv == k], yv[gv == k]) for k in range(8)])
    pred = base_test >= th[gt]

    def gaps(i):
        tpr = [pred[i][(gt[i] == k) & (yt[i] == 1)].mean() for k in range(8)]
        fpr = [pred[i][(gt[i] == k) & (yt[i] == 0)].mean() for k in range(8)]
        return {"eopp1": np.ptp(tpr), "eopp0": np.ptp(fpr)}

    return {"point": gaps(np.arange(len(yt))), "boot": pd.DataFrame([gaps(i) for i in draws])}


def metric_rows(conditions: dict, y, g, draws) -> tuple[pd.DataFrame, dict]:
    """条件ごとの点推定と bootstrap の値。"""
    rows, boot = [], {}
    for name, runs in conditions.items():
        boot[name] = pd.DataFrame([seed_mean_metrics(runs, y, g, i) for i in draws])
        point = seed_mean_metrics(runs, y, g)
        per_seed_auroc = [roc_auc_score(y, s) for s, _ in runs]
        row = {
            "condition": name,
            "overall_auroc": float(np.mean(per_seed_auroc)),
            "overall_auroc_sd": np.std(per_seed_auroc),
        }
        for k, v in point.items():
            lo, hi = boot[name][k].quantile([0.025, 0.975])
            row |= {k: v, f"{k}_lo": lo, f"{k}_hi": hi}
        rows.append(row)
        print(name, flush=True)
    return pd.DataFrame(rows), boot


def add_diffs(table: pd.DataFrame, boot: dict, ref_of) -> pd.DataFrame:
    """ref_of(name) が返す条件との差と区間を足す。"""
    point = table.set_index("condition")
    rows = []
    for name in table.condition:
        ref = ref_of(name)
        row = {"diff_vs": ref}
        if ref is None:
            rows.append(row)
            continue
        for k in Q2_METRICS:
            diff = boot[name][k] - boot[ref][k]
            row[f"{k}_diff"] = point.loc[name, k] - point.loc[ref, k]
            row[f"{k}_diff_lo"], row[f"{k}_diff_hi"] = diff.quantile([0.025, 0.975])
        rows.append(row)
    return pd.concat([table, pd.DataFrame(rows)], axis=1)


def weight_rows(procedures) -> pd.DataFrame:
    """学習の終わりの群の重みが、held-out で最も悪い群に向いていたか（損失が働いていたかの確認）。"""
    rows = []
    for procedure in procedures:
        for loss in ("eo", "minimax_auc"):
            for seed in SEEDS:
                log = RUNS / procedure / f"{loss}_{CONDITION}_s{seed}" / "log.json"
                last = json.loads(log.read_text())["epochs"][-1]
                rows.append({"procedure": procedure, "loss": loss, "seed": seed} | weight_summary(loss, last))
    return pd.DataFrame(rows)


def weight_summary(loss: str, last: dict) -> dict:
    """EO：λ のうち task 以外に載った割合と、TPR が最も低い群を引き上げる制約に載った割合。
    minimax AUC：held-out で順位損失が最大のペアに載った q と、q が最大のペアがそれと一致するか。
    q の初期値は件数の積で偏っているので、一様との比では読まない。"""
    if loss == "eo":
        lam = np.array(last["lambda"])
        held = np.array(last["holdout_rates"])  # [G, 2]：列 0 が陰性の陽性判定率（FPR）、列 1 が TPR
        worst = int(np.argmin(held[:, 1]))
        constraints = [(c, a, b) for c in (1, 0) for a in range(8) for b in range(8) if a != b]
        lift = sum(w for w, (c, _a, b) in zip(lam[1:], constraints, strict=True) if c == 1 and b == worst)
        return {"constraint_mass": 1 - lam[0], "mass_lifting_worst_tpr": lift, "worst_tpr_group": worst}
    q = np.array(last["q"]).ravel()
    risk = np.array(last["holdout_risk"])
    return {"q_on_worst_pair": q[risk.argmax()], "q_argmax_is_worst_pair": bool(q.argmax() == risk.argmax())}


def collect_conditions(procedures, val) -> dict:
    """手順 × 損失 × epoch の選び方ごとの run。2 段は補正前（1 段目そのもの）も置く。"""
    conditions = {}
    for procedure in procedures:
        if procedure != "onestage":
            run = RUNS / procedure / f"erm_{CONDITION}_s{SEEDS[0]}"
            base_val, base_test = (np.load(run / f"epoch_-1_{s}.npy") for s in ("val", "test"))
            conditions[f"{procedure}:before"] = [(base_test, youden(base_val, val.target.to_numpy()))]
        for rule in ("auroc", "target"):
            for loss in LOSSES:
                if loss == "erm" and rule == "target":
                    continue
                conditions[f"{procedure}:{loss}:{rule}"] = load_condition(procedure, loss, rule, val)
    return conditions


def reference_of(name: str) -> str | None:
    """差を取る相手。公平性損失は同じ手順の ERM、2 段の ERM は補正前、1 段の ERM と補正前は無し。"""
    procedure, *rest = name.split(":")
    if rest == ["before"]:
        return None
    if rest[0] != "erm":
        return f"{procedure}:erm:auroc"
    return None if procedure == "onestage" else f"{procedure}:before"


def threshold_rows(procedures, val, test, draws) -> list[dict]:
    """2 段の補正前の 1 段目に、群別しきい値を当てた参照の行。"""
    rows = []
    for procedure in procedures:
        if procedure == "onestage":
            continue
        run = RUNS / procedure / f"erm_{CONDITION}_s{SEEDS[0]}"
        base_val, base_test = (np.load(run / f"epoch_-1_{s}.npy") for s in ("val", "test"))
        ref = group_threshold_reference(base_val, base_test, val, test, draws)
        row = {"condition": f"{procedure}:group_threshold"}
        for k in ("eopp1", "eopp0"):
            lo, hi = ref["boot"][k].quantile([0.025, 0.975])
            row |= {k: ref["point"][k], f"{k}_lo": lo, f"{k}_hi": hi}
        rows.append(row)
    return rows


def main() -> None:
    val, test = pd.read_csv(SPLITS / "val.csv"), pd.read_csv(SPLITS / "test.csv")
    y, g = test.target.to_numpy(), demo8(test)
    draws = bootstrap_index(test.image.str.split("/").str[1].to_numpy(), np.random.default_rng(0))
    procedures = [p for p in PROCEDURES if (RUNS / p).is_dir()]
    table, boot = metric_rows(collect_conditions(procedures, val), y, g, draws)
    table = add_diffs(table, boot, reference_of)
    table = pd.concat([table, pd.DataFrame(threshold_rows(procedures, val, test, draws))], ignore_index=True)
    table.to_csv(PACKAGE / "results" / "q2_fairness_loss.csv", index=False)
    weight_rows(procedures).to_csv(PACKAGE / "results" / "q2_fairness_weights.csv", index=False)
    with pd.option_context("display.width", 250):
        print(table[["condition", "overall_auroc", *Q2_METRICS]].round(3).to_string(index=False))


if __name__ == "__main__":
    main()
