"""Q2 の HAM10000 版：属性条件付き hypernet に公平性損失をかけると、性別 × 年齢の群の差が縮むか。ResNet-50 と比べる。

CheXpert の Q2（`q2_fairness_loss.py`・`q2_threshold_equivalence.py`・`q2_auroc_decomposition.py`）の 1 段の設定を、
HAM10000 で ResNet-50 と属性条件付き Spatial LoRA の 2 モデルに当てる。CheXpert と違い、HAM10000 では hypernet が
属性の条件を実際に使っている（条件付けで全体 AUROC が上がる）ので、群ごとにパラメータを変えられるモデルなら
損失が群内の順位まで直せるかを見られる。

- run：hypernet_e2e の `*_ham10000_*_dual_holdout` preset。モデル 2 × 損失 3（ERM・EO rate・minimax AUC）× seed 3。
  重みは train の病変 80% で学習し、群の重みは残り 20% の held-out で更新する（ERM も同じ 80% で学習）
- 予測：`export_onestage.py` が書き出した `cache/ham10000_q2/<model>_<loss>_s<seed>/epoch_XX_{val,test}.npy`（z1 − z0）
- 群：sex × age_group_65 の 4 群（ID は学習時と同じ `2 * sex + (age ≥ 65)`）
- 評価：test のうち、施設が既知で sex・age が揃う 1,185 行（external は年齢欠損の符号化で崩れるので外す。
  `analysis/ham10000_architecture_erm`）。val は全行（sex・age の欠損は無い）
- 指標：CheXpert の Q2 と同じ（最悪群 AUROC・AUROC gap・Eopp1・Eopp0。しきい値は val の Youden）
- 区間：病変単位の bootstrap 95%。seed 3 本の平均に対して求め、差は同じ bootstrap 標本の上で求める

epoch の選び方は CheXpert の Q2 と同じ 2 通り（`auroc`：val の全体 AUROC が最大、`target`：損失自身の狙いが
val で最もよい）。学習中に記録された `val/auroc` は bf16 の softmax で同点が増えて低く出るので使わず、
書き出した fp32 の予測から選び直す。

比べる差は次の 3 つ。
- 同じモデルの中で「公平性損失 − ERM」（CheXpert の Q2 と同じ読み方）
- ERM どうしで「Spatial LoRA − ResNet」（条件付けの効果。C1 の再確認）
- 損失の効きのモデル差：(Spatial LoRA の損失 − ERM) − (ResNet の損失 − ERM)

出力:
- `results/q2_ham10000_fairness_loss.csv`：指標と差
- `results/q2_ham10000_threshold.csv`：ERM に群別しきい値を当てた場合と、損失のモデル、その平行移動の近似
- `results/q2_ham10000_auroc_decomposition.csv`：全体 AUROC の、群の中・群をまたぐ組への分解
- `results/q2_ham10000_weights.csv`：学習の終わりの群の重みが held-out で最も悪い群に向いていたか

    uv run python -m analysis.archive.latent_bias_hypernet.q2_ham10000
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import roc_auc_score

from analysis.archive.latent_bias_hypernet.data import PACKAGE, ROOT
from analysis.archive.latent_bias_hypernet.q1_definitions import (
    METRICS,
    MIN_NEG,
    MIN_POS,
    bootstrap_index,
    group_rows,
    summarize,
    youden,
)
from analysis.conditioning_mechanism.metrics import within_between

RUNS = PACKAGE / "cache" / "ham10000_q2"
SPLITS = ROOT / "data" / "ham10000" / "splits_dual_holdout"
RESULTS = PACKAGE / "results"
MODELS = ("resnet", "meta")
LOSSES = ("erm", "eo", "minimax_auc")
SEEDS = (42, 43, 44)
G = 4
# 外部施設（dataset == 0）は年齢欠損が集中し、属性条件付きモデルの入力が崩れるので評価から外す。
EXTERNAL_SITE = 0


def groups(frame: pd.DataFrame) -> np.ndarray:
    return frame.sex.to_numpy().astype(int) * 2 + (frame.age.to_numpy() >= 65).astype(int)


def load_splits() -> tuple[pd.DataFrame, pd.DataFrame, np.ndarray]:
    """val（全行）と、評価する test の行と、その行番号（書き出した test の予測は全 1,511 行）。"""
    val, test = pd.read_csv(SPLITS / "val.csv"), pd.read_csv(SPLITS / "test.csv")
    keep = (test.dataset != EXTERNAL_SITE) & (test.sex_missing == 0) & (test.age_missing == 0)
    return val, test[keep].reset_index(drop=True), np.flatnonzero(keep.to_numpy())


def rates(score, y, g, threshold) -> tuple[np.ndarray, np.ndarray]:
    pred = score >= threshold
    return (
        np.array([pred[(g == k) & (y == 1)].mean() for k in range(G)]),
        np.array([pred[(g == k) & (y == 0)].mean() for k in range(G)]),
    )


def pair_auroc_min(score, y, g) -> float:
    """陽性群 a と陰性群 b の全 16 ペアの AUROC の最小値（minimax AUC が上げようとする量）。"""
    pos = [score[(g == a) & (y == 1)] for a in range(G)]
    neg = [score[(g == b) & (y == 0)] for b in range(G)]
    return min((p[:, None] > n[None, :]).mean() + 0.5 * (p[:, None] == n[None, :]).mean() for p in pos for n in neg)


def select_epoch(run, loss: str, rule: str, val: pd.DataFrame) -> int:
    """val だけで epoch を選ぶ。規則は CheXpert の Q2 と同じ。"""
    epochs = sorted(int(p.stem.split("_")[1]) for p in run.glob("epoch_*_val.npy"))
    y, g = val.target.to_numpy(), groups(val)
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


def scores(model: str, loss: str, seed: int, rule: str, val, rows) -> tuple[np.ndarray, np.ndarray]:
    """選んだ epoch の (val, 評価する test の行) のスコア。"""
    run = RUNS / f"{model}_{loss}_s{seed}"
    e = select_epoch(run, loss, rule, val)
    return np.load(run / f"epoch_{e:02d}_val.npy"), np.load(run / f"epoch_{e:02d}_test.npy")[rows]


# ---- 指標と差 ----


def seed_metrics(score, threshold, y, g) -> dict[str, float]:
    return summarize(group_rows(g, y, score, threshold, False)) | {"overall_auroc": roc_auc_score(y, score)}


KEYS = ("overall_auroc", *METRICS)


def seed_mean(runs, y, g, idx) -> dict[str, float]:
    per_seed = [seed_metrics(s[idx], th, y[idx], g[idx]) for s, th in runs]
    return {k: float(np.mean([m[k] for m in per_seed])) for k in KEYS}


def interval(values: pd.Series) -> tuple[float, float]:
    lo, hi = values.quantile([0.025, 0.975])
    return float(lo), float(hi)


def fairness_loss_table(val, test, rows, draws) -> pd.DataFrame:
    y, g, yv = test.target.to_numpy(), groups(test), val.target.to_numpy()
    conditions = {}
    for model in MODELS:
        for loss in LOSSES:
            for rule in ("auroc", "target"):
                if loss == "erm" and rule == "target":
                    continue
                runs = []
                for seed in SEEDS:
                    s_val, s_test = scores(model, loss, seed, rule, val, rows)
                    runs.append((s_test, youden(s_val, yv)))
                conditions[f"{model}:{loss}:{rule}"] = runs
    point = {name: seed_mean(runs, y, g, np.arange(len(y))) for name, runs in conditions.items()}
    boot = {name: pd.DataFrame([seed_mean(runs, y, g, i) for i in draws]) for name, runs in conditions.items()}
    print("metrics done", flush=True)

    def diff(a: str, b: str, c: str | None = None, d: str | None = None) -> dict:
        """a − b、c・d があれば (a − b) − (c − d)。"""
        out = {}
        for k in KEYS:
            p = point[a][k] - point[b][k]
            bt = boot[a][k] - boot[b][k]
            if c is not None:
                p -= point[c][k] - point[d][k]
                bt = bt - (boot[c][k] - boot[d][k])
            out[f"{k}_diff"] = p
            out[f"{k}_diff_lo"], out[f"{k}_diff_hi"] = interval(bt)
        return out

    table = []
    for name in conditions:
        model, loss, rule = name.split(":")
        row = {"condition": name, "model": model, "loss": loss, "select": rule, "contrast": "value"}
        for k in KEYS:
            row[k] = point[name][k]
            row[f"{k}_lo"], row[f"{k}_hi"] = interval(boot[name][k])
        table.append(row)
    for name in conditions:
        model, loss, rule = name.split(":")
        if loss == "erm":
            continue
        base = f"{model}:erm:auroc"
        label = {"condition": name, "model": model, "loss": loss, "select": rule}
        table.append(label | {"contrast": "loss − erm"} | diff(name, base))
        if model == "meta":
            other = f"resnet:{loss}:{rule}"
            table.append(
                {
                    "condition": name,
                    "model": "meta − resnet",
                    "loss": loss,
                    "select": rule,
                    "contrast": "(loss − erm) meta − resnet",
                }
                | diff(name, base, other, "resnet:erm:auroc")
            )
    table.append(
        {
            "condition": "meta:erm:auroc",
            "model": "meta − resnet",
            "loss": "erm",
            "select": "auroc",
            "contrast": "meta − resnet",
        }
        | diff("meta:erm:auroc", "resnet:erm:auroc")
    )
    return pd.DataFrame(table)


# ---- 群別しきい値との比較（CheXpert の q2_threshold_equivalence と同じ規則） ----

THRESHOLD_KEYS = ("overall_auroc", "tpr", "fpr", "eopp1", "eopp0")


def group_match(s_val, y, g) -> np.ndarray:
    """群ごとに、全体 Youden の動作点（TPR*, FPR*）に最も近い点のしきい値（2 つのずれの大きいほうが最小）。"""
    pred = s_val >= youden(s_val, y)
    target_tpr, target_fpr = pred[y == 1].mean(), pred[y == 0].mean()
    out = np.empty(G)
    for k in range(G):
        sk, yk = s_val[g == k], y[g == k]
        candidates = np.unique(sk)
        tpr = (sk[yk == 1][None, :] >= candidates[:, None]).mean(1)
        fpr = (sk[yk == 0][None, :] >= candidates[:, None]).mean(1)
        out[k] = candidates[np.argmin(np.maximum(abs(tpr - target_tpr), abs(fpr - target_fpr)))]
    return out


def offset_fit(erm_val, loss_val, g_val, erm_test, g_test) -> tuple[np.ndarray, np.ndarray, float]:
    """損失のスコアを ERM のスコアの定数倍 + 群ごとの定数で近似する（val で回帰、test に当てる）。"""

    def design(s, g):
        return np.c_[s, np.eye(G)[g]]

    x_val = design(erm_val, g_val)
    model = LinearRegression(fit_intercept=False).fit(x_val, loss_val)
    return model.predict(x_val), model.predict(design(erm_test, g_test)), model.score(x_val, loss_val)


def threshold_metrics(score, threshold, y, g) -> dict[str, float]:
    cut = threshold[g] if np.ndim(threshold) else threshold
    pred = (score >= cut).astype(float)
    pos, neg = np.bincount(g[y == 1], minlength=G), np.bincount(g[y == 0], minlength=G)
    keep = (pos >= MIN_POS) & (neg >= MIN_NEG)
    tpr = np.bincount(g[y == 1], pred[y == 1], minlength=G)[keep] / pos[keep]
    fpr = np.bincount(g[y == 0], pred[y == 0], minlength=G)[keep] / neg[keep]
    return {
        "overall_auroc": roc_auc_score(y, score),
        "tpr": pred[y == 1].mean(),
        "fpr": pred[y == 0].mean(),
        "eopp1": np.ptp(tpr),
        "eopp0": np.ptp(fpr),
    }


def threshold_table(val, test, rows, draws) -> pd.DataFrame:
    yv, gv, y, g = val.target.to_numpy(), groups(val), test.target.to_numpy(), groups(test)
    out = []
    for model in MODELS:
        rules: dict[str, list] = {}
        r2: dict[str, list] = {}
        for seed in SEEDS:
            erm_val, erm_test = scores(model, "erm", seed, "auroc", val, rows)
            rules.setdefault("erm", []).append((erm_test, youden(erm_val, yv)))
            rules.setdefault("erm_group_match", []).append((erm_test, group_match(erm_val, yv, gv)))
            for loss in ("eo", "minimax_auc"):
                loss_val, loss_test = scores(model, loss, seed, "auroc", val, rows)
                fit_val, fit_test, fit_r2 = offset_fit(erm_val, loss_val, gv, erm_test, g)
                rules.setdefault(loss, []).append((loss_test, youden(loss_val, yv)))
                rules.setdefault(f"{loss}_as_erm_offset", []).append((fit_test, youden(fit_val, yv)))
                r2.setdefault(f"{loss}_as_erm_offset", []).append(fit_r2)
        for rule, runs in rules.items():

            def mean(idx, runs=runs):
                per_seed = [threshold_metrics(s[idx], th, y[idx], g[idx]) for s, th in runs]
                return {k: float(np.mean([m[k] for m in per_seed])) for k in THRESHOLD_KEYS}

            point, boot = mean(np.arange(len(y))), pd.DataFrame([mean(i) for i in draws])
            row = {"model": model, "rule": rule}
            for k in THRESHOLD_KEYS:
                row[k] = point[k]
                row[f"{k}_lo"], row[f"{k}_hi"] = interval(boot[k])
            if rule in r2:
                row["offset_r2_val"] = float(np.mean(r2[rule]))
            out.append(row)
        print("threshold", model, flush=True)
    return pd.DataFrame(out)


# ---- 全体 AUROC の分解（CheXpert の q2_auroc_decomposition と同じ） ----


def decompose(test_scores, y, g, idx) -> dict[str, float]:
    parts = []
    for s in test_scores:
        within, between, share = within_between(s[idx], y[idx], g[idx])
        parts.append((share * within + (1 - share) * between, within, between, share))
    mean = np.mean(parts, axis=0)
    return {"overall": mean[0], "within": mean[1], "between": mean[2], "within_share": mean[3]}


def decomposition_table(val, test, rows, draws) -> pd.DataFrame:
    y, g = test.target.to_numpy(), groups(test)
    out = []
    for model in MODELS:
        runs = {loss: [scores(model, loss, seed, "auroc", val, rows)[1] for seed in SEEDS] for loss in LOSSES}
        point = {loss: decompose(runs[loss], y, g, np.arange(len(y))) for loss in LOSSES}
        boot = {loss: pd.DataFrame([decompose(runs[loss], y, g, i) for i in draws]) for loss in LOSSES}
        for loss in LOSSES:
            row = {"model": model, "loss": loss, **point[loss]}
            if loss != "erm":
                for part in ("overall", "within", "between"):
                    row[f"{part}_diff"] = point[loss][part] - point["erm"][part]
                    row[f"{part}_diff_lo"], row[f"{part}_diff_hi"] = interval(boot[loss][part] - boot["erm"][part])
            out.append(row)
    return pd.DataFrame(out)


# ---- 群の重み ----


def weight_table() -> pd.DataFrame:
    """EO：λ のうち task 以外に載った割合と、held-out で TPR が最も低い群を引き上げる制約に載った割合。
    minimax AUC：held-out で順位損失が最大のペアに載った q と、q が最大のペアがそれと一致するか。"""
    constraints = [(c, a, b) for c in (1, 0) for a in range(G) for b in range(G) if a != b]
    out = []
    for model in MODELS:
        for loss in ("eo", "minimax_auc"):
            for seed in SEEDS:
                last = json.loads((RUNS / f"{model}_{loss}_s{seed}" / "log.json").read_text())["epochs"][-1]
                row = {"model": model, "loss": loss, "seed": seed}
                if loss == "eo":
                    lam = np.array(last["lambda"])
                    held = np.array(last["holdout_rates"])  # [G, 2]：列 0 が FPR、列 1 が TPR
                    worst = int(np.argmin(held[:, 1]))
                    lift = sum(w for w, (c, _a, b) in zip(lam[1:], constraints, strict=True) if c == 1 and b == worst)
                    row |= {"constraint_mass": 1 - lam[0], "mass_lifting_worst_tpr": lift, "worst_tpr_group": worst}
                else:
                    q, risk = np.array(last["q"]).ravel(), np.array(last["holdout_risk"])
                    worst_pair = risk.argmax()
                    row |= {"q_on_worst_pair": q[worst_pair], "q_argmax_is_worst_pair": bool(q.argmax() == worst_pair)}
                out.append(row)
    return pd.DataFrame(out)


def main() -> None:
    val, test, rows = load_splits()
    draws = bootstrap_index(test.lesion_id.to_numpy(), np.random.default_rng(0))
    loss_table = fairness_loss_table(val, test, rows, draws)
    loss_table.to_csv(RESULTS / "q2_ham10000_fairness_loss.csv", index=False)
    threshold_table(val, test, rows, draws[:200]).to_csv(RESULTS / "q2_ham10000_threshold.csv", index=False)
    decomposition = decomposition_table(val, test, rows, draws[:200])
    decomposition.to_csv(RESULTS / "q2_ham10000_auroc_decomposition.csv", index=False)
    weight_table().to_csv(RESULTS / "q2_ham10000_weights.csv", index=False)
    cols = ["condition", "contrast", *[c for k in KEYS for c in (k, f"{k}_diff", f"{k}_diff_lo", f"{k}_diff_hi")]]
    with pd.option_context("display.width", 250, "display.max_columns", 40):
        print(loss_table.reindex(columns=cols).round(3).to_string(index=False))


if __name__ == "__main__":
    main()
