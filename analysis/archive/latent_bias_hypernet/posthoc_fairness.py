"""ERM の凍結特徴に後から掛ける公平性手法を、属性の群と潜在の群で比べる（HAM10000）。

学習時の手法（`resnet_ham10000_fg_*`）と対にする事後の手法。どれも ERM の backbone 表現（分類 head の直前）を
固定し、val だけを使う。val を病変単位で半分に分け、片方で学習、もう片方で正則化の強さを選ぶ（5 通りの分け方の平均）。

- `llr`：群を使わない最終層の再学習（DFR の対照。群の効果と「val で学び直す」効果を分ける）
- `dfr_<群>`：群 × クラスが均等になる重みで最終層を学び直す（Kirichenko et al. ICLR 2023 の DFR）
- 群ごとのしきい値：val で群ごとの TPR を揃える（機会均等の後処理）。AUROC は変えず、TPR・FPR の gap だけを動かす

正則化の選び方は 2 通り。属性を使わない選び方（holdout の全体 AUROC）と、
使う選び方（holdout の性別 × 年齢の最悪群 AUROC）。
群は性別 × 年齢（4 群）と、DINOv2 の GMM 8 クラスタ（学習前に固定、属性なし）。

入力: `cache/<run-id>_{val,test}{,_features}.npz`（`predictions.py --features`）、
`data/ham10000/splits_latent/{val,test}.csv`
出力: `results/ham10000_e2e/posthoc.csv`

    uv run python -m analysis.archive.latent_bias_hypernet.posthoc_fairness
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.preprocessing import StandardScaler

from analysis.archive.latent_bias_hypernet.data import PACKAGE, ROOT
from analysis.archive.latent_bias_hypernet.e2e_eval import (
    CACHE,
    axis_metrics,
    conditions,
    group_aucs,
    metrics,
    runs,
    sex_age,
)
from analysis.common.predictions import features_cache_path, load_cache, load_features

CS = (0.001, 0.01, 0.1, 1.0)
SPLITS = 5
ERM = ("resnet-ham10000", "resnet-ham10000-fg-erm")


def groups_of(frame: pd.DataFrame, name: str) -> np.ndarray:
    return sex_age(frame, "sex") if name == "sexage" else frame.gmm8.to_numpy()


def balanced_weights(y: np.ndarray, g: np.ndarray) -> np.ndarray:
    """群 × クラスの各セルが同じ総重みを持つ重み。欠損群（-1）は 1 つの群として扱う。"""
    cells = g * 2 + y
    _, inverse, counts = np.unique(cells, return_inverse=True, return_counts=True)
    return len(y) / (len(counts) * counts[inverse])


def holdout_split(val: pd.DataFrame, seed: int) -> np.ndarray:
    units = val.lesion_id.fillna(val.image).to_numpy(str)
    uniq = np.unique(units)
    chosen = set(np.random.default_rng(seed).choice(uniq, len(uniq) // 2, replace=False))
    return np.array([u in chosen for u in units])


def refit(hv, yv, w, fit, hold, val_frame, select: str):
    """fit 側で最終層を学び、hold 側の基準で C を選ぶ。選んだモデルを返す。"""
    best, best_score = None, -np.inf
    for c in CS:
        model = LogisticRegression(C=c, max_iter=2000).fit(hv[fit], yv[fit], sample_weight=w[fit])
        s = model.decision_function(hv[hold])
        if select == "overall":
            score = roc_auc_score(yv[hold], s)
        else:
            score = min(group_aucs(yv[hold], s, sex_age(val_frame[hold], "sex"), min_n=5).values())
        if score > best_score:
            best, best_score = model, score
    return best


def youden(y, s) -> float:
    fpr, tpr, t = roc_curve(y, s)
    return t[np.argmax(tpr - fpr)]


def group_threshold_gaps(val, test, sv, st, name: str) -> dict:
    """群ごとのしきい値で TPR を揃えたときの、性別 × 年齢の TPR・FPR gap（Hardt et al. の機会均等の後処理）。

    目標の TPR は、val 全体の Youden しきい値での TPR。各群のしきい値は、val のその群の陽性の得点の分位点で決める。
    陽性が 5 件未満の群と、群の欠けた行（external の性別欠損）は全体のしきい値を使う。
    """
    gv, gt = groups_of(val, name), groups_of(test, name)
    yv, yt = val.target.to_numpy(), test.target.to_numpy()
    overall = youden(yv, sv)
    target_tpr = (sv[yv == 1] >= overall).mean()
    thr = np.full(len(test), overall)
    for k in np.unique(gv[gv >= 0]):
        positives = sv[(gv == k) & (yv == 1)]
        if len(positives) >= 5:
            thr[gt == k] = np.quantile(positives, 1 - target_tpr)
    row = axis_metrics("sexage", yt, st, st >= thr, sex_age(test, "sex"))
    return {"sexage_tpr_gap": row["sexage_tpr_gap"], "sexage_fpr_gap": row["sexage_fpr_gap"]}


def one_run(run_dir, val, test) -> list[dict]:
    images_v, images_t = val.image.to_numpy(str), test.image.to_numpy(str)
    hv = load_features(features_cache_path(CACHE, run_dir.name, "val"), images_v)["features"].astype(np.float32)
    ht = load_features(features_cache_path(CACHE, run_dir.name, "test"), images_t)["features"].astype(np.float32)
    scaler = StandardScaler().fit(hv)
    hv, ht = scaler.transform(hv), scaler.transform(ht)
    yv = val.target.to_numpy()
    sv = load_cache(CACHE / f"{run_dir.name}_val.npz", images_v)["logits"]
    st = load_cache(CACHE / f"{run_dir.name}_test.npz", images_t)["logits"]
    sv, st = sv[:, 1] - sv[:, 0], st[:, 1] - st[:, 0]
    rows = [{"method": "erm", "select": "—", **metrics("ham10000", test, st, youden(yv, sv))}]
    for name in ("sexage", "gmm8"):
        rows.append({"method": f"thr_{name}", "select": "—", **group_threshold_gaps(val, test, sv, st, name)})
    methods = {"llr": np.ones(len(yv)), "dfr_sexage": None, "dfr_gmm8": None}
    for method, weight in methods.items():
        w = weight if weight is not None else balanced_weights(yv, groups_of(val, method.removeprefix("dfr_")))
        for select in ("overall", "worst_sexage"):
            parts = []
            for seed in range(SPLITS):
                fit = holdout_split(val, seed)
                model = refit(hv, yv, w, fit, ~fit, val, select)
                s_val, s_test = model.decision_function(hv), model.decision_function(ht)
                parts.append(metrics("ham10000", test, s_test, youden(yv[~fit], s_val[~fit])))
            mean = pd.DataFrame(parts).mean().to_dict()
            rows.append({"method": method, "select": select, **mean})
    return rows


def main() -> None:
    names = conditions("ham10000")
    names["resnet-ham10000-fg-erm"] = "ResNet-50 ERM（公平性比較の基準）"
    table = runs(names)
    table = table[table.exp.isin(ERM)]
    splits = ROOT / "data" / "ham10000" / "splits_latent"
    val, test = pd.read_csv(splits / "val.csv"), pd.read_csv(splits / "test.csv")
    rows = []
    for r in table.itertuples():
        if not features_cache_path(CACHE, r.run_dir.name, "val").exists():
            continue
        for row in one_run(r.run_dir, val, test):
            rows.append({"exp": r.exp, "seed": r.seed, **row})
    out = pd.DataFrame(rows)
    out.to_csv(PACKAGE / "results" / "ham10000_e2e" / "posthoc.csv", index=False)
    cols = ["auroc_all", "auroc_external", "sexage_min", "sexage_gap", "sexage_tpr_gap", "sexage_fpr_gap", "gmm8_min"]
    print(out.groupby(["exp", "method", "select"])[cols].mean().round(4).to_string())


if __name__ == "__main__":
    main()
