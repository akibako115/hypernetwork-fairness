"""発見の質が是正の効果を予測するかを、群の切り方を並べて確かめる（HAM10000、ERM の凍結特徴）。

発見（監査）側で試してきた切り方を、そのまま是正（事後の DFR）の群として使い、両者の順位が対応するかを見る。

切り方（K = 8。群の数が違うものは注記）：
- `sexage`：性別 × 年齢 4 群（属性。是正の対象そのもの）
- `site`：撮影施設（属性でない既知の要因。val では 3 施設）
- `dino_gmm8`・`dino_gmm8_lo`・`random8`：`latent_conditions.py` の列（DINOv2 の GMM、ラベル直交版、ランダム）
- `h_km8`：ERM の特徴（PCA 32）の k-means
- `h_lo_km8`：ERM の特徴から、ラベルと ERM の得点で説明できる部分を除いてからの k-means
- `he_km8`：ERM の特徴と DINOv2 を並べた [h; e] の k-means
- `score_q8`：ERM の得点の 8 分位（難しさだけで切る対照）
- `jtt4`：クラス × ERM の誤り（val の Youden しきい値）の 4 群（JTT 型）

k-means と分位は val だけで決める（test のラベルも特徴も使わない）。

発見の採点は、監査者が手にする val の上で、ERM の val 得点（ERM は val を学習していない）を使う。
- `oe_sd`：val で全体をスプライン較正したあとの、群ごとの log(O/E) の件数重み付き SD（群ごとの系統的なずれ）。
  Platt では較正曲線の曲がりが分位の群に残るので使わない
- `auc_gap`：全体 AUROC − 群内 AUROC の最小（陽性・陰性とも 10 件以上の群）
- `nmi_score`・`nmi_label`：得点の 8 分位・ラベルとの NMI（難しさ・ラベルで切れている度合い）
- `nmi_sexage`・`nmi_site`：属性・施設との NMI

是正は DFR（群 × クラス均等の重みで最終層を学び直す）。val を病変単位で半分に分け、片方で学習、
もう片方の全体 AUROC で正則化を選ぶ（属性を使わない。5 通りの分け方の平均）。効果は、群を使わない学び直し（`llr`）との
差で測る（「val で学び直す」効果を除き、群の効果だけを見る）。

入力: `cache/<run-id>_{val,test}{,_features}.npz`（ERM の run）、`data/ham10000/splits_latent/{val,test}.csv`
出力: `results/ham10000_e2e/discovery_remedy.csv`（run × 切り方）、`discovery_remedy_rank.csv`（順位相関）

    uv run python -m analysis.archive.latent_bias_hypernet.discovery_to_remedy
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy.stats import spearmanr
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import normalized_mutual_info_score as nmi
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import SplineTransformer, StandardScaler

from analysis.archive.latent_bias_hypernet.data import PACKAGE, ROOT
from analysis.archive.latent_bias_hypernet.e2e_eval import CACHE, conditions, group_aucs, metrics, runs, sex_age
from analysis.archive.latent_bias_hypernet.posthoc_fairness import ERM, balanced_weights, holdout_split, youden
from analysis.common.predictions import features_cache_path, load_cache, load_features

K = 8
SPLITS = 5
CS = (0.001, 0.01, 0.1, 1.0)
DINO = [f"e{i}" for i in range(16)]
DISCOVERY = ["oe_sd", "auc_gap", "nmi_score", "nmi_label", "nmi_sexage", "nmi_site"]
REMEDY = ["auroc_all", "auroc_external", "sexage_min", "sexage_gap", "gmm8_min"]


def kmeans(x: np.ndarray, seed: int) -> np.ndarray:
    return KMeans(K, n_init=10, random_state=seed).fit_predict(StandardScaler().fit_transform(x))


def quantiles(s: np.ndarray, k: int = K) -> np.ndarray:
    return np.searchsorted(np.quantile(s, np.linspace(0, 1, k + 1)[1:-1]), s)


def groupings(val: pd.DataFrame, h: np.ndarray, s: np.ndarray, seed: int) -> dict[str, np.ndarray]:
    """val の各行の群。k-means と分位は val だけで決める。"""
    y = val.target.to_numpy()
    hp = PCA(32, random_state=seed).fit_transform(StandardScaler().fit_transform(h))
    side = np.column_stack([y, s])
    h_lo = hp - LinearRegression().fit(side, hp).predict(side)
    pred = s >= youden(y, s)
    return {
        "sexage": sex_age(val, "sex"),
        "site": val.dataset.to_numpy(),
        "dino_gmm8": val.gmm8.to_numpy(),
        "dino_gmm8_lo": val.gmm8_lo.to_numpy(),
        "random8": val.random8.to_numpy(),
        "h_km8": kmeans(hp, seed),
        "h_lo_km8": kmeans(h_lo, seed),
        "he_km8": kmeans(np.column_stack([hp, val[DINO].to_numpy()]), seed),
        "score_q8": quantiles(s),
        "jtt4": y * 2 + (pred != y),
    }


def discovery(val: pd.DataFrame, s: np.ndarray, g: np.ndarray) -> dict:
    y = val.target.to_numpy()
    basis = SplineTransformer(n_knots=5, degree=3).fit_transform(s[:, None])
    p = LogisticRegression(C=10.0, max_iter=2000).fit(basis, y).predict_proba(basis)[:, 1]
    keys, counts = np.unique(g, return_counts=True)
    log_oe = np.array([np.log((y[g == k].sum() + 0.5) / (p[g == k].sum() + 0.5)) for k in keys])
    mean = np.average(log_oe, weights=counts)
    aucs = group_aucs(y, s, g, min_n=10)
    return {
        "oe_sd": float(np.sqrt(np.average((log_oe - mean) ** 2, weights=counts))),
        "auc_gap": roc_auc_score(y, s) - min(aucs.values()) if aucs else np.nan,
        "nmi_score": nmi(quantiles(s), g),
        "nmi_label": nmi(y, g),
        "nmi_sexage": nmi(sex_age(val, "sex"), g),
        "nmi_site": nmi(val.dataset, g),
        "n_groups": len(keys),
        "min_cell_pos": int(min(y[g == k].sum() for k in keys)),
    }


def refit_overall(hv, yv, w, fit, hold) -> LogisticRegression:
    best, best_score = None, -np.inf
    for c in CS:
        model = LogisticRegression(C=c, max_iter=2000).fit(hv[fit], yv[fit], sample_weight=w[fit])
        score = roc_auc_score(yv[hold], model.decision_function(hv[hold]))
        if score > best_score:
            best, best_score = model, score
    return best


def remedy(hv, ht, val, test, w) -> dict:
    """群の重み w で最終層を学び直したときの test の指標（5 通りの分け方の平均）。"""
    yv = val.target.to_numpy()
    parts = []
    for split in range(SPLITS):
        fit = holdout_split(val, split)
        model = refit_overall(hv, yv, w, fit, ~fit)
        s_val, s_test = model.decision_function(hv), model.decision_function(ht)
        parts.append(metrics("ham10000", test, s_test, youden(yv[~fit], s_val[~fit])))
    return pd.DataFrame(parts)[REMEDY].mean().to_dict()


def one_run(run_dir, exp: str, seed: int, val: pd.DataFrame, test: pd.DataFrame) -> list[dict]:
    images_v, images_t = val.image.to_numpy(str), test.image.to_numpy(str)
    hv = load_features(features_cache_path(CACHE, run_dir.name, "val"), images_v)["features"].astype(np.float32)
    ht = load_features(features_cache_path(CACHE, run_dir.name, "test"), images_t)["features"].astype(np.float32)
    logits = load_cache(CACHE / f"{run_dir.name}_val.npz", images_v)["logits"]
    s = logits[:, 1] - logits[:, 0]
    groups = groupings(val, hv, s, seed)
    scaler = StandardScaler().fit(hv)
    hv_s, ht_s = scaler.transform(hv), scaler.transform(ht)
    yv = val.target.to_numpy()
    base = remedy(hv_s, ht_s, val, test, np.ones(len(yv)))
    rows = [{"exp": exp, "seed": seed, "grouping": "llr", **base}]
    for name, g in groups.items():
        out = remedy(hv_s, ht_s, val, test, balanced_weights(yv, g))
        delta = {f"d_{k}": out[k] - base[k] for k in REMEDY}
        rows.append({"exp": exp, "seed": seed, "grouping": name, **discovery(val, s, g), **out, **delta})
    return rows


def rank_table(frame: pd.DataFrame) -> pd.DataFrame:
    """切り方ごとの seed 平均で、発見の指標と是正の効果の Spearman 順位相関。"""
    mean = frame[frame.grouping != "llr"].groupby("grouping").mean(numeric_only=True)
    rows = []
    for d in DISCOVERY:
        for r in ("d_sexage_min", "d_sexage_gap", "d_auroc_external", "d_auroc_all"):
            rho, p = spearmanr(mean[d], mean[r], nan_policy="omit")
            n = int(mean[[d, r]].notna().all(axis=1).sum())
            rows.append({"discovery": d, "remedy": r, "spearman": rho, "p": p, "n": n})
    return pd.DataFrame(rows)


def main() -> None:
    names = conditions("ham10000")
    names["resnet-ham10000-fg-erm"] = "ResNet-50 ERM（公平性比較の基準）"
    table = runs(names)
    table = table[table.exp.isin(ERM)]
    table = table[[features_cache_path(CACHE, d.name, "val").exists() for d in table.run_dir]]
    splits = ROOT / "data" / "ham10000" / "splits_latent"
    val, test = pd.read_csv(splits / "val.csv"), pd.read_csv(splits / "test.csv")
    jobs = (delayed(one_run)(r.run_dir, r.exp, r.seed, val, test) for r in table.itertuples())
    rows = [row for part in Parallel(n_jobs=len(table))(jobs) for row in part]
    frame = pd.DataFrame(rows)
    dest = PACKAGE / "results" / "ham10000_e2e"
    frame.to_csv(dest / "discovery_remedy.csv", index=False)
    cols = DISCOVERY + ["min_cell_pos"] + [f"d_{k}" for k in REMEDY]
    print(frame.groupby("grouping")[cols].mean().round(3).to_string())
    ranks = rank_table(frame)
    ranks.to_csv(dest / "discovery_remedy_rank.csv", index=False)
    print(ranks.round(3).to_string())


if __name__ == "__main__":
    main()
