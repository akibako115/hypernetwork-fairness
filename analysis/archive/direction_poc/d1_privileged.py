"""方向 1：画像から計算できない条件（学習時にしか無い情報）で部分群を作り、推論時は画像から推定して振り分ける。

問い：学習時だけの情報で作った区分は、同じ数の GMM 区分やランダム区分より、条件付きの頭として良いか。
推論時の条件は画像から推定するので、結局は画像の関数になる。それでも効くなら「この区分で規則を分けろ」という
帰納バイアスが効いている、と読む。対照として、推論時にも使えて画像からは計算できない条件（過去検査の有無）も並べる。

場と区分：
- CheXpert（1 段目 `a_s42`・`a_s43` の h、train_b → test）。学習時だけの情報はレポート種別（affirmed_normal /
  focused_followup / other、`label_construct.py` の分類）。推論時にも使える非画像の条件は追跡撮影（過去検査あり）。
- HAM10000（DINOv2、train → test）。学習時だけの情報は診断の確定方法（病理か、経過観察・合議・共焦点か）。
  悪性はすべて病理で確定しているので、この区分はラベルとほぼ重なる。

頭（すべて h を標準化して PCA 64 次元に落とした z の上のロジスティック回帰）：
- `global`：z だけ。`mlp`：z だけの 2 層 MLP（区分を使わない非線形の頭）
- `oracle`：学習時も test でも本当の区分で切り替える（test で使えない情報を使う参考値。ラベルの漏れを含む）
- `pred`：区分を z から推定し（学習側は患者単位 5 分割の cross-fit）、学習時も test でも推定した区分で切り替える
- `lupi_soft` / `lupi_hard`：学習時は本当の区分で頭を学び、test では推定した区分の事後確率で頭を混ぜる / 最大の区分を使う
- `lupi_std`：学習時は本当の区分で頭を学び、test では全員を標準の区分（CheXpert は affirmed_normal）として読む
- 対照：同じ数の GMM 区分（基盤モデルの埋め込み）、ランダム区分、属性（CheXpert は年齢 × race、HAM は性別 × 年齢）

切り替えは「切片＋群ごとの重みの差（z ⊗ 群）」で表し、正則化は fit 側の患者（病変）単位の取り分けで選ぶ。
区間は test の患者（病変）単位 bootstrap（`BOOT` 回）、global との対応のある差。

出力: `results/d1/{scores_<setting>.npz,summary.csv,router.csv}`

    uv run python -m analysis.archive.direction_poc.d1_privileged
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import rootutils
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.neural_network import MLPClassifier

ROOT = rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from analysis.archive.direction_poc.d5_label_semantics import weighted_auc  # noqa: E402
from analysis.archive.latent_bias_hypernet import audit  # noqa: E402
from analysis.archive.latent_bias_hypernet.data import CACHE, load  # noqa: E402
from analysis.archive.latent_bias_hypernet.heterogeneity import random_groups, sex_age  # noqa: E402

OUT = ROOT / "analysis" / "archive" / "direction_poc" / "results" / "d1"
DIMS = 64
CS = (0.01, 0.1, 1.0)
BOOT = 300
MIN_POS = 10
REPORT_TYPES = ("affirmed_normal", "focused_followup", "other")


def onehot(g: np.ndarray, K: int) -> np.ndarray:
    return np.eye(K, dtype=np.float32)[g]


def design(z: np.ndarray, G: np.ndarray) -> np.ndarray:
    """切片と、群ごとの重みの差。G は one-hot でも事後確率でもよい（事後確率なら頭の線形混合になる）。"""
    inter = (z[:, :, None] * G[:, None, :]).reshape(len(z), -1)
    return np.concatenate([z, G, inter], axis=1)


def held_mask(units: np.ndarray, seed: int = 0) -> np.ndarray:
    uniq = np.unique(units)
    held = np.random.default_rng(seed).permutation(uniq)[: len(uniq) // 5]
    return np.isin(units, held)


def fit_logistic(X: np.ndarray, y: np.ndarray, units: np.ndarray) -> LogisticRegression:
    """fit 側を単位で 80/20 に分けて C を選び、全 fit で学習し直す。"""
    h = held_mask(units)
    best, best_c = -1.0, CS[0]
    for c in CS:
        m = LogisticRegression(C=c, max_iter=2000).fit(X[~h], y[~h])
        auc = roc_auc_score(y[h], m.decision_function(X[h]))
        if auc > best:
            best, best_c = auc, c
    return LogisticRegression(C=best_c, max_iter=2000).fit(X, y)


def router(zf: np.ndarray, gf: np.ndarray, uf: np.ndarray, zt: np.ndarray, K: int) -> tuple[np.ndarray, np.ndarray]:
    """区分を z から推定する。fit 側は患者単位 5 分割の cross-fit、test は全 fit で学習した版の事後確率。"""
    pf = np.zeros((len(zf), K), dtype=np.float32)
    for tr, te in GroupKFold(5).split(zf, gf, uf):
        m = LogisticRegression(C=0.1, max_iter=2000).fit(zf[tr], gf[tr])
        pf[np.ix_(te, m.classes_)] = m.predict_proba(zf[te])
    m = LogisticRegression(C=0.1, max_iter=2000).fit(zf, gf)
    pt = np.zeros((len(zt), K), dtype=np.float32)
    pt[:, m.classes_] = m.predict_proba(zt)
    return pf, pt


def project(xf: np.ndarray, xt: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mu, sd = xf.mean(0), xf.std(0) + 1e-6
    pca = PCA(DIMS, random_state=0).fit((xf - mu) / sd)
    zf, zt = pca.transform((xf - mu) / sd), pca.transform((xt - mu) / sd)
    s = zf.std(0)
    return (zf / s).astype(np.float32), (zt / s).astype(np.float32)


def privileged_heads(zf, yf, uf, zt, priv_f, priv_t, K, standard):
    """学習時だけの区分に関わる頭の test score と、推定の質（test の one-vs-rest AUROC）。"""
    pf, pt = router(zf, priv_f, uf, zt, K)
    scores = {}
    oracle = fit_logistic(design(zf, onehot(priv_f, K)), yf, uf)
    scores["oracle"] = oracle.decision_function(design(zt, onehot(priv_t, K)))
    scores["lupi_soft"] = oracle.decision_function(design(zt, pt))
    scores["lupi_hard"] = oracle.decision_function(design(zt, onehot(pt.argmax(1), K)))
    if standard is not None:
        scores["lupi_std"] = oracle.decision_function(design(zt, onehot(np.full(len(zt), standard), K)))
    pred = fit_logistic(design(zf, onehot(pf.argmax(1), K)), yf, uf)
    scores["pred"] = pred.decision_function(design(zt, onehot(pt.argmax(1), K)))
    pred_soft = fit_logistic(design(zf, pf), yf, uf)
    scores["pred_soft"] = pred_soft.decision_function(design(zt, pt))
    quality = {}
    for k in range(K):
        if 0 < (priv_t == k).sum() < len(priv_t):
            quality[f"router_auc_{k}"] = roc_auc_score(priv_t == k, pt[:, k])
    quality["router_acc"] = float((pt.argmax(1) == priv_t).mean())
    return scores, quality


def fixed_heads(zf, yf, uf, zt, partitions):
    """推論時にもそのまま使える区分（GMM・ランダム・属性・非画像の条件）で切り替えた頭。"""
    scores = {}
    for name, (gf, gt, K) in partitions.items():
        m = fit_logistic(design(zf, onehot(gf, K)), yf, uf)
        scores[name] = m.decision_function(design(zt, onehot(gt, K)))
    return scores


def baseline_heads(zf, yf, uf, zt):
    scores = {"global": fit_logistic(zf, yf, uf).decision_function(zt)}
    mlp = []
    for seed in range(3):
        m = MLPClassifier((256,), alpha=1e-2, early_stopping=True, max_iter=300, random_state=seed).fit(zf, yf)
        mlp.append(m.predict_proba(zt)[:, 1])
    scores["mlp"] = np.log(np.mean(mlp, axis=0) + 1e-7) - np.log(1 - np.mean(mlp, axis=0) + 1e-7)
    return scores


def evaluate(scores: dict, y: np.ndarray, demo: np.ndarray, young: np.ndarray, units: np.ndarray, extra: dict) -> list:
    """全体 AUROC、属性の最悪群 AUROC、年齢 2 区分の差、各部分集合の AUROC と、global との差の bootstrap 区間。"""
    codes, uniq = pd.factorize(units)
    rng = np.random.default_rng(0)
    weights = [rng.multinomial(len(uniq), np.full(len(uniq), 1 / len(uniq)))[codes].astype(float) for _ in range(BOOT)]

    def stats(s, w):
        out = {"auroc": weighted_auc(s, y, w)}
        worst = []
        for k in np.unique(demo[demo >= 0]):
            m = demo == k
            if (y[m] == 1).sum() >= MIN_POS and (y[m] == 0).sum() >= MIN_POS:
                worst.append(weighted_auc(s[m], y[m], w[m]))
        out["worst_demo"] = min(worst)
        if young is not None:
            out["age_gap"] = weighted_auc(s[young], y[young], w[young]) - weighted_auc(s[~young], y[~young], w[~young])
        for name, mask in extra.items():
            out[name] = weighted_auc(s[mask], y[mask], w[mask])
        return out

    ones = np.ones(len(y))
    base_boot = [stats(scores["global"], w) for w in weights]
    rows = []
    for name, s in scores.items():
        row = {"head": name, **stats(s, ones)}
        if name != "global":
            boot = [stats(s, w) for w in weights]
            for key in list(row):
                if key == "head":
                    continue
                diffs = np.array([b[key] - bb[key] for b, bb in zip(boot, base_boot, strict=True)])
                row[f"d_{key}_lo"], row[f"d_{key}_hi"] = np.nanpercentile(diffs, [2.5, 97.5])
        rows.append(row)
    return rows


def chexpert(stage1: str) -> tuple[list, dict]:
    F = load(stage1, "train_b", "raddino")
    T = load(stage1, "test", "raddino")
    uf = F.context.image.str.split("/").str[1].to_numpy(str)
    ut = T.context.image.str.split("/").str[1].to_numpy(str)
    zf, zt = project(F.features, T.features)
    rt_f = F.context.report_type.map({t: i for i, t in enumerate(REPORT_TYPES)}).to_numpy()
    rt_t = T.context.report_type.map({t: i for i, t in enumerate(REPORT_TYPES)}).to_numpy()
    fu_f = (F.context.patient_report_date_order.to_numpy() > 1).astype(int)
    fu_t = (T.context.patient_report_date_order.to_numpy() > 1).astype(int)
    gmm3 = audit.GMMSlicer(F.route, 3, dims=32)
    gmm8 = audit.GMMSlicer(F.route, 8, dims=32)
    partitions = {
        "gmm3": (gmm3(F.route), gmm3(T.route), 3),
        "random3": (random_groups(uf, 0) % 3, random_groups(ut, 1) % 3, 3),
        "followup": (fu_f, fu_t, 2),
        "random2": (random_groups(uf, 2) % 2, random_groups(ut, 3) % 2, 2),
        "attr8": (F.group, T.group, 8),
        "gmm8": (gmm8(F.route), gmm8(T.route), 8),
        "random8": (random_groups(uf, 0), random_groups(ut, 1), 8),
    }
    scores = baseline_heads(zf, F.target, uf, zt)
    priv, quality = privileged_heads(zf, F.target, uf, zt, rt_f, rt_t, 3, standard=0)
    scores.update(priv)
    scores.update(fixed_heads(zf, F.target, uf, zt, partitions))
    y = T.target
    affirmed = (y == 0) | (rt_t == 0)
    young = T.context.age.to_numpy() < 65
    extra = {"auroc_affirmed": affirmed, "auroc_affirmed_old": affirmed & ~young, "auroc_affirmed_young": affirmed & young}
    np.savez_compressed(OUT / f"scores_chexpert_{stage1}.npz", y=y, **{k: v for k, v in scores.items()})
    rows = evaluate(scores, y, T.group, young, ut, extra)
    for r in rows:
        r["setting"] = f"CheXpert（{stage1}）"
    return rows, {"setting": f"CheXpert（{stage1}）", **quality}


def ham() -> tuple[list, dict]:
    lat = ROOT / "data" / "ham10000" / "splits_latent"
    f, t = pd.read_csv(lat / "train.csv"), pd.read_csv(lat / "test.csv")
    ef = np.load(CACHE / "embed" / "dinov2" / "ham_train.npz")
    et = np.load(CACHE / "embed" / "dinov2" / "ham_test.npz")
    assert np.array_equal(ef["image"], f.image.to_numpy(str)) and np.array_equal(et["image"], t.image.to_numpy(str))
    zf, zt = project(ef["embedding"].astype(np.float32), et["embedding"].astype(np.float32))
    uf, ut = f.lesion_id.to_numpy(str), t.lesion_id.to_numpy(str)
    histo_f, histo_t = (f.dx_type.to_numpy() == 3).astype(int), (t.dx_type.to_numpy() == 3).astype(int)
    gmm2 = audit.GMMSlicer(ef["embedding"].astype(np.float32), 2, dims=32)
    partitions = {
        "gmm2": (gmm2(ef["embedding"].astype(np.float32)), gmm2(et["embedding"].astype(np.float32)), 2),
        "random2": (random_groups(uf, 0) % 2, random_groups(ut, 1) % 2, 2),
        "attr4": (np.maximum(sex_age(f), 0), np.maximum(sex_age(t), 0), 4),
        "gmm8": (f.gmm8.to_numpy(), t.gmm8.to_numpy(), 8),
        "random8": (random_groups(uf, 0), random_groups(ut, 1), 8),
    }
    yf, y = f.target.to_numpy(), t.target.to_numpy()
    scores = baseline_heads(zf, yf, uf, zt)
    priv, quality = privileged_heads(zf, yf, uf, zt, histo_f, histo_t, 2, standard=None)
    scores.update(priv)
    scores.update(fixed_heads(zf, yf, uf, zt, partitions))
    extra = {"auroc_histo": histo_t == 1, "auroc_external": t.site_missing.to_numpy() == 1}
    np.savez_compressed(OUT / "scores_ham.npz", y=y, **{k: v for k, v in scores.items()})
    rows = evaluate(scores, y, sex_age(t), None, ut, extra)
    for r in rows:
        r["setting"] = "HAM10000（DINOv2）"
    return rows, {"setting": "HAM10000（DINOv2）", **quality}


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    rows, quality = [], []
    for fn, arg in ((chexpert, "a_s42"), (chexpert, "a_s43"), (ham, None)):
        r, q = fn(arg) if arg else fn()
        rows += r
        quality.append(q)
        print(pd.DataFrame(r).set_index("head").drop(columns="setting").round(4).to_string(), flush=True)
        print(q, flush=True)
    pd.DataFrame(rows).to_csv(OUT / "summary.csv", index=False)
    pd.DataFrame(quality).to_csv(OUT / "router.csv", index=False)


if __name__ == "__main__":
    main()
