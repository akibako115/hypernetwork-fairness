"""部分群ごとに判定規則が違うか（規則の異質性）を、凍結特徴上の頭の比べ合いで測る（段階 0）。

条件付きモデル（hypernet・MoE）が単一モデルに勝てるのは、部分群ごとに最適な判定規則が違うときだけ、という
仮説（主張 C3）の事前指標。同じ特徴 x と群 g で、3 つのロジスティック頭を fit 側で学習し、test で比べる。

- `global`：x だけ
- `intercept`：x ＋ 群の one-hot（群ごとに事前確率だけずれる）
- `interaction`：x ＋ 群の one-hot ＋ x ⊗ 群（群ごとに重みも違う）

- `mlp`：x だけの 2 層 MLP（群を使わない非線形の頭）。クラスタ別の頭は特徴空間を区分ごとに線形にするので、
  その利得が「部分群」ではなく「非線形性」で説明できるかを見る対照

規則の異質性は interaction − intercept の test AUROC 差。群の数だけ頭が増える効果を除くため、
同じ大きさのランダム群で同じ差を測り、差し引いたものを主指標にする。正則化 C は fit 側の病変・患者単位の
取り分けで選ぶ。区間は test の病変・患者単位の対応のある bootstrap（200 回）。

場と特徴（x は fit 側で標準化 → PCA 32）:
- HAM10000：DINOv2（ham_train → ham_test）。群は施設、GMM 8、ラベル直交 GMM 8、性別 × 年齢、ランダム 8
- ISIC 施設間：1 段目 `isic_bcn_s42` の h（isic_audit → isic_test）。群は部位、GMM 8（DINOv2）、性別 × 年齢、ランダム 8
- FairVLMed：DINOv2（fvl_train → fvl_test）。群は GMM 8、ラベル直交 GMM 8、人種、ランダム 8
- CheXpert：1 段目 `a_s42` の h（train_b → test）。群は GMM 8（RAD-DINO）、年齢 × 人種（8）、AP/PA、ランダム 8

出力: `results/heterogeneity.csv`

    uv run python -m analysis.archive.latent_bias_hypernet.heterogeneity
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.neural_network import MLPClassifier

from analysis.archive.latent_bias_hypernet import audit
from analysis.archive.latent_bias_hypernet.data import CACHE, PACKAGE, ROOT, load

DIMS = 32
CS = (0.01, 0.1, 1.0)
BOOT = 200
HEADS = ("global", "intercept", "interaction")


def design(x: np.ndarray, g: np.ndarray, K: int, head: str) -> np.ndarray:
    if head == "global":
        return x
    onehot = np.eye(K, dtype=np.float32)[g]
    if head == "intercept":
        return np.concatenate([x, onehot], axis=1)
    inter = (x[:, :, None] * onehot[:, None, :]).reshape(len(x), -1)
    return np.concatenate([x, onehot, inter], axis=1)


def fit_head(xf, gf, yf, units_f, K, head, seed=0):
    """fit 側を単位で 80/20 に分けて C を選び、全 fit で学習し直す。"""
    uniq = np.unique(units_f)
    held = set(np.random.default_rng(seed).permutation(uniq)[: len(uniq) // 5])
    h = np.isin(units_f, list(held))
    best, best_c = -1.0, CS[0]
    for c in CS:
        m = LogisticRegression(C=c, max_iter=3000).fit(design(xf[~h], gf[~h], K, head), yf[~h])
        auc = roc_auc_score(yf[h], m.decision_function(design(xf[h], gf[h], K, head)))
        if auc > best:
            best, best_c = auc, c
    return LogisticRegression(C=best_c, max_iter=3000).fit(design(xf, gf, K, head), yf)


def min_auc(y, s, demo):
    vals = []
    for k in np.unique(demo):
        m = demo == k
        if k < 0 or y[m].sum() < 15 or (1 - y[m]).sum() < 15:
            continue
        vals.append(roc_auc_score(y[m], s[m]))
    return min(vals)


def run_setting(name, xf, yf, uf, xt, yt, ut, demo_t, partitions):
    """partitions: {群名: (g_fit, g_test, K)}。"""
    mu, sd = xf.mean(0), xf.std(0) + 1e-6
    pca = PCA(DIMS, random_state=0).fit((xf - mu) / sd)
    zf, zt = pca.transform((xf - mu) / sd), pca.transform((xt - mu) / sd)
    s = zf.std(0)
    zf, zt = zf / s, zt / s
    rows = []
    scores = {}
    mlp_scores = []
    for seed in range(3):
        mlp = MLPClassifier((256,), alpha=1e-2, early_stopping=True, max_iter=300, random_state=seed).fit(zf, yf)
        mlp_scores.append(mlp.predict_proba(zt)[:, 1])
    mlp_t = np.mean(mlp_scores, axis=0)
    for pname, (gf, gt, K) in partitions.items():
        for head in HEADS:
            m = fit_head(zf, gf, yf, uf, K, head)
            scores[(pname, head)] = m.decision_function(design(zt, gt, K, head))
    uniq = np.unique(ut)
    idx_by_unit = pd.Series(np.arange(len(ut))).groupby(ut).apply(np.array).to_dict()
    rng = np.random.default_rng(0)
    boots = [np.concatenate([idx_by_unit[u] for u in rng.choice(uniq, len(uniq))]) for _ in range(BOOT)]
    for pname in partitions:
        st = {h: scores[(pname, h)] for h in HEADS}
        row = {"setting": name, "partition": pname, "K": partitions[pname][2]}
        for h in HEADS:
            row[f"auroc_{h}"] = roc_auc_score(yt, st[h])
            row[f"min_demo_{h}"] = min_auc(yt, st[h], demo_t)
        d_int = [roc_auc_score(yt[b], st["interaction"][b]) - roc_auc_score(yt[b], st["intercept"][b]) for b in boots]
        d_icp = [roc_auc_score(yt[b], st["intercept"][b]) - roc_auc_score(yt[b], st["global"][b]) for b in boots]
        row["rule_gain"] = row["auroc_interaction"] - row["auroc_intercept"]
        row["rule_gain_lo"], row["rule_gain_hi"] = np.percentile(d_int, [2.5, 97.5])
        row["prior_gain"] = row["auroc_intercept"] - row["auroc_global"]
        row["prior_gain_lo"], row["prior_gain_hi"] = np.percentile(d_icp, [2.5, 97.5])
        row["demo_rule_gain"] = row["min_demo_interaction"] - row["min_demo_intercept"]
        row["auroc_mlp"] = roc_auc_score(yt, mlp_t)
        row["min_demo_mlp"] = min_auc(yt, mlp_t, demo_t)
        d_mlp = [roc_auc_score(yt[b], st["interaction"][b]) - roc_auc_score(yt[b], mlp_t[b]) for b in boots]
        row["vs_mlp"] = row["auroc_interaction"] - row["auroc_mlp"]
        row["vs_mlp_lo"], row["vs_mlp_hi"] = np.percentile(d_mlp, [2.5, 97.5])
        rows.append(row)
        print({k: (round(v, 4) if isinstance(v, float) else v) for k, v in row.items()}, flush=True)
    out = pd.DataFrame(rows)
    rand = out.loc[out.partition == "ランダム 8"].iloc[0]
    out["rule_gain_vs_random"] = out.rule_gain - rand.rule_gain
    out["demo_rule_gain_vs_random"] = out.demo_rule_gain - rand.demo_rule_gain
    return out


def sex_age(c: pd.DataFrame) -> np.ndarray:
    sex = c.sex.to_numpy()
    age = c.age.to_numpy()
    ok = (sex >= 0) & ~np.isnan(age) & (age > 0)
    return np.where(ok, sex.clip(0, 1) * 2 + (np.nan_to_num(age) >= 65), -1).astype(int)


def random_groups(units: np.ndarray, seed: int) -> np.ndarray:
    uniq = np.unique(units)
    lab = dict(zip(uniq, np.random.default_rng(seed).integers(0, 8, len(uniq)), strict=True))
    return np.array([lab[u] for u in units])


def ham() -> pd.DataFrame:
    lat = ROOT / "data" / "ham10000" / "splits_latent"
    f, t = pd.read_csv(lat / "train.csv"), pd.read_csv(lat / "test.csv")
    ef = np.load(CACHE / "embed" / "dinov2" / "ham_train.npz")["embedding"].astype(np.float32)
    et = np.load(CACHE / "embed" / "dinov2" / "ham_test.npz")["embedding"].astype(np.float32)
    known = t.site_missing.to_numpy() == 0
    uf, ut = f.lesion_id.to_numpy(str), t.lesion_id.to_numpy(str)
    parts = {
        "施設": (f.site.to_numpy(), t.site.to_numpy(), 4),
        "GMM 8": (f.gmm8.to_numpy(), t.gmm8.to_numpy(), 8),
        "ラベル直交 GMM 8": (f.gmm8_lo.to_numpy(), t.gmm8_lo.to_numpy(), 8),
        "性別 × 年齢": (sex_age(f), np.maximum(sex_age(t), 0), 4),
        "ランダム 8": (random_groups(uf, 0), random_groups(ut, 1), 8),
    }
    # 施設を持たない external は施設の比較から外すため、全ての群で既知施設の行だけを使う
    parts = {k: (gf, gt[known], K) for k, (gf, gt, K) in parts.items()}
    return run_setting(
        "HAM10000（DINOv2）",
        ef,
        f.target.to_numpy(),
        uf,
        et[known],
        t.target.to_numpy()[known],
        ut[known],
        sex_age(t)[known],
        parts,
    )


def fairvlmed() -> pd.DataFrame:
    lat = ROOT / "data" / "fairvlmed" / "splits_latent"
    f, t = pd.read_csv(lat / "train.csv"), pd.read_csv(lat / "test.csv")
    ef = np.load(CACHE / "embed" / "dinov2" / "fvl_train.npz")["embedding"].astype(np.float32)
    et = np.load(CACHE / "embed" / "dinov2" / "fvl_test.npz")["embedding"].astype(np.float32)
    uf, ut = f.image.to_numpy(str), t.image.to_numpy(str)
    race_f, race_t = f.race.to_numpy().clip(0, 2), t.race.to_numpy().clip(0, 2)
    parts = {
        "GMM 8": (f.gmm8.to_numpy(), t.gmm8.to_numpy(), 8),
        "ラベル直交 GMM 8": (f.gmm8_lo.to_numpy(), t.gmm8_lo.to_numpy(), 8),
        "人種": (race_f, race_t, 3),
        "ランダム 8": (random_groups(uf, 0), random_groups(ut, 1), 8),
    }
    demo = np.where(t.race_missing.to_numpy() == 0, t.race.to_numpy(), -1)
    return run_setting("FairVLMed（DINOv2）", ef, f.target.to_numpy(), uf, et, t.target.to_numpy(), ut, demo, parts)


def isic() -> pd.DataFrame:
    F = load("isic_bcn_s42", "isic_audit", "dinov2")
    T = load("isic_bcn_s42", "isic_test", "dinov2")
    uf = F.context.lesion_id.fillna(F.context.image).to_numpy(str)
    ut = T.context.lesion_id.fillna(T.context.image).to_numpy(str)
    gmm = audit.GMMSlicer(F.route, 8, dims=32)
    site_f = F.context.anatom_site_general.to_numpy().astype(int)
    site_t = T.context.anatom_site_general.to_numpy().astype(int)
    Ks = int(max(site_f.max(), site_t.max())) + 2
    parts = {
        "部位": (site_f + 1, site_t + 1, Ks),
        "GMM 8": (gmm(F.route), gmm(T.route), 8),
        "性別 × 年齢": (np.maximum(sex_age(F.context), 0), np.maximum(sex_age(T.context), 0), 4),
        "ランダム 8": (random_groups(uf, 0), random_groups(ut, 1), 8),
    }
    return run_setting(
        "ISIC 施設間（1 段目 h）", F.features, F.target, uf, T.features, T.target, ut, sex_age(T.context), parts
    )


def chexpert() -> pd.DataFrame:
    F = load("a_s42", "train_b", "raddino")
    T = load("a_s42", "test", "raddino")
    uf = F.context.image.str.split("/").str[1].to_numpy(str)
    ut = T.context.image.str.split("/").str[1].to_numpy(str)
    gmm = audit.GMMSlicer(F.route, 8, dims=32)
    parts = {
        "GMM 8": (gmm(F.route), gmm(T.route), 8),
        "年齢 × 人種": (F.group, T.group, 8),
        "AP / PA": (F.context.ap_pa.to_numpy().astype(int), T.context.ap_pa.to_numpy().astype(int), 2),
        "ランダム 8": (random_groups(uf, 0), random_groups(ut, 1), 8),
    }
    demo = (T.context.age.to_numpy() >= 65).astype(int)
    return run_setting("CheXpert（1 段目 h）", F.features, F.target, uf, T.features, T.target, ut, demo, parts)


def main() -> None:
    out = pd.concat([ham(), fairvlmed(), isic(), chexpert()], ignore_index=True)
    dest = PACKAGE / "results"
    out.to_csv(dest / "heterogeneity.csv", index=False)
    cols = [
        "setting",
        "partition",
        "auroc_global",
        "prior_gain",
        "rule_gain",
        "rule_gain_lo",
        "rule_gain_hi",
        "rule_gain_vs_random",
        "demo_rule_gain_vs_random",
        "auroc_mlp",
        "vs_mlp",
        "vs_mlp_lo",
        "vs_mlp_hi",
        "min_demo_interaction",
        "min_demo_mlp",
    ]
    print(out[cols].round(4).to_string(index=False))


if __name__ == "__main__":
    main()
