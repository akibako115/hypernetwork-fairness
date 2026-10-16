"""ISIC 2019 の施設間のずれ（BCN20000 → HAM10000）で、監査セットによる補正と属性・潜在の評価を行う。

植え込みに頼らない検証。1 段目は BCN だけで学習した `isic_bcn_s42`、e(x) は DINOv2。
HAM の監査セット（`isic_audit`）を大きさごとに取り、bgrid.py と同じ補正手法を学習する（監査セットの病変 20% で選択）。
test（`isic_test`）で次を測る。
- 全体 AUROC
- 属性：性別（2）、65 歳で 2 区分の年齢、性別 × 年齢（4）の最小 AUROC と TPR・FPR gap
  （しきい値は監査器用 split の Youden）
- 部位（`anatom_site_general`、属性ではない要因）の最小 AUROC
- 潜在スライスの偏り：`isic_audit_v` で学習した独立な監査器（h＋DINOv2、3 seed）の 16 分位での log O/E の SD

出力: `results/isic/grid.csv`、各条件のスコアは `cache/isic_scores/<method>_<size>_s<seed>.npz`

    uv run python -m analysis.archive.latent_bias_hypernet.isic_eval
"""

from __future__ import annotations

import argparse
import itertools

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, roc_curve

from analysis.archive.latent_bias_hypernet import audit
from analysis.archive.latent_bias_hypernet.audit_corrected import logit
from analysis.archive.latent_bias_hypernet.bgrid import corrector, last_layer, ma_tree, split_holdout, take
from analysis.archive.latent_bias_hypernet.data import PACKAGE, Split, load

STAGE1 = "isic_bcn_s42"
EMBED = "dinov2"
METHODS = ("none", "global", "gmm16", "router_he", "llr", "dfr_demo", "ma_tree", "lhcf_cluster")


def sex_age(s: Split) -> np.ndarray:
    sex = s.context.sex.to_numpy().clip(0, 1).astype(int)
    return sex * 2 + (s.context.age.fillna(50).to_numpy() >= 65).astype(int)


def groupings(s: Split) -> dict[str, np.ndarray]:
    c = s.context
    old = (c.age.fillna(50).to_numpy() >= 65).astype(int)
    return {
        "sex": c.sex.to_numpy().astype(int),
        "age": old,
        "sex×age": sex_age(s),
        "site": c.anatom_site_general.to_numpy().astype(int),
    }


def group_stats(y, score, pred, g) -> dict[str, float]:
    aucs, tprs, fprs = [], [], []
    for k in np.unique(g):
        m = g == k
        if k < 0 or y[m].sum() < 15 or (1 - y[m]).sum() < 15:
            continue
        aucs.append(roc_auc_score(y[m], score[m]))
        tprs.append(pred[m & (y == 1)].mean())
        fprs.append(pred[m & (y == 0)].mean())
    return {
        "min_auroc": min(aucs),
        "auroc_gap": max(aucs) - min(aucs),
        "tpr_gap": max(tprs) - min(tprs),
        "fpr_gap": max(fprs) - min(fprs),
    }


def evaluate(score_v, score_t, V: Split, T: Split, xv, xt, seeds=(0, 1, 2)) -> dict:
    y = T.target
    fpr, tpr, thr = roc_curve(V.target, score_v)
    pred = score_t >= thr[np.argmax(tpr - fpr)]
    row = {"auroc": roc_auc_score(y, score_t)}
    row["bacc"] = (pred[y == 1].mean() + (1 - pred[y == 0].mean())) / 2
    for name, g in groupings(T).items():
        for k, v in group_stats(y, score_t, pred, g).items():
            row[f"{name}|{k}"] = v
    cal = audit.Calibrator(score_v, V.target)
    pv, pt = cal(score_v), cal(score_t)
    patients = V.context.lesion_id.fillna(V.context.image).to_numpy()
    sds = []
    for seed in seeds:
        r = audit.ResidualAuditor(xv, logit(pv), V.target, patients, seed)(xt)
        slices = audit.quantile_slicer(r, 16)(r)
        table = audit.group_metrics(slices, y, pt, cal.threshold, 100, 10.0)
        lesion_t = T.context.lesion_id.fillna(T.context.image).to_numpy()
        sds.append(audit.dispersion(table, slices, y, pt, lesion_t)["excess_sd"])
    row["latent_sd"] = float(np.mean(sds))
    return row


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sizes", nargs="+", default=["500", "1000", "2000", "all"])
    parser.add_argument("--methods", nargs="+", default=list(METHODS))
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    args = parser.parse_args(argv)

    pool = load(STAGE1, "isic_audit", "h+" + EMBED)
    e_pool = load(STAGE1, "isic_audit", EMBED).route
    V = load(STAGE1, "isic_audit_v", "h+" + EMBED)
    T = load(STAGE1, "isic_test", "h+" + EMBED)
    e_v, e_t = load(STAGE1, "isic_audit_v", EMBED).route, load(STAGE1, "isic_test", EMBED).route
    xv, (ph, pe) = audit.auditor_inputs(V.features, e_v, V.features, e_v, dims=32)
    xt = np.concatenate([ph.transform(T.features), pe.transform(e_t)], axis=1)
    # 病変を患者の代わりに使う（bgrid の患者単位の取り分けは image の 2 番目の要素を見るため差し替える）
    for s in (pool, V, T):
        s.context = s.context.assign(image=("x/" + s.context.lesion_id.fillna(s.context.image) + "/" + s.context.image))

    out = PACKAGE / "results" / "isic"
    out.mkdir(parents=True, exist_ok=True)
    path = out / "grid.csv"
    rows = pd.read_csv(path).to_dict("records") if path.exists() else []
    have = {(str(r["size"]), r["method"], int(r["seed"])) for r in rows}
    for size, seed in itertools.product(args.sizes, args.seeds):
        B, e = take(pool, e_pool, None if size == "all" else int(size), seed)
        fit, e_fit, hold, e_hold = split_holdout(B, e, seed)
        evals = [(T, e_t), (V, e_v)]
        for method in args.methods:
            if (size, method, seed) in have:
                continue
            if method == "none":
                st, sv = T.base, V.base
            elif method in ("llr", "dfr_demo"):
                st, sv = last_layer(method, fit, hold, evals, groups=sex_age)
            elif method == "ma_tree":
                st, sv = ma_tree(fit, e_fit, hold, e_hold, evals, seed)
            else:
                st, sv = corrector(method, fit, e_fit, hold, e_hold, evals, seed)
            scores_dir = PACKAGE / "cache" / "isic_scores"
            scores_dir.mkdir(parents=True, exist_ok=True)
            np.savez(scores_dir / f"{method}_{size}_s{seed}.npz", test=st, audit_v=sv)
            row = {"size": size, "method": method, "seed": seed, "n_B": len(B.target)}
            row.update(evaluate(sv, st, V, T, xv, xt))
            rows.append(row)
            print({k: (round(v, 3) if isinstance(v, float) else v) for k, v in row.items()}, flush=True)
            pd.DataFrame(rows).to_csv(path, index=False)


if __name__ == "__main__":
    main()
