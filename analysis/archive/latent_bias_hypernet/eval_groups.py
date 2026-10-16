"""補正後のモデルを、粒度の違う属性の群で評価する（最小 AUROC・gap・BAcc・EO 系）。

しきい値は各モデルの val スコアで決めた Youden（全体で 1 つ）。test で次を測る。
- 全体：AUROC、BAcc（(TPR + TNR) / 2）
- 群ごと：AUROC、TPR、FPR、BAcc。群の切り方ごとに最小値と、最大 − 最小の gap
群は n ≥ 100 かつ陽性・陰性とも 20 以上のものだけ使う。

出力: `results/eval_groups/im_<stage1>.csv`

    uv run python -m analysis.archive.latent_bias_hypernet.eval_groups --stage1 a_s42
"""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, roc_curve

from analysis.archive.latent_bias_hypernet.audit import demo_cells
from analysis.archive.latent_bias_hypernet.audit_corrected import scores
from analysis.archive.latent_bias_hypernet.data import PACKAGE, load
from analysis.archive.latent_bias_hypernet.planting import hidden_subgroup

MIN_N, MIN_CLASS = 100, 20


def groupings(context: pd.DataFrame) -> dict[str, np.ndarray]:
    sex = context.sex.to_numpy().astype(int)
    old = (context.age.to_numpy() >= 65).astype(int)
    race = context.race_group.to_numpy().astype(int)
    return {
        "sex(2)": sex,
        "age(2)": old,
        "race(4)": race,
        "sex×age(4)": sex * 2 + old,
        "age×race(8)": old * 4 + race,
        "sex×age×race(16)": sex * 8 + old * 4 + race,
        "demo32": demo_cells(context, "demo32"),
    }


def youden(y: np.ndarray, s: np.ndarray) -> float:
    fpr, tpr, thr = roc_curve(y, s)
    return float(thr[np.argmax(tpr - fpr)])


def evaluate(run, V, T, subgroup) -> dict:
    sv, st = scores(run, V, T)
    thr = youden(V.target, sv)
    y, pred = T.target, st >= thr
    tpr, tnr = pred[y == 1].mean(), (~pred[y == 0]).mean()
    row = {"run": run, "auroc": roc_auc_score(y, st), "bacc": (tpr + tnr) / 2, "tpr": tpr, "fpr": 1 - tnr}
    for name, g in groupings(T.context).items():
        stats = []
        for k in np.unique(g):
            m = g == k
            pos, neg = int(y[m].sum()), int((1 - y[m]).sum())
            if m.sum() < MIN_N or pos < MIN_CLASS or neg < MIN_CLASS:
                continue
            gt, gf = pred[m & (y == 1)].mean(), pred[m & (y == 0)].mean()
            stats.append((roc_auc_score(y[m], st[m]), gt, gf, (gt + 1 - gf) / 2))
        a = np.array(stats)
        row[f"{name}|groups"] = len(a)
        row[f"{name}|min_auroc"] = a[:, 0].min()
        row[f"{name}|auroc_gap"] = a[:, 0].max() - a[:, 0].min()
        row[f"{name}|tpr_gap"] = a[:, 1].max() - a[:, 1].min()
        row[f"{name}|fpr_gap"] = a[:, 2].max() - a[:, 2].min()
        row[f"{name}|eo_gap"] = max(row[f"{name}|tpr_gap"], row[f"{name}|fpr_gap"])
        row[f"{name}|min_bacc"] = a[:, 3].min()
    if subgroup:
        S = hidden_subgroup(T.context, subgroup)
        row["S|tpr"] = pred[S & (y == 1)].mean()
        row["S|fpr"] = pred[S & (y == 0)].mean()
        row["S|auroc"] = roc_auc_score(y[S], st[S])
    return row


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage1", default="a_s42")
    parser.add_argument("--subgroup", default=None)
    args = parser.parse_args(argv)
    V, T = load(args.stage1, "val"), load(args.stage1, "test")
    runs = sorted(p.name for p in (PACKAGE / "results" / "latent_codes").glob(f"im_{args.stage1}_*"))
    rows = [evaluate(r, V, T, args.subgroup) for r in ["base", *runs]]
    out = PACKAGE / "results" / "eval_groups"
    out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out / f"im_{args.stage1}.csv", index=False)


if __name__ == "__main__":
    main()
