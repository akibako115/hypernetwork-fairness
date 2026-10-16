"""ResNet-50・Spatial LoRA（定数）・Spatial LoRA（属性）の全体性能と公平性指標（HAM10000）。

「属性で条件付けた hypernet が性能を底上げする」を図で見るための値を書く。全体性能は AUROC と BAcc、
公平性指標は性別 × 年齢 4 群の AUROC gap・Eopp1（TPR の gap）・Eopp0（FPR の gap）と、群ごとの AUROC・TPR・FPR。しきい値は各 run の val の Youden。
定数の版は、パラメータの数が同じで条件だけがない構成で、利得がアダプタの容量から来るかの対照になる。

全体性能は属性（性別・年齢）が揃った test の行で求める。属性が欠ける行は別施設（external）に集中し、
属性で条件付けたモデルには学習中に見ていない欠損の入力が入るため（`analysis/ham10000_architecture_erm` の主比較と同じ扱い）。
参考に全行の全体 AUROC も書く（`overall_auroc_all`）。

値は 3 seed の平均で、区間は病変単位の bootstrap（500 回）。差は ResNet-50 との対応のある差で、同じ bootstrap 標本で求める。

入力: `e2e_eval.py` と同じ予測 cache（`resnet-ham10000`・`spatial-lora-ham10000{,-const}`）
出力: `results/ham10000_e2e/architecture_groups.csv`（model × metric ごとの値・区間・ResNet-50 との差・区間）

    uv run python -m analysis.archive.latent_bias_hypernet.architecture_groups
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from analysis.archive.latent_bias_hypernet.data import PACKAGE, ROOT
from analysis.archive.latent_bias_hypernet.e2e_eval import (
    axis_metrics,
    bootstrap_index,
    runs,
    scores,
    sex_age,
    thresholds,
)

MODELS = {
    "resnet-ham10000": "ResNet-50",
    "spatial-lora-ham10000-const": "Spatial LoRA（定数で条件付け）",
    "spatial-lora-ham10000": "Spatial LoRA（属性で条件付け）",
}
REFERENCE = "resnet-ham10000"
# sex_age の符号：性別（female 0・male 1）× 2 ＋ 65 歳以上
GROUPS = {0: "female_lt65", 1: "female_ge65", 2: "male_lt65", 3: "male_ge65"}
METRICS = ["overall_auroc", "overall_bacc", "overall_auroc_all", "auroc_gap", "eopp1", "eopp0"]
METRICS += [f"{m}_{g}" for m in ("auroc", "tpr", "fpr") for g in GROUPS.values()]


def values(y, s, g, thr: float) -> dict[str, float]:
    complete = g >= 0
    yc, pred = y[complete], s[complete] >= thr
    axis = axis_metrics("sexage", y, s, s >= thr, g)
    groups = {}
    for k, name in GROUPS.items():
        m = g == k
        groups[f"auroc_{name}"] = roc_auc_score(y[m], s[m])
        groups[f"tpr_{name}"] = (s[m & (y == 1)] >= thr).mean()
        groups[f"fpr_{name}"] = (s[m & (y == 0)] >= thr).mean()
    return groups | {
        "overall_auroc": roc_auc_score(yc, s[complete]),
        "overall_bacc": (pred[yc == 1].mean() + (~pred[yc == 0]).mean()) / 2,
        "overall_auroc_all": roc_auc_score(y, s),
        "auroc_gap": axis["sexage_gap"],
        "eopp1": axis["sexage_tpr_gap"],
        "eopp0": axis["sexage_fpr_gap"],
    }


def main() -> None:
    table = runs(MODELS)
    splits = ROOT / "data" / "ham10000" / "splits_latent"
    val, test = pd.read_csv(splits / "val.csv"), pd.read_csv(splits / "test.csv")
    thr = thresholds(val, scores(val, "val", table))
    st = scores(test, "test", table)
    seeds_of = table.groupby("exp").seed.apply(list).to_dict()
    y, g = test.target.to_numpy(), sex_age(test, "sex")

    def seed_mean(exp, idx):
        vals = [values(y[idx], st[(exp, s)][idx], g[idx], thr[(exp, s)]) for s in seeds_of[exp]]
        return {k: np.mean([v[k] for v in vals]) for k in METRICS}

    full = {exp: seed_mean(exp, np.arange(len(test))) for exp in MODELS}
    draws = {exp: [] for exp in MODELS}
    for b in bootstrap_index(test, "ham10000"):
        try:
            cur = {exp: seed_mean(exp, b) for exp in MODELS}
        except ValueError:
            continue
        for exp in MODELS:
            draws[exp].append(cur[exp])
    rows = []
    for exp, label in MODELS.items():
        for k in METRICS:
            v = np.array([d[k] for d in draws[exp]])
            diff = v - np.array([d[k] for d in draws[REFERENCE]])
            lo, hi = np.percentile(v, [2.5, 97.5])
            dlo, dhi = np.percentile(diff, [2.5, 97.5])
            rows.append(
                {
                    "exp": exp,
                    "model": label,
                    "metric": k,
                    "value": full[exp][k],
                    "lo": lo,
                    "hi": hi,
                    "diff": full[exp][k] - full[REFERENCE][k],
                    "diff_lo": dlo,
                    "diff_hi": dhi,
                }
            )
    out = pd.DataFrame(rows)
    out.to_csv(PACKAGE / "results" / "ham10000_e2e" / "architecture_groups.csv", index=False)
    out = out.assign(
        v=out.apply(lambda r: f"{r.value:.3f} [{r.lo:.3f}, {r.hi:.3f}]", axis=1),
        d=out.apply(lambda r: f"{r['diff']:+.3f} [{r.diff_lo:+.3f}, {r.diff_hi:+.3f}]", axis=1),
    )
    print(out.pivot(index="metric", columns="model", values="v").to_string())
    print(out.pivot(index="metric", columns="model", values="d").to_string())


if __name__ == "__main__":
    main()
