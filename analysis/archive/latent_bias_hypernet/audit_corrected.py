"""補正後のモデルを、属性セルと、補正手法とは独立に学習した監査器の潜在スライスで測る。

各条件（`base` は 1 段目そのもの、ほかは latent_codes の run 名）について:
1. val のスコアをスプライン較正し、Youden しきい値を決める
2. val の残差で監査器（`audit.ResidualAuditor`、3 seed）を学習し、test を予測残差の分位（8・32）で切る。
   監査器の入力は h(x)＋RAD-DINO（`aud_he`）、RAD-DINO だけ（`aud_e`）、DINOv2 だけ（`aud_d`）。補正手法の
   振り分け係と同じ入力の監査器だけで測ると、その手法に有利になるため
3. test で、属性セル（demo8・demo32・demo96）、既知の文脈、監査器のスライスの log O/E のばらつきを測る

出力: `results/audit_corrected/<tag>.csv`

    uv run python -m analysis.archive.latent_bias_hypernet.audit_corrected --stage1 a_s42 --tag a_s42 \
        --runs gd_a_s42_global_r4_s0 ...
"""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.metrics import roc_auc_score

from analysis.archive.latent_bias_hypernet import audit
from analysis.archive.latent_bias_hypernet.data import CACHE, PACKAGE, load
from analysis.archive.latent_bias_hypernet.planting import hidden_subgroup

MIN_N, MIN_EXP = 150, 15.0


def logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def scores(run: str, V, T) -> tuple[np.ndarray, np.ndarray]:
    if run == "base":
        return V.base, T.base
    return np.load(CACHE / "assign" / f"{run}_val_score.npy"), np.load(CACHE / "assign" / f"{run}_test_score.npy")


def evaluate(run: str, V, T, inputs: dict, seeds, subgroup) -> dict:
    sv, st = scores(run, V, T)
    cal = audit.Calibrator(sv, V.target)
    pv, pt = cal(sv), cal(st)
    y = T.target
    base_tpr = float((pt >= cal.threshold)[y == 1].mean())
    row = {"run": run, "auroc": roc_auc_score(y, st)}
    fam = {name: audit.demo_cells(T.context, name) for name in ("demo8", "demo32", "demo96")}
    fam["context"] = audit.context_cells(T.context)
    patients = V.context.image.str.split("/").str[1].to_numpy()
    audited = []
    for inp, (xv, xt) in inputs.items():
        for seed in seeds:
            g = audit.ResidualAuditor(xv, logit(pv), V.target, patients, seed)(xt)
            for K in (8, 32):
                fam[f"{inp}{K}_s{seed}"] = audit.quantile_slicer(g, K)(g)
        audited += [f"{inp}{K}" for K in (8, 32)]
    for name, groups in fam.items():
        table = audit.group_metrics(groups, y, pt, cal.threshold, MIN_N, MIN_EXP)
        s = audit.summarize_family(name, table, base_tpr, groups, y, pt)
        for stat, key in (
            ("sd", "excess_sd"),
            ("max", "max_log_oe"),
            ("min", "min_log_oe"),
            ("worst_auroc", "worst_auroc"),
        ):
            row[f"{name}_{stat}"] = s[key]
    for name in audited:
        for stat in ("sd", "max", "min", "worst_auroc"):
            row[f"{name}_{stat}"] = float(np.mean([row.pop(f"{name}_s{seed}_{stat}") for seed in seeds]))
    if subgroup:
        S = hidden_subgroup(T.context, subgroup)
        row["S_log_oe"] = audit.log_oe(y[S], pt[S])
        row["S_tpr"] = float((pt >= cal.threshold)[S & (y == 1)].mean())
        row["S_auroc"] = roc_auc_score(y[S], st[S])
    return row


def auditor_inputs(stage1: str, names: list[str]) -> dict:
    """監査器の入力の組。`he` は h(x) と RAD-DINO、`e` は RAD-DINO だけ、`d` は DINOv2 だけ。"""
    out = {}
    V, T = load(stage1, "val", "raddino"), load(stage1, "test", "raddino")
    for name in names:
        if name == "he":
            xv, (ph, pe) = audit.auditor_inputs(V.features, V.route, V.features, V.route)
            xt = np.concatenate([ph.transform(T.features), pe.transform(T.route)], axis=1)
        else:
            route = "raddino" if name == "e" else "dinov2"
            ev, et = load(stage1, "val", route).route, load(stage1, "test", route).route
            pca = PCA(128, random_state=0).fit(ev)
            xv, xt = pca.transform(ev), pca.transform(et)
        out[f"aud_{name}"] = (xv, xt)
    return out


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage1", default="a_s42")
    parser.add_argument("--route", default="raddino")
    parser.add_argument("--runs", nargs="+", default=["base"])
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--subgroup", default=None)
    parser.add_argument("--auditors", nargs="+", default=["he", "e", "d"])
    parser.add_argument("--tag", required=True)
    args = parser.parse_args(argv)

    V = load(args.stage1, "val", args.route)
    T = load(args.stage1, "test", args.route)
    inputs = auditor_inputs(args.stage1, args.auditors)
    rows = []
    for run in ["base", *[r for r in args.runs if r != "base"]]:
        rows.append(evaluate(run, V, T, inputs, args.seeds, args.subgroup))
        print(run, round(rows[-1]["auroc"], 4), flush=True)
    out = PACKAGE / "results" / "audit_corrected"
    out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out / f"{args.tag}.csv", index=False)


if __name__ == "__main__":
    main()
