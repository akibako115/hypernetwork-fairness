"""潜在スライスのうち偏りの大きいものが、どの属性・文脈の画像でできているかを test で調べる。

スライスは val の残差で学習した監査器（`audit.ResidualAuditor`、入力は DINOv2 か h＋RAD-DINO）の予測残差の
分位で切る。各スライスについて、log O/E と、属性・文脈の割合を test 全体と比べて書き出す。

出力: `results/profile/<stage1>_<auditor>.csv`

    uv run python -m analysis.archive.latent_bias_hypernet.profile_slices --stage1 a_s42 --auditor d
"""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from analysis.archive.latent_bias_hypernet import audit
from analysis.archive.latent_bias_hypernet.audit_corrected import auditor_inputs, logit
from analysis.archive.latent_bias_hypernet.data import PACKAGE, load


def factors(context: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "female": context.sex == 1,
            "age<40": context.age < 40,
            "age≥80": context.age >= 80,
            "Black": context.race_group == 2,
            "Asian": context.race_group == 1,
            "Medicaid等": context.insurance_type == 1,
            "lateral": context.frontal_lateral == 1,
            "AP": context.ap_pa == 1,
            "device": context.device.astype(bool),
            "pacer": context.pacer.astype(bool),
            "followup": context.followup.astype(bool),
            "focused": context.report_type == "focused_followup",
            "affirmed_normal": context.report_type == "affirmed_normal",
        }
    ).astype(float)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage1", default="a_s42")
    parser.add_argument("--auditor", default="d", choices=("he", "e", "d"))
    parser.add_argument("--K", type=int, default=16)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)

    V, T = load(args.stage1, "val", "raddino"), load(args.stage1, "test", "raddino")
    xv, xt = auditor_inputs(args.stage1, [args.auditor])[f"aud_{args.auditor}"]
    cal = audit.Calibrator(V.base, V.target)
    pv, pt = cal(V.base), cal(T.base)
    patients = V.context.image.str.split("/").str[1].to_numpy()
    g = audit.ResidualAuditor(xv, logit(pv), V.target, patients, args.seed)(xt)
    slices = audit.quantile_slicer(g, args.K)(g)
    f = factors(T.context)
    y = T.target
    rows = []
    for k in range(args.K):
        m = slices == k
        row = {"slice": k, "n": int(m.sum()), "pos_rate": y[m].mean(), "log_oe": audit.log_oe(y[m], pt[m])}
        row["pred_residual"] = float(np.mean(g[m]))
        row.update((f.loc[m].mean() - f.mean()).add_prefix("Δ").to_dict())
        rows.append(row)
    table = pd.DataFrame(rows)
    out = PACKAGE / "results" / "profile"
    out.mkdir(parents=True, exist_ok=True)
    table.to_csv(out / f"{args.stage1}_{args.auditor}.csv", index=False)
    with pd.option_context("display.width", 250, "display.max_columns", 30):
        print("test 全体の割合:", f.mean().round(3).to_dict())
        print(table.round(2).to_string(index=False))


if __name__ == "__main__":
    main()
