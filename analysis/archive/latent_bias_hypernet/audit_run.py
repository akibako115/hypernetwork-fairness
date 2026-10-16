"""1 段目（と、あれば hypernet の補正）を、属性セルと潜在スライスの両方で監査する。

スライスは発見用 split（既定 train_b、1 段目が見ていない）で決め、test で測る。結果は
`results/audit/<stage1>/` に、族ごとの要約（`families.csv`）、群ごとの表（`groups_<family>.csv`）、
属性で補正した後にも残る偏り（`families_demo_adjusted.csv`）、植え込みの答え合わせ（`planted.csv`）を書く。

    uv run python -m analysis.archive.latent_bias_hypernet.audit_run --stage1 a_s42 \
        --field gd_a_s42_router_hraddino_r4_s0 gd_a_s42_router_hraddino_r4_s1 gd_a_s42_router_hraddino_r4_s2
"""

from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd

from analysis.archive.latent_bias_hypernet import audit
from analysis.archive.latent_bias_hypernet.data import CACHE, PACKAGE, load
from analysis.archive.latent_bias_hypernet.planting import hidden_subgroup

MIN_N, MIN_EXP = 150, 15.0
DEMO_FAMILIES = ("demo8", "demo32", "demo96")


def logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def demo_adjust(cells_d, y_d, p_d, cells_t, p_t, k: float = 20.0) -> np.ndarray:
    """発見用 split で属性セルごとの切片（縮約付き log O/E）を求め、test の予測に足す。

    属性の掛け合わせで直せる偏りを取り除いた後に、潜在スライスにまだ偏りが残るかを見るため。
    """
    t = pd.DataFrame({"c": cells_d, "y": y_d, "p": p_d}).groupby("c").sum()
    # 発見用 split 全体の O/E を引き、セル間の相対的なずれだけを足す（較正用 split とのずれを持ち込まない）
    overall = np.log(y_d.sum() / p_d.sum())
    offset = (np.log((t.y + k * y_d.mean() / p_d.mean()) / (t.p + k)) - overall).to_dict()
    delta = np.array([offset.get(c, 0.0) for c in cells_t])
    return audit.sigmoid(logit(p_t) + delta)


def slice_families(args, D, T, p_d, residual_d) -> dict[str, np.ndarray]:
    """test の各行に、族ごとの群番号を振る。決めるのに使うのは発見用 split（と test の x）だけ。"""
    fam = {name: audit.demo_cells(T.context, name) for name in DEMO_FAMILIES}
    fam["context"] = audit.context_cells(T.context)
    rng = np.random.default_rng(0)
    fam["random32"] = rng.integers(0, 32, len(T.target))
    fam["score32"] = audit.quantile_slicer(D.base, 32)(T.base)
    for K in args.K:
        fam[f"gmm{K}"] = audit.GMMSlicer(D.route, K)(T.route)
        min_leaf = max(MIN_N, len(D.target) // (4 * K))
        fam[f"tree{K}"] = audit.TreeSlicer(D.route, residual_d, K, min_leaf)(T.route)
    for name in args.field:
        corr = np.load(CACHE / "assign" / f"{name}_test_score.npy") - T.base
        for K in args.K:
            fam[f"field{K}:{name}"] = audit.quantile_slicer(corr, K)(corr)
        fam[f"router:{name}"] = np.load(CACHE / "assign" / f"{name}_test.npy")
    return fam


def audit_families(fam, T, p_t, threshold, base_tpr, out, suffix=""):
    rows, tables = [], {}
    y = T.target
    cells32 = audit.demo_cells(T.context, "demo32")
    # スコア帯（test の 1 段目スコアの 16 分位）。この内側でも偏りが残れば、スコアの高さでは説明できない
    score_bins = audit.quantile_slicer(T.base, 16)(T.base)
    for name, g in fam.items():
        table = audit.group_metrics(g, y, p_t, threshold, MIN_N, MIN_EXP)
        tables[name] = table
        row = audit.summarize_family(name, table, base_tpr, g, y, p_t)
        row["residual_r2"] = audit.residual_explained(g, y - p_t)
        if not name.startswith("demo"):
            w = audit.within_cell(cells32, g, y, p_t, 50, 5.0)
            row["within_demo32_excess_sd"] = w["within_excess_sd"]
            row["within_pairs"] = w["within_pairs"]
            row["within_score_excess_sd"] = audit.within_cell(score_bins, g, y, p_t, 50, 5.0)["within_excess_sd"]
        rows.append(row)
        table.to_csv(out / f"groups{suffix}_{name.replace(':', '__')}.csv", index=False)
    return pd.DataFrame(rows), tables


def planted_report(rule: str, fam, T, p_t, threshold) -> pd.DataFrame:
    """植え込んだ部分集団 S を、各族の群がどれだけ切り出せるか。"""
    S = hidden_subgroup(T.context, rule)
    y = T.target
    rows = []
    pred = p_t >= threshold
    rows.append(
        {
            "family": "oracle_S",
            "share_S": 1.0,
            "recall_S": 1.0,
            "n": int(S.sum()),
            "log_oe": audit.log_oe(y[S], p_t[S]),
            "tpr": float(pred[S & (y == 1)].mean()),
        }
    )
    for name, g in fam.items():
        best = None
        for k in np.unique(g):
            m = g == k
            if m.sum() < MIN_N or p_t[m].sum() < MIN_EXP:
                continue
            share = S[m].mean()
            if best is None or share > best["share_S"]:
                best = {
                    "family": name,
                    "share_S": float(share),
                    "recall_S": float((m & S).sum() / S.sum()),
                    "n": int(m.sum()),
                    "log_oe": audit.log_oe(y[m], p_t[m]),
                    "tpr": float(pred[m & (y == 1)].mean()),
                }
        if best:
            rows.append(best)
    return pd.DataFrame(rows)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage1", default="a_s42")
    parser.add_argument("--discover", default="train_b")
    parser.add_argument("--route", default="raddino")
    parser.add_argument("--K", type=int, nargs="+", default=[8, 32])
    parser.add_argument("--field", nargs="*", default=[], help="latent_codes の run 名（test の補正量を読む）")
    parser.add_argument("--subgroup", default=None, help="planting.HIDDEN_SUBGROUPS の規則（植え込みの答え合わせ）")
    parser.add_argument("--calibrate-on", default="val", help="全体較正としきい値を決める split（test と同じ母集団）")
    parser.add_argument("--calib", choices=("spline", "isotonic", "platt"), default="spline")
    parser.add_argument("--tag", default="")
    args = parser.parse_args(argv)

    out = PACKAGE / "results" / "audit" / (args.stage1 + (f"_{args.tag}" if args.tag else ""))
    out.mkdir(parents=True, exist_ok=True)
    D = load(args.stage1, args.discover, args.route)
    T = load(args.stage1, "test", args.route)
    C = load(args.stage1, args.calibrate_on) if args.calibrate_on != args.discover else D
    cal = audit.Calibrator(C.base, C.target, args.calib)
    p_d, p_t = cal(D.base), cal(T.base)
    pred = p_t >= cal.threshold
    base_tpr = float(pred[T.target == 1].mean())
    fam = slice_families(args, D, T, p_d, D.target - p_d)

    summary, _ = audit_families(fam, T, p_t, cal.threshold, base_tpr, out)
    summary.to_csv(out / "families.csv", index=False)

    # 属性セル（demo96）の切片で直した後の予測で、もう一度監査する
    p_adj = demo_adjust(
        audit.demo_cells(D.context, "demo96"), D.target, p_d, audit.demo_cells(T.context, "demo96"), p_t
    )
    adj_tpr = float((p_adj >= cal.threshold)[T.target == 1].mean())
    adjusted, _ = audit_families(fam, T, p_adj, cal.threshold, adj_tpr, out, "_adj")
    adjusted.to_csv(out / "families_demo_adjusted.csv", index=False)

    if args.subgroup:
        planted_report(args.subgroup, fam, T, p_t, cal.threshold).to_csv(out / "planted.csv", index=False)
    (out / "meta.json").write_text(
        json.dumps({"args": vars(args), "threshold": cal.threshold, "base_tpr": base_tpr}, indent=1, ensure_ascii=False)
    )
    cols = ["family", "groups", "excess_sd", "above_null", "max_log_oe", "max_log_oe_n", "worst_tpr_gap", "residual_r2"]
    cols += ["within_demo32_excess_sd", "within_score_excess_sd"]
    with pd.option_context("display.width", 200, "display.max_columns", 20):
        print(summary[cols].round(3).to_string(index=False))
        print("--- demo96 で補正した後")
        print(adjusted[cols].round(3).to_string(index=False))


if __name__ == "__main__":
    main()
