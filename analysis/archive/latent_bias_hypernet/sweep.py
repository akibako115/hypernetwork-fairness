"""latent_codes.py の条件をまとめて逐次実行する。既に summary.json がある条件は飛ばす。

uv run python -m analysis.archive.latent_bias_hypernet.sweep crossfit --stage1 a_s42
uv run python -m analysis.archive.latent_bias_hypernet.sweep improve --stage1 a_s42
"""

from __future__ import annotations

import argparse
import itertools

from analysis.archive.latent_bias_hypernet import latent_codes
from analysis.archive.latent_bias_hypernet.data import PACKAGE


def crossfit(stage1: str, seeds: list[int]) -> list[list[str]]:
    """Q1：同じ補正を、1 段目が学習した split（in-sample）と見ていない split（cross-fit）で学習して比べる。"""
    fit_splits = {"a": "train_a", "b": "train_b"}
    conditions = [
        ("global", "none", 1),
        ("groups8", "none", 8),
        ("router", "meta", 16),
        ("router", "raddino", 16),
    ]
    runs = []
    for (tag, fit), (assign, route, K), rank, seed in itertools.product(fit_splits.items(), conditions, (0, 4), seeds):
        name = f"cf_{stage1}_fit{tag}_{assign}_{route}_K{K}_r{rank}_s{seed}"
        runs.append(
            [
                *("--name", name, "--stage1", stage1, "--fit", fit, "--assign", assign, "--route", route),
                *("--K", str(K), "--rank", str(rank), "--seed", str(seed)),
            ]
        )
    return runs


def discovery(stage1: str, seeds: list[int], routes: list[str], lams: list[float]) -> list[list[str]]:
    """Q2：振り分け係と k-means で、渡していない要因をどれだけ回収するか。fit は cross-fit 側に固定する。"""
    runs = []
    for route, lam, seed in itertools.product(routes, lams, seeds):
        for assign in ("router", "kmeans"):
            if assign == "kmeans" and route == "meta":
                continue
            name = f"dv_{stage1}_{assign}_{route}_K16_r4_lam{lam:g}_s{seed}"
            runs.append(
                [
                    *("--name", name, "--stage1", stage1, "--fit", "train_b", "--assign", assign, "--route", route),
                    *("--K", "16", "--rank", "4", "--lam", str(lam), "--seed", str(seed)),
                ]
            )
    return runs


# 改善の比較：（補正の群, 振り分けの入力, K, DRO の群）
IMPROVE = {
    "global": ("global", "raddino", 1, "none"),
    "demo32": ("demo32", "none", 32, "none"),
    "dro_demo32": ("global", "none", 1, "demo32"),
    "lhcf": ("global", "raddino", 1, "gmm"),
    "gmm32": ("gmm", "raddino", 32, "none"),
    "router_meta": ("router", "meta", 16, "none"),
    "router_he": ("router", "h+raddino", 16, "none"),
}


def improve(stage1: str, seeds: list[int], conditions: list[str], dro_eta: float) -> list[list[str]]:
    """Q3：属性セルの補正・GroupDRO、LHCF 型（GMM クラスタの GroupDRO）、振り分け係を同じ条件で比べる。"""
    runs = []
    for cond, seed in itertools.product(conditions, seeds):
        assign, route, K, dro = IMPROVE[cond]
        name = f"im_{stage1}_{cond}_s{seed}"
        argv = [*("--name", name, "--stage1", stage1, "--fit", "train_b", "--assign", assign, "--route", route)]
        argv += [*("--K", str(K), "--rank", "4", "--seed", str(seed), "--dro", dro, "--dro-eta", str(dro_eta))]
        runs.append(argv)
    return runs


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("plan", choices=("crossfit", "discovery", "improve"))
    parser.add_argument("--stage1", default="a_s42")
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--routes", nargs="+", default=["raddino", "meta"])
    parser.add_argument("--lams", type=float, nargs="+", default=[1e-3])
    parser.add_argument("--conditions", nargs="+", default=list(IMPROVE))
    parser.add_argument("--dro-eta", type=float, default=0.05)
    args = parser.parse_args()
    if args.plan == "crossfit":
        runs = crossfit(args.stage1, args.seeds)
    elif args.plan == "improve":
        runs = improve(args.stage1, args.seeds, args.conditions, args.dro_eta)
    else:
        runs = discovery(args.stage1, args.seeds, args.routes, args.lams)
    for argv in runs:
        name = argv[1]
        if (PACKAGE / "results" / "latent_codes" / name / "summary.json").exists():
            continue
        print("run", name, flush=True)
        latent_codes.main(argv)


if __name__ == "__main__":
    main()
