"""方向 4：Multiaccuracy-Boost（Kim et al. 2019）と、[h; e] で振り分ける hypernet 補正を同じ物差しで並べる。

1 段目（凍結）の上に、1 段目が見ていない train_b（監査セット）で事後補正を学習し、test で比べる。
補正の学習は train_b の患者 80%、停止・選択は残り 20%（bgrid.py と同じ分け方、seed ごとに分け直す）。

手法:
- `none`：補正なし。単調な全体の再較正（Platt など）は順位もしきい値の位置も変えないので、これと同じ値になる
- `global`：全員共通の rank 4 の ΔW と Δb（latent_codes の K = 1）。全体の「向き」の補正
- `cells8`：年齢 × race の 8 群ごとの補正（rank 4）。評価の群そのものを補正の群にする
- `router_he_r0` / `router_he_r4`：[h; e] を受ける振り分け係（16 群）の hypernet 補正。rank 0 は切片だけ
- `ma_kim_{h,e,he}`：Kim et al. の Multiaccuracy-Boost に寄せた版。監査器は属性を使わない ridge 回帰で、
  入力は h・e・[h; e] の主成分。各周で、全体・現在の予測 p < 0.5・p ≥ 0.5 の 3 領域それぞれに監査器を当て、
  残差との相関が最大の領域だけを logit で加法補正する。step は fit 側の線探索、停止は取り分けた側の損失
- `ma_tree`：bgrid.py の Multiaccuracy 型（[h; e] の主成分上の深さ 3 の回帰木を監査器にした反復補正）

場（1 段目）:
- 実データ：`a_s42`・`a_s43`。Q1 と同じ 8 群（年齢 65 歳で 2 区分 × race）と 4 指標
- 植え込み：`a_pacersub_s42`（ペースメーカー陽性の間引き）、`a_plant_s42`（マーカー）、`a_yfdev_s42`
  （65 歳未満女性 × デバイスの陽性の間引き）。植え込んだ部分集団 S の回復と、補正量で S を当てる読み出し

読み出し（`detect`）：1 枚ごとの補正量（補正後 − 補正前の logit）で S を当てる AUROC。学び損ねた陽性（間引き）は
補正が正、ショートカット（マーカー）は負に向くので、マーカーは符号を反転する。`detect_abs` は |補正量|。
これは「どこが偏っていたかを読み出せる」という主張を、手法に依らない同じ量で比べるためのもの。
既存の「ΔW で AUROC 1.00」は全員共通の rank 1 の ΔW の値で、`global` と同じ種類の読み出しにあたる。

出力:
- `results/d4/scores/<stage1>/<method>_s<seed>.npz`（test・val のスコア）
- `results/d4/runs_<stage1>.csv`（seed ごとの指標）
- `results/d4/q1.csv`（seed 平均の logit での Q1 指標と、患者単位 bootstrap 区間、ma_kim_he との差）

    OMP_NUM_THREADS=8 uv run python -m analysis.archive.direction_poc.d4_multiaccuracy run
    OMP_NUM_THREADS=8 uv run python -m analysis.archive.direction_poc.d4_multiaccuracy q1
"""

from __future__ import annotations

import argparse
import math
from types import SimpleNamespace

import numpy as np
import pandas as pd
import rootutils
import torch
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge
from sklearn.metrics import roc_auc_score

ROOT = rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from analysis.archive.latent_bias_hypernet import audit, bgrid, latent_codes, q1_definitions  # noqa: E402
from analysis.archive.latent_bias_hypernet.data import Split, load  # noqa: E402
from analysis.archive.latent_bias_hypernet.planting import hidden_subgroup  # noqa: E402

OUT = ROOT / "analysis" / "archive" / "direction_poc" / "results" / "d4"
EMBED = {"a_plant_s42": "raddino_plant"}
SUBGROUP = {"a_pacersub_s42": "pacer", "a_yfdev_s42": "young_female_device", "a_plant_s42": "marker"}
STAGE1S = ("a_s42", "a_s43", "a_pacersub_s42", "a_plant_s42", "a_yfdev_s42")
METHODS = (
    "none",
    "global",
    "cells8",
    "router_he_r0",
    "router_he_r4",
    "ma_kim_h",
    "ma_kim_e",
    "ma_kim_he",
    "ma_tree",
)
SEEDS = (0, 1, 2)
BOOT = 500


def load_setting(stage1: str) -> dict:
    """train_b（監査セット）・val・test の Split と e(x)。route には [h; e] を入れる。"""
    embed = EMBED.get(stage1, "raddino")
    out = {}
    for split in ("train_b", "val", "test"):
        s = load(stage1, split, "h+" + embed)
        out[split] = (s, load(stage1, split, embed).route)
    return out


def corrector(method: str, fit, e_fit, hold, e_hold, evals: list, seed: int) -> list[np.ndarray]:
    """latent_codes の補正を学習し、evals それぞれのスコアを返す。"""
    assign, K, rank = {
        "global": ("global", 1, 4),
        "cells8": ("groups8", 8, 4),
        "router_he_r0": ("router", 16, 0),
        "router_he_r4": ("router", 16, 4),
    }[method]
    splits = {"fit": Split(**fit.__dict__), "val": Split(**hold.__dict__)}
    splits |= {f"eval{i}": Split(**t.__dict__) for i, (t, _) in enumerate(evals)}
    fixed = latent_codes.fixed_assignment(assign, splits, K, seed) if assign != "router" else None
    stats = {"h_mu": fit.features.mean(0), "h_sd": fit.features.std(0) + 1e-6}
    stats |= {"r_mu": fit.route.mean(0), "r_sd": fit.route.std(0) + 1e-6}
    data = {k: latent_codes.Tensors(s, stats, fixed[k] if fixed else None, "cuda") for k, s in splits.items()}
    steps = math.ceil(len(fit.target) / 2048)
    args = SimpleNamespace(
        assign=assign, K=K, rank=rank, lam=1e-3, beta=0.05, router_hidden=256, penalty_weight="sqrt",
        epochs=max(30, math.ceil(600 / steps)), batch=2048, lr=1e-3, seed=seed, device="cuda", dro_eta=0.05,
    )  # fmt: skip
    torch.manual_seed(seed)
    model, _ = latent_codes.fit_corrector(args, data, float(fit.target.mean()))
    return [latent_codes.predict(model, data[f"eval{i}"], K)[0] for i in range(len(evals))]


def weighted_loss(f: np.ndarray, y: np.ndarray, w: np.ndarray) -> float:
    p = audit.sigmoid(f)
    return float(-np.average(y * np.log(p + 1e-9) + (1 - y) * np.log(1 - p + 1e-9), weights=w))


def auditor_space(kind: str, fit, e_fit):
    """監査器の入力。h・e はそれぞれ fit 側の主成分 64 次元を標準化したもの、he は両方の連結。"""
    parts = []
    if kind in ("h", "he"):
        parts.append(("h", PCA(64, random_state=0).fit(fit.features)))
    if kind in ("e", "he"):
        parts.append(("e", PCA(64, random_state=0).fit(e_fit)))

    def x(s, e):
        cols = [p.transform(s.features if name == "h" else e) for name, p in parts]
        return np.concatenate(cols, axis=1)

    ref = x(fit, e_fit)
    mu, sd = ref.mean(0), ref.std(0) + 1e-6
    return lambda s, e: (x(s, e) - mu) / sd


def regions(f: np.ndarray) -> dict[str, np.ndarray]:
    """Kim et al. の 3 領域：全体と、現在の予測で 2 つに分けた側（class weight 付きの logit なので境は 0）。"""
    return {"all": np.ones(len(f), dtype=bool), "low": f < 0, "high": f >= 0}


def ma_kim(kind: str, fit, e_fit, hold, e_hold, evals: list, seed: int, max_rounds: int = 100):
    """Multiaccuracy-Boost。evals のスコアと、1 周目の補正量（読み出し用）を返す。

    1 周ごとに、3 領域それぞれで ridge の監査器を fit 側の class weight 付き残差 y − p に当て、
    取り分けた側で監査器と残差の相関を測る。相関が最大の領域だけを、fit 側で線探索した step で補正する。
    取り分けた側の損失が 5 周続けて下がらなければ止め、最良の周の補正を返す。
    """
    space = auditor_space(kind, fit, e_fit)
    xf, xh = space(fit, e_fit), space(hold, e_hold)
    xts = [space(t, e) for t, e in evals]
    yf, yh = fit.target, hold.target
    pos = yf.mean()
    wf = np.where(yf == 1, 0.5 / pos, 0.5 / (1 - pos))
    wh = np.where(yh == 1, 0.5 / pos, 0.5 / (1 - pos))
    ff, fh = fit.base.astype(np.float64), hold.base.astype(np.float64)
    fts = [t.base.astype(np.float64) for t, _ in evals]
    best = (weighted_loss(fh, yh, wh), [f.copy() for f in fts])
    first, stale = None, 0
    for _ in range(max_rounds):
        rf, rh = regions(ff), regions(fh)
        cand = []
        for name in rf:
            if rf[name].sum() < 200 or rh[name].sum() < 50:
                continue
            m = rf[name]
            aud = Ridge(alpha=10.0).fit(xf[m], (yf - audit.sigmoid(ff))[m], sample_weight=wf[m])
            mh = rh[name]
            corr = np.corrcoef(aud.predict(xh[mh]), (yh - audit.sigmoid(fh))[mh])[0, 1]
            cand.append((corr, name, aud))
        corr, name, aud = max(cand, key=lambda c: c[0])
        if not corr > 0:
            break
        step_f = np.where(regions(ff)[name], aud.predict(xf), 0.0)
        etas = (0.25, 0.5, 1.0, 2.0, 4.0, 8.0)
        eta = min(etas, key=lambda a: weighted_loss(ff + a * step_f, yf, wf))
        ff = ff + eta * step_f
        fh = fh + eta * np.where(regions(fh)[name], aud.predict(xh), 0.0)
        updates = [eta * np.where(regions(f)[name], aud.predict(xt), 0.0) for f, xt in zip(fts, xts, strict=True)]
        fts = [f + u for f, u in zip(fts, updates, strict=True)]
        if first is None:
            first = updates
        cur = weighted_loss(fh, yh, wh)
        if cur < best[0] - 1e-5:
            best, stale = (cur, [f.copy() for f in fts]), 0
        else:
            stale += 1
            if stale >= 5:
                break
    return best[1], first


def scores_for(method: str, S: dict, seed: int) -> dict[str, np.ndarray]:
    B, e_B = S["train_b"]
    fit, e_fit, hold, e_hold = bgrid.split_holdout(B, e_B, seed)
    evals = [S["test"], S["val"]]
    if method == "none":
        return {"test": S["test"][0].base, "val": S["val"][0].base}
    first = None
    if method.startswith("ma_kim_"):
        out, first = ma_kim(method.removeprefix("ma_kim_"), fit, e_fit, hold, e_hold, evals, seed)
    elif method == "ma_tree":
        out = bgrid.ma_tree(fit, e_fit, hold, e_hold, evals, seed)
    else:
        out = corrector(method, fit, e_fit, hold, e_hold, evals, seed)
    res = {"test": np.asarray(out[0], dtype=np.float32), "val": np.asarray(out[1], dtype=np.float32)}
    if first is not None:
        res["first_test"] = np.asarray(first[0], dtype=np.float32)
    return res


def subgroup_of(stage1: str, s: Split) -> np.ndarray | None:
    rule = SUBGROUP.get(stage1)
    if rule is None:
        return None
    return s.marked if rule == "marker" else hidden_subgroup(s.context, rule)


def plant_metrics(stage1: str, score: np.ndarray, first: np.ndarray | None, test: Split, cal: tuple) -> dict:
    """植え込んだ部分集団 S の回復と読み出し。"""
    S = subgroup_of(stage1, test)
    if S is None:
        return {}
    y = test.target
    corr = score - test.base
    sign = -1.0 if stage1 == "a_plant_s42" else 1.0
    out = {}
    if stage1 == "a_plant_s42":
        neg = y == 0
        out["shortcut"] = roc_auc_score(S[neg], score[neg])
        out["S_auroc"] = roc_auc_score(y[S], score[S])
    else:
        keep = (y == 0) | S
        out["S_pos_auroc"] = roc_auc_score(y[keep], score[keep])
        p = audit.Calibrator(*cal)(score)
        out["S_log_oe"] = audit.log_oe(y[S], p[S])
    flat = np.ptp(corr) == 0
    out["detect"] = 0.5 if flat else roc_auc_score(S, sign * corr)
    out["detect_abs"] = 0.5 if flat else roc_auc_score(S, np.abs(corr))
    # 陽性の中だけで当てる（ラベルで補正量が変わる分を除く）。マーカーは陰性の中で当てる
    within = (y == 0) if stage1 == "a_plant_s42" else (y == 1)
    out["detect_within"] = 0.5 if flat else roc_auc_score(S[within], sign * corr[within])
    if first is not None:
        out["detect_first_round"] = roc_auc_score(S, sign * first) if np.ptp(first) > 0 else 0.5
    return out


def q1_metrics(score: np.ndarray, test: Split, val_score: np.ndarray, val: Split) -> dict:
    thr = q1_definitions.youden(val_score, val.target)
    g = audit.demo_cells(test.context, "demo8")
    t = q1_definitions.group_rows(g, test.target, score, thr, False)
    return {"auroc": roc_auc_score(test.target, score)} | q1_definitions.summarize(t)


def run(stage1s, methods, seeds) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    for stage1 in stage1s:
        # 1 段目ごとに別の CSV にして、1 段目の単位で並列に回せるようにする
        path = OUT / f"runs_{stage1}.csv"
        rows = pd.read_csv(path).to_dict("records") if path.exists() else []
        have = {(r["stage1"], r["method"], int(r["seed"])) for r in rows}
        S = load_setting(stage1)
        test, val = S["test"][0], S["val"][0]
        sdir = OUT / "scores" / stage1
        sdir.mkdir(parents=True, exist_ok=True)
        for method in methods:
            for seed in seeds:
                if (stage1, method, seed) in have:
                    continue
                res = scores_for(method, S, seed)
                np.savez(sdir / f"{method}_s{seed}.npz", **res)
                row = {"stage1": stage1, "method": method, "seed": seed}
                row |= q1_metrics(res["test"], test, res["val"], val)
                row |= plant_metrics(stage1, res["test"], res.get("first_test"), test, (res["val"], val.target))
                rows.append(row)
                print({k: (round(v, 3) if isinstance(v, float) else v) for k, v in row.items()}, flush=True)
                pd.DataFrame(rows).to_csv(path, index=False)
                if method == "none":
                    break


def mean_scores(stage1: str, method: str, split: str) -> np.ndarray:
    seeds = (0,) if method == "none" else SEEDS
    return np.mean([np.load(OUT / "scores" / stage1 / f"{method}_s{s}.npz")[split] for s in seeds], axis=0)


def q1_table(stage1s, methods, base: str = "ma_kim_he") -> None:
    """seed 平均の logit で Q1 の 4 指標＋全体 AUROC を出し、患者単位の対応のある bootstrap で区間をつける。"""
    rows = []
    for stage1 in stage1s:
        S = load_setting(stage1)
        test, val = S["test"][0], S["val"][0]
        y, g = test.target, audit.demo_cells(test.context, "demo8")
        units = test.context.image.str.split("/").str[1].to_numpy()
        draws = q1_definitions.bootstrap_index(units, np.random.default_rng(0))[:BOOT]
        Sg = subgroup_of(stage1, test)
        boot = {}
        for method in methods:
            score, vscore = mean_scores(stage1, method, "test"), mean_scores(stage1, method, "val")
            thr = q1_definitions.youden(vscore, val.target)

            def stat(i, score=score, thr=thr):
                t = q1_definitions.group_rows(g[i], y[i], score[i], thr, False)
                out = {"auroc": roc_auc_score(y[i], score[i])} | q1_definitions.summarize(t)
                if Sg is not None and stage1 != "a_plant_s42":
                    keep = (y[i] == 0) | Sg[i]
                    out["S_pos_auroc"] = roc_auc_score(y[i][keep], score[i][keep])
                elif Sg is not None:
                    out["shortcut"] = roc_auc_score(Sg[i][y[i] == 0], score[i][y[i] == 0])
                return out

            point = stat(np.arange(len(y)))
            boot[method] = pd.DataFrame([stat(i) for i in draws])
            row = {"stage1": stage1, "method": method} | point
            for k in point:
                lo, hi = boot[method][k].quantile([0.025, 0.975])
                row |= {f"{k}_lo": lo, f"{k}_hi": hi}
            rows.append(row)
            print(stage1, method, {k: round(v, 3) for k, v in point.items()}, flush=True)
        for row in rows:
            if row["stage1"] != stage1:
                continue
            for k in boot[base].columns:
                d = boot[row["method"]][k] - boot[base][k]
                row |= {f"{k}_vs_{base}": row[k] - point_of(rows, stage1, base, k)}
                row |= {f"{k}_vs_{base}_lo": d.quantile(0.025), f"{k}_vs_{base}_hi": d.quantile(0.975)}
        pd.DataFrame(rows).to_csv(OUT / "q1.csv", index=False)


def point_of(rows, stage1, method, k):
    return next(r[k] for r in rows if r["stage1"] == stage1 and r["method"] == method)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("step", choices=("run", "q1"))
    parser.add_argument("--stage1", nargs="+", default=list(STAGE1S))
    parser.add_argument("--methods", nargs="+", default=list(METHODS))
    parser.add_argument("--seeds", type=int, nargs="+", default=list(SEEDS))
    args = parser.parse_args(argv)
    if args.step == "run":
        run(args.stage1, args.methods, args.seeds)
    else:
        q1_table(args.stage1, args.methods)


if __name__ == "__main__":
    main()
