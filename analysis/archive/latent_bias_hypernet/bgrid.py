"""監査用データ（B）の偏りの量と大きさを変えて、補正と発見がどこまでできるかを比べる。

植え込み実験では B が偏りを含まない大きなデータだったため、補正できて当然という面があった。ここでは
B にも A と同じ偏りを一部・全部残し、B の大きさも 1,000 枚から全量まで変える。

- ペースメーカー（`a_pacersub_s42`）：B のペースメーカー陽性を残す割合を 1 − 0.9·bias にする（bias = 1 で A と同じ 10%）
- マーカー（`a_plant_s42`）：B の各画像は、確率 bias で A と同じ相関の規則（陽性 50%・陰性 5%）、
  それ以外は展開先の規則（ラベルと独立に 20%）で印を付ける。印あり・なしの表現は `probes/export_marked.py` と
  `probes/embed.py --mark-all` で書き出した 2 版から画像ごとに選ぶ

補正（凍結した 1 段目の上、B の患者 80% で学習し、残り 20% で選ぶ）:
- `global`・`gmm16`・`router_meta`・`router_he`：latent_codes.py の補正（rank 4、K = 16）
- `lhcf_cluster`：GMM 16 クラスタごとに補正を持たせ、同じクラスタの GroupDRO（η = 0.05）で学習する（LHCF 型）
- `llr`：最終層の再学習（h(x) のロジスティック回帰、LaBonte et al. 2023 の設定に近い）
- `dfr_demo`：年齢 × race × クラスで重みを均等にした最終層の再学習（DFR、Kirichenko et al. 2023 の群に属性を使う版）
- `ma_tree`：Multiaccuracy 型（Kim et al. 2019）。[h; e] の主成分上の浅い回帰木を監査器にして、残差を反復して補正する

評価（test）:
- 全体 AUROC
- ペースメーカー：S の陽性を全陰性と比べた AUROC（`S_pos_auroc`）、S の log O/E（val で較正）
- マーカー：陰性の中で印あり・なしをスコアで見分ける AUROC（`shortcut`、0.5 が理想）、印あり群の AUROC
- 発見：S（ペースメーカー・印あり）を補正量で当てる AUROC（`detect`。マーカーは補正量の符号を反転）

出力: `results/bgrid/<plant>.csv`

    uv run python -m analysis.archive.latent_bias_hypernet.bgrid --plant pacer
"""

from __future__ import annotations

import argparse
import itertools
import math
from types import SimpleNamespace

import numpy as np
import pandas as pd
import torch
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.tree import DecisionTreeRegressor

from analysis.archive.latent_bias_hypernet import audit, latent_codes
from analysis.archive.latent_bias_hypernet.data import CACHE, PACKAGE, Split, load, metadata_onehot, subset
from analysis.archive.latent_bias_hypernet.planting import CORRELATED, INDEPENDENT, _uniform

STAGE1 = {"pacer": "a_pacersub_s42", "marker": "a_plant_s42"}
TEST_EMBED = {"pacer": "raddino", "marker": "raddino_plant"}
METHODS = ("none", "global", "gmm16", "router_meta", "router_he", "llr", "dfr_demo", "ma_tree", "lhcf_cluster")


def patients(s: Split) -> np.ndarray:
    return s.context.image.str.split("/").str[1].to_numpy()


def he_route(h: np.ndarray, e: np.ndarray) -> np.ndarray:
    """data.load の `h+<embed>` と同じ連結（それぞれ全体の標準偏差で割る）。"""
    return np.concatenate([h / (h.std() + 1e-6), e / (e.std() + 1e-6)], axis=1)


class Pool:
    """B の全量と、評価用の val・test。route には [h; e] を持たせ、e(x) は別に持つ。"""

    def __init__(self, plant: str):
        self.plant = plant
        stage1 = STAGE1[plant]
        self.val = load(stage1, "val", "h+" + TEST_EMBED[plant])
        self.test = load(stage1, "test", "h+" + TEST_EMBED[plant])
        self.e_test = load(stage1, "test", TEST_EMBED[plant]).route
        self.e_val = load(stage1, "val", TEST_EMBED[plant]).route
        if plant == "pacer":
            self.B = load(stage1, "train_b", "h+raddino")
            self.e_B = load(stage1, "train_b", "raddino").route
        else:
            context = pd.read_csv(CACHE / "context" / "train_b.csv")
            parts = {}
            for tag in ("all", "none"):
                npz = np.load(CACHE / "stage1" / stage1 / f"train_b_mark{tag}.npz")
                if not np.array_equal(npz["image"], context.image.to_numpy(dtype=str)):
                    raise ValueError("train_b の行順が一致しない")
                parts[tag] = npz
            self.marked_parts = parts
            self.e_parts = {
                "all": np.load(CACHE / "embed" / "raddino_markall" / "train_b.npz")["embedding"].astype(np.float32),
                "none": np.load(CACHE / "embed" / "raddino" / "train_b.npz")["embedding"].astype(np.float32),
            }
            self.context_B = context

    def biased_B(self, bias: float, seed: int) -> tuple[Split, np.ndarray]:
        """偏りの量 bias の B 全量と、その e(x) を返す。"""
        rng = np.random.default_rng(1000 + seed)
        if self.plant == "pacer":
            B = self.B
            hit = B.context.pacer.to_numpy(dtype=bool) & (B.target == 1)
            keep = ~(hit & (rng.random(len(B.target)) >= 1 - 0.9 * bias))
            return subset(B, keep), self.e_B[keep]
        c = self.context_B
        y = c.target.to_numpy()
        u = c.image.map(_uniform).to_numpy()
        correlated = rng.random(len(y)) < bias
        rate = np.where(correlated, np.where(y == 1, CORRELATED[1], CORRELATED[0]), INDEPENDENT)
        marked = u < rate
        a, n = self.marked_parts["all"], self.marked_parts["none"]
        feats = np.where(marked[:, None], a["features"], n["features"]).astype(np.float32)
        logits = np.where(marked[:, None], a["logits"], n["logits"])
        e = np.where(marked[:, None], self.e_parts["all"], self.e_parts["none"])
        B = Split(
            features=feats,
            base=(logits[:, 1] - logits[:, 0]).astype(np.float32),
            target=y.astype(np.int64),
            route=he_route(feats, e),
            context=c,
            marked=marked,
        )
        return B, e


def take(B: Split, e: np.ndarray, size: int | None, seed: int) -> tuple[Split, np.ndarray]:
    """患者単位で size 枚以上になるまで患者を取る。"""
    if size is None:
        return B, e
    pat = patients(B)
    order = np.random.default_rng(2000 + seed).permutation(np.unique(pat))
    counts = pd.Series(pat).value_counts()
    chosen = order[: int(np.searchsorted(np.cumsum(counts[order].to_numpy()), size)) + 1]
    keep = np.isin(pat, chosen)
    return subset(B, keep), e[keep]


def split_holdout(B: Split, e: np.ndarray, seed: int):
    pat = patients(B)
    uniq = np.unique(pat)
    held = np.isin(pat, np.random.default_rng(3000 + seed).permutation(uniq)[: max(1, len(uniq) // 5)])
    return subset(B, ~held), e[~held], subset(B, held), e[held]


def corrector(method: str, fit: Split, e_fit, hold: Split, e_hold, evals: list, seed: int) -> list[np.ndarray]:
    """latent_codes の補正を学習し、evals（(Split, e) の並び）それぞれのスコアを返す。"""
    assign = {
        "global": "global",
        "gmm16": "gmm",
        "lhcf_cluster": "gmm",
        "router_meta": "router",
        "router_he": "router",
    }[method]
    K = 1 if assign == "global" else 16
    fit, hold = Split(**fit.__dict__), Split(**hold.__dict__)
    targets = [Split(**t.__dict__) for t, _ in evals]
    if method == "router_meta":
        for s in (fit, hold, *targets):
            s.route = metadata_onehot(s.context)
    elif method in ("gmm16", "lhcf_cluster"):
        fit.route, hold.route = e_fit, e_hold
        for s, (_, e) in zip(targets, evals, strict=True):
            s.route = e
    splits = {"fit": fit, "val": hold, **{f"eval{i}": t for i, t in enumerate(targets)}}
    fixed = latent_codes.fixed_assignment(assign, splits, K, seed) if assign != "router" else None
    stats = {"h_mu": fit.features.mean(0), "h_sd": fit.features.std(0) + 1e-6}
    stats.update(r_mu=fit.route.mean(0), r_sd=fit.route.std(0) + 1e-6)
    data = {k: latent_codes.Tensors(s, stats, fixed[k] if fixed else None, "cuda") for k, s in splits.items()}
    if method == "lhcf_cluster":
        # LHCF 型の公平な版：クラスタごとに別の補正を持たせ、同じクラスタの GroupDRO で学習する
        data["fit"].dro_group = torch.as_tensor(fixed["fit"], dtype=torch.long, device="cuda")
    steps = math.ceil(len(fit.target) / 2048)
    args = SimpleNamespace(
        assign=assign,
        K=K,
        rank=4,
        lam=1e-3,
        beta=0.05,
        router_hidden=256,
        penalty_weight="sqrt",
        epochs=max(30, math.ceil(600 / steps)),
        batch=2048,
        lr=1e-3,
        seed=seed,
        device="cuda",
        dro_eta=0.05,
    )
    torch.manual_seed(seed)
    model, _ = latent_codes.fit_corrector(args, data, float(fit.target.mean()))
    return [latent_codes.predict(model, data[f"eval{i}"], K)[0] for i in range(len(targets))]


def demo8(s: Split) -> np.ndarray:
    return audit.demo_cells(s.context, "demo8")


def last_layer(method: str, fit: Split, hold: Split, evals: list, groups=demo8) -> list[np.ndarray]:
    """最終層の再学習。C は取り分けた側の損失で選ぶ。`dfr_demo` は属性群（既定は年齢 × race）× クラスを等重みにする。"""
    mu, sd = fit.features.mean(0), fit.features.std(0) + 1e-6

    def z(s):
        return (s.features - mu) / sd

    def weights(s):
        y = s.target
        if method == "llr":
            return np.where(y == 1, 0.5 / y.mean(), 0.5 / (1 - y.mean()))
        cell = groups(s) * 2 + y
        counts = pd.Series(cell).value_counts()
        return (len(y) / (len(counts) * counts[cell])).to_numpy()

    best = None
    for C in (0.003, 0.03, 0.3):
        lr = LogisticRegression(C=C, max_iter=500).fit(z(fit), fit.target, sample_weight=weights(fit))
        p = lr.predict_proba(z(hold))[:, 1]
        loss = -np.average(
            hold.target * np.log(p + 1e-9) + (1 - hold.target) * np.log(1 - p + 1e-9), weights=weights(hold)
        )
        if best is None or loss < best[0]:
            best = (loss, lr)
    return [best[1].decision_function(z(t)) for t, _ in evals]


def ma_tree(fit: Split, e_fit, hold: Split, e_hold, evals: list, seed: int) -> list[np.ndarray]:
    """Multiaccuracy 型：浅い回帰木の監査器で残差を探し、logit を反復して補正する（取り分けた側で停止）。"""
    ph = PCA(32, random_state=0).fit(fit.features)
    pe = PCA(32, random_state=0).fit(e_fit)

    def x(s, e):
        return np.concatenate([ph.transform(s.features), pe.transform(e)], axis=1)

    xf, xh = x(fit, e_fit), x(hold, e_hold)
    xts = [x(t, e) for t, e in evals]
    yf, yh = fit.target, hold.target
    pos = yf.mean()
    wf = np.where(yf == 1, 0.5 / pos, 0.5 / (1 - pos))
    wh = np.where(yh == 1, 0.5 / pos, 0.5 / (1 - pos))
    ff, fh = fit.base.astype(np.float64), hold.base.astype(np.float64)
    fts = [t.base.astype(np.float64) for t, _ in evals]

    def loss(f, y, w):
        p = audit.sigmoid(f)
        return -np.average(y * np.log(p + 1e-9) + (1 - y) * np.log(1 - p + 1e-9), weights=w)

    best, best_t, history = loss(fh, yh, wh), [f.copy() for f in fts], 0
    min_leaf = max(20, len(yf) // 50)
    for t in range(200):
        p = audit.sigmoid(ff)
        tree = DecisionTreeRegressor(max_depth=3, min_samples_leaf=min_leaf, random_state=seed + t)
        tree.fit(xf, yf - p, sample_weight=wf)
        leaf = tree.apply(xf)
        num = pd.Series(wf * (yf - p)).groupby(leaf).sum()
        den = pd.Series(wf * p * (1 - p)).groupby(leaf).sum() + 1e-6
        value = (num / den).clip(-2, 2)
        step = 0.3
        ff = ff + step * pd.Series(leaf).map(value).to_numpy()
        fh = fh + step * pd.Series(tree.apply(xh)).map(value).fillna(0).to_numpy()
        fts = [
            f + step * pd.Series(tree.apply(xt)).map(value).fillna(0).to_numpy() for f, xt in zip(fts, xts, strict=True)
        ]
        cur = loss(fh, yh, wh)
        if cur < best - 1e-5:
            best, best_t, history = cur, [f.copy() for f in fts], 0
        else:
            history += 1
            if history >= 10:
                break
    return best_t


def metrics(plant: str, score: np.ndarray, test: Split, cal_val: tuple) -> dict:
    y = test.target
    corr = score - test.base
    out = {"auroc": roc_auc_score(y, score)}
    if plant == "pacer":
        S = test.context.pacer.to_numpy(dtype=bool)
        keep = (y == 0) | S
        out["S_pos_auroc"] = roc_auc_score(y[keep], score[keep])
        sv, yv = cal_val
        p = audit.Calibrator(sv, yv)(score)
        out["S_log_oe"] = audit.log_oe(y[S], p[S])
        out["detect"] = roc_auc_score(S, corr) if np.ptp(corr) > 0 else 0.5
    else:
        M = test.marked
        neg = y == 0
        out["shortcut"] = roc_auc_score(M[neg], score[neg])
        out["auroc_marked"] = roc_auc_score(y[M], score[M])
        out["detect"] = roc_auc_score(M, -corr) if np.ptp(corr) > 0 else 0.5
    return out


def scores_for(method, pool, fit, e_fit, hold, e_hold, seed) -> tuple[np.ndarray, np.ndarray]:
    """1 回の学習で test と val（S の log O/E の較正用）のスコアを返す。"""
    evals = [(pool.test, pool.e_test), (pool.val, pool.e_val)]
    if method == "none":
        return pool.test.base, pool.val.base
    if method in ("llr", "dfr_demo"):
        out = last_layer(method, fit, hold, evals)
    elif method == "ma_tree":
        out = ma_tree(fit, e_fit, hold, e_hold, evals, seed)
    else:
        out = corrector(method, fit, e_fit, hold, e_hold, evals, seed)
    return out[0], out[1]


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--plant", choices=tuple(STAGE1), required=True)
    parser.add_argument("--biases", type=float, nargs="+", default=[0.0, 0.5, 1.0])
    parser.add_argument("--sizes", nargs="+", default=["1000", "5000", "20000", "all"])
    parser.add_argument("--methods", nargs="+", default=list(METHODS))
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--tag", default="")
    args = parser.parse_args(argv)

    pool = Pool(args.plant)
    out = PACKAGE / "results" / "bgrid"
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"{args.plant}{('_' + args.tag) if args.tag else ''}.csv"
    done = pd.read_csv(path) if path.exists() else pd.DataFrame(columns=["bias", "size", "method", "seed"])
    rows = done.to_dict("records")
    have = {(r["bias"], str(r["size"]), r["method"], int(r["seed"])) for r in rows}
    for bias, size, seed in itertools.product(args.biases, args.sizes, args.seeds):
        full, e_full = pool.biased_B(bias, seed)
        B, e = take(full, e_full, None if size == "all" else int(size), seed)
        fit, e_fit, hold, e_hold = split_holdout(B, e, seed)
        for method in args.methods:
            if (bias, size, method, seed) in have:
                continue
            score, score_val = scores_for(method, pool, fit, e_fit, hold, e_hold, seed)
            cal = (score_val, pool.val.target) if args.plant == "pacer" else None
            row = {"bias": bias, "size": size, "method": method, "seed": seed, "n_B": len(B.target)}
            row.update(metrics(args.plant, score, pool.test, cal))
            rows.append(row)
            print({k: (round(v, 3) if isinstance(v, float) else v) for k, v in row.items()}, flush=True)
            pd.DataFrame(rows).to_csv(path, index=False)


if __name__ == "__main__":
    main()
