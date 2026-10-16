"""固定した 1 段目のモデルに、グループ別の補正（ΔW_k, Δb_k）を hypernet で足して学習し、グループと補正を読む。

    補正後の logit = base(x) + ΔW_k · h(x) + Δb_k,   k = グループ（振り分け係が決める、または既知の群）

- base(x)・h(x)：1 段目（固定）の logit 差と、分類 head 直前の表現（fit split の平均・標準偏差で標準化）
- 補正係：コードブック c_k から低ランクの ΔW_k = U·v(c_k) と Δb_k を作る（rank 0 なら切片だけ）
- 振り分け係：e(x) を受ける MLP。Gumbel-softmax の straight-through で 1 つのグループを選ぶ
- 損失：class weight 付き BCE ＋ λ Σ_k √p̄_k ‖(ΔW_k, Δb_k)‖ − β H(p̄)

`--assign` で、グループの決め方を振り分け係（`router`）、既知の 8 群（`groups8`）、属性 32 セル（`demo32`）、
e(x) の k-means（`kmeans`）・GMM（`gmm`）、全員 1 群（`global`）から選ぶ。`--dro` を付けると損失を
GroupDRO にする（群は属性 32 セル、または LHCF と同じ e(x) の GMM クラスタ）。
val の損失で epoch を選び、test で評価する。

出力: `results/latent_codes/<name>/{summary.json,codes.csv}` と `cache/assign/<name>_{val,test}.npy`

    uv run python analysis/archive/latent_bias_hypernet/latent_codes.py --name demo --stage1 a_s42 --fit train_b \
        --assign router --route raddino --K 16 --rank 4 --lam 1e-3
"""

from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd
import torch
from sklearn.cluster import MiniBatchKMeans
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_mutual_info_score, roc_auc_score
from sklearn.mixture import GaussianMixture
from torch import nn

from analysis.archive.latent_bias_hypernet.audit import demo_cells
from analysis.archive.latent_bias_hypernet.data import GROUP_NAMES, PACKAGE, Split, holdout_by_patient, load

FACTORS = ("pacer", "focused", "followup", "device", "old", "sex", "lateral", "ap", "nonwhite", "marked")


class Corrector(nn.Module):
    """コードブックから各グループの補正を生成する hypernet と、振り分け係。"""

    def __init__(self, feat_dim: int, route_dim: int, K: int, rank: int, code_dim: int = 16, hidden: int = 256):
        super().__init__()
        self.rank = rank
        if not route_dim:
            self.router = None
        elif hidden:
            self.router = nn.Sequential(nn.Linear(route_dim, hidden), nn.ReLU(), nn.Linear(hidden, K))
        else:
            self.router = nn.Linear(route_dim, K)
        self.codebook = nn.Parameter(torch.randn(K, code_dim))
        self.to_bias = nn.Linear(code_dim, 1)
        nn.init.zeros_(self.to_bias.weight)
        nn.init.zeros_(self.to_bias.bias)
        if rank:
            self.to_v = nn.Linear(code_dim, rank)
            self.U = nn.Parameter(torch.randn(feat_dim, rank) * 0.01)

    def deltas(self) -> tuple[torch.Tensor, torch.Tensor]:
        """[K, F] の ΔW と [K] の Δb。"""
        db = self.to_bias(self.codebook).squeeze(-1)
        if not self.rank:
            return torch.zeros(len(db), 0, device=db.device), db
        return self.to_v(self.codebook) @ self.U.T, db

    def route(self, x: torch.Tensor, tau: float, hard_eval: bool) -> tuple[torch.Tensor, torch.Tensor]:
        """one-hot の割り当て q と、振り分け確率 p を返す。"""
        logits = self.router(x)
        p = logits.softmax(-1)
        if hard_eval:
            q = nn.functional.one_hot(logits.argmax(-1), logits.shape[-1]).float()
        else:
            q = nn.functional.gumbel_softmax(logits, tau=tau, hard=True)
        return q, p

    def correct(self, h: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
        dW, db = self.deltas()
        out = q @ db
        if self.rank:
            out = out + ((q @ dW) * h).sum(-1)
        return out


def weighted_bce(score: torch.Tensor, y: torch.Tensor, pos_rate: float, reduction: str = "mean") -> torch.Tensor:
    """1 段目と同じ global inverse class weight の BCE（既定はサンプル平均）。"""
    w = torch.where(y == 1, 0.5 / pos_rate, 0.5 / (1 - pos_rate))
    loss = w * nn.functional.binary_cross_entropy_with_logits(score, y.float(), reduction="none")
    return loss.mean() if reduction == "mean" else loss


class GroupDRO:
    """Sagawa et al. 2020 の online GroupDRO：群の重み q を群損失の指数で更新し、Σ q_g L_g を最小化する。

    LHCF（Masroor et al. 2026）は群を e(x) の GMM クラスタにした同じ形の minimax にあたる。
    """

    def __init__(self, n_groups: int, eta: float, device: str):
        self.q = torch.full((n_groups,), 1.0 / n_groups, device=device)
        self.eta = eta
        self.n = n_groups

    def __call__(self, per_sample: torch.Tensor, group: torch.Tensor) -> torch.Tensor:
        sums = torch.zeros(self.n, device=per_sample.device).index_add_(0, group, per_sample)
        counts = torch.zeros(self.n, device=per_sample.device).index_add_(0, group, torch.ones_like(per_sample))
        present = counts > 0
        group_loss = torch.where(present, sums / counts.clamp(min=1), torch.zeros_like(sums))
        with torch.no_grad():
            self.q = self.q * torch.exp(self.eta * group_loss.detach() * present)
            self.q = self.q / self.q.sum()
        return (self.q * group_loss).sum()


def fixed_assignment(kind: str, splits: dict[str, Split], K: int, seed: int) -> dict[str, np.ndarray]:
    """振り分け係を使わない条件の割り当て。k-means と GMM は fit split の e(x) に当てる。

    `demo32` は性別 × 年齢 4 区分 × race 4 区分のセル（番号は 0〜31）。`gmm` は LHCF と同じく
    e(x) の主成分に当てた GMM のクラスタ（属性もラベルも使わない）。
    """
    if kind == "global":
        return {name: np.zeros(len(s.target), dtype=np.int64) for name, s in splits.items()}
    if kind == "groups8":
        return {name: s.group for name, s in splits.items()}
    if kind == "demo32":
        return {name: demo_cells(s.context, "demo32").astype(np.int64) for name, s in splits.items()}
    if kind == "gmm":
        pca = PCA(64, random_state=seed).fit(splits["fit"].route)
        gmm = GaussianMixture(K, covariance_type="diag", random_state=seed, max_iter=200)
        gmm.fit(pca.transform(splits["fit"].route))
        return {name: gmm.predict(pca.transform(s.route)).astype(np.int64) for name, s in splits.items()}
    if kind == "kmeans":
        pca = PCA(64, random_state=seed).fit(splits["fit"].route)
        km = MiniBatchKMeans(K, random_state=seed, n_init=5, batch_size=4096).fit(pca.transform(splits["fit"].route))
        return {name: km.predict(pca.transform(s.route)).astype(np.int64) for name, s in splits.items()}
    raise ValueError(kind)


class Tensors:
    """1 split を GPU に置いた形。h と route は fit split の統計で標準化する。"""

    def __init__(self, s: Split, stats: dict[str, np.ndarray], assign: np.ndarray | None, device: str):
        def t(a, dtype=torch.float32):
            return torch.as_tensor(a, dtype=dtype, device=device)

        self.h = t((s.features - stats["h_mu"]) / stats["h_sd"])
        self.base = t(s.base)
        self.y = t(s.target, torch.long)
        self.route = t((s.route - stats["r_mu"]) / stats["r_sd"]) if s.route is not None else None
        self.assign = t(assign, torch.long) if assign is not None else None
        self.dro_group = None
        self.n = len(s.target)


def assignment_of(model: Corrector, d: Tensors, K: int, tau: float, train: bool):
    if d.assign is not None:
        q = nn.functional.one_hot(d.assign, K).float()
        return q, q
    return model.route(d.route, tau, hard_eval=not train)


def penalty(model: Corrector, p_bar: torch.Tensor, weighting: str) -> torch.Tensor:
    dW, db = model.deltas()
    norms = torch.sqrt((dW**2).sum(-1) + db**2 + 1e-12)
    scale = torch.sqrt(p_bar.detach() + 1e-6) if weighting == "sqrt" else torch.ones_like(norms)
    return (scale * norms).sum()


def predict(model: Corrector, d: Tensors, K: int) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    with torch.no_grad():
        scores, assigns = [], []
        for i in range(0, d.n, 8192):
            sl = slice(i, i + 8192)
            part = Tensors.__new__(Tensors)
            part.h, part.base, part.y = d.h[sl], d.base[sl], d.y[sl]
            part.route = d.route[sl] if d.route is not None else None
            part.assign = d.assign[sl] if d.assign is not None else None
            q, _ = assignment_of(model, part, K, 1.0, train=False)
            scores.append((part.base + model.correct(part.h, q)).cpu().numpy())
            assigns.append(q.argmax(-1).cpu().numpy())
    model.train()
    return np.concatenate(scores), np.concatenate(assigns)


def fit_corrector(args, data: dict[str, Tensors], pos_rate: float) -> tuple[Corrector, dict]:
    fit, val = data["fit"], data["val"]
    route_dim = fit.route.shape[1] if (args.assign == "router") else 0
    model = Corrector(fit.h.shape[1], route_dim, args.K, args.rank, hidden=args.router_hidden).to(args.device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    best = {"val_loss": np.inf}
    dro = GroupDRO(int(fit.dro_group.max()) + 1, args.dro_eta, args.device) if fit.dro_group is not None else None
    gen = torch.Generator(device=args.device).manual_seed(args.seed)
    history = []
    for epoch in range(args.epochs):
        tau = max(0.3, 1.0 * (0.9**epoch))
        perm = torch.randperm(fit.n, device=args.device, generator=gen)
        for i in range(0, fit.n, args.batch):
            idx = perm[i : i + args.batch]
            part = Tensors.__new__(Tensors)
            part.h, part.base, part.y = fit.h[idx], fit.base[idx], fit.y[idx]
            part.route = fit.route[idx] if fit.route is not None else None
            part.assign = fit.assign[idx] if fit.assign is not None else None
            q, p = assignment_of(model, part, args.K, tau, train=True)
            p_bar = p.mean(0)
            if dro is None:
                loss = weighted_bce(part.base + model.correct(part.h, q), part.y, pos_rate)
            else:
                per = weighted_bce(part.base + model.correct(part.h, q), part.y, pos_rate, reduction="none")
                loss = dro(per, fit.dro_group[idx])
            loss = loss + args.lam * penalty(model, p_bar, args.penalty_weight)
            if model.router is not None:
                loss = loss + args.beta * (p_bar * torch.log(p_bar + 1e-9)).sum()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
        score, _ = predict(model, val, args.K)
        val_loss = float(weighted_bce(torch.as_tensor(score), val.y.cpu(), pos_rate))
        history.append(val_loss)
        if val_loss < best["val_loss"]:
            best = {
                "val_loss": val_loss,
                "epoch": epoch,
                "state": {k: v.clone() for k, v in model.state_dict().items()},
            }
    model.load_state_dict(best["state"])
    return model, {"best_epoch": best["epoch"], "val_loss": best["val_loss"], "val_history": history}


def factor_table(s: Split) -> pd.DataFrame:
    """答え合わせに使う 2 値の要因（モデルにも振り分け係にも渡していない、または既知の属性）。"""
    c = s.context
    return pd.DataFrame(
        {
            "focused": (c.report_type == "focused_followup").to_numpy(),
            "followup": c.followup.to_numpy(),
            "device": c.device.to_numpy(),
            "pacer": c.pacer.to_numpy(),
            "old": (c.age >= 65).to_numpy(),
            "sex": (c.sex == 1).to_numpy(),
            "lateral": (c.frontal_lateral == 1).to_numpy(),
            "ap": (c.ap_pa == 1).to_numpy(),
            "nonwhite": (c.race_group != 0).to_numpy(),
            "marked": s.marked,
        }
    )


def performance(score: np.ndarray, s: Split, pos_rate: float) -> dict[str, float]:
    y = s.target
    c = s.context
    out = {
        "loss": float(weighted_bce(torch.as_tensor(score), torch.as_tensor(y), pos_rate)),
        "auroc": roc_auc_score(y, score),
    }
    group_auc = [roc_auc_score(y[s.group == g], score[s.group == g]) for g in range(len(GROUP_NAMES))]
    out["worst8"] = float(np.min(group_auc))
    out["mean8"] = float(np.mean(group_auc))
    old = c.age.to_numpy() >= 65
    out["auroc_old"] = roc_auc_score(y[old], score[old])
    out["auroc_young"] = roc_auc_score(y[~old], score[~old])
    for kind in ("focused_followup", "affirmed_normal"):
        keep = (y == 0) | (c.report_type.to_numpy() == kind)
        out[f"auroc_pos_{kind}"] = roc_auc_score(y[keep], score[keep])
    if s.marked.any():
        for name, m in (("marked", s.marked), ("unmarked", ~s.marked)):
            out[f"auroc_{name}"] = roc_auc_score(y[m], score[m])
            out[f"fpr0_{name}"] = float((score[m & (y == 0)] > 0).mean())
            out[f"tpr0_{name}"] = float((score[m & (y == 1)] > 0).mean())
    return out


def recovery(assign_val: np.ndarray, assign_test: np.ndarray, val: Split, test: Split) -> dict[str, float]:
    """各要因を、グループ割り当てからどれだけ当てられるか。

    `auc_code`：val で求めたグループ別の要因の割合を test のスコアにした AUROC。
    `auc_known`：既知の 8 群 × 撮影方向 × 性別のセルで同じことをした値。
    `auc_both`：両者を掛け合わせたセル。`auc_both − auc_known` が既知の属性を超えて回収した分。
    """
    fv, ft = factor_table(val), factor_table(test)
    known_v = val.group * 8 + val.context.frontal_lateral.to_numpy() * 4 + val.context.ap_pa.to_numpy() * 2
    known_v = known_v + val.context.sex.to_numpy()
    known_t = test.group * 8 + test.context.frontal_lateral.to_numpy() * 4 + test.context.ap_pa.to_numpy() * 2
    known_t = known_t + test.context.sex.to_numpy()
    out = {}
    for f in FACTORS:
        if not ft[f].any():
            continue
        yv, yt = fv[f].to_numpy(), ft[f].to_numpy()
        out[f"ami_{f}"] = adjusted_mutual_info_score(yt, assign_test)
        for label, cv, ct in (
            ("code", assign_val, assign_test),
            ("known", known_v, known_t),
            ("both", assign_val * 1000 + known_v, assign_test * 1000 + known_t),
        ):
            rate = pd.Series(yv).groupby(cv).mean()
            pred = pd.Series(ct).map(rate).fillna(yv.mean()).to_numpy()
            out[f"auc_{label}_{f}"] = roc_auc_score(yt, pred)
    return out


def code_table(model: Corrector, base: np.ndarray, score: np.ndarray, assign: np.ndarray, s: Split, pos_rate: float):
    dW, db = (t.detach().cpu().numpy() for t in model.deltas())
    factors = factor_table(s)
    y = torch.as_tensor(s.target)
    rows = []
    for k in range(len(db)):
        m = assign == k
        row = {"code": k, "n": int(m.sum()), "share": float(m.mean()), "db": float(db[k])}
        row["dw_norm"] = float(np.linalg.norm(dW[k])) if dW.shape[1] else 0.0
        if m.sum() > 0:
            ym = y[m]
            row["pos_rate"] = float(s.target[m].mean())
            row["loss_base"] = float(weighted_bce(torch.as_tensor(base[m]), ym, pos_rate))
            row["loss_corr"] = float(weighted_bce(torch.as_tensor(score[m]), ym, pos_rate))
            for f in FACTORS:
                row[f"rate_{f}"] = float(factors[f][m].mean())
        rows.append(row)
    return pd.DataFrame(rows)


def main(argv: list[str] | None = None) -> dict:
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", required=True)
    parser.add_argument("--stage1", default="a_s42")
    parser.add_argument("--fit", default="train_b")
    parser.add_argument(
        "--assign", choices=("router", "groups8", "demo32", "kmeans", "gmm", "global"), default="router"
    )
    parser.add_argument("--route", default="raddino", help="embed/<name>、`meta`、または `none`")
    parser.add_argument("--K", type=int, default=16)
    parser.add_argument("--rank", type=int, default=4)
    parser.add_argument("--lam", type=float, default=1e-3)
    parser.add_argument("--beta", type=float, default=0.05)
    parser.add_argument("--route-pca", type=int, default=0, help="0 なら e(x) をそのまま使う")
    parser.add_argument("--router-hidden", type=int, default=256, help="0 なら線形の振り分け係")
    parser.add_argument("--penalty-weight", choices=("sqrt", "none"), default="sqrt")
    parser.add_argument(
        "--dro", choices=("none", "demo32", "gmm"), default="none", help="GroupDRO の群（none なら ERM）"
    )
    parser.add_argument("--dro-K", type=int, default=32, help="`--dro gmm` のクラスタ数")
    parser.add_argument("--dro-eta", type=float, default=0.01)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch", type=int, default=2048)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--select", choices=("val", "holdout"), default="val", help="epoch 選択と答え合わせの率に使う split"
    )
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args(argv)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    route = None if args.route == "none" else args.route
    if args.select == "val":
        splits = {"fit": load(args.stage1, args.fit, route), "val": load(args.stage1, "val", route)}
    else:
        # 選択用の split を fit 側（展開先と同じ分布）から患者単位で取り分ける。植え込み実験の val は
        # 1 段目の学習側と同じくマーカーがラベルと相関しており、そこで選ぶとショートカットを残す補正が選ばれる。
        rest, held = holdout_by_patient(load(args.stage1, args.fit, route), 0.2)
        splits = {"fit": rest, "val": held}
    splits["test"] = load(args.stage1, "test", route)
    if args.assign == "groups8":
        args.K = len(GROUP_NAMES)
    if args.assign == "global":
        args.K = 1
    fixed = fixed_assignment(args.assign, splits, args.K, args.seed) if args.assign != "router" else None
    if args.route_pca and splits["fit"].route is not None:
        # 振り分け係の入力を e(x) の主成分に絞り、ラベルに効く細かな方向で分けにくくする
        pca = PCA(args.route_pca, random_state=0).fit(splits["fit"].route)
        for s in splits.values():
            s.route = pca.transform(s.route).astype(np.float32)
    fit = splits["fit"]
    stats = {"h_mu": fit.features.mean(0), "h_sd": fit.features.std(0) + 1e-6}
    if fit.route is not None:
        stats.update(r_mu=fit.route.mean(0), r_sd=fit.route.std(0) + 1e-6)
    data = {k: Tensors(s, stats, fixed[k] if fixed else None, args.device) for k, s in splits.items()}
    if args.dro != "none":
        # DRO の群は補正の群とは別に決める（LHCF は e(x) の GMM、GroupDRO は属性セル）
        dro_groups = fixed_assignment(args.dro, splits, args.dro_K, args.seed)
        data["fit"].dro_group = torch.as_tensor(dro_groups["fit"], dtype=torch.long, device=args.device)
    pos_rate = float(fit.target.mean())
    model, train_log = fit_corrector(args, data, pos_rate)

    test = splits["test"]
    score_test, assign_test = predict(model, data["test"], args.K)
    score_val, assign_val = predict(model, data["val"], args.K)
    summary = {
        "args": vars(args),
        **train_log,
        "base": performance(test.base, test, pos_rate),
        "corrected": performance(score_test, test, pos_rate),
        "recovery": recovery(assign_val, assign_test, splits["val"], test),
        "codes_used": int(len(np.unique(assign_test))),
    }
    out = PACKAGE / "results" / "latent_codes" / args.name
    out.mkdir(parents=True, exist_ok=True)
    (out / "summary.json").write_text(json.dumps(summary, indent=1, ensure_ascii=False, default=float))
    code_table(model, test.base, score_test, assign_test, test, pos_rate).to_csv(out / "codes.csv", index=False)
    assign_dir = PACKAGE / "cache" / "assign"
    assign_dir.mkdir(parents=True, exist_ok=True)
    np.save(assign_dir / f"{args.name}_val.npy", assign_val)
    np.save(assign_dir / f"{args.name}_test.npy", assign_test)
    np.save(assign_dir / f"{args.name}_test_score.npy", score_test)
    if args.select == "val":
        np.save(assign_dir / f"{args.name}_val_score.npy", score_val)
    return summary


if __name__ == "__main__":
    main()
