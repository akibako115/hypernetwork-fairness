"""属性の掛け合わせで切る監査と、属性を使わずに見つける潜在スライスの監査を、同じ物差しで比べる。

物差しは群の O/E（観測した陽性数 ÷ 予測確率の和）。予測確率は 1 段目の logit を発見用 split で
全体較正（既定は logit のスプライン上のロジスティック回帰）したもの。log O/E > 0 は、群の陽性を
過少に予測している（補正の切片で直せる偏り）ことを表す。TPR・FPR は較正用 split で決めた全体のしきい値
（Youden）で測る。

スライスはすべて発見用 split（1 段目が見ていない train_b、または補正後なら val）で決め、test で測る。
同じデータで決めて測ると、雑音の大きい群を選ぶだけで偏りが大きく見えるため。

族（family）:
- `demo8`：年齢 2 区分 × race 4 区分（現行の 8 群）
- `demo32`：性別 × 年齢 4 区分 × race 4 区分
- `demo96`：`demo32` × 保険種別 3 区分（3 は 2 にまとめる）
- `gmm`：e(x) の主成分 64 次元に当てた GMM（LHCF と同じく属性もラベルも使わない）
- `tree`：e(x) から残差 y − p を回帰する決定木の葉（ラベルを使う誤差スライス発見）
- `field`：hypernet の補正量（1 枚ごとの logit 補正）の分位で切った群
- `context`：属性に無い既知の要因（デバイス・追跡撮影・レポート種別・撮影方向）の掛け合わせ
- `score`：1 段目のスコアの分位（スコアの高さだけで分けた対照）
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.mixture import GaussianMixture
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import SplineTransformer
from sklearn.tree import DecisionTreeRegressor

from analysis.archive.latent_bias_hypernet.data import Split

AGE_EDGES = (40, 65, 80)


def sigmoid(z: np.ndarray) -> np.ndarray:
    return 1 / (1 + np.exp(-z))


class Calibrator:
    """発見用 split の logit に当てた全体の較正と Youden しきい値。

    `spline`（既定）は logit の自然スプライン（節 6 個）の上のロジスティック回帰で、スコアの水準ごとの
    較正のずれまで滑らかに取り除く。Platt だと 2 パラメータで直しきれない較正曲線の曲がりが残り、
    スコアの高さで分けただけの群にも O/E の差が出る。isotonic は裾で確率が 0 近くに張り付き O/E が暴れる。
    """

    def __init__(self, logit: np.ndarray, y: np.ndarray, kind: str = "spline"):
        self.kind = kind
        if kind == "platt":
            self.model = LogisticRegression(C=1e6).fit(logit[:, None], y)
        elif kind == "spline":
            self.model = make_pipeline(
                SplineTransformer(n_knots=6, degree=3, knots="quantile", extrapolation="linear"),
                LogisticRegression(C=1e4, max_iter=2000),
            ).fit(logit[:, None].astype(np.float64), y)
        else:
            self.model = IsotonicRegression(y_min=1e-4, y_max=1 - 1e-4, out_of_bounds="clip").fit(logit, y)
        p = self(logit)
        fpr, tpr, thr = roc_curve(y, p)
        self.threshold = float(thr[np.argmax(tpr - fpr)])

    def __call__(self, logit: np.ndarray) -> np.ndarray:
        if self.kind == "isotonic":
            return self.model.predict(logit)
        return self.model.predict_proba(logit[:, None].astype(np.float64))[:, 1]


def demo_cells(context: pd.DataFrame, kind: str) -> np.ndarray:
    age_bin = np.digitize(context.age.to_numpy(), AGE_EDGES)
    race = context.race_group.to_numpy().astype(int)
    old = (context.age.to_numpy() >= 65).astype(int)
    if kind == "demo8":
        return old * 4 + race
    if kind == "demo16":
        return context.sex.to_numpy().astype(int) * 8 + old * 4 + race
    cell = context.sex.to_numpy().astype(int) * 16 + age_bin * 4 + race
    if kind == "demo32":
        return cell
    if kind == "demo96":
        return cell * 3 + np.minimum(context.insurance_type.to_numpy().astype(int), 2)
    raise ValueError(kind)


def context_cells(context: pd.DataFrame) -> np.ndarray:
    """デバイス × 追跡撮影 × レポート種別（3）× 正面・側面。"""
    rt = pd.Categorical(context.report_type, categories=["affirmed_normal", "focused_followup", "other"]).codes
    return (
        context.device.to_numpy().astype(int) * 12
        + context.followup.to_numpy().astype(int) * 6
        + rt * 2
        + context.frontal_lateral.to_numpy().astype(int)
    )


def imaging_context_cells(context: pd.DataFrame) -> np.ndarray:
    """撮影文脈：撮影方向（正面 PA・正面 AP・側面）× 追跡撮影 × レポート種別（3）。

    context_cells と違い、デバイスを入れない。デバイスなど画像に写る部分集団は要因ごとに別の族にし、
    「撮り方・レポートの書かれ方」と「画像に写る物」を交絡の候補として分けて読む。
    """
    view = context.frontal_lateral.to_numpy().astype(int) * 2 + context.ap_pa.to_numpy().astype(int)
    rt = pd.Categorical(context.report_type, categories=["affirmed_normal", "focused_followup", "other"]).codes
    return view * 6 + context.followup.to_numpy().astype(int) * 3 + rt


class GMMSlicer:
    """LHCF 型：e(x) の主成分に GMM を当て、属性もラベルも使わずに群を作る。"""

    def __init__(self, route: np.ndarray, K: int, seed: int = 0, dims: int = 64):
        self.pca = PCA(dims, random_state=seed).fit(route)
        z = self.pca.transform(route)
        self.gmm = GaussianMixture(K, covariance_type="diag", random_state=seed, max_iter=200).fit(z)

    def __call__(self, route: np.ndarray) -> np.ndarray:
        return self.gmm.predict(self.pca.transform(route))


class TreeSlicer:
    """誤差スライス発見：e(x) の主成分から残差 y − p を回帰する決定木の葉を群にする。"""

    def __init__(self, route: np.ndarray, residual: np.ndarray, K: int, min_leaf: int, seed: int = 0, dims: int = 64):
        self.pca = PCA(dims, random_state=seed).fit(route)
        self.tree = DecisionTreeRegressor(max_leaf_nodes=K, min_samples_leaf=min_leaf, random_state=seed)
        self.tree.fit(self.pca.transform(route), residual)

    def __call__(self, route: np.ndarray) -> np.ndarray:
        return self.tree.apply(self.pca.transform(route))


def quantile_slicer(ref: np.ndarray, K: int):
    edges = np.quantile(ref, np.linspace(0, 1, K + 1)[1:-1])
    return lambda v: np.digitize(v, edges)


def log_oe(y: np.ndarray, p: np.ndarray) -> float:
    """log((O + 0.5) / (E + 0.5))。O = 0 の群でも有限にする。"""
    return float(np.log((y.sum() + 0.5) / (p.sum() + 0.5)))


def group_metrics(group: np.ndarray, y: np.ndarray, p: np.ndarray, threshold: float, min_n: int, min_exp: float):
    """群ごとの n・陽性数・期待陽性数・log O/E・TPR・FPR・AUROC。

    群の選別は n と期待陽性数 E = Σp で行う。観測陽性数で選ぶと、低スコア帯ではたまたま陽性の多い群だけが
    残り、O/E が上振れする。標準誤差も帰無仮説（較正が合っている）の下の 1/√E を使う。
    """
    rows = []
    pred = p >= threshold
    for g in np.unique(group):
        m = group == g
        n, pos, exp = int(m.sum()), int(y[m].sum()), float(p[m].sum())
        if n < min_n or exp < min_exp:
            continue
        row = {"group": g, "n": n, "pos": pos, "exp": exp, "log_oe": log_oe(y[m], p[m]), "se_log_oe": 1 / np.sqrt(exp)}
        ok = 10 <= pos < n
        row["tpr"] = float(pred[m & (y == 1)].mean()) if ok else np.nan
        row["fpr"] = float(pred[m & (y == 0)].mean())
        row["auroc"] = float(roc_auc_score(y[m], p[m])) if ok else np.nan
        rows.append(row)
    return pd.DataFrame(rows)


NULL_DRAWS = 20


def _weighted_var(group: np.ndarray, keep: np.ndarray, y: np.ndarray, p: np.ndarray) -> float:
    """keep の群について、log O/E の期待陽性数重みの分散。"""
    m = np.isin(group, keep)
    g, yy, pp = group[m], y[m], p[m]
    _, inv = np.unique(g, return_inverse=True)
    obs = np.bincount(inv, weights=yy)
    exp = np.bincount(inv, weights=pp)
    v = np.log((obs + 0.5) / (exp + 0.5))
    w = exp / exp.sum()
    return float((w * (v - (w * v).sum()) ** 2).sum())


def null_labels(p: np.ndarray, rng: np.random.Generator, cluster: np.ndarray | None = None) -> np.ndarray:
    """較正が合っている帰無仮説の下のラベル y* ~ Bernoulli(p)。

    `cluster`（病変など、ラベルを共有する単位）を渡すと、同じ単位の画像に同じ一様乱数を使う。
    同じ病変の複数画像はラベルが一致するので、画像ごとに独立に引くと標本誤差を過小に見積もる。
    """
    if cluster is None:
        return (rng.random(len(p)) < p).astype(float)
    _, inv = np.unique(cluster, return_inverse=True)
    return (rng.random(inv.max() + 1)[inv] < p).astype(float)


def dispersion(
    table: pd.DataFrame, group: np.ndarray, y: np.ndarray, p: np.ndarray, cluster: np.ndarray | None = None
) -> dict[str, float]:
    """群間の log O/E のばらつき（期待陽性数で重み付け）から、標本誤差による分を差し引いた量。

    標本誤差の分は、較正が合っている帰無仮説の下で y* を引き直したときの同じ分散の平均
    （パラメトリックな帰無分布、`null_labels`）。群の選別は p だけで決まるので、引き直しても同じ群の組で比べられる。
    """
    keep = table.group.to_numpy()
    raw = _weighted_var(group, keep, y, p)
    rng = np.random.default_rng(0)
    null = [_weighted_var(group, keep, null_labels(p, rng, cluster), p) for _ in range(NULL_DRAWS)]
    noise = float(np.mean(null))
    return {
        "raw_var": raw,
        "noise_var": noise,
        "excess_sd": float(np.sqrt(max(raw - noise, 0.0))),
        "null_q95_sd": float(np.sqrt(max(np.quantile(null, 0.95) - noise, 0.0))),
        "above_null": bool(raw > np.quantile(null, 0.95)),
    }


def summarize_family(name: str, table: pd.DataFrame, base_tpr: float, group, y, p) -> dict:
    worst_under = table.sort_values("log_oe", ascending=False).iloc[0]
    worst_tpr = table.dropna(subset=["tpr"]).sort_values("tpr").iloc[0]
    return {
        "family": name,
        "groups": len(table),
        "coverage": int(table.n.sum()),
        **dispersion(table, group, y, p),
        "max_log_oe": float(worst_under.log_oe),
        "max_log_oe_n": int(worst_under.n),
        "min_log_oe": float(table.log_oe.min()),
        "worst_tpr": float(worst_tpr.tpr),
        "worst_tpr_gap": float(base_tpr - worst_tpr.tpr),
        "worst_tpr_n": int(worst_tpr.n),
        "max_fpr": float(table.fpr.max()),
        "worst_auroc": float(table.auroc.min()),
        "max_fpr_n": int(table.sort_values("fpr").iloc[-1].n),
    }


def _within_var(cells, slices, y, p, pairs) -> float:
    rows = []
    for c, sl in pairs:
        mc = cells == c
        m = mc & (slices == sl)
        rows.append((float(p[m].sum()), log_oe(y[m], p[m]) - log_oe(y[mc], p[mc])))
    e, d = np.array(rows).T
    w = e / e.sum()
    return float((w * d**2).sum())


def within_cell(cells: np.ndarray, slices: np.ndarray, y: np.ndarray, p: np.ndarray, min_n: int, min_exp: float):
    """属性セルの内側に、潜在スライスで分けたときの log O/E のばらつきがどれだけ隠れているか。

    セル × スライスの交差群について、セル全体の log O/E からのずれの二乗（期待陽性数で重み付け）を求め、
    y* ~ Bernoulli(p) で引き直した帰無分布の平均を差し引く。
    """
    pairs = []
    for c in np.unique(cells):
        mc = cells == c
        if p[mc].sum() < min_exp:
            continue
        for sl in np.unique(slices[mc]):
            m = mc & (slices == sl)
            if m.sum() >= min_n and p[m].sum() >= min_exp:
                pairs.append((c, sl))
    if not pairs:
        return {"within_excess_sd": float("nan"), "within_pairs": 0}
    raw = _within_var(cells, slices, y, p, pairs)
    rng = np.random.default_rng(0)
    null = [_within_var(cells, slices, (rng.random(len(p)) < p).astype(float), p, pairs) for _ in range(NULL_DRAWS)]
    return {
        "within_excess_sd": float(np.sqrt(max(raw - np.mean(null), 0.0))),
        "within_pairs": len(pairs),
        "within_above_null": bool(raw > np.quantile(null, 0.95)),
    }


def residual_explained(group: np.ndarray, residual: np.ndarray) -> float:
    """test の残差 y − p の分散のうち、群平均で説明できる割合（群を test で決めていないので過学習しない）。"""
    s = pd.Series(residual)
    means = s.groupby(group).transform("mean").to_numpy()
    return float(1 - ((residual - means) ** 2).sum() / ((residual - residual.mean()) ** 2).sum())


def contains(split: Split) -> dict[str, np.ndarray]:
    """答え合わせ用の、属性に無い要因。"""
    c = split.context
    return {
        "device": c.device.to_numpy(dtype=bool),
        "followup": c.followup.to_numpy(dtype=bool),
        "focused": (c.report_type == "focused_followup").to_numpy(),
        "pacer": c.pacer.to_numpy(dtype=bool),
    }


class ResidualAuditor:
    """補正手法とは独立に学習する誤差予測器。val の残差だけを見て、test を予測残差の分位で切る。

    入力は 1 段目の表現 h(x) と e(x) をそれぞれ主成分 64 次元にして連結したもの。較正後の logit を
    offset にして、その上に足す logit 補正 g(x) を小さな MLP で学習する（患者単位 20% で early stopping）。
    """

    def __init__(self, x: np.ndarray, logit_offset: np.ndarray, y: np.ndarray, patients: np.ndarray, seed: int):
        import torch
        from torch import nn

        self.torch = torch
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        rng = np.random.default_rng(seed)
        uniq = np.unique(patients)
        held = np.isin(patients, rng.permutation(uniq)[: len(uniq) // 5])
        self.mu, self.sd = x[~held].mean(0), x[~held].std(0) + 1e-6
        torch.manual_seed(seed)
        self.net = nn.Sequential(nn.Linear(x.shape[1], 128), nn.ReLU(), nn.Dropout(0.2), nn.Linear(128, 1))
        self.net = self.net.to(self.device)
        opt = torch.optim.Adam(self.net.parameters(), lr=1e-3, weight_decay=1e-4)
        xt = torch.as_tensor((x - self.mu) / self.sd, dtype=torch.float32, device=self.device)
        ot = torch.as_tensor(logit_offset, dtype=torch.float32, device=self.device)
        yt = torch.as_tensor(y, dtype=torch.float32, device=self.device)
        tr = torch.as_tensor(np.flatnonzero(~held), device=self.device)
        va = torch.as_tensor(np.flatnonzero(held), device=self.device)
        gen = torch.Generator(device=self.device).manual_seed(seed)
        best, best_state = np.inf, None
        for _ in range(60):
            self.net.train()
            order = tr[torch.randperm(len(tr), device=self.device, generator=gen)]
            for i in range(0, len(tr), 1024):
                idx = order[i : i + 1024]
                loss = nn.functional.binary_cross_entropy_with_logits(ot[idx] + self.net(xt[idx]).squeeze(-1), yt[idx])
                opt.zero_grad()
                loss.backward()
                opt.step()
            self.net.eval()
            with torch.no_grad():
                v = nn.functional.binary_cross_entropy_with_logits(ot[va] + self.net(xt[va]).squeeze(-1), yt[va])
            if v < best:
                best, best_state = float(v), {k: t.clone() for k, t in self.net.state_dict().items()}
        self.net.load_state_dict(best_state)
        self.net.eval()

    def __call__(self, x: np.ndarray) -> np.ndarray:
        with self.torch.no_grad():
            xt = self.torch.as_tensor((x - self.mu) / self.sd, dtype=self.torch.float32, device=self.device)
            return self.net(xt).squeeze(-1).cpu().numpy()


def auditor_inputs(h: np.ndarray, e: np.ndarray, h_ref: np.ndarray, e_ref: np.ndarray, dims: int = 64):
    """監査器の入力：h と e を、参照 split で当てた主成分に落として連結する。"""
    ph = PCA(dims, random_state=0).fit(h_ref)
    pe = PCA(dims, random_state=0).fit(e_ref)
    return np.concatenate([ph.transform(h), pe.transform(e)], axis=1), (ph, pe)
