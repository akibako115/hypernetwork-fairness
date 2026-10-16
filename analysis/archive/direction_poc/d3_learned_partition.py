"""方向 3：区分を判定の境界に合わせる方法を、凍結特徴の上の 2 段目の head で当たり付けする。

場は HAM10000 の植え込み（`rule_planting.py --dose strong`）。少数側 B（DINOv2 の GMM 8 のクラスタ 2・4）では
マーカーの意味が反転する。1 段目は cross-fit の ResNet-50（stage1 で学習、seed 42〜44）で、2 段目に当たる head を
1 段目が見ていない stage2 の train で学習し、val で選び、test で測る。植え込みのない実データ（`real`）でも同じ方法を回す。

head は共通の形 `logit = s1(x) + w0·x + b0 + Σ_k π_k(x) (W_k·x + c_k)` をとる。
- x：1 段目の表現 h を stage2 の train で標準化 → PCA 64（マーカーの方向を残す。`marker_auc` で確かめる）
- s1：1 段目の logit 差（凍結）。2 段目は補正だけを学ぶ
- π：区分。固定（one-hot）か、振り分け係 softmax(G·z) を補正と同時に学ぶ（mixture of experts）

区分の作り方:
- 固定：定数（K=1）、ランダム 8、GMM 8（植え込みの境界を含む区分。上限の目安）
- (a) 同時学習（`moe_*`）：z は DINOv2（PCA 32）か [h; e]。K=2・8。初期値を 5 通り変え、val の損失で選ぶ
- (b) 傾きの差で選ぶ（`search`）：ラベルを使わない候補区分（DINOv2 の PCA 32 に GMM、K∈{2,3,4,6,8,10,12}×5 seed）
  それぞれで固定区分の head を学び、val で「固定区分の head − 同じ K のランダム区分の head」の損失の減りが最大の候補を
  選ぶ。候補には元の GMM 8 と同じ作り方（seed 0、K=8）が入るので、これは生成ではなく選び方の検証になる

誤りから区分を作り直す反復（`crossfit.py`）は既存の結果（B との ARI 0.035 以下）を引用し、ここでは回さない。

評価（test）：AUROC 全体・A・B、区分と B の ARI、B が過半の区分の B 被覆率。定数との差の区間は、
seed 平均の差を病変単位の対応のある bootstrap（500 回）で求める。区分の seed 間の安定性は、1 段目の seed が違う
run どうしの test での区分の ARI。実データでは全体と性別 × 年齢の最悪群 AUROC、区分と施設・ラベル・性別 × 年齢の NMI。

入力: `latent_bias_hypernet/cache/*-cf-stage1-*_cf_{train,val,test}.npz`、`latent_bias_hypernet/cache/embed/dinov2/ham_*.npz`、
`data/ham10000/splits_crossfit/stage2/`
出力: `results/d3/{runs,summary,contrast,stability}.csv`、`cache/d3/<field>_s<seed>.npz`（test の score と区分）

    OMP_NUM_THREADS=8 uv run python -m analysis.archive.direction_poc.d3_learned_partition
    OMP_NUM_THREADS=8 uv run python -m analysis.archive.direction_poc.d3_learned_partition --alignment   # 選び方の検算
"""

from __future__ import annotations

import glob

import numpy as np
import pandas as pd
import rootutils
import torch
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score as ari
from sklearn.metrics import normalized_mutual_info_score as nmi
from sklearn.metrics import roc_auc_score
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler

ROOT = rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from analysis.archive.latent_bias_hypernet.e2e_eval import group_aucs, sex_age  # noqa: E402
from analysis.archive.latent_bias_hypernet.rule_planting import marker_mask, minority  # noqa: E402

LBH = ROOT / "analysis" / "archive" / "latent_bias_hypernet"
OUT = ROOT / "analysis" / "archive" / "direction_poc" / "results" / "d3"
CACHE_OUT = ROOT / "analysis" / "archive" / "direction_poc" / "cache" / "d3"
SPLITS = ("train", "val", "test")
FRAMES = ROOT / "data" / "ham10000" / "splits_crossfit" / "stage2"
STAGE1 = {"rule": "resnet-ham10000-rule-strong-cf-stage1", "real": "resnet-ham10000-cf-stage1"}
SEEDS = (42, 43, 44)
DX, DZ = 64, 32
L2S = (1e-3, 1e-2, 1e-1)
RESTARTS = 5
SEARCH_K = (2, 3, 4, 6, 8, 10, 12)
SEARCH_SEEDS = range(5)
BOOT = 500
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ---------------------------------------------------------------- 入力


def load(field: str, seed: int) -> dict:
    """stage2 の 3 split の 1 段目の表現・logit、DINOv2、ラベル、B、マーカーを行対応を検算して読む。"""
    pattern = str(LBH / "cache" / f"*-{STAGE1[field]}-s{seed}-*_cf_train.npz")
    (train_path,) = glob.glob(pattern)
    prefix = train_path.removesuffix("_cf_train.npz")
    data = {}
    for s in SPLITS:
        frame = pd.read_csv(FRAMES / f"{s}.csv")
        cf = np.load(f"{prefix}_cf_{s}.npz")
        np.testing.assert_array_equal(cf["target"], frame.target.to_numpy())
        emb_split = "train" if s == "train" else s
        emb = np.load(LBH / "cache" / "embed" / "dinov2" / f"ham_{emb_split}.npz")
        index = pd.Series(np.arange(len(emb["image"])), index=emb["image"])
        rows = index.loc[frame.image].to_numpy()
        data[s] = {
            "h": cf["features"].astype(np.float64),
            "s1": cf["score"].astype(np.float64),
            "y": cf["target"].astype(np.int64),
            "e": emb["embedding"][rows].astype(np.float64),
            "frame": frame,
            "b": minority(frame),
            "marked": marker_mask(frame, "strong") if field == "rule" else np.zeros(len(frame), bool),
        }
    return data


def reduce(data: dict) -> None:
    """h と e を stage2 の train で標準化 → PCA に落とす（x：補正の入力、z：振り分けの入力）。"""
    for key, dim, out in (("h", DX, "x"), ("e", DZ, "z")):
        scaler = StandardScaler().fit(data["train"][key])
        pca = PCA(dim, random_state=0).fit(scaler.transform(data["train"][key]))
        scale = pca.transform(scaler.transform(data["train"][key])).std(axis=0).mean()
        for s in SPLITS:
            data[s][out] = pca.transform(scaler.transform(data[s][key])) / scale
    for s in SPLITS:
        data[s]["hz"] = np.concatenate([data[s]["x"], data[s]["z"]], axis=1)


def candidate_partitions() -> dict[str, dict[str, pd.Series]]:
    """ラベルを使わない候補区分。DINOv2 を HAM の train 全体で標準化 → PCA 32 → GMM（gmm8 と同じ作り方）。"""
    embs = {s: np.load(LBH / "cache" / "embed" / "dinov2" / f"ham_{s}.npz") for s in SPLITS}
    raw = {s: embs[s]["embedding"].astype(np.float64) for s in SPLITS}
    mu, sd = raw["train"].mean(0), raw["train"].std(0) + 1e-6
    pca = PCA(32, random_state=0).fit((raw["train"] - mu) / sd)
    z = {s: pca.transform((raw[s] - mu) / sd) for s in SPLITS}
    out = {}
    for k in SEARCH_K:
        for seed in SEARCH_SEEDS:
            gmm = GaussianMixture(k, covariance_type="diag", random_state=seed, n_init=3).fit(z["train"])
            out[f"gmm{k}_s{seed}"] = {s: pd.Series(gmm.predict(z[s]), index=embs[s]["image"]) for s in SPLITS}
    return out


# ---------------------------------------------------------------- head


class Head(torch.nn.Module):
    """s1 ＋ 共有の線形補正 ＋ 区分ごとの線形補正。`gate_dim` を渡すと区分を softmax で同時に学ぶ。"""

    def __init__(self, k: int, dx: int, gate_dim: int | None = None, seed: int = 0):
        super().__init__()
        gen = torch.Generator().manual_seed(seed)
        self.shared = torch.nn.Linear(dx, 1)
        torch.nn.init.zeros_(self.shared.weight)
        torch.nn.init.zeros_(self.shared.bias)
        self.w = torch.nn.Parameter(torch.zeros(k, dx))
        self.c = torch.nn.Parameter(torch.zeros(k))
        self.gate = None
        if gate_dim is not None:
            self.gate = torch.nn.Linear(gate_dim, k)
            with torch.no_grad():
                self.gate.weight.copy_(torch.randn(k, gate_dim, generator=gen) * 1.0)
                self.gate.bias.zero_()
            # 専門家が同じ値から始まると対称性が崩れないので、補正にも小さな乱数を入れる
            with torch.no_grad():
                self.w.copy_(torch.randn(k, dx, generator=gen) * 0.01)

    def weights(self, assign: torch.Tensor) -> torch.Tensor:
        return torch.softmax(self.gate(assign), dim=1) if self.gate is not None else assign

    def forward(self, x, s1, assign):
        pi = self.weights(assign)
        expert = x @ self.w.T + self.c
        return s1 + self.shared(x).squeeze(1) + (pi * expert).sum(1)

    def penalty(self):
        return (self.w**2).sum() + (self.shared.weight**2).sum()


def tensors(d: dict, assign: np.ndarray) -> tuple:
    def t(v):
        return torch.as_tensor(np.asarray(v), dtype=torch.float32, device=DEVICE)

    return t(d["x"]), t(d["s1"]), t(assign), t(d["y"])


def fit_head(data: dict, assign: dict, k: int, gate_dim: int | None, l2: float, seed: int) -> tuple[Head, float]:
    """train で全バッチの Adam、val の損失で早期終了。val の最小損失を返す。"""
    torch.manual_seed(seed)
    model = Head(k, DX, gate_dim, seed).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=1e-2)
    tr, va = tensors(data["train"], assign["train"]), tensors(data["val"], assign["val"])
    bce = torch.nn.functional.binary_cross_entropy_with_logits
    best, best_state, wait = np.inf, None, 0
    n = len(tr[3])
    for step in range(3000):
        opt.zero_grad()
        loss = bce(model(*tr[:3]), tr[3]) + l2 * model.penalty() / n
        loss.backward()
        opt.step()
        if step % 10:
            continue
        with torch.no_grad():
            val = bce(model(*va[:3]), va[3]).item()
        if val < best - 1e-5:
            best, wait = val, 0
            best_state = {key: v.detach().clone() for key, v in model.state_dict().items()}
        else:
            wait += 1
            if wait > 30:
                break
    model.load_state_dict(best_state)
    return model, best


def select(data: dict, assign: dict, k: int, gate_dim: int | None) -> tuple[Head, float]:
    """L2 と（振り分けを学ぶときは）初期値を val の損失で選ぶ。"""
    restarts = RESTARTS if gate_dim is not None else 1
    best = (None, np.inf)
    for l2 in L2S:
        for r in range(restarts):
            model, val = fit_head(data, assign, k, gate_dim, l2, seed=r)
            if val < best[1]:
                best = (model, val)
    return best


def predict(model: Head, d: dict, assign: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    x, s1, a, _ = tensors(d, assign)
    with torch.no_grad():
        score = model(x, s1, a).cpu().numpy()
        part = model.weights(a).argmax(1).cpu().numpy()
    return score, part


# ---------------------------------------------------------------- 方法


def onehot(labels: np.ndarray, k: int) -> np.ndarray:
    return np.eye(k, dtype=np.float32)[labels]


def fixed(data: dict, column: str | dict) -> dict:
    """固定区分。`column` は stage2 の列名か、image → 区分 の Series の dict。"""
    labels = {}
    for s in SPLITS:
        f = data[s]["frame"]
        labels[s] = f[column].to_numpy() if isinstance(column, str) else column[s].loc[f.image].to_numpy()
    k = int(max(v.max() for v in labels.values())) + 1
    return {s: onehot(labels[s], k) for s in SPLITS}, k


def random_partition(data: dict, k: int, seed: int) -> dict:
    rng = np.random.default_rng(1000 + seed)
    return {s: onehot(rng.integers(0, k, len(data[s]["y"])), k) for s in SPLITS}


def search(data: dict, candidates: dict) -> tuple[str, pd.DataFrame]:
    """候補区分ごとに、固定区分の head と同じ K のランダム区分の head の val 損失の差を測り、最大のものを選ぶ。"""
    rows = []
    random_loss = {}
    for k in SEARCH_K:
        random_loss[k] = np.mean([select(data, random_partition(data, k, r), k, None)[1] for r in range(2)])
    for name, cols in candidates.items():
        assign, k = fixed(data, cols)
        _, loss = select(data, assign, k, None)
        rows.append({"candidate": name, "k": k, "val_loss": loss, "gain": random_loss[k] - loss})
    table = pd.DataFrame(rows)
    return table.loc[table.gain.idxmax(), "candidate"], table


def methods(data: dict, candidates: dict, seed: int) -> dict:
    """方法ごとに (assign, k, gate_dim) を返す。search は候補を val で選んだ結果。"""
    out = {"const": (*fixed(data, "const"), None), "random8": (*fixed(data, "random8"), None)}
    out["gmm8"] = (*fixed(data, "gmm8"), None)
    for gate in ("z", "hz"):
        for k in (2, 8):
            assign = {s: data[s][gate] for s in SPLITS}
            out[f"moe_{gate}_k{k}"] = (assign, k, data["train"][gate].shape[1])
    chosen, table = search(data, candidates)
    table.assign(seed=seed).to_csv(OUT / f"search_{data['field']}_s{seed}.csv", index=False)
    out["search"] = (*fixed(data, candidates[chosen]), None)
    out["search"] += (chosen,)
    return out


# ---------------------------------------------------------------- 評価


def safe_auc(y, s):
    return roc_auc_score(y, s) if 0 < y.sum() < len(y) else np.nan


def describe(d: dict, score: np.ndarray, part: np.ndarray, field: str) -> dict:
    y, b, f = d["y"], d["b"], d["frame"]
    row = {"auroc": roc_auc_score(y, score), "n_parts_used": int(len(np.unique(part)))}
    if field == "rule":
        share = pd.Series(b).groupby(part).mean()
        covered = np.isin(part, share.index[share > 0.5])
        row.update(
            {
                "auroc_a": safe_auc(y[~b], score[~b]),
                "auroc_b": safe_auc(y[b], score[b]),
                "ari_b": ari(b, part),
                "b_coverage": float(covered[b].mean()),
                "b_purity": float(b[covered].mean()) if covered.any() else np.nan,
            }
        )
    else:
        g = sex_age(f, "sex")
        aucs = group_aucs(y, score, g)
        known = g >= 0
        row.update(
            {
                "worst_sexage": min(aucs.values()),
                "nmi_site": nmi(f.dataset, part),
                "nmi_label": nmi(y, part),
                "nmi_sexage": nmi(g[known], part[known]),
            }
        )
    return row


def marker_auc(d: dict) -> float:
    """x からマーカーの有無を当てる線形 probe の test AUROC（x がマーカーの方向を保っているかの確認）。"""
    from sklearn.linear_model import LogisticRegression

    probe = LogisticRegression(max_iter=2000).fit(d["train"]["x"], d["train"]["marked"])
    return roc_auc_score(d["test"]["marked"], probe.decision_function(d["test"]["x"]))


def run_field(field: str, candidates: dict) -> tuple[pd.DataFrame, dict]:
    rows, scores = [], {}
    for seed in SEEDS:
        data = load(field, seed)
        data["field"] = field
        reduce(data)
        check = marker_auc(data) if field == "rule" else np.nan
        for name, spec in methods(data, candidates, seed).items():
            assign, k, gate_dim = spec[:3]
            model, val = select(data, assign, k, gate_dim)
            score, part = predict(model, data["test"], assign["test"])
            row = {"field": field, "seed": seed, "method": name, "k": k, "val_loss": val, "marker_auc": check}
            row["chosen"] = spec[3] if len(spec) > 3 else ""
            row.update(describe(data["test"], score, part, field))
            rows.append(row)
            scores[(seed, name)] = (score, part)
            print(field, seed, name, {key: round(v, 3) for key, v in row.items() if isinstance(v, float)})
        scores[(seed, "_y")] = data["test"]["y"]
        scores[(seed, "_frame")] = data["test"]["frame"]
    return pd.DataFrame(rows), scores


def contrast(field: str, scores: dict, names: list[str]) -> pd.DataFrame:
    """定数との seed 平均の差を、病変単位の対応のある bootstrap で。rule は B・A・全体、real は全体・性別 × 年齢の最悪群。"""
    frame = scores[(SEEDS[0], "_frame")]
    y = scores[(SEEDS[0], "_y")]
    lesions = frame.lesion_id.to_numpy()
    uniq = np.unique(lesions)
    members = {lesion: np.flatnonzero(lesions == lesion) for lesion in uniq}
    b = minority(frame)
    g = sex_age(frame, "sex")
    rng = np.random.default_rng(0)

    def metrics(idx, score):
        yy, ss = y[idx], score[idx]
        if field == "rule":
            bb = b[idx]
            return {"auroc": safe_auc(yy, ss), "auroc_b": safe_auc(yy[bb], ss[bb]), "auroc_a": safe_auc(yy[~bb], ss[~bb])}
        aucs = group_aucs(yy, ss, g[idx])
        return {"auroc": safe_auc(yy, ss), "worst_sexage": min(aucs.values()) if aucs else np.nan}

    def diff(idx, name):
        per_seed = []
        for seed in SEEDS:
            m = metrics(idx, scores[(seed, name)][0])
            c = metrics(idx, scores[(seed, "const")][0])
            per_seed.append({key: m[key] - c[key] for key in m})
        return pd.DataFrame(per_seed).mean()

    full = np.arange(len(y))
    boots = [np.concatenate([members[u] for u in rng.choice(uniq, len(uniq))]) for _ in range(BOOT)]
    rows = []
    for name in names:
        point = diff(full, name)
        dist = pd.DataFrame([diff(idx, name) for idx in boots])
        for metric in point.index:
            lo, hi = np.nanpercentile(dist[metric], [2.5, 97.5])
            rows.append({"field": field, "method": name, "metric": metric, "diff": point[metric], "lo": lo, "hi": hi})
    return pd.DataFrame(rows)


def stability(field: str, scores: dict, names: list[str]) -> pd.DataFrame:
    """1 段目の seed が違う run どうしの、test での区分の ARI の平均。"""
    rows = []
    for name in names:
        parts = [scores[(seed, name)][1] for seed in SEEDS]
        pairs = [ari(parts[i], parts[j]) for i in range(len(SEEDS)) for j in range(i + 1, len(SEEDS))]
        rows.append({"field": field, "method": name, "seed_ari": float(np.mean(pairs))})
    return pd.DataFrame(rows)


def alignment() -> None:
    """選び方の検算：候補ごとの val の利得と、B との重なりの関係。正解（gmm8 と ARI>0.9）を候補から外したら何を選ぶか。

    出力: `results/d3/search_alignment.csv`（候補ごと）、`results/d3/search_without_truth.csv`（外したときの選択と test）
    """
    from scipy.stats import spearmanr

    candidates = candidate_partitions()
    train = pd.read_csv(FRAMES / "train.csv")
    b = minority(train)
    props = []
    for name, cols in candidates.items():
        part = cols["train"].loc[train.image].to_numpy()
        share = pd.Series(b).groupby(part).mean()
        covered = np.isin(part, share.index[share > 0.5])
        props.append(
            {
                "candidate": name,
                "ari_gmm8": ari(train.gmm8, part),
                "nmi_b": nmi(b, part),
                "b_coverage": float(covered[b].mean()),
                "b_purity": float(b[covered].mean()) if covered.any() else np.nan,
            }
        )
    props = pd.DataFrame(props)
    rows, picks = [], []
    for seed in SEEDS:
        table = pd.read_csv(OUT / f"search_rule_s{seed}.csv").merge(props, on="candidate")
        rho = spearmanr(table.gain, table.nmi_b).statistic
        rows.append(table.assign(spearman_gain_nmi_b=rho))
        rest = table[table.ari_gmm8 < 0.9]
        chosen = rest.loc[rest.gain.idxmax()]
        data = load("rule", seed)
        reduce(data)
        out = {}
        for name, spec in (("const", fixed(data, "const")), ("chosen", fixed(data, candidates[chosen.candidate]))):
            model, _ = select(data, spec[0], spec[1], None)
            score, part = predict(model, data["test"], spec[0]["test"])
            out[name] = describe(data["test"], score, part, "rule")
        picks.append(
            {
                "seed": seed,
                "chosen": chosen.candidate,
                "ari_gmm8": chosen.ari_gmm8,
                "nmi_b": chosen.nmi_b,
                "b_coverage_train": chosen.b_coverage,
                "b_purity_train": chosen.b_purity,
                "spearman_gain_nmi_b": rho,
                "auroc_b_const": out["const"]["auroc_b"],
                "auroc_b_chosen": out["chosen"]["auroc_b"],
                "auroc_chosen_minus_const": out["chosen"]["auroc"] - out["const"]["auroc"],
            }
        )
    pd.concat(rows, ignore_index=True).to_csv(OUT / "search_alignment.csv", index=False)
    picks = pd.DataFrame(picks)
    picks.to_csv(OUT / "search_without_truth.csv", index=False)
    print(picks.round(3).to_string(index=False))


def main() -> None:
    import sys

    if "--alignment" in sys.argv:
        alignment()
        return
    torch.set_num_threads(8)
    OUT.mkdir(parents=True, exist_ok=True)
    CACHE_OUT.mkdir(parents=True, exist_ok=True)
    candidates = candidate_partitions()
    tables, contrasts, stabs = [], [], []
    for field in ("rule", "real"):
        table, scores = run_field(field, candidates)
        names = [m for m in table.method.unique() if m != "const"]
        tables.append(table)
        contrasts.append(contrast(field, scores, names))
        stabs.append(stability(field, scores, list(table.method.unique())))
        for seed in SEEDS:
            arrays = {f"{m}_score": scores[(seed, m)][0] for m in table.method.unique()}
            arrays.update({f"{m}_part": scores[(seed, m)][1] for m in table.method.unique()})
            np.savez(CACHE_OUT / f"{field}_s{seed}.npz", **arrays)
    runs = pd.concat(tables, ignore_index=True)
    runs.to_csv(OUT / "runs.csv", index=False)
    numeric = runs.select_dtypes("number").columns.drop(["seed", "k"])
    summary = runs.groupby(["field", "method"], sort=False)[list(numeric)].mean().reset_index()
    summary.to_csv(OUT / "summary.csv", index=False)
    pd.concat(contrasts, ignore_index=True).to_csv(OUT / "contrast.csv", index=False)
    pd.concat(stabs, ignore_index=True).to_csv(OUT / "stability.csv", index=False)
    print(summary.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
