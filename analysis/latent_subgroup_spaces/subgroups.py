"""潜在サブグループを、空間 × 群の作り方ごとに作り、同じ物差しで測る関数群。

3 本の notebook（`01_initial_spaces`・`02_space_comparison`・`03_calibration_bias`）が共有する。どの空間・K・閾値を使うかの判断は
notebook が持ち、ここは計算だけを置く。

群は val で作り（クラスタの当てはめ、監査器の学習、最悪群の選択）、test で評価する。同じデータで作って
測ると、最悪の群を選んだことだけで値が悪く出る（勝者の呪い）。

物差しは `docs/research-premises.md` に合わせる。CheXpert の target は No Finding（1 = 所見なし）なので、
害の大きい側は FPR（所見のある画像を「所見なし」と出す見落とし）。しきい値は 0.5（logit 差 0）で固定する。
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import rootutils
from scipy.special import expit
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, roc_auc_score
from sklearn.mixture import GaussianMixture
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeRegressor

ROOT = rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

PACKAGE = Path(__file__).resolve().parent
CACHE = PACKAGE / "cache"
SPLITS_DIR = ROOT / "data" / "chexpert" / "splits_race_known"
# 文脈ラベルと RAD-DINO は latent_bias_hypernet が作った cache を読み取り専用で使う。
# val・test は race-known と同じ行順（load_split が検算する）。
CONTEXT_DIR = ROOT / "analysis" / "archive" / "latent_bias_hypernet" / "cache" / "context"
RADDINO_DIR = ROOT / "analysis" / "archive" / "latent_bias_hypernet" / "cache" / "embed" / "raddino"
# 病種は impression から CheXbert で付けたラベル。target（No Finding）と 98% 一致するのはこの file だけ。
DISEASE_LABELS = ROOT / "data" / "chexpert" / "label" / "impression_fixed.json"
DISEASES = (
    "Enlarged Cardiomediastinum",
    "Cardiomegaly",
    "Lung Opacity",
    "Lung Lesion",
    "Edema",
    "Consolidation",
    "Pneumonia",
    "Atelectasis",
    "Pneumothorax",
    "Pleural Effusion",
    "Pleural Other",
    "Fracture",
    "Support Devices",
)
CONTEXT_FLAGS = ("followup", "device", "pacer", "picc", "central_line", "sternotomy", "surgical_clip")
RACE_NAMES = {0: "White", 1: "Others", 2: "Asian", 3: "Black"}
# 群別の率を数える下限（data_pipeline/README.md の最小セル）。FPR は陰性、TPR は陽性の件数で数える。
MIN_CELL = 20


# ----- 読み込み -----


@dataclass
class Split:
    """1 run・1 split の行ごとの表と空間。`frame` の行は split CSV と同じ順。"""

    frame: pd.DataFrame
    condition: np.ndarray
    features: np.ndarray
    raddino: np.ndarray


def _patient(image: pd.Series) -> pd.Series:
    return image.str.split("/").str[1]


def load_context(split: str) -> pd.DataFrame:
    """属性・文脈ラベル・病種を、split CSV の行順で返す。

    病種の列は `dz_<名前>` で、CheXbert の 1（陽性）だけを真にする。不確実（-1）と未言及は偽。
    """
    base = pd.read_csv(SPLITS_DIR / f"{split}.csv")
    context = pd.read_csv(CONTEXT_DIR / f"{split}.csv")
    if not (context.image.to_numpy() == base.image.to_numpy()).all():
        raise ValueError(f"{split}: 文脈ラベルの行順が split CSV と違う")
    labels = pd.read_json(DISEASE_LABELS, lines=True).rename(columns={"path_to_image": "image"})
    labels = labels[["image", *DISEASES]].drop_duplicates("image")
    merged = base[["image"]].merge(labels, on="image", how="left", validate="one_to_one")
    frame = base[["image", "target", "age", "sex", "race_group", "frontal_lateral", "ap_pa", "ethnicity"]].copy()
    frame["patient"] = _patient(frame.image)
    frame["age65"] = (frame.age >= 65).astype(int)
    frame["attr8"] = frame.age65 * 4 + frame.race_group
    frame["report_type"] = context.report_type.to_numpy()
    for flag in CONTEXT_FLAGS:
        frame[flag] = context[flag].to_numpy(dtype=bool)
    for disease in DISEASES:
        frame[f"dz_{disease}"] = (merged[disease] == 1).to_numpy()
    return frame


def load_split(run_id: str, split: str) -> Split:
    """`extract.py` の cache と文脈ラベル・RAD-DINO を、行順を検算して束ねる。"""
    with np.load(CACHE / f"{run_id}_{split}.npz") as cached:
        arrays = {key: cached[key] for key in cached.files}
    frame = load_context(split)
    raddino = np.load(RADDINO_DIR / f"{split}.npz", allow_pickle=True)
    for images in (arrays["image"], raddino["image"].astype(str)):
        if not (images == frame.image.to_numpy()).all():
            raise ValueError(f"{run_id} {split}: cache の行順が split CSV と違う")
    if not (arrays["target"] == frame.target.to_numpy()).all():
        raise ValueError(f"{run_id} {split}: target が split CSV と違う")
    frame["margin"] = arrays["margin"]
    frame["p"] = expit(arrays["margin"])
    frame["pred"] = (arrays["margin"] >= 0).astype(int)
    frame["cell"] = arrays["cell"]
    return Split(
        frame, arrays["condition"], arrays["features"].astype(np.float32), raddino["embedding"].astype(np.float32)
    )


def load_cells(run_id: str) -> tuple[pd.DataFrame, np.ndarray, dict[str, np.ndarray]]:
    """セル表・セルごとの condition・stage ごとの ΔW Gram。"""
    with np.load(CACHE / f"{run_id}_cells.npz") as cached:
        columns = json.loads(str(cached["metadata"]))["columns"]
        table = pd.DataFrame({key: cached[f"cell_{key}"] for key in [*columns, "train_count", "train_positives"]})
        grams = {key.removeprefix("gram_"): cached[key] for key in cached.files if key.startswith("gram_")}
        condition = cached["condition"]
    return table, condition, grams


# ----- 空間 -----


def kernel_coordinates(gram: np.ndarray, weight: np.ndarray, variance: float = 0.99, max_dims: int = 32) -> np.ndarray:
    """ΔW の Gram 行列から、ΔW の距離を保つセルの座標を作る（重み付き kernel PCA）。

    train の件数で重み付けして中心化する。件数の多いセルが座標軸を決めるのは、行単位で k-means を
    当てるときと同じ扱いにするため。累積寄与 `variance` までの成分（最大 `max_dims`）を残す。

    Args:
        gram: `[n_cells, n_cells]`。
        weight: セルごとの重み（train の件数）。0 のセルも座標は持つ。
    Returns:
        `[n_cells, d]` の座標。
    """
    w = weight / weight.sum()
    mean_row = gram @ w
    centered = gram - mean_row[:, None] - mean_row[None, :] + w @ gram @ w
    root = np.sqrt(w)
    values, vectors = np.linalg.eigh(root[:, None] * centered * root[None, :])
    order = np.argsort(values)[::-1]
    values, vectors = np.clip(values[order], 0, None), vectors[:, order]
    keep = min(int(np.searchsorted(np.cumsum(values) / values.sum(), variance)) + 1, max_dims)
    # 重み付き中心化した Gram を、射影 centered @ (√w V) / √λ で全セルへ広げる。
    projection = root[:, None] * vectors[:, :keep] / np.sqrt(values[:keep])
    return centered @ projection


def reduce(val: np.ndarray, test: np.ndarray, dims: int, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """val で標準化と PCA を当て、test に同じ変換を当てる。"""
    scaler = StandardScaler().fit(val)
    pca = PCA(dims, random_state=seed).fit(scaler.transform(val))
    return pca.transform(scaler.transform(val)), pca.transform(scaler.transform(test))


def attribute_matrix(frame: pd.DataFrame) -> np.ndarray:
    """属性（年齢、性別、race 6 区分、ethnicity、正面・側面、AP・PA）の one-hot ＋ 年齢。"""
    categorical = pd.get_dummies(frame[["sex", "race_group", "ethnicity", "frontal_lateral", "ap_pa"]].astype(str))
    return np.column_stack([frame.age.to_numpy() / 100.0, categorical.to_numpy(dtype=float)])


# ----- 群の作り方（val で作り、val・test の群番号を返す） -----


def kmeans_groups(val: np.ndarray, test: np.ndarray, k: int, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """k-means。属性だけの空間では同じセルの行が同じ点に重なるので、行単位の当てはめは件数で重み付けしたのと同じ。"""
    model = KMeans(k, n_init=10, random_state=seed).fit(val)
    return model.labels_, model.predict(test)


def gmm_groups(val: np.ndarray, test: np.ndarray, k: int, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """全共分散の GMM（LHCF と同じ形）。"""
    model = GaussianMixture(k, covariance_type="full", n_init=3, random_state=seed, reg_covar=1e-4).fit(val)
    return model.predict(val), model.predict(test)


def gmm_bic_k(val: np.ndarray, ks: range, seed: int = 0) -> pd.DataFrame:
    """K ごとの BIC。LHCF は BIC 最小の K を使う。"""
    rows = []
    for k in ks:
        model = GaussianMixture(k, covariance_type="full", n_init=1, random_state=seed, reg_covar=1e-4).fit(val)
        rows.append({"k": k, "bic": model.bic(val)})
    return pd.DataFrame(rows)


def george_groups(val, test, y_val, y_test, k: int, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """GEORGE：真のクラスごとに、学習済みモデルの特徴へ GMM を当てて部分クラスに分ける。

    原典は UMAP の上で K を選ぶが、ここでは PCA 後の空間で K を固定して振る。群番号は クラス × K + 部分クラス。
    """
    out_val = np.empty(len(val), dtype=int)
    out_test = np.empty(len(test), dtype=int)
    for label in (0, 1):
        fit_rows, apply_rows = y_val == label, y_test == label
        a, b = gmm_groups(val[fit_rows], test[apply_rows], k, seed)
        out_val[fit_rows], out_test[apply_rows] = label * k + a, label * k + b
    return out_val, out_test


def miss_auditor(val, test, frame_val: pd.DataFrame, kind: str, param: int, seed: int = 0):
    """監査器：val の陰性で「見落とし（0.5 で所見なしと出す）」を入力から予測し、その予測で群に切る。

    評価の主指標が FPR なので、監査の対象も FPR に合わせて陰性の見落としにする（overview の「誤差の信号」）。
    - `ridge`：予測した見落とし率の分位で `param` 個に切る（Kim et al. の ridge 監査器の形）
    - `tree`：回帰木の葉を群にする。`param` は葉の最小件数（陰性の数）。深さは制限しない

    Returns:
        val・test の群番号と、test の予測見落とし率。
    """
    negatives = frame_val.target.to_numpy() == 0
    miss = frame_val.pred.to_numpy()[negatives].astype(float)
    if kind == "ridge":
        model = Ridge(alpha=10.0).fit(val[negatives], miss)
        score_val, score_test = model.predict(val), model.predict(test)
        edges = np.quantile(score_val[negatives], np.linspace(0, 1, param + 1)[1:-1])
        return np.digitize(score_val, edges), np.digitize(score_test, edges), score_test
    model = DecisionTreeRegressor(min_samples_leaf=param, random_state=seed).fit(val[negatives], miss)
    leaves = {leaf: index for index, leaf in enumerate(np.unique(model.apply(val[negatives])))}
    to_group = np.vectorize(leaves.get)
    return to_group(model.apply(val)), to_group(model.apply(test)), model.predict(test)


# ----- 物差し -----


def group_table(groups: np.ndarray, frame: pd.DataFrame) -> pd.DataFrame:
    """群ごとの件数・陽性率・0.5 での FPR と TPR・AUROC。下限未満の率は NaN にする。"""
    rows = []
    for group in np.unique(groups):
        part = frame[groups == group]
        y, pred = part.target.to_numpy(), part.pred.to_numpy()
        n_pos, n_neg = int(y.sum()), int((1 - y).sum())
        both = n_pos >= MIN_CELL and n_neg >= MIN_CELL
        rows.append(
            {
                "group": group,
                "n": len(part),
                "n_pos": n_pos,
                "n_neg": n_neg,
                "prevalence": y.mean(),
                "fpr": pred[y == 0].mean() if n_neg >= MIN_CELL else np.nan,
                "tpr": pred[y == 1].mean() if n_pos >= MIN_CELL else np.nan,
                "auroc": roc_auc_score(y, part.margin) if both else np.nan,
            }
        )
    return pd.DataFrame(rows)


def worst_fpr_null(groups: np.ndarray, frame: pd.DataFrame, draws: int = 200, seed: int = 0) -> np.ndarray:
    """群の大きさを保って陰性の群番号を並べ替えたときの、最悪 FPR の分布（帰無分布）。

    小さい群ほど FPR は偶然で大きく振れるので、最悪群の値はそれだけで全体より悪く出る。ここを超えた分だけを
    「群が偏りを捉えた」と読む。陰性だけを並べ替えるので、陽性率の違いは帰無分布に入らない。
    患者内の相関は並べ替えで崩れるため、帰無分布はやや狭めに出る（行単位）。
    """
    rng = np.random.default_rng(seed)
    negatives = frame.target.to_numpy() == 0
    g, miss = groups[negatives], frame.pred.to_numpy()[negatives]
    labels, counts = np.unique(g, return_counts=True)
    eligible = counts >= MIN_CELL
    worst = np.empty(draws)
    for draw in range(draws):
        shuffled = rng.permutation(miss)
        sums = np.bincount(np.searchsorted(labels, g), weights=shuffled, minlength=len(labels))
        worst[draw] = (sums / counts)[eligible].max()
    return worst


def summarize_partition(name: str, val_groups, test_groups, val_frame, test_frame, null_draws: int = 200) -> dict:
    """1 つの群の作り方を、test の最悪 FPR と帰無分布・val で選んだ最悪群の test 値で要約する。"""
    val_table = group_table(val_groups, val_frame)
    test_table = group_table(test_groups, test_frame).set_index("group")
    chosen = val_table.loc[val_table.fpr.idxmax(), "group"]
    null = worst_fpr_null(test_groups, test_frame, null_draws)
    overall = test_frame.pred[test_frame.target == 0].mean()
    valid = test_table.dropna(subset=["fpr"])
    return {
        "partition": name,
        "groups": len(test_table),
        "groups_valid": len(valid),
        "min_n_neg": int(valid.n_neg.min()),
        "overall_fpr": overall,
        "worst_fpr": valid.fpr.max(),
        "null_worst_mean": null.mean(),
        "null_worst_p95": np.quantile(null, 0.95),
        "excess_over_null": valid.fpr.max() - null.mean(),
        "p_value": (1 + (null >= valid.fpr.max()).sum()) / (1 + len(null)),
        "val_chosen_test_fpr": test_table.fpr.get(chosen, np.nan),
        "val_chosen_n_neg": int(test_table.n_neg.get(chosen, 0)),
        "fpr_sd": np.sqrt(np.average((valid.fpr - overall) ** 2, weights=valid.n_neg)),
        "prevalence_range": valid.prevalence.max() - valid.prevalence.min(),
        "worst_auroc": valid.auroc.min(),
    }


# ----- 群の中身 -----


def explained_by(groups: np.ndarray, frame: pd.DataFrame, columns: list[str]) -> float:
    """群の番号が、指定した列の組み合わせ（交差セル）でどれだけ決まるか（NMI）。"""
    cells = frame[columns].astype(str).agg("|".join, axis=1)
    return normalized_mutual_info_score(cells, groups)


def standardized_excess(groups: np.ndarray, frame: pd.DataFrame, strata: list[str]) -> pd.DataFrame:
    """群ごとの見落とし数の O/E。E は「層ごとの全体の FPR」を群の陰性に当てた期待値。

    層を属性にすれば「属性の構成で説明できる分」、病種や文脈にすれば「所見の構成で説明できる分」を除いた
    残りが O/E に残る。O/E が 1 に近づくほど、その層で群の FPR の違いが説明できる。
    """
    negatives = frame[frame.target == 0].copy()
    negatives["group"] = groups[frame.target.to_numpy() == 0]
    key = negatives[strata].astype(str).agg("|".join, axis=1)
    negatives["expected"] = negatives.groupby(key).pred.transform("mean")
    table = negatives.groupby("group").agg(
        n_neg=("pred", "size"), observed=("pred", "sum"), expected=("expected", "sum")
    )
    table["o_e"] = table.observed / table.expected
    return table


def enrichment(mask: np.ndarray, frame: pd.DataFrame, factors: list[str]) -> pd.DataFrame:
    """陰性のうち、群の中での各要因の割合と、陰性全体での割合の比。"""
    negatives = frame.target.to_numpy() == 0
    inside, everyone = frame[negatives & mask], frame[negatives]
    rows = [
        {"factor": factor, "share_in_group": inside[factor].mean(), "share_overall": everyone[factor].mean()}
        for factor in factors
    ]
    table = pd.DataFrame(rows)
    table["ratio"] = table.share_in_group / table.share_overall
    return table


def seed_agreement(groups_by_seed: dict[int, np.ndarray]) -> float:
    """seed 間の ARI の平均（test の同じ行どうし）。"""
    seeds = sorted(groups_by_seed)
    pairs = [(a, b) for i, a in enumerate(seeds) for b in seeds[i + 1 :]]
    return float(np.mean([adjusted_rand_score(groups_by_seed[a], groups_by_seed[b]) for a, b in pairs]))


# ----- 局所の偏り（クラスタより小さい単位） -----


def knn_scan(val_space, test_space, val_frame, test_frame, k: int, draws: int = 99, seed: int = 0) -> dict:
    """val の陰性を中心に k 近傍の球を作り、見落としが最も集中する球を走査統計で選んで test で測る。

    球の得点は二項の尤度比（Kulldorff の走査統計、球の中の率が全体より高い向きだけ）。val の陰性の
    見落としを並べ替えて最大得点の帰無分布を作り、p 値を出す。選んだ球は半径をそのまま test に当てる
    （test の陰性のうち、中心からの距離が半径以内のもの）。

    Returns:
        val での得点・p 値・球の中の FPR と、test の同じ球の陰性数・FPR、中心の行番号。
    """
    rng = np.random.default_rng(seed)
    neg_val = val_frame.target.to_numpy() == 0
    centers = val_space[neg_val]
    miss = val_frame.pred.to_numpy()[neg_val].astype(float)
    neighbors = NearestNeighbors(n_neighbors=k).fit(centers)
    distance, index = neighbors.kneighbors(centers)
    base = miss.mean()
    observed = _scan_scores(miss[index].sum(1), k, len(miss), miss.sum(), base)
    best = int(np.argmax(observed))
    null = np.array(
        [_scan_scores(rng.permutation(miss)[index].sum(1), k, len(miss), miss.sum(), base).max() for _ in range(draws)]
    )
    neg_test = test_frame.target.to_numpy() == 0
    test_distance = np.linalg.norm(test_space[neg_test] - centers[best], axis=1)
    inside = test_distance <= distance[best, -1]
    return {
        "k": k,
        "val_fpr_in_ball": miss[index[best]].mean(),
        "val_overall_fpr": base,
        "scan_score": observed[best],
        "p_value": (1 + (null >= observed[best]).sum()) / (1 + draws),
        "test_n_neg_in_ball": int(inside.sum()),
        "test_fpr_in_ball": test_frame.pred.to_numpy()[neg_test][inside].mean() if inside.any() else np.nan,
        "center_row": int(np.flatnonzero(neg_val)[best]),
        "ball_rows_val": np.flatnonzero(neg_val)[index[best]],
        "ball_rows_test": np.flatnonzero(neg_test)[inside],
    }


def _scan_scores(hits: np.ndarray, size: int, total: int, total_hits: float, base: float) -> np.ndarray:
    """球ごとの二項尤度比（率が全体より高い球だけ正、ほかは 0）。"""
    inside_rate = np.clip(hits / size, 1e-12, 1 - 1e-12)
    outside_rate = np.clip((total_hits - hits) / (total - size), 1e-12, 1 - 1e-12)
    overall = np.clip(base, 1e-12, 1 - 1e-12)
    loglik = (
        hits * np.log(inside_rate)
        + (size - hits) * np.log(1 - inside_rate)
        + (total_hits - hits) * np.log(outside_rate)
        + (total - size - total_hits + hits) * np.log(1 - outside_rate)
    )
    null = total_hits * np.log(overall) + (total - total_hits) * np.log(1 - overall)
    return np.where(inside_rate > overall, loglik - null, 0.0)


# ----- 得点を補正したずれ（較正のずれ） -----
#
# しきい値での FPR は得点の関数なので、画像を見る空間（モデルの特徴・RAD-DINO）で FPR の悪い群を探すと、
# 決定境界の「所見なし」側の画像（難しい陰性）を切り出すだけになる。群に特有のずれは、得点をそろえたうえで
# 残る分、つまり再較正した確率に対する陽性数の O/E で測る（multiaccuracy・calibration-in-the-small の形）。
# O/E < 1 は、その群でモデルが No Finding を出しすぎている（見落としの側に偏っている）ことを表す。


def recalibrate(val_frame: pd.DataFrame, *frames: pd.DataFrame) -> list[np.ndarray]:
    """val 全体で logit 差を再較正し、各 frame の確率を返す。

    学習の inverse class weight で得点は陽性側にずれているので、そのままの p では O/E が全体で 1 にならない。
    Platt（logit 差の 1 次）では得点の帯ごとのずれが残る（得点の十分位で O/E の SD が帰無分布の 2 倍）ため、
    logit 差の 3 次スプライン（節点 8）のロジスティック回帰にする。群を見ない単調に近い変換で、群間の差は作らない。
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import SplineTransformer

    model = make_pipeline(
        SplineTransformer(n_knots=8, degree=3, extrapolation="linear"), LogisticRegression(C=100.0, max_iter=2000)
    )
    model.fit(val_frame[["margin"]].to_numpy(), val_frame.target.to_numpy())
    return [model.predict_proba(frame[["margin"]].to_numpy())[:, 1] for frame in (val_frame, *frames)]


MIN_EXPECTED = 10.0  # O/E を数える群の期待陽性数の下限


def calibration_table(groups: np.ndarray, y: np.ndarray, p: np.ndarray) -> pd.DataFrame:
    """群ごとの陽性数 O、期待陽性数 E、log O/E と平均残差 (O − E)/n。期待陽性数が下限未満の群は NaN にする。

    multiaccuracy（Kim et al.）が抑える量そのものは平均残差（差）のほう。log O/E は比なので、
    同じ差でも陽性率の低い群ほど大きく出る。どちらで群を選ぶかで、選ばれる群が変わりうる。
    """
    frame = pd.DataFrame({"group": groups, "y": y, "p": p})
    table = frame.groupby("group").agg(n=("y", "size"), observed=("y", "sum"), expected=("p", "sum"))
    enough = table.expected >= MIN_EXPECTED
    table["log_oe"] = np.where(enough, np.log((table.observed + 0.5) / (table.expected + 0.5)), np.nan)
    table["mean_residual"] = np.where(enough, (table.observed - table.expected) / table.n, np.nan)
    return table


def _oe_dispersion(table: pd.DataFrame) -> float:
    valid = table.dropna(subset=["log_oe"])
    return float(np.sqrt(np.average(valid.log_oe**2, weights=valid.expected)))


def summarize_calibration(name, val_groups, test_groups, y_val, p_val, y_test, p_test, draws: int = 200, seed: int = 0):
    """群の較正のずれを要約する。

    - `oe_sd`：test の log O/E の、期待陽性数で重み付けした SD
    - `oe_sd_null`：y を Bernoulli(p) で引き直したときの同じ量の平均（較正が完全でも出る分）
    - `val_chosen_test_log_oe`：val で O/E が最小（No Finding を最も出しすぎ）の群の、test の log O/E
    """
    rng = np.random.default_rng(seed)
    test_table = calibration_table(test_groups, y_test, p_test)
    val_table = calibration_table(val_groups, y_val, p_val)
    chosen = val_table.log_oe.idxmin()
    null = [
        _oe_dispersion(calibration_table(test_groups, rng.random(len(p_test)) < p_test, p_test)) for _ in range(draws)
    ]
    observed = _oe_dispersion(test_table)
    return {
        "partition": name,
        "groups_valid": int(test_table.log_oe.notna().sum()),
        "oe_sd": observed,
        "oe_sd_null": float(np.mean(null)),
        "oe_sd_p": (1 + sum(value >= observed for value in null)) / (1 + draws),
        "min_log_oe": test_table.log_oe.min(),
        "val_chosen_val_log_oe": val_table.log_oe.min(),
        "val_chosen_test_log_oe": test_table.log_oe.get(chosen, np.nan),
        "val_chosen_test_n": int(test_table.n.get(chosen, 0)),
        "val_chosen_test_expected": float(test_table.expected.get(chosen, np.nan)),
    }


def rates(y: np.ndarray, pred: np.ndarray, margin: np.ndarray) -> dict:
    """0.5 での FPR・TPR と AUROC。下限未満の率は NaN にする。"""
    n_pos, n_neg = int(y.sum()), int((1 - y).sum())
    both = n_pos >= MIN_CELL and n_neg >= MIN_CELL
    return {
        "n": len(y),
        "n_pos": n_pos,
        "n_neg": n_neg,
        "prevalence": y.mean() if len(y) else np.nan,
        "fpr": pred[y == 0].mean() if n_neg >= MIN_CELL else np.nan,
        "tpr": pred[y == 1].mean() if n_pos >= MIN_CELL else np.nan,
        "auroc": roc_auc_score(y, margin) if both else np.nan,
    }


def performance_of(mask: np.ndarray, frame: pd.DataFrame, p: np.ndarray) -> dict:
    """1 つの群の性能と、補集合との差（gap = 群 − 補集合）。較正のずれ（log O/E・平均残差）も並べる。

    較正のずれが大きい群が、主指標（0.5 の FPR）でも悪いとは限らない。得点の低い群で確率を高めに出しても、
    0.5 を超えなければ見落としは増えない。群の AUROC は、得点で選んだ群ほど得点の幅が狭く、低めに出る。

    Args:
        mask: test の行のうち群に入るもの。
        frame: test の行（`target`・`pred`・`margin`）。
        p: 再較正した確率。

    Returns:
        dict: 群の `n`・`fpr`・`tpr`・`auroc` など、補集合の `rest_*`、差の `gap_*`、`log_oe`・`mean_residual`。
    """
    y, pred, margin = frame.target.to_numpy(), frame.pred.to_numpy(), frame.margin.to_numpy()
    inside = rates(y[mask], pred[mask], margin[mask])
    rest = rates(y[~mask], pred[~mask], margin[~mask])
    observed, expected = y[mask].sum(), p[mask].sum()
    row = inside | {f"rest_{key}": value for key, value in rest.items() if key != "n"}
    for key in ("fpr", "tpr", "auroc"):
        row[f"gap_{key}"] = inside[key] - rest[key]
    row["log_oe"] = np.log((observed + 0.5) / (expected + 0.5))
    row["mean_residual"] = (observed - expected) / max(mask.sum(), 1)
    return row


def residual_auditor(val, test, y_val, p_val, kind: str, param: int, seed: int = 0):
    """Kim et al. の形の監査器：再較正した残差 y − p を入力から予測し、その予測で群に切る。

    `ridge` は予測残差の分位で `param` 個、`tree` は葉（最小件数 `param`）を群にする。
    """
    residual = y_val - p_val
    if kind == "ridge":
        model = Ridge(alpha=10.0).fit(val, residual)
        score_val, score_test = model.predict(val), model.predict(test)
        edges = np.quantile(score_val, np.linspace(0, 1, param + 1)[1:-1])
        return np.digitize(score_val, edges), np.digitize(score_test, edges)
    model = DecisionTreeRegressor(min_samples_leaf=param, random_state=seed).fit(val, residual)
    leaves = {leaf: index for index, leaf in enumerate(np.unique(model.apply(val)))}
    to_group = np.vectorize(leaves.get)
    return to_group(model.apply(val)), to_group(model.apply(test))


def knn_residual_scan(
    val_space, test_space, y_val, p_val, y_test, p_test, k: int, draws: int = 99, seed: int = 0
) -> dict:
    """k 近傍の球ごとの残差の z 値（Σ(y − p) / √Σp(1 − p)）で、No Finding を最も出しすぎている球を val で選ぶ。

    帰無分布は y を Bernoulli(p) で引き直した最小 z 値。選んだ球の半径を test に当て、test の O/E を測る。
    """
    rng = np.random.default_rng(seed)
    _, index = NearestNeighbors(n_neighbors=k).fit(val_space).kneighbors(val_space)
    distance = np.linalg.norm(val_space[index[:, -1]] - val_space, axis=1)
    variance = (p_val * (1 - p_val))[index].sum(1)

    def z_scores(y):
        return (y[index].sum(1) - p_val[index].sum(1)) / np.sqrt(variance)

    observed = z_scores(y_val.astype(float))
    best = int(np.argmin(observed))
    null = np.array([z_scores((rng.random(len(p_val)) < p_val).astype(float)).min() for _ in range(draws)])
    inside = np.linalg.norm(test_space - val_space[best], axis=1) <= distance[best]
    observed_test, expected_test = y_test[inside].sum(), p_test[inside].sum()
    return {
        "k": k,
        "val_z": observed[best],
        "p_value": (1 + (null <= observed[best]).sum()) / (1 + draws),
        "val_log_oe": np.log((y_val[index[best]].sum() + 0.5) / (p_val[index[best]].sum() + 0.5)),
        "test_n_in_ball": int(inside.sum()),
        "test_expected": float(expected_test),
        "test_log_oe": np.log((observed_test + 0.5) / (expected_test + 0.5)) if inside.any() else np.nan,
        "ball_rows_val": index[best],
        "ball_rows_test": np.flatnonzero(inside),
    }


# ----- 空間の組み立て（02・03 で共有） -----


def add_view(frame: pd.DataFrame) -> pd.DataFrame:
    """撮影の向き（0 = AP、1 = PA、2 = 側面）と、年齢 65 歳 × 性別 × race × 向きの 48 セルを足す。"""
    view = np.where(frame.frontal_lateral == 1, 2, frame.ap_pa)
    return frame.assign(view=view, fine48=((frame.age65 * 2 + frame.sex) * 4 + frame.race_group) * 3 + view)


def build_spaces(run_id: str, cluster_dims: int = 32, audit_dims: int = 64) -> tuple[dict[str, Split], dict]:
    """1 run の val・test と、各空間の (val, test) の座標。

    空間：`attributes`（one-hot ＋ 年齢）、`c`、`dW`（3 stage の Gram の和の kernel PCA）、
    `feat32`・`feat64`（モデル自身の特徴の PCA）、`rad32`・`rad64`（RAD-DINO の PCA）。PCA は val で当てる。
    """
    splits = {split: load_split(run_id, split) for split in ("val", "test")}
    for data in splits.values():
        data.frame = add_view(data.frame)
    val, test = splits["val"], splits["test"]
    cells, _, grams = load_cells(run_id)
    coordinates = kernel_coordinates(
        grams["stage3"] + grams["stage4"] + grams["fc"], cells.train_count.to_numpy() + 1e-3
    )
    attributes = attribute_matrix(pd.concat([val.frame, test.frame]))
    spaces = {
        "attributes": (attributes[: len(val.frame)], attributes[len(val.frame) :]),
        "c": (val.condition, test.condition),
        "dW": (coordinates[val.frame.cell.to_numpy()], coordinates[test.frame.cell.to_numpy()]),
        "feat32": reduce(val.features, test.features, cluster_dims),
        "feat64": reduce(val.features, test.features, audit_dims),
        "rad32": reduce(val.raddino, test.raddino, cluster_dims),
        "rad64": reduce(val.raddino, test.raddino, audit_dims),
    }
    return splits, spaces
