"""Q1 の報告用の表：部分群の定義を変えると、属性で切った場合と比べて見える性能差がどう変わるか。

同じ 1 段目の test 予測を、次の定義で群に分け、標準の公平性指標を求める。

- 基準：属性。CheXpert は年齢（65 歳で 2 区分）× race の 8 群、ISIC は性別 × 年齢の 4 群（race が無い）。
- 特徴空間のクラスタ：群の数は基準の属性とそろえる。
  - k-means・GMM：基盤モデルの埋め込み（CheXpert は RAD-DINO と DINOv2、ISIC は DINOv2）に当てる。
    GMM は LHCF（Masroor et al. 2026）と同じ。1 段目自身の表現 h には当てない
    （スコアの高さで群が分かれ、群内の AUROC が機械的に下がる）。
  - GEORGE 型：原典（Sohoni et al. 2020）どおり 1 段目自身の表現 h を使い、陽性・陰性それぞれの中で GMM を当てる
    （原典の UMAP は、他の方法とそろえて主成分 64 次元に替えた）。クラスの中で切るのでスコアの高さで分かれてもよく、
    「同じクラスの中の難しい群」を探すのが目的である。群がクラスを 1 つしか含まないので、群の AUROC は定義できず
    求めない（「陽性の群 vs 全陰性」で代えると、低いスコアの陽性を集めた群が切り方から自動的に 0.5 を下回り、
    他の定義の群内 AUROC と比べられない）。Eopp1 は陽性の群だけ、Eopp0 は陰性の群だけで求める。
    test の群への割り当てに test のラベルを使う。基盤モデルの埋め込みに当てた版も試したが、他のクラスタと
    違う傾向が無く、先行研究にも無い組み合わせなので外した。
- 交絡の候補：撮影文脈（撮影方向 × 追跡撮影 × レポート種別）と、画像に写る部分集団（要因ごとに有無の 2 群）。
  ISIC は部位。群の数は基準とそろわないので、属性との差ではなく、それ自体で差があるかを読む。

クラスタは test 以外（CheXpert は 1 段目が見ていない train_b、ISIC は監査セットの母集団）に当てる。

指標：最悪群 AUROC、AUROC gap（最大 − 最小）、Eopp1（TPR の最大 − 最小）、Eopp0（FPR の最大 − 最小）。
しきい値は val の Youden。較正は使わない（順位としきい値しか見ないので、単調な較正は結果を変えない）。
区間は患者（ISIC は病変）単位の bootstrap の 95%。属性との差は同じ bootstrap 標本の上で求める（対応のある比較）。

最悪群や gap は、偏りがなくても群が小さく不ぞろいなほど悪く出る。表には最小群の大きさと有病率の幅を添える。

出力: `results/q1_definitions.csv`

    uv run python -m analysis.archive.latent_bias_hypernet.q1_definitions
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.mixture import GaussianMixture

from analysis.archive.latent_bias_hypernet import audit, isic_discovery
from analysis.archive.latent_bias_hypernet.data import PACKAGE, load
from analysis.archive.latent_bias_hypernet.planting import hidden_subgroup

STAGE1 = "a_s42"
BOOT = 1000
MIN_POS = MIN_NEG = 10
METRICS = ("worst_auroc", "auroc_gap", "eopp1", "eopp0")


def youden(score: np.ndarray, y: np.ndarray) -> float:
    fpr, tpr, thr = roc_curve(y, score)
    return float(thr[np.argmax(tpr - fpr)])


def cluster(x_fit: np.ndarray, x_test: np.ndarray, method: str, K: int) -> np.ndarray:
    """主成分 64 次元に落としてから、k-means か GMM で K 群に分ける。"""
    pca = PCA(64, random_state=0).fit(x_fit)
    z_fit, z_test = pca.transform(x_fit), pca.transform(x_test)
    if method == "kmeans":
        model = KMeans(K, n_init=10, random_state=0).fit(z_fit)
    else:
        model = GaussianMixture(K, covariance_type="diag", random_state=0, max_iter=200).fit(z_fit)
    return model.predict(z_test)


def george(x_fit, y_fit, x_test, y_test, K: int) -> np.ndarray:
    """クラスごとに GMM で K/2 群に分ける。陰性の群が 0..K/2-1、陽性の群が K/2..K-1。"""
    out = np.empty(len(y_test), dtype=int)
    for c in (0, 1):
        sub = cluster(x_fit[y_fit == c], x_test[y_test == c], "gmm", K // 2)
        out[y_test == c] = sub + c * (K // 2)
    return out


def group_rows(g: np.ndarray, y: np.ndarray, score: np.ndarray, threshold: float, class_pure: bool) -> pd.DataFrame:
    """群ごとの n・陽性数・AUROC・TPR・FPR。陽性・陰性が少ない群と、片方のクラスだけの群の AUROC は NaN にする。"""
    rows = []
    pred = score >= threshold
    for k in np.unique(g[g >= 0]):
        m = g == k
        pos, neg = int(y[m].sum()), int((1 - y[m]).sum())
        row = {"group": k, "n": int(m.sum()), "pos": pos, "auroc": np.nan, "tpr": np.nan, "fpr": np.nan}
        if class_pure and pos >= MIN_POS and neg == 0:
            row["tpr"] = pred[m].mean()
        elif class_pure and neg >= MIN_NEG and pos == 0:
            row["fpr"] = pred[m].mean()
        elif not class_pure and pos >= MIN_POS and neg >= MIN_NEG:
            row["auroc"] = roc_auc_score(y[m], score[m])
            row["tpr"] = pred[m & (y == 1)].mean()
            row["fpr"] = pred[m & (y == 0)].mean()
        rows.append(row)
    return pd.DataFrame(rows)


def summarize(t: pd.DataFrame) -> dict[str, float]:
    return {
        "worst_auroc": t.auroc.min(),
        "auroc_gap": t.auroc.max() - t.auroc.min(),
        "eopp1": t.tpr.max() - t.tpr.min(),
        "eopp0": t.fpr.max() - t.fpr.min(),
    }


def bootstrap_index(units: np.ndarray, rng: np.random.Generator) -> list[np.ndarray]:
    """単位（患者・病変）を復元抽出した行番号を BOOT 本作る。"""
    order = np.argsort(units, kind="stable")
    _, starts = np.unique(units[order], return_index=True)
    rows_of_unit = np.split(order, starts[1:])
    draws = []
    for _ in range(BOOT):
        picked = rng.integers(0, len(rows_of_unit), len(rows_of_unit))
        draws.append(np.concatenate([rows_of_unit[u] for u in picked]))
    return draws


def evaluate(definitions: dict, y, score, threshold, units, base: str, dataset: str) -> pd.DataFrame:
    """定義ごとの指標・区間と、基準（属性）との差・区間。"""
    draws = bootstrap_index(units, np.random.default_rng(0))
    boot = {}
    rows = []
    for name, (kind, g) in definitions.items():
        pure = kind == "GEORGE 型"
        t = group_rows(g, y, score, threshold, pure)
        valid = t.dropna(subset=["auroc", "tpr", "fpr"], how="all")
        boot[name] = pd.DataFrame([summarize(group_rows(g[i], y[i], score[i], threshold, pure)) for i in draws])
        prev = valid.pos / valid.n
        row = {"dataset": dataset, "definition": name, "kind": kind, "groups": len(valid), "min_n": valid.n.min()}
        row |= {"min_pos": valid.pos.min(), "prev_min": prev.min(), "prev_max": prev.max()}
        for k, v in summarize(t).items():
            lo, hi = boot[name][k].quantile([0.025, 0.975])
            row |= {k: v, f"{k}_lo": lo, f"{k}_hi": hi}
        rows.append(row)
        print(name, flush=True)
    out = pd.DataFrame(rows)
    for k in METRICS:
        diff = {name: boot[name][k] - boot[base][k] for name in definitions}
        out[f"{k}_diff"] = out[k] - out.loc[out.definition == base, k].item()
        out[f"{k}_diff_lo"] = [diff[n].quantile(0.025) for n in out.definition]
        out[f"{k}_diff_hi"] = [diff[n].quantile(0.975) for n in out.definition]
    return out


def chexpert() -> pd.DataFrame:
    space = {name: {s: load(STAGE1, s, name) for s in ("train_b", "val", "test")} for name in ("raddino", "dinov2")}
    D, V, T = (space["raddino"][s] for s in ("train_b", "val", "test"))
    base = "年齢 × race"
    defs = {base: ("属性", audit.demo_cells(T.context, "demo8"))}
    for sp, S in space.items():
        x_fit, x_test = S["train_b"].route, S["test"].route
        defs[f"k-means（{sp}）"] = ("クラスタ", cluster(x_fit, x_test, "kmeans", 8))
        defs[f"GMM（{sp}）"] = ("クラスタ", cluster(x_fit, x_test, "gmm", 8))
    defs["GEORGE 型（h）"] = ("GEORGE 型", george(D.features, D.target, T.features, T.target, 8))
    defs["撮影文脈"] = ("交絡", audit.imaging_context_cells(T.context))
    for rule in ("pacer", "picc", "central_line", "sternotomy", "surgical_clip"):
        defs[f"部分集団：{rule}"] = ("交絡", hidden_subgroup(T.context, rule).astype(int))
    units = T.context.image.str.split("/").str[1].to_numpy()
    return evaluate(defs, T.target, T.base, youden(V.base, V.target), units, base, "chexpert")


def isic() -> pd.DataFrame:
    A, V, T = (load(isic_discovery.STAGE1, s, "dinov2") for s in ("isic_audit", "isic_audit_v", "isic_test"))
    sex = T.context.sex.to_numpy().astype(int)
    old = (T.context.age.to_numpy() >= 65).astype(int)
    base = "性別 × 年齢"
    defs = {
        base: ("属性", np.where(sex < 0, -1, sex * 2 + old)),
        "k-means（dinov2）": ("クラスタ", cluster(A.route, T.route, "kmeans", 4)),
        "GMM（dinov2）": ("クラスタ", cluster(A.route, T.route, "gmm", 4)),
        "GEORGE 型（h）": ("GEORGE 型", george(A.features, A.target, T.features, T.target, 4)),
        "部位": ("交絡", T.context.anatom_site_general.to_numpy().astype(int)),
    }
    units = T.context.lesion_id.fillna(T.context.image).to_numpy()
    return evaluate(defs, T.target, T.base, youden(V.base, V.target), units, base, "isic")


def main() -> None:
    out = pd.concat([chexpert(), isic()], ignore_index=True)
    out.to_csv(PACKAGE / "results" / "q1_definitions.csv", index=False)
    cols = ["dataset", "definition", "groups", "min_n", "prev_min", "prev_max", *METRICS]
    with pd.option_context("display.width", 250):
        print(out[cols].round(3).to_string(index=False))


if __name__ == "__main__":
    main()
