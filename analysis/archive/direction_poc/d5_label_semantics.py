"""方向 5：CheXpert No Finding の年齢差を、ラベルの意味の違い（レポート種別の構成）として分解する。

問い：ラベルの意味がそろった部分集合（レポートが正常を明言した陽性）で測り直すと、属性間の差はどれだけ残るか。
残る差のうち、陽性のレポート種別の構成の違いで説明できる分と、同じ種別の中の差はそれぞれどれだけか。

陽性の AUROC は「陽性 1 件が陰性全体の何割より上にいるか」の平均なので、陰性を固定すれば陽性の種別ごとに
AUROC_g = Σ_t w_{g,t} · A_{g,t} と厳密に分かれる（w は群 g の陽性に占める種別 t の割合、A は種別 t の陽性 vs 群 g の全陰性）。
年齢差 A_young − A_old を Oaxaca 型に、構成の差 Σ_t Δw_t · Ā_t と、種別の中の差 Σ_t w̄_t · ΔA_t に分ける
（w̄・Ā は 2 群の平均で、2 通りの順序の平均にあたる対称な分け方）。

レポート種別は `fairness_mechanism_probe/label_construct.py` の正規表現による分類（書かれ方の分類で、臨床的な正誤ではない）。
対象のモデル：
- M01〜M07（race-known の既存の主比較、3 seed の logit 平均）。val の予測が M01 にしか無いので、Eopp は M01 だけ
- Q1 の 1 段目 `a_s42`・`a_s43`（train_a で学習した ResNet-50 ERM、test は同じ 19,617 件）

指標は Q1 と同じ 4 つ（最悪群 AUROC、AUROC gap、Eopp1、Eopp0。しきい値は val の Youden）。陽性を絞るときは、陰性は全件残す。
区間は患者単位 bootstrap（`BOOT` 回）の 95%。

出力: `results/d5/{metrics,decomposition,bootstrap}.csv`

    uv run python -m analysis.archive.direction_poc.d5_label_semantics
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import rootutils
from sklearn.metrics import roc_curve

ROOT = rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from analysis.archive.latent_bias_hypernet.data import CACHE as LBH_CACHE  # noqa: E402
from analysis.archive.latent_bias_hypernet.data import GROUP_NAMES  # noqa: E402

OUT = ROOT / "analysis" / "archive" / "direction_poc" / "results" / "d5"
RK_CACHE = ROOT / "analysis" / "archive" / "race_known_architecture_objective_comparison" / "cache"
FMP_CACHE = ROOT / "analysis" / "archive" / "fairness_mechanism_probe" / "cache"
CONDITIONS = {
    "M01 ResNet ERM": "resnet-chexpert-race-known-s4",
    "M02 ResNet DRO": "resnet-chexpert-age-race-group-dro-race-known-s4",
    "M03 Spatial ERM": "spatial-lora-chexpert-race-known-s4",
    "M04 Spatial DRO": "spatial-lora-chexpert-age-race-group-dro-race-known-s4",
    "M06 Spatial 2-stage DRO": "spatial-lora-chexpert-from-resnet-age-race-group-dro-race-known-s4",
    "M07 HyperAdapt ERM": "hyperadapt-chexpert-race-known-s4",
}
STAGE1 = ("a_s42", "a_s43")
TYPES = ("affirmed_normal", "focused_followup", "other")
SUBSETS = ("all",) + TYPES
BOOT = 500
MIN_POS = 10


def weighted_auc(score: np.ndarray, y: np.ndarray, w: np.ndarray) -> float:
    """重み付き AUROC（Mann–Whitney）。bootstrap の重複回数を重みとして渡す。同点は 0.5 で数える。"""
    pos, neg = y == 1, y == 0
    s_neg, w_neg = score[neg], w[neg]
    order = np.argsort(s_neg)
    s_neg, cum = s_neg[order], np.concatenate([[0.0], np.cumsum(w_neg[order])])
    lo = np.searchsorted(s_neg, score[pos], side="left")
    hi = np.searchsorted(s_neg, score[pos], side="right")
    below = cum[lo] + 0.5 * (cum[hi] - cum[lo])
    total_pos, total_neg = w[pos].sum(), cum[-1]
    if total_pos == 0 or total_neg == 0:
        return np.nan
    return float((w[pos] * below).sum() / (total_pos * total_neg))


def youden(score: np.ndarray, y: np.ndarray) -> float:
    fpr, tpr, thr = roc_curve(y, score)
    return float(thr[np.argmax(tpr - fpr)])


def load_test() -> pd.DataFrame:
    """test の属性・レポート種別に、各モデルの test score を列として足す。"""
    frame = pd.read_csv(LBH_CACHE / "context" / "test.csv")
    frame["group"] = (frame.age >= 65).astype(int) * 4 + frame.race_group.astype(int)
    frame["patient"] = frame.image.str.split("/").str[1]
    for name, stem in CONDITIONS.items():
        frame[name] = mean_logit(RK_CACHE, stem, "test", frame.image.to_numpy(dtype=str))
    for name in STAGE1:
        z = np.load(LBH_CACHE / "stage1" / name / "test.npz")
        assert np.array_equal(z["image"], frame.image.to_numpy(dtype=str))
        frame[name] = z["logits"][:, 1] - z["logits"][:, 0]
    return frame


def mean_logit(cache, stem: str, split: str, images: np.ndarray) -> np.ndarray:
    paths = sorted(p for p in cache.glob(f"*-{stem}?-*_{split}.npz") if "features" not in p.name)
    if len(paths) != 3:
        raise ValueError(f"{stem} の {split} 予測が 3 seed そろっていない: {len(paths)}")
    logits = []
    for path in paths:
        z = np.load(path, allow_pickle=True)
        if not np.array_equal(z["image"], images):
            raise ValueError(f"{path.name} の行順が test と一致しない")
        logits.append(z["logits"][:, 1] - z["logits"][:, 0])
    return np.mean(logits, axis=0)


def thresholds() -> dict[str, float]:
    """val の Youden しきい値。val の予測があるモデルだけ。"""
    val = pd.read_csv(LBH_CACHE / "context" / "val.csv")
    images = val.image.to_numpy(dtype=str)
    out = {"M01 ResNet ERM": youden(mean_logit(FMP_CACHE, CONDITIONS["M01 ResNet ERM"], "val", images), val.target)}
    for name in STAGE1:
        z = np.load(LBH_CACHE / "stage1" / name / "val.npz")
        out[name] = youden(z["logits"][:, 1] - z["logits"][:, 0], val.target.to_numpy())
    return out


def keep_mask(frame: pd.DataFrame, subset: str) -> np.ndarray:
    """陰性は全件、陽性は種別 `subset` のものだけ残す。"""
    pos = frame.target.to_numpy() == 1
    if subset == "all":
        return np.ones(len(frame), dtype=bool)
    return ~pos | (frame.report_type.to_numpy() == subset)


def group_metrics(frame: pd.DataFrame, score: str, w: np.ndarray, subset: str, thr: float | None) -> dict:
    """8 群の AUROC・TPR・FPR から Q1 の 4 指標と、年齢 2 区分の AUROC 差を求める。"""
    keep = keep_mask(frame, subset)
    s, y, g = frame[score].to_numpy(), frame.target.to_numpy(), frame.group.to_numpy()
    old = frame.age.to_numpy() >= 65
    row = {}
    aucs, tprs, fprs = [], [], []
    for k, name in enumerate(GROUP_NAMES):
        m = keep & (g == k)
        n_pos = int((m & (y == 1)).sum())
        auc = weighted_auc(s[m], y[m], w[m]) if n_pos >= MIN_POS else np.nan
        row[f"auc {name}"] = auc
        aucs.append(auc)
        if thr is not None:
            pos_w, neg_w = w[m & (y == 1)], w[m & (y == 0)]
            tprs.append((pos_w * (s[m & (y == 1)] >= thr)).sum() / pos_w.sum() if n_pos >= MIN_POS else np.nan)
            fprs.append((neg_w * (s[m & (y == 0)] >= thr)).sum() / neg_w.sum())
    row["worst_auroc"] = np.nanmin(aucs)
    row["auroc_gap"] = np.nanmax(aucs) - np.nanmin(aucs)
    if thr is not None:
        row["eopp1"] = np.nanmax(tprs) - np.nanmin(tprs)
        row["eopp0"] = np.nanmax(fprs) - np.nanmin(fprs)
    row["auc <65"] = weighted_auc(s[keep & ~old], y[keep & ~old], w[keep & ~old])
    row["auc ≥65"] = weighted_auc(s[keep & old], y[keep & old], w[keep & old])
    row["age_gap"] = row["auc <65"] - row["auc ≥65"]
    return row


def decompose(frame: pd.DataFrame, score: str, w: np.ndarray, young: np.ndarray, old: np.ndarray) -> dict:
    """年齢差を、陽性の種別構成の差と、種別の中の差に分ける（Oaxaca 型、対称）。"""
    s, y, t = frame[score].to_numpy(), frame.target.to_numpy(), frame.report_type.to_numpy()
    parts = {}
    for side, m in (("young", young), ("old", old)):
        pos_w = w[m & (y == 1)]
        for typ in TYPES:
            sel = m & ((y == 0) | (t == typ))
            parts[(side, typ, "A")] = weighted_auc(s[sel], y[sel], w[sel])
            parts[(side, typ, "w")] = (w[m & (y == 1) & (t == typ)]).sum() / pos_w.sum()
    row = {"gap": 0.0, "composition": 0.0, "within": 0.0}
    for typ in TYPES:
        a_y, a_o = parts[("young", typ, "A")], parts[("old", typ, "A")]
        w_y, w_o = parts[("young", typ, "w")], parts[("old", typ, "w")]
        row["composition"] += (w_y - w_o) * (a_y + a_o) / 2
        row["within"] += (w_y + w_o) / 2 * (a_y - a_o)
        row[f"within {typ}"] = (w_y + w_o) / 2 * (a_y - a_o)
        row[f"A young {typ}"], row[f"A old {typ}"] = a_y, a_o
        row[f"w young {typ}"], row[f"w old {typ}"] = w_y, w_o
    row["gap"] = row["composition"] + row["within"]
    return row


def boot_weights(frame: pd.DataFrame, rng: np.random.Generator) -> np.ndarray:
    """患者単位の bootstrap を、画像ごとの重複回数として返す。"""
    codes, uniq = pd.factorize(frame.patient)
    counts = rng.multinomial(len(uniq), np.full(len(uniq), 1 / len(uniq)))
    return counts[codes].astype(float)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    frame = load_test()
    thr = thresholds()
    ones = np.ones(len(frame))
    models = list(CONDITIONS) + list(STAGE1)
    age = frame.age.to_numpy()
    bands = {"<65 vs ≥65": (age < 65, age >= 65), "<50 vs 80+": (age < 50, age >= 80)}

    metric_rows, decomp_rows = [], []
    for model in models:
        for subset in SUBSETS:
            row = {"model": model, "positives": subset}
            row.update(group_metrics(frame, model, ones, subset, thr.get(model)))
            metric_rows.append(row)
        for band, (young, old) in bands.items():
            decomp_rows.append({"model": model, "band": band, **decompose(frame, model, ones, young, old)})
    pd.DataFrame(metric_rows).to_csv(OUT / "metrics.csv", index=False)
    pd.DataFrame(decomp_rows).to_csv(OUT / "decomposition.csv", index=False)

    # bootstrap：Q1 の 1 段目と M01 で、陽性全体と affirmed_normal の指標、その差、分解の各項
    rng = np.random.default_rng(0)
    boot_rows = []
    for b in range(BOOT):
        w = boot_weights(frame, rng)
        for model in ("a_s42", "M01 ResNet ERM"):
            full = group_metrics(frame, model, w, "all", thr[model])
            aff = group_metrics(frame, model, w, "affirmed_normal", thr[model])
            dec = decompose(frame, model, w, *bands["<65 vs ≥65"])
            row = {"boot": b, "model": model}
            for key in ("worst_auroc", "auroc_gap", "eopp1", "eopp0", "age_gap"):
                row[f"all {key}"], row[f"aff {key}"] = full[key], aff[key]
                row[f"diff {key}"] = aff[key] - full[key]
            row.update({f"dec {k}": dec[k] for k in ("gap", "composition", "within", "within focused_followup")})
            boot_rows.append(row)
    boot = pd.DataFrame(boot_rows)
    boot.to_csv(OUT / "bootstrap.csv", index=False)
    summary = boot.drop(columns="boot").groupby("model").quantile([0.025, 0.5, 0.975]).T
    pd.set_option("display.width", 250)
    print(pd.DataFrame(metric_rows).set_index(["model", "positives"])[
        ["worst_auroc", "auroc_gap", "eopp1", "eopp0", "auc <65", "auc ≥65", "age_gap"]].round(3).to_string())
    print(pd.DataFrame(decomp_rows).set_index(["model", "band"])[["gap", "composition", "within"]].round(3).to_string())
    print(summary.round(3).to_string())


if __name__ == "__main__":
    main()
