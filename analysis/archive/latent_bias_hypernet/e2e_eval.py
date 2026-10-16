"""E2E の条件付きモデル（段階 1〜3）を、条件ごとに test で比べる（HAM10000・FairVLMed）。

比べるのは、Spatial LoRA の条件だけを替えた run、ResNet-50、群を意識した目的関数（GroupDRO）の run。
予測は `analysis/common/predictions.py` が `cache/<run-id>_<split>.npz` に書いたもの（このスクリプトは推論しない。
足りないときは作るコマンドを表示して止まる）。しきい値は val の Youden。

指標（test）:
- 全体 AUROC
- 属性：HAM は性別 × 年齢（4 群、sex・age が揃う行）、FairVLMed は race（3 群）と性別 × 年齢（4 群）の
  最小 AUROC と gap、TPR・FPR gap。4 群以下に絞るのは、細かいセルの最小 AUROC が標本誤差と勝者の呪いで
  比較に使えないため（granularity.py）
- 潜在：test の GMM 8 クラスタ（陽性・陰性とも 15 件以上）の最小 AUROC
- HAM のみ：施設が既知の行と external（学習にない施設、属性欠損が集中）の AUROC、rosendahl・vidir_modern の AUROC
区間は、対照（定数）との seed 平均の差を、病変（FairVLMed は画像）単位の対応のある bootstrap（500 回）で求める。

出力: `results/<dataset>_e2e/summary.csv`（run ごと）、`results/<dataset>_e2e/contrast.csv`（対照との差）

    uv run python -m analysis.archive.latent_bias_hypernet.e2e_eval --dataset ham10000
"""

from __future__ import annotations

import argparse
import json
import re
import shlex
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, roc_curve

from analysis.archive.latent_bias_hypernet.data import PACKAGE, ROOT
from analysis.common.predictions import cache_path, load_cache, stale_reason

RUNS = PACKAGE / "runs"
CACHE = PACKAGE / "cache"
BOOT = 500
RUN_ID = re.compile(r"^\d{8}T\d{6}Z-(?P<exp>.+)-s(?P<seed>\d+)-[0-9a-f]{4}$")
LABELS = {
    "": "Spatial LoRA（属性で条件付け）",
    "-const": "定数（構成だけ）",
    "-random8": "ランダム 8 区分",
    "-site": "撮影施設（上限）",
    "-gmm8": "DINOv2 GMM 8",
    "-gmm8lo": "ラベル直交 GMM 8",
    "-emb16": "DINOv2 埋め込み（連続）",
    "-emb16lo": "ラベル直交埋め込み（連続）",
    "-gmm8-group-dro": "GMM 8 で条件付け＋GMM 8 の GroupDRO",
    "-const-gmm8-group-dro": "定数＋GMM 8 の GroupDRO",
    "-fc": "属性で分類層だけ条件付け",
}
BASELINES = {
    "resnet-{d}": "ResNet-50（条件なし）",
    "resnet-{d}-gmm8-group-dro": "ResNet-50＋GMM 8 の GroupDRO（LHCF 型）",
    "resnet-{d}-sex-age-group-dro": "ResNet-50＋性別 × 年齢の GroupDRO",
}


def conditions(dataset: str) -> dict[str, str]:
    out = {f"spatial-lora-{dataset}{suffix}": label for suffix, label in LABELS.items()}
    out.update({key.format(d=dataset): label for key, label in BASELINES.items()})
    return out


def runs(names: dict[str, str]) -> pd.DataFrame:
    rows = []
    for d in sorted(RUNS.iterdir()):
        m = RUN_ID.match(d.name)
        if not m or m["exp"] not in names:
            continue
        if json.loads((d / "run.json").read_text())["status"] != "succeeded":
            continue
        rows.append({"exp": m["exp"], "seed": int(m["seed"]), "run_dir": d})
    # 同じ条件・seed が複数あれば最新を使う
    return pd.DataFrame(rows).groupby(["exp", "seed"]).tail(1).reset_index(drop=True)


def scores(frame: pd.DataFrame, split: str, table: pd.DataFrame) -> dict:
    out, missing = {}, []
    for r in table.itertuples():
        path = cache_path(CACHE, r.run_dir.name, split)
        if stale_reason(path, r.run_dir) is not None:
            missing.append(r.run_dir)
            continue
        cached = load_cache(path, frame.image.to_numpy(dtype=str))
        np.testing.assert_array_equal(cached["target"], frame.target.to_numpy())
        out[(r.exp, r.seed)] = cached["probabilities"][:, 1]
    if missing:
        parts = ["uv", "run", "python", "analysis/common/predictions.py", "--study", "latent_bias_hypernet"]
        parts += ["--split", split]
        for p in missing:
            parts += ["--run-dir", str(p.relative_to(ROOT))]
        print("予測 cache が足りない:\n" + shlex.join(parts))
        sys.exit(1)
    return out


def sex_age(f: pd.DataFrame, sex: str) -> np.ndarray:
    ok = (f[f"{sex}_missing"] == 0) & (f.age_missing == 0)
    return np.where(ok, f[sex].to_numpy() * 2 + (f.age.to_numpy() >= 65), -1).astype(int)


def group_aucs(y, s, g, min_n=15):
    out = {}
    for k in np.unique(g):
        m = g == k
        if k < 0 or y[m].sum() < min_n or (1 - y[m]).sum() < min_n:
            continue
        out[k] = roc_auc_score(y[m], s[m])
    return out


def axis_metrics(name: str, y, s, pred, g) -> dict:
    aucs = group_aucs(y, s, g)
    keys = list(aucs)
    tpr = [pred[(g == k) & (y == 1)].mean() for k in keys]
    fpr = [pred[(g == k) & (y == 0)].mean() for k in keys]
    return {
        f"{name}_min": min(aucs.values()),
        f"{name}_gap": max(aucs.values()) - min(aucs.values()),
        f"{name}_tpr_gap": max(tpr) - min(tpr),
        f"{name}_fpr_gap": max(fpr) - min(fpr),
    }


def ham_extra(test: pd.DataFrame, y, s) -> dict:
    known = test.site_missing.to_numpy() == 0
    row = {"auroc_known": roc_auc_score(y[known], s[known]), "auroc_external": roc_auc_score(y[~known], s[~known])}
    for k, name in ((0, "rosendahl"), (1, "vidir_modern")):
        m = known & (test.site.to_numpy() == k)
        row[f"site_{name}"] = roc_auc_score(y[m], s[m])
    return row


def metrics(dataset: str, test: pd.DataFrame, s: np.ndarray, thr: float) -> dict:
    y = test.target.to_numpy()
    pred = s >= thr
    row = {"auroc_all": roc_auc_score(y, s)}
    if dataset == "ham10000":
        row.update(ham_extra(test, y, s))
        row.update(axis_metrics("sexage", y, s, pred, sex_age(test, "sex")))
    else:
        race = np.where(test.race_missing == 0, test.race, -1)
        row.update(axis_metrics("race", y, s, pred, race))
        row.update(axis_metrics("sexage", y, s, pred, sex_age(test, "gender")))
    row["gmm8_min"] = min(group_aucs(y, s, test.gmm8.to_numpy()).values())
    if "lhcf" in test:
        # LHCF の隠れコホート（lhcf_cohorts.py）の最悪 AUROC
        row["lhcf_min"] = min(group_aucs(y, s, test.lhcf.to_numpy()).values())
    return row


def thresholds(val: pd.DataFrame, sv: dict) -> dict:
    out = {}
    for key, s in sv.items():
        fpr, tpr, t = roc_curve(val.target, s)
        out[key] = t[np.argmax(tpr - fpr)]
    return out


def bootstrap_index(test: pd.DataFrame, dataset: str) -> list[np.ndarray]:
    units = test.lesion_id.fillna(test.image) if dataset == "ham10000" else test.image
    units = units.to_numpy(str)
    uniq = np.unique(units)
    by = pd.Series(np.arange(len(test))).groupby(units).apply(np.array).to_dict()
    rng = np.random.default_rng(0)
    return [np.concatenate([by[u] for u in rng.choice(uniq, len(uniq))]) for _ in range(BOOT)]


def contrasts(dataset, test, st, thr, seeds_of, names, reference, keys) -> pd.DataFrame:
    """対照との seed 平均の差と、その対応のある bootstrap 区間。"""

    def seed_mean(exp, idx):
        part = test.iloc[idx].reset_index(drop=True)
        vals = [metrics(dataset, part, st[(exp, s)][idx], thr[(exp, s)]) for s in seeds_of[exp]]
        return {k: np.mean([v[k] for v in vals]) for k in keys}

    boots = bootstrap_index(test, dataset)
    ref_boot = []
    for b in boots:
        try:
            ref_boot.append(seed_mean(reference, b))
        except ValueError:
            ref_boot.append(None)
    full_ref = seed_mean(reference, np.arange(len(test)))
    rows = []
    for exp in seeds_of:
        if exp == reference:
            continue
        diffs = {k: [] for k in keys}
        for b, ref in zip(boots, ref_boot, strict=True):
            try:
                cur = seed_mean(exp, b) if ref is not None else None
            except ValueError:
                cur = None
            for k in keys if cur is not None else ():
                diffs[k].append(cur[k] - ref[k])
        full = seed_mean(exp, np.arange(len(test)))
        for k in keys:
            lo, hi = np.percentile(diffs[k], [2.5, 97.5])
            rows.append({"condition": names[exp], "metric": k, "delta": full[k] - full_ref[k], "lo": lo, "hi": hi})
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="ham10000", choices=("ham10000", "fairvlmed"))
    args = parser.parse_args()
    names = conditions(args.dataset)
    splits = ROOT / "data" / args.dataset / "splits_latent"
    table = runs(names)
    val, test = pd.read_csv(splits / "val.csv"), pd.read_csv(splits / "test.csv")
    sv, st = scores(val, "val", table), scores(test, "test", table)
    thr = thresholds(val, sv)
    rows = []
    for key, s in st.items():
        row = {"exp": key[0], "condition": names[key[0]], "seed": key[1]}
        row.update(metrics(args.dataset, test, s, thr[key]))
        rows.append(row)
    summary = pd.DataFrame(rows).sort_values(["exp", "seed"])
    dest = PACKAGE / "results" / f"{args.dataset}_e2e"
    dest.mkdir(parents=True, exist_ok=True)
    summary.to_csv(dest / "summary.csv", index=False)
    cols = [c for c in summary.columns if c not in ("exp", "condition", "seed")]
    print(summary.groupby("condition").seed.count().to_string())
    print(summary.groupby("condition")[cols].mean().round(4).to_string())
    print(summary.groupby("condition")[cols].std().round(4).to_string())

    reference = f"spatial-lora-{args.dataset}-const"
    seeds_of = summary.groupby("exp").seed.apply(list).to_dict()
    if reference not in seeds_of:
        return
    keys = ["auroc_all", "sexage_min", "sexage_gap", "gmm8_min"]
    keys += ["auroc_known", "auroc_external"] if args.dataset == "ham10000" else ["race_min", "race_gap"]
    contrast = contrasts(args.dataset, test, st, thr, seeds_of, names, reference, keys)
    contrast.to_csv(dest / "contrast.csv", index=False)
    print(contrast.pivot(index="condition", columns="metric", values="delta").round(4).to_string())
    ci = contrast.assign(ci=contrast.apply(lambda r: f"[{r.lo:+.3f}, {r.hi:+.3f}]", axis=1))
    print(ci.pivot(index="condition", columns="metric", values="ci").to_string())


if __name__ == "__main__":
    main()
