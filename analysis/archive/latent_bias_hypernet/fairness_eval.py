"""公平性手法の比較（`resnet_ham10000_fg_*`）を、属性の群と潜在の群、2 通りの選び方で評価する（HAM10000）。

問いは 2 つ。
1. 属性を使う公平性手法のどれかが、ERM より性別 × 年齢の最悪群 AUROC・gap を改善するか
   （改善しないなら、比べる土俵がない）
2. 群を属性（性別 × 年齢）から潜在（DINOv2 の GMM 8 クラスタ）に替えると、その改善はどれだけ残るか

Spatial LoRA（E2E）に LHCF 型の損失と LHCF のコホートでの条件付けを足した run（`spatial_lora_ham10000_fg_*`）も
同じ表に並べ、振り分けを足した効果を `PAIRS` で読む。

選び方は、val 全体 AUROC で選んだ checkpoint（属性なし）と、val の性別 × 年齢の最悪群 AUROC で選んだ checkpoint
（val で属性を使う）。区間は、同じ選び方の ERM との seed 平均の差を、病変単位の対応のある bootstrap で求める。

あわせて、各手法の群の重み（GroupDRO の q、FairDRO・minimax AUC・EO の双対）が学習の終わりにどれだけ偏り、
同じ seed の ERM で val の AUROC が最も低い群に向いたかを読む（`objective_weights.py`）。

入力: `runs/` の fg run、`cache/<run-id>[_<checkpoint>_<split>]_{val,test}.npz`、`data/ham10000/splits_lhcf/`
出力: `results/ham10000_e2e/fairness_{summary,contrast,weights}.csv`

足りない予測 cache は、作るコマンドを表示して止まる。

    uv run python -m analysis.archive.latent_bias_hypernet.fairness_eval
"""

from __future__ import annotations

import shlex
import sys

import pandas as pd

from analysis.archive.latent_bias_hypernet.data import PACKAGE, ROOT
from analysis.archive.latent_bias_hypernet.e2e_eval import (
    CACHE,
    contrasts,
    group_aucs,
    metrics,
    runs,
    sex_age,
    thresholds,
)
from analysis.archive.latent_bias_hypernet.objective_weights import describe
from analysis.common.predictions import load_cache, read_metadata

METHODS = {
    "erm": "ERM",
    "dann-sex-age": "属性の敵対的除去（DANN）",
    "gdro": "GroupDRO（q 学習率 0.1）",
    "ug": "群の再重み付け",
    "resample": "群 × クラス均等抽出",
    "fairdro": "FairDRO",
    "minimaxauc": "minimax AUC",
    "popauc": "AUC 損失（q 固定、minimax の対照）",
    "eo": "EO 制約",
    "wst": "LHCF の最悪コホート損失",
    "fis": "FIS",
    "fami": "FaMI",
    "swad": "SWAD",
    "swadwst": "SWAD ＋ 最悪コホート損失",
}
GROUPS = {
    "sex-age": "性別 × 年齢",
    "gmm8": "GMM 8",
    "he8": "[h; e] の k-means 8",
    "lhcf": "LHCF の隠れコホート",
    "lhcfrand": "大きさだけ揃えたランダムなコホート",
    "label": "クラス",
}
# Spatial LoRA（E2E）に LHCF 型の損失を足した run。条件は定数か LHCF のコホート
SPATIAL = {
    "spatial-lora-ham10000-fg-erm": "Spatial LoRA：定数、ERM",
    "spatial-lora-ham10000-fg-swadwst-lhcf": "Spatial LoRA：定数、SWAD ＋ 最悪コホート損失（群：LHCF の隠れコホート）",
    "spatial-lora-ham10000-fg-lhcfcond-erm": "Spatial LoRA：LHCF のコホートで条件付け、ERM",
    "spatial-lora-ham10000-fg-lhcfcond-swadwst-lhcf": (
        "Spatial LoRA：LHCF のコホートで条件付け、SWAD ＋ 最悪コホート損失（群：LHCF の隠れコホート）"
    ),
}
R, SL = "resnet-ham10000-fg-", "spatial-lora-ham10000-fg-"
# 手法どうしの直接の比較（対象, 基準）。重みの学習の有無、群の替え方、SWAD への組み込みの効果、
# 振り分け（Spatial LoRA の条件）を足した効果を切り分ける。
PAIRS = [
    (R + "minimaxauc-sex-age", R + "popauc-sex-age"),
    (R + "minimaxauc-gmm8", R + "popauc-gmm8"),
    (R + "wst-lhcf", R + "wst-sex-age"),
    (R + "fis-lhcf", R + "fis-sex-age"),
    (R + "fami-lhcf", R + "fami-sex-age"),
    (R + "swadwst-lhcf", R + "swad"),
    (R + "swadwst-lhcf", R + "swadwst-sex-age"),
    (R + "swadwst-lhcf", R + "swadwst-lhcfrand"),
    (R + "swadwst-lhcf", R + "swadwst-label"),
    (SL + "erm", R + "erm"),
    (SL + "swadwst-lhcf", SL + "erm"),
    (SL + "lhcfcond-erm", SL + "erm"),
    (SL + "lhcfcond-swadwst-lhcf", SL + "swadwst-lhcf"),
    (SL + "lhcfcond-swadwst-lhcf", R + "swadwst-lhcf"),
]
SELECT = {"overall": "全体 AUROC で選択", "worst": "最悪群 AUROC で選択"}
KEYS = [
    "auroc_all",
    "auroc_external",
    "sexage_min",
    "sexage_gap",
    "sexage_tpr_gap",
    "sexage_fpr_gap",
    "gmm8_min",
    "lhcf_min",
]


def names() -> dict[str, str]:
    out = {"resnet-ham10000-fg-erm": METHODS["erm"], "resnet-ham10000-fg-dann-sex-age": METHODS["dann-sex-age"]}
    out["resnet-ham10000-fg-swad"] = METHODS["swad"]
    for method, label in METHODS.items():
        for group, glabel in GROUPS.items():
            out[f"resnet-ham10000-fg-{method}-{group}"] = f"{label}（群：{glabel}）"
    return out | SPATIAL


def is_swad(run_dir) -> bool:
    return "-swad" in run_dir.name


def cache_for(run_dir, split: str, select: str):
    """選び方に対応する予測 cache の path。最悪群で選んだものは、metadata の checkpoint 名で探す。

    SWAD は val で checkpoint を選ばず平均した重みを使うので、どちらの選び方でも `swad.ckpt` の予測を返す。
    """
    if select == "overall" and not is_swad(run_dir):
        return CACHE / f"{run_dir.name}_{split}.npz"
    name = "swad.ckpt" if is_swad(run_dir) else "best_val_worst_sexage"
    for path in CACHE.glob(f"{run_dir.name}_*_{split}.npz"):
        if name in read_metadata(path)["checkpoint"]:
            return path
    return None


def load_scores(table, val, test) -> tuple[dict, dict]:
    sv, st, missing = {}, {}, []
    for r in table.itertuples():
        for select in SELECT:
            paths = {split: cache_for(r.run_dir, split, select) for split in ("val", "test")}
            if any(p is None or not p.exists() for p in paths.values()):
                missing.append((r.run_dir, select))
                continue
            key = (f"{r.exp}|{select}", r.seed)
            sv[key] = load_cache(paths["val"], val.image.to_numpy(str))["probabilities"][:, 1]
            st[key] = load_cache(paths["test"], test.image.to_numpy(str))["probabilities"][:, 1]
    if missing:
        for select in SELECT:
            for flag in ("--swad", "--worst" if select == "worst" else None):
                dirs = [
                    str(d.relative_to(ROOT)) for d, s in missing if s == select and is_swad(d) == (flag == "--swad")
                ]
                for split in ("val", "test") if dirs else ():
                    parts = ["uv", "run", "python", "-m", "analysis.archive.latent_bias_hypernet.predict_ungrouped"]
                    parts += ["--split", split] + ([flag] if flag else [])
                    for d in dirs:
                        parts += ["--run-dir", d]
                    print(shlex.join(parts))
        sys.exit(1)
    return sv, st


def weight_table(table, sv, train, val) -> pd.DataFrame:
    """各 run の群の重みの偏りと、それが同じ seed の ERM の val 最悪群に向いたか。"""
    groups = {"sexage": sex_age(train, "sex"), "gmm8": train.gmm8.to_numpy()}
    groups_val = {"sexage": sex_age(val, "sex"), "gmm8": val.gmm8.to_numpy()}
    rows = []
    for r in table.itertuples():
        erm = sv[("resnet-ham10000-fg-erm|overall", r.seed)]
        worst = {}
        for name, g in groups_val.items():
            aucs = group_aucs(val.target.to_numpy(), erm, g)
            worst[name] = min(aucs, key=aucs.get)
        rows.append({"exp": r.exp, "seed": r.seed, **describe(r.run_dir, r.exp, train, groups, worst)})
    return pd.DataFrame(rows)


def pair_contrasts(test, st, thr, summary, labels) -> pd.DataFrame:
    """PAIRS の各組で、対象と基準の seed 平均の差（病変単位の対応のある bootstrap）。"""
    out = []
    for select in SELECT:
        for exp, ref in PAIRS:
            keys = {f"{e}|{select}": e for e in (exp, ref)}
            seeds_of = {
                k: summary[(summary.exp == e) & (summary.select == select)].seed.tolist() for k, e in keys.items()
            }
            if not all(seeds_of.values()):
                continue
            part = {k: v for k, v in st.items() if k[0] in keys}
            cond = {k: labels[e] for k, e in keys.items()}
            frame = contrasts("ham10000", test, part, thr, seeds_of, cond, f"{ref}|{select}", KEYS)
            out.append(frame.assign(select=select, reference=labels[ref]))
    return pd.concat(out, ignore_index=True)


def main() -> None:
    labels = names()
    table = runs(labels)
    # splits_lhcf は splits_latent に LHCF のコホート列を足したもの
    splits = ROOT / "data" / "ham10000" / "splits_lhcf"
    val, test = pd.read_csv(splits / "val.csv"), pd.read_csv(splits / "test.csv")
    sv, st = load_scores(table, val, test)
    thr = thresholds(val, sv)
    rows = []
    for (key, seed), s in st.items():
        exp, select = key.split("|")
        rows.append({"exp": exp, "condition": labels[exp], "select": select, "seed": seed})
        rows[-1].update(metrics("ham10000", test, s, thr[(key, seed)]))
    summary = pd.DataFrame(rows)
    dest = PACKAGE / "results" / "ham10000_e2e"
    summary.to_csv(dest / "fairness_summary.csv", index=False)
    print(summary.groupby(["select", "condition"])[KEYS].mean().round(4).to_string())

    weights = weight_table(table, sv, pd.read_csv(splits / "train.csv"), val)
    weights.to_csv(dest / "fairness_weights.csv", index=False)
    print(weights.groupby("exp").mean(numeric_only=True).round(3).to_string())

    out = []
    for select in SELECT:
        part = {k: v for k, v in st.items() if k[0].endswith(f"|{select}")}
        seeds_of = summary[summary.select == select].groupby("exp").seed.apply(list)
        seeds_of = {f"{e}|{select}": s for e, s in seeds_of.items()}
        cond = {f"{e}|{select}": labels[e] for e in labels}
        frame = contrasts("ham10000", test, part, thr, seeds_of, cond, f"resnet-ham10000-fg-erm|{select}", KEYS)
        out.append(frame.assign(select=select))
    contrast = pd.concat(out, ignore_index=True)
    pairs = pair_contrasts(test, st, thr, summary, labels)
    pairs.to_csv(dest / "fairness_pairs.csv", index=False)
    print(
        pairs.assign(ci=pairs.apply(lambda r: f"{r.delta:+.3f} [{r.lo:+.3f}, {r.hi:+.3f}]", axis=1))
        .pivot(index=["select", "condition", "reference"], columns="metric", values="ci")
        .to_string()
    )
    contrast.to_csv(dest / "fairness_contrast.csv", index=False)
    ci = contrast.assign(ci=contrast.apply(lambda r: f"{r.delta:+.3f} [{r.lo:+.3f}, {r.hi:+.3f}]", axis=1))
    print(ci.pivot(index=["select", "condition"], columns="metric", values="ci").to_string())


if __name__ == "__main__":
    main()
