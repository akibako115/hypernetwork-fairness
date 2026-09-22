"""通常 ResNet と attribute-invariant ResNet を、同じ split・同じ群・同じ seed 集合で比べる。

seed 43 / 44 / 45 の 3 本ずつを読む。どちらも val AUROC で選んだ checkpoint を使い、評価は val に
使っていない test split で行う。invariant 化で分類性能をどれだけ払ったのか
（`classification_performance`）と、その見返りに属性ごとの公平性がどう動いたのか
（`fairness_metrics`）を、別の表として残す。片方だけを見ると「gap は縮んだが全体も落ちた」を
見落とす。

**seed 42 は混ぜない。** s42 の invariant は adversary が全属性（sex / race / ethnicity /
frontal_lateral / ap_pa / age）にかかっており、ここで読む 3 本（sex / race / age のみ）とは
別の手法になる。同じ系列として平均を取ると、手法の差と設定の差が混ざる。

seed の畳み方（この package の判断）:

- **手法ごとに平均と標準偏差**を出す（`*_by_method_<split>.csv`）。標本 SD（`ddof=1`）で、
  n=3 なので自由度は 2 しかない。**SD は「どれくらい揺れるか」の目安であって、検定ではない。**
- per-seed の表も必ず残す。n=3 の平均は 1 本の外れで動くので、**3 点の散らばりを見ずに
  平均だけ読ませない**。

群の切り方（この package の判断）:

- **split は test**。checkpoint は 6 本とも val AUROC で選ばれているので、val で群別に比べると
  選択の効いた側へ寄る。
- **全体性能は test 全行**で測る。属性の欠損で母集団を削ると、「invariant 化でいくら払ったか」が
  属性の欠損パターンに依存してしまう。
- **公平性の母集団は属性ごとに取る**。sex / ethnicity / age_group_65 は各属性の非欠損行すべて、
  race だけ **White / Asian / Black** に絞る。残り 3 カテゴリは test で n=3,280 / 356 / 49 と
  偏りが大きく、陽性が 32 例・5 例しかない群の TPR が gap を決めてしまう。属性を交差させない
  ここでは母集団を揃える必要が無いので、**定義を変えたのが race だけ**になるこの取り方にする。
  その代わり race の値は学習側が記録した `val/race/*` とは別定義になる。突き合わせない。
- gap だけでなく **worst と best、各群の n も出す**。gap が縮んでも、worst が上がったのか
  best が下がったのかで意味が逆になるし、動いた群が何人かを見ないと差を読めない。

指標は `projects/` から import せず、群ごとの TPR / FPR からここで組み立てる。学習側と同じ
定義であることは `analysis/tests/test_fairness_agreement.py` が golden データで固定する。
実装どうしを import して比べると、どちらが正しいのか分からないまま両方が動く。

入力: `cache/<run-id>_<split>.npz`
出力: `results/classification_performance_<split>.csv`、`results/group_metrics_<split>.csv`、
      `results/fairness_metrics_<split>.csv`、`results/classification_performance_by_method_<split>.csv`、
      `results/fairness_metrics_by_method_<split>.csv`

使い方:
    uv run python analysis/initial_resnet_vs_invariant/groups.py --split test
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import rootutils
from sklearn.metrics import accuracy_score, balanced_accuracy_score, log_loss, roc_auc_score

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from analysis.common.paths import split_csv  # noqa: E402
from analysis.common.predictions import cache_path, load_cache  # noqa: E402

PACKAGE = Path(__file__).parent
CACHE, RESULTS = PACKAGE / "cache", PACKAGE / "results"

RESNET, INVARIANT = "ResNet", "attribute-invariant ResNet"
# 比較する 6 run。条件の違いは第1段の loss だけで、seed 以外は揃っている。
RUNS = [
    (RESNET, 43, "20260922T063725Z-resnet-chexpert-s43-efcd"),
    (RESNET, 44, "20260922T063727Z-resnet-chexpert-s44-6f0c"),
    (RESNET, 45, "20260922T063727Z-resnet-chexpert-s45-fb77"),
    (INVARIANT, 43, "20260922T063725Z-resnet-chexpert-attribute-invariant-s43-546c"),
    (INVARIANT, 44, "20260922T063728Z-resnet-chexpert-attribute-invariant-s44-8e6b"),
    (INVARIANT, 45, "20260922T063728Z-resnet-chexpert-attribute-invariant-s45-404a"),
]
MODELS = [RESNET, INVARIANT]

# split CSV の符号。旧 repo の `create_cv_chexpert.py` が付けた対応で、元の文字列は
# `data/chexpert/df_chexpert_plus_240401.csv` にある。符号表に無い値は欠損として落ちるので、
# race はここに 3 つだけ書くことが「White / Asian / Black に絞る」の実装になる。
SEX_LABELS = {0: "Male", 1: "Female"}
RACE_LABELS = {0: "White", 2: "Asian", 3: "Black"}
ETHNICITY_LABELS = {0: "Non-Hispanic", 1: "Hispanic"}
AGE_LABELS = {False: "<65", True: ">=65"}
# 学習時に使ったのと同じ 4 属性。年齢群の境界 65 歳も学習時と同じ。
ATTRIBUTES = ["sex", "race", "ethnicity", "age_group_65"]

# seed 方向へ畳む列。群の n のように seed で変わらない列は畳まない。
PERFORMANCE_METRICS = ["accuracy", "balanced_accuracy", "auroc", "cross_entropy"]
FAIRNESS_METRICS = [
    "worst AUROC",
    "best AUROC",
    "AUROC gap",
    "worst bACC",
    "best bACC",
    "bACC gap",
    "Eopp1",
    "Eopp0",
    "Eodds",
]


def classification_metrics(model: str, seed: int, cached: dict[str, Any], split: str) -> dict[str, Any]:
    """1 run 分の全体性能を返す。

    cross entropy も出すのは、AUROC が同じでも確率の較正が崩れていることがあるため。

    Args:
        model: 表に出す手法名
        seed: その run の seed
        cached: `load_cache` が返した予測 cache
        split: 評価した split

    Returns:
        dict[str, Any]: accuracy / balanced accuracy / AUROC / cross entropy
    """
    target, probabilities = cached["target"], cached["probabilities"]
    return {
        "model": model,
        "seed": seed,
        "split": split,
        "run_id": cached["metadata"]["run_id"],
        "n_examples": len(target),
        "accuracy": accuracy_score(target, cached["predictions"]),
        "balanced_accuracy": balanced_accuracy_score(target, cached["predictions"]),
        "auroc": roc_auc_score(target, probabilities[:, 1]),
        "cross_entropy": log_loss(target, probabilities, labels=np.arange(probabilities.shape[1])),
    }


def demographics_of(split: str) -> pd.DataFrame:
    """split CSV から、属性 1 つにつき 1 列の表示用ラベル表を作る。

    欠損と、符号表に無い値（race の残り 3 カテゴリ）は欠損にする。落とさず欠損のまま残すのは、
    属性ごとに母集団が違うためで、`group_rows` が列ごとに落とす。index は split CSV の行番号の
    ままにしておく。予測 cache は CSV と同じ順で並んでいるので、この index がそのまま
    cache の添字になる。

    Args:
        split: `val` または `test`

    Returns:
        pd.DataFrame: 列が `ATTRIBUTES`、値が表示用ラベルか欠損
    """
    frame = pd.read_csv(split_csv("chexpert", split))
    labelled = pd.DataFrame(
        {
            "sex": frame["sex"].map(SEX_LABELS),
            "race": frame["race"].map(RACE_LABELS),
            "ethnicity": frame["ethnicity"].map(ETHNICITY_LABELS),
            "age_group_65": (frame["age"] >= 65).map(AGE_LABELS),
        }
    )
    missing = pd.DataFrame(
        {
            "sex": frame["sex_missing"].astype(bool),
            "race": frame["race_missing"].astype(bool),
            "ethnicity": frame["ethnicity_missing"].astype(bool),
            "age_group_65": frame["age_missing"].astype(bool) | frame["age"].isna(),
        }
    )
    return labelled.mask(missing)


def group_metrics(target: np.ndarray, probability: np.ndarray, prediction: np.ndarray) -> dict[str, float]:
    """1 つの群の性能を返す。

    `n` が小さい群では AUROC も TPR も揺れる。群の値を読む前に `n` と `n_positive` を
    見られるよう、同じ行に入れておく。

    片方のクラスしか持たない群では AUROC も balanced accuracy も定義できない。`sklearn` は
    balanced accuracy を「居るクラスだけの recall 平均」として返してしまうので、ここで
    欠損にする。学習側の `compute_fairness_metrics` も同じ場合に値を出さない。

    Args:
        target: 正解ラベル
        probability: 陽性クラスの確率
        prediction: 予測ラベル

    Returns:
        dict[str, float]: `n` / `n_positive` / `positive_rate` / `auroc` / `bacc` / `tpr` / `fpr`
    """
    positive, negative = target == 1, target == 0
    both_classes = positive.any() and negative.any()
    return {
        "n": len(target),
        "n_positive": int(positive.sum()),
        "positive_rate": positive.mean(),
        "auroc": roc_auc_score(target, probability) if both_classes else np.nan,
        "bacc": balanced_accuracy_score(target, prediction) if both_classes else np.nan,
        "tpr": prediction[positive].mean() if positive.any() else np.nan,
        "fpr": prediction[negative].mean() if negative.any() else np.nan,
    }


def group_rows(model: str, seed: int, cached: dict[str, Any], demographics: pd.DataFrame) -> list[dict[str, Any]]:
    """1 run 分を、属性 × 群の行に展開する。

    Args:
        model: 表に出す手法名
        seed: その run の seed
        cached: `load_cache` が返した予測 cache
        demographics: `demographics_of` が返した属性表

    Returns:
        list[dict[str, Any]]: 手法・seed・属性・群名と性能
    """
    rows: list[dict[str, Any]] = []
    for attribute in ATTRIBUTES:
        column = demographics[attribute].dropna()
        for name, part in column.groupby(column, sort=True):
            # 行番号はそのまま cache の添字になる（cache は CSV 順、属性表は欠損を空けただけ）。
            index = part.index.to_numpy()
            probability = cached["probabilities"][index, 1]
            metrics = group_metrics(cached["target"][index], probability, cached["predictions"][index])
            rows.append({"model": model, "seed": seed, "attribute": attribute, "group": name, **metrics})
    return rows


def _across_groups(values: pd.Series) -> tuple[float, float, float]:
    """群をまたいだ worst / best / gap を返す。

    群が 1 つしか無い、または値を出せない群がある属性では、いずれも定義しない。`pandas` の
    `max` / `min` は欠損を飛ばすので、そのまま使うと**残った群だけで測った差**が出てしまう。
    学習側の `compute_fairness_metrics` も同じ場合に値を出さない。

    Args:
        values: 1 つの (model, seed, 属性) に属する群の値

    Returns:
        tuple[float, float, float]: worst・best・gap
    """
    if len(values) < 2 or values.isna().any():
        return np.nan, np.nan, np.nan
    return values.min(), values.max(), values.max() - values.min()


def summarize(part: pd.DataFrame) -> pd.Series:
    """1 つの (model, seed, 属性) を、worst / best / gap の 1 行に畳む。

    `Eopp1` は群間の TPR の最大差、`Eopp0` は TNR の最大差（FPR の最大差と同値）、
    `Eodds` は `(TPR gap + FPR gap) / 2` で、`projects/*/utils/metrics.py` と同じ定義になる。

    Args:
        part: `group_metrics_*.csv` の部分表

    Returns:
        pd.Series: worst・best・gap と Eopp0 / Eopp1 / Eodds
    """
    worst_auroc, best_auroc, auroc_gap = _across_groups(part["auroc"])
    worst_bacc, best_bacc, bacc_gap = _across_groups(part["bacc"])
    tpr_gap = _across_groups(part["tpr"])[2]
    fpr_gap = _across_groups(part["fpr"])[2]
    return pd.Series(
        {
            "groups": len(part),
            "min n": int(part["n"].min()),
            "min n positive": int(part["n_positive"].min()),
            "worst AUROC": worst_auroc,
            "best AUROC": best_auroc,
            "AUROC gap": auroc_gap,
            "worst bACC": worst_bacc,
            "best bACC": best_bacc,
            "bACC gap": bacc_gap,
            "Eopp1": tpr_gap,
            "Eopp0": fpr_gap,
            "Eodds": (tpr_gap + fpr_gap) / 2,
        }
    )


def across_seeds(frame: pd.DataFrame, keys: list[str], metrics: list[str]) -> pd.DataFrame:
    """seed 方向へ平均と標準偏差に畳み、1 指標 1 行の long 形式で返す。

    SD は標本標準偏差（`ddof=1`）。n=3 なので自由度は 2 で、**揺れの目安であって検定ではない**。
    幅の広い表に `*_mean` / `*_std` を横へ足すのではなく long にするのは、指標ごとに
    mean と std が必ず隣り合う形にして、片方だけ引用されるのを防ぐため。

    Args:
        frame: per-seed の表
        keys: seed 以外の群化キー（`model` や `model` + `attribute`）
        metrics: 畳む列

    Returns:
        pd.DataFrame: `keys` + `metric` / `mean` / `std` / `n_seeds`
    """
    long = frame.melt(id_vars=[*keys, "seed"], value_vars=metrics, var_name="metric", value_name="value")
    values = long.groupby([*keys, "metric"], sort=False)["value"]
    summary = pd.DataFrame({"mean": values.mean(), "std": values.std(ddof=1), "n_seeds": values.count()})
    return summary.reset_index()


def main() -> None:
    """6 run の全体性能・群指標・公平性と、その seed 集約を `results/` へ書く。

    Args:
        なし

    Returns:
        None
    """
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--split", default="test", choices=("val", "test"))
    args = parser.parse_args()

    images = pd.read_csv(split_csv("chexpert", args.split))["image"].to_numpy(dtype=str)
    caches = []
    for model, seed, run_id in RUNS:
        caches.append((model, seed, load_cache(cache_path(CACHE, run_id, args.split), images=images)))
    demographics = demographics_of(args.split)
    RESULTS.mkdir(exist_ok=True)

    performance_rows = [classification_metrics(model, seed, cached, args.split) for model, seed, cached in caches]
    performance = pd.DataFrame(performance_rows)
    performance.to_csv(RESULTS / f"classification_performance_{args.split}.csv", index=False)

    group_records: list[dict[str, Any]] = []
    for model, seed, cached in caches:
        group_records.extend(group_rows(model, seed, cached, demographics))
    groups = pd.DataFrame(group_records)
    groups.to_csv(RESULTS / f"group_metrics_{args.split}.csv", index=False)

    # `sort=False` で初出順を保つ。行は `RUNS` の順 × `ATTRIBUTES` の順に積んであるので、
    # ここで並べ直さなくても表の並びが読む順になる。
    keys = ["model", "seed", "attribute"]
    fairness = groups.groupby(keys, sort=False).apply(summarize, include_groups=False).reset_index()
    fairness.to_csv(RESULTS / f"fairness_metrics_{args.split}.csv", index=False)

    by_method = across_seeds(performance, ["model"], PERFORMANCE_METRICS)
    by_method.to_csv(RESULTS / f"classification_performance_by_method_{args.split}.csv", index=False)
    fairness_by_method = across_seeds(fairness, ["model", "attribute"], FAIRNESS_METRICS)
    fairness_by_method.to_csv(RESULTS / f"fairness_metrics_by_method_{args.split}.csv", index=False)

    print(f"{len(caches)} runs / {len(groups)} group rows -> {RESULTS}")


if __name__ == "__main__":
    main()
