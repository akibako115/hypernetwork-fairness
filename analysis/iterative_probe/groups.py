"""demographic 群と交差群ごとの性能を、予測 cache から表にする。

run artifact に残るのは属性ごとの worst と gap までで、**群そのものの性能**も
**属性を掛け合わせた交差群**も無い。ここで群の切り方を決めて、予測 cache から群別指標を出す。

群の切り方（この package の判断）:

- **split は test**。checkpoint は全 model が val AUROC で選ばれているので、val で群別に
  比べると選択の効いた側へ寄る。test はどの model も見ていない。
- **属性は age group（65 歳）× sex × race**。race は White / Asian / Black に絞る。
  Other / Pacific Islander / Native American は test で n が小さく、交差させると評価が崩れる。
- **母集団はどの粒度でも同じ**。3 属性すべてが非欠損で race が上の 3 つに入る行だけを使う。
  粒度を変えても母集団が変わらないので、単独属性と交差群を並べて読める。
- gap だけでなく **worst と best の値も出す**。gap が縮んでも、worst が上がったのか
  best が下がったのかで意味が逆になる。

入力: `cache/<run-id>_<split>.npz`、`results/epoch_metrics.csv`、`results/baseline_epoch_metrics.csv`
出力: `results/group_metrics_<split>.csv`、`results/fairness_summary_<split>.csv`

ここで出す Eopp0 / Eopp1 / Eodds と worst / gap が学習側の `compute_fairness_metrics` と
同じ定義であることは、`analysis/tests/test_fairness_agreement.py` が golden データで固定する。
突き合わせを実行時に 1 model だけ行うのをやめ、毎回の `pytest` で全 case を確かめる。

使い方:
    uv run python analysis/iterative_probe/groups.py --split test
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import rootutils
from sklearn.metrics import balanced_accuracy_score, roc_auc_score

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from analysis.common.paths import split_csv  # noqa: E402
from analysis.common.predictions import cache_path, load_cache  # noqa: E402

PACKAGE = Path(__file__).parent
CACHE, RESULTS = PACKAGE / "cache", PACKAGE / "results"

# split CSV の符号。旧 repo の `create_cv_chexpert.py` が付けた対応で、race は 6 値のうち 3 つを使う。
AGE_LABELS = {0: "<65", 1: ">=65"}
SEX_LABELS = {0: "Male", 1: "Female"}
RACE_LABELS = {0: "White", 2: "Asian", 3: "Black"}
GROUPINGS = [("age",), ("sex",), ("race",), ("age", "sex"), ("age", "race"), ("sex", "race"), ("age", "sex", "race")]
GROUPING_NAMES = [" x ".join(keys) for keys in GROUPINGS]
BASELINE_LABEL = "ResNet (ERM)"


def demographics_of(split: str) -> pd.DataFrame:
    """split CSV から、評価に使う行の属性表を作る。

    年齢群は学習時と同じ定義（65 歳境界、`age_missing` はそのまま欠損）で切る。
    欠損のある行と、絞り込んだ race 以外の行を落とす。index は split CSV の行番号のままに
    しておく。予測 cache は CSV と同じ順で並んでいるので、この index がそのまま cache の
    添字になる。

    Args:
        split: `val` または `test`

    Returns:
        pd.DataFrame: 表示用ラベルの属性表
    """
    frame = pd.read_csv(split_csv("chexpert", split))
    labelled = pd.DataFrame(
        {
            "age": (frame["age"] >= 65).map(lambda flag: AGE_LABELS[int(flag)]),
            "sex": frame["sex"].map(SEX_LABELS),
            "race": frame["race"].map(RACE_LABELS),
        }
    )
    age_missing = frame["age_missing"].astype(bool) | frame["age"].isna()
    missing = age_missing | frame["sex_missing"].astype(bool) | frame["race_missing"].astype(bool)
    valid = ~missing & labelled.notna().all(axis=1)
    return labelled[valid]


def models_under_test(split: str) -> list[dict[str, Any]]:
    """比較する model を、baseline を先頭にして並べる。

    どの run を比べるかは `collect.py` が書いた表が持つ。ここで run-id を書き直さない。

    Args:
        split: `val` または `test`

    Returns:
        list[dict[str, Any]]: `run_id` / `label` / `modulation` / `step_size` と予測 cache
    """
    epochs = pd.read_csv(RESULTS / "epoch_metrics.csv")
    baseline = pd.read_csv(RESULTS / "baseline_epoch_metrics.csv")
    models = [{"run_id": baseline["run_id"].iloc[0], "label": BASELINE_LABEL, "modulation": "", "step_size": np.nan}]
    for row in epochs.drop_duplicates("run_id")[["run_id", "modulation", "step_size"]].itertuples():
        models.append(
            {
                "run_id": row.run_id,
                "label": f"{row.modulation} / {row.step_size:g}",
                "modulation": row.modulation,
                "step_size": row.step_size,
            }
        )
    images = pd.read_csv(split_csv("chexpert", split))["image"].to_numpy(dtype=str)
    for model in models:
        model["cache"] = load_cache(cache_path(CACHE, model["run_id"], split), images=images)
    return models


def group_metrics(target: np.ndarray, probability: np.ndarray, prediction: np.ndarray) -> dict[str, float]:
    """1 つの群の性能を返す。

    `n` が小さい群では AUROC も TPR も揺れる。群の値を読む前に `n` と `positive_rate` を
    見られるよう、同じ行に入れておく。

    片方のクラスしか持たない群では AUROC も balanced accuracy も定義できない。`sklearn` は
    balanced accuracy を「居るクラスだけの recall 平均」として返してしまうので、ここで
    欠損にする。学習側の `compute_fairness_metrics` も同じ場合に値を出さない。

    Args:
        target: 正解ラベル
        probability: 陽性クラスの確率
        prediction: 予測ラベル

    Returns:
        dict[str, float]: `n` / `positive_rate` / `auroc` / `bacc` / `tpr` / `fpr`
    """
    positive, negative = target == 1, target == 0
    both_classes = positive.any() and negative.any()
    return {
        "n": len(target),
        "positive_rate": positive.mean(),
        "auroc": roc_auc_score(target, probability) if both_classes else np.nan,
        "bacc": balanced_accuracy_score(target, prediction) if both_classes else np.nan,
        "tpr": prediction[positive].mean() if positive.any() else np.nan,
        "fpr": prediction[negative].mean() if negative.any() else np.nan,
    }


def group_rows(model: dict[str, Any], demographics: pd.DataFrame) -> list[dict[str, Any]]:
    """1 model 分を、全粒度 × 全群の行に展開する。

    Args:
        model: `models_under_test` の 1 要素
        demographics: `demographics_of` が返す属性表

    Returns:
        list[dict[str, Any]]: model・粒度・群名と性能
    """
    cached = model["cache"]
    rows: list[dict[str, Any]] = []
    for keys in GROUPINGS:
        for values, part in demographics.groupby(list(keys), sort=True):
            # 行番号はそのまま cache の添字になる（cache は CSV 順、demographics は絞っただけ）。
            index = part.index.to_numpy()
            names = values if isinstance(values, tuple) else (values,)
            probability = cached["probabilities"][index, 1]
            metrics = group_metrics(cached["target"][index], probability, cached["predictions"][index])
            rows.append(
                {
                    "model": model["label"],
                    "modulation": model["modulation"],
                    "step_size": model["step_size"],
                    "grouping": " x ".join(keys),
                    "group": " / ".join(names),
                    **metrics,
                }
            )
    return rows


def _across_groups(values: pd.Series) -> tuple[float, float, float]:
    """群をまたいだ worst / best / gap を返す。

    群が 1 つしか無い、または値を出せない群がある粒度では、いずれも定義しない。`pandas` の
    `max` / `min` は欠損を飛ばすので、そのまま使うと**残った群だけで測った差**が出てしまう。
    学習側の `compute_fairness_metrics` も同じ場合に値を出さない。

    Args:
        values: 1 つの (model, 粒度) に属する群の値

    Returns:
        tuple[float, float, float]: worst・best・gap
    """
    if len(values) < 2 or values.isna().any():
        return np.nan, np.nan, np.nan
    return values.min(), values.max(), values.max() - values.min()


def summarize(part: pd.DataFrame) -> pd.Series:
    """1 つの (model, 粒度) を、worst / best / gap の 1 行に畳む。

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


def main() -> None:
    """群別指標と、粒度ごとの worst / best / gap を `results/` へ書く。

    Args:
        なし

    Returns:
        None
    """
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--split", default="test", choices=("val", "test"))
    args = parser.parse_args()

    demographics = demographics_of(args.split)
    models = models_under_test(args.split)
    groups = pd.DataFrame([row for model in models for row in group_rows(model, demographics)])
    RESULTS.mkdir(exist_ok=True)
    groups.to_csv(RESULTS / f"group_metrics_{args.split}.csv", index=False)

    summary = groups.groupby(["grouping", "model"]).apply(summarize, include_groups=False)
    labels = [model["label"] for model in models]
    order = pd.MultiIndex.from_product([GROUPING_NAMES, labels], names=["grouping", "model"])
    summary = summary.reindex(order)
    summary.to_csv(RESULTS / f"fairness_summary_{args.split}.csv")

    print(f"{len(models)} models / {len(groups)} group rows ({len(demographics)} 行) -> {RESULTS}")


if __name__ == "__main__":
    main()
