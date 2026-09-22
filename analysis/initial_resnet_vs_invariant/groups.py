"""通常 ResNet と attribute-invariant ResNet を、同じ split・同じ群で比べる。

どちらも val AUROC で選んだ checkpoint を使い、評価は val に使っていない test split で行う。
invariant 化で分類性能をどれだけ払ったのか（`classification_performance`）と、その見返りに
属性ごとの公平性がどう動いたのか（`fairness_metrics`）を、別の表として残す。片方だけを見ると
「gap は縮んだが全体も落ちた」を見落とす。

公平性指標は学習側と同じ `compute_fairness_metrics` を使う。分析側で定義し直すと、学習中に
記録した値と付き合わせられなくなる。属性は学習時に使ったのと同じ 4 つで、年齢群も同じ
`add_age_groups` で切る。

入力: `cache/<run-id>_<split>.npz`
出力: `results/classification_performance_<split>.csv`、`results/fairness_metrics_<split>.csv`

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
import torch
from sklearn.metrics import accuracy_score, balanced_accuracy_score, log_loss, roc_auc_score

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from analysis.common.paths import split_csv  # noqa: E402
from analysis.common.predictions import cache_path, load_cache  # noqa: E402
from projects.hypernet_e2e.data.evaluation_attributes import add_age_groups  # noqa: E402
from projects.hypernet_e2e.utils.metrics import compute_fairness_metrics  # noqa: E402

PACKAGE = Path(__file__).parent
CACHE, RESULTS = PACKAGE / "cache", PACKAGE / "results"

# 比較する 2 run。どちらも seed 42 の 1 本で、違いは第1段の loss だけになる。
RUNS = {
    "ResNet": "20260921T103036Z-resnet-chexpert-s42-5538",
    "attribute-invariant ResNet": "20260921T125711Z-resnet-chexpert-attribute-invariant-s42-9fd3",
}
FAIRNESS_COLUMNS = ["sex", "race", "ethnicity", "age_group_65"]
AGE_GROUPS = {"age_group_65": {"source": "age", "boundaries": [65]}}


def classification_metrics(label: str, cached: dict[str, Any], split: str) -> dict[str, Any]:
    """1 model 分の全体性能を返す。

    cross entropy も出すのは、AUROC が同じでも確率の較正が崩れていることがあるため。

    Args:
        label: 表に出す model 名
        cached: `load_cache` が返した予測 cache
        split: 評価した split

    Returns:
        dict[str, Any]: accuracy / balanced accuracy / AUROC / cross entropy
    """
    target, probabilities = cached["target"], cached["probabilities"]
    return {
        "model": label,
        "split": split,
        "run_id": cached["metadata"]["run_id"],
        "n_examples": len(target),
        "accuracy": accuracy_score(target, cached["predictions"]),
        "balanced_accuracy": balanced_accuracy_score(target, cached["predictions"]),
        "auroc": roc_auc_score(target, probabilities[:, 1]),
        "cross_entropy": log_loss(target, probabilities, labels=np.arange(probabilities.shape[1])),
    }


def attributes_of(split: str) -> dict[str, torch.Tensor]:
    """`compute_fairness_metrics` が取る属性 tensor を split CSV から作る。

    Args:
        split: `val` または `test`

    Returns:
        dict[str, torch.Tensor]: `categorical` と `categorical_missing`
    """
    frame = pd.read_csv(split_csv("chexpert", split))
    frame = add_age_groups(frame, AGE_GROUPS)
    missing_columns = [f"{name}_missing" for name in FAIRNESS_COLUMNS]
    return {
        "categorical": torch.as_tensor(frame[FAIRNESS_COLUMNS].to_numpy(), dtype=torch.long),
        "categorical_missing": torch.as_tensor(frame[missing_columns].to_numpy(dtype=bool), dtype=torch.bool),
    }


def fairness_rows(
    label: str, cached: dict[str, Any], attributes: dict[str, torch.Tensor], split: str
) -> list[dict[str, Any]]:
    """1 model 分を、属性 1 つにつき 1 行へ展開する。

    Args:
        label: 表に出す model 名
        cached: `load_cache` が返した予測 cache
        attributes: `attributes_of` が返した属性
        split: 評価した split

    Returns:
        list[dict[str, Any]]: 属性ごとの Eopp0 / Eopp1 / Eodds / worst-group 指標
    """
    metrics = compute_fairness_metrics(
        torch.from_numpy(cached["logits"]),
        torch.from_numpy(cached["target"]),
        attributes,
        {"categorical": FAIRNESS_COLUMNS},
    )
    return [{"model": label, "split": split, "attribute": name, **values} for name, values in metrics.items()]


def main() -> None:
    """2 run の全体性能と属性別の公平性を `results/` へ書く。

    Args:
        なし

    Returns:
        None
    """
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--split", default="test", choices=("val", "test"))
    args = parser.parse_args()

    images = pd.read_csv(split_csv("chexpert", args.split))["image"].to_numpy(dtype=str)
    caches = {label: load_cache(cache_path(CACHE, run_id, args.split), images=images) for label, run_id in RUNS.items()}
    attributes = attributes_of(args.split)

    RESULTS.mkdir(exist_ok=True)
    performance = pd.DataFrame(classification_metrics(label, cached, args.split) for label, cached in caches.items())
    performance.to_csv(RESULTS / f"classification_performance_{args.split}.csv", index=False)
    rows = [row for label, cached in caches.items() for row in fairness_rows(label, cached, attributes, args.split)]
    pd.DataFrame(rows).to_csv(RESULTS / f"fairness_metrics_{args.split}.csv", index=False)
    print(f"{len(caches)} models -> {RESULTS}")


if __name__ == "__main__":
    main()
