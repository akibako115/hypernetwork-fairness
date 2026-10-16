"""固定動作点の群別変化を分解する。入力は検証済み予測cacheと8群指標。

07 notebookから使用し、結果の保存・可視化はnotebookが担う。
"""

import numpy as np
import pandas as pd

from analysis.archive.race_known_architecture_objective_comparison.comparison import GROUP_LABELS, group_ids


def paired_group_changes(subgroup, pairs):
    """同seed・同群の率の差と、全体BACCへの寄与を返す。

    Args:
        subgroup: 各条件・seed・群の指標と陽性/陰性数。
        pairs: 固定したbaselineと比較条件。
    Returns:
        DataFrame: 群別差、ラベル別寄与、群平均BACCへの寄与。
    """
    rows = []
    keys = ["seed", "group_id"]
    for baseline, treatment in pairs:
        left = subgroup[subgroup.condition == baseline].set_index(keys).sort_index()
        right = subgroup[subgroup.condition == treatment].set_index(keys).sort_index()
        if left.index.has_duplicates or right.index.has_duplicates or not left.index.equals(right.index):
            raise ValueError("同seed・同群の対応が不正")
        np.testing.assert_array_equal(left[["n_positive", "n_negative"]], right[["n_positive", "n_negative"]])
        change = right[["tpr", "tnr", "bacc"]] - left[["tpr", "tnr", "bacc"]]
        change = change.rename(columns=lambda name: f"delta_{name}")
        change["group"] = left.group
        for label in ("positive", "negative"):
            change[f"n_{label}"] = left[f"n_{label}"]
        for metric, label in (("tpr", "positive"), ("tnr", "negative")):
            totals = left.groupby(level="seed")[f"n_{label}"].transform("sum")
            change[f"{metric}_contribution"] = 0.5 * left[f"n_{label}"] / totals * change[f"delta_{metric}"]
        change["bacc_contribution"] = change.tpr_contribution + change.tnr_contribution
        change["macro_bacc_contribution"] = change.delta_bacc / left.groupby(level="seed")["group"].transform("size")
        change.insert(0, "pair", f"{baseline} → {treatment}")
        rows.append(change.reset_index())
    return pd.concat(rows, ignore_index=True)


def gap_extrema(subgroup):
    """各seedでgapを決める群を列挙する。同率の群をすべて残す。

    Args:
        subgroup: 全8群のTPR/TNR。
    Returns:
        DataFrame: 条件・seed・指標の最小/最大値と群名。
    """
    rows = []
    for (condition, seed), data in subgroup.groupby(["condition", "seed"], sort=False):
        for metric in ("tpr", "tnr"):
            valid = data[metric].notna().all()
            low, high = (data[metric].min(), data[metric].max()) if valid else (np.nan, np.nan)
            rows.append(
                {
                    "condition": condition,
                    "seed": seed,
                    "metric": metric,
                    "minimum": low,
                    "maximum": high,
                    "gap": high - low,
                    "minimum_groups": "; ".join(data.loc[data[metric] == low, "group"]),
                    "maximum_groups": "; ".join(data.loc[data[metric] == high, "group"]),
                }
            )
    return pd.DataFrame(rows)


def prediction_transitions(caches, frame, pairs, seeds):
    """同じ症例のFN↔TP・FP↔TNを数える。陽性/陰性を別々に扱う。

    Args:
        caches: 検証済み(条件, seed)別予測。
        frame: 同じ行順のtest CSV。
        pairs: 固定した比較ペア。
        seeds: 固定したseed。
    Returns:
        DataFrame: 群別件数。率の差は正味改善件数/対象ラベル件数。
    """
    target, ids = frame.target.to_numpy(), group_ids(frame)
    rows = []
    for baseline, treatment in pairs:
        for seed in seeds:
            left, right = caches[baseline, seed], caches[treatment, seed]
            np.testing.assert_array_equal(left["target"], target)
            np.testing.assert_array_equal(right["target"], target)
            before, after = left["predictions"], right["predictions"]
            for group, name in enumerate(GROUP_LABELS):
                row = {"pair": f"{baseline} → {treatment}", "seed": seed, "group_id": group, "group": name}
                for label, prefix, improved, worsened in (
                    (1, "positive", "FN_to_TP", "TP_to_FN"),
                    (0, "negative", "FP_to_TN", "TN_to_FP"),
                ):
                    mask = (ids == group) & (target == label)
                    row[f"n_{prefix}"] = int(mask.sum())
                    row[improved] = int((mask & (before != label) & (after == label)).sum())
                    row[worsened] = int((mask & (before == label) & (after != label)).sum())
                    row[f"net_{prefix}"] = row[improved] - row[worsened]
                rows.append(row)
    return pd.DataFrame(rows)


def summarize_best_q(history):
    """評価checkpointまでのqを集計する。選択後のepochは平均に入れない。

    Args:
        history: read_q_historyで検証したepoch・群別診断。
    Returns:
        DataFrame: epoch等重みのq_mean平均とbest epochのq_end・q_mean・loss。
            q平均は学習の記述量であり、勾配寄与や改善の因果効果ではない。
    """
    rows = []
    for (condition, seed, group), data in history.groupby(["condition", "seed", "group_id"], sort=False):
        if data.best_epoch.nunique() != 1:
            raise ValueError("best epochが一意でない")
        best = int(data.best_epoch.iloc[0])
        before = data[data.epoch <= best].sort_values("epoch")
        if before.epoch.tolist() != list(range(best + 1)):
            raise ValueError("bestまでのepochが欠損または重複")
        at_best = before.iloc[-1]
        rows.append(
            {
                "condition": condition,
                "seed": seed,
                "group_id": group,
                "group": at_best.group,
                "best_epoch": best,
                "epochs_to_best": len(before),
                "q_mean_to_best": before.q_mean.mean(),
                "q_mean_at_best": at_best.q_mean,
                "q_end_at_best": at_best.q_end,
                "loss_at_best": at_best.loss,
                "count_at_best": at_best["count"],
            }
        )
    return pd.DataFrame(rows)
