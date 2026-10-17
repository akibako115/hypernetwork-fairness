"""Q2 の補足：公平性損失で全体 AUROC が下がるのは、群をまたぐ組の順位が崩れたためか。

全体 AUROC は陽性・陰性の全組の一致率で、組を「同じ群（年齢 × race の 8 群）の中」と「群をまたぐ」に分けると、
組数で重み付けた平均が元に戻る（`analysis/dataset_gain_sources/metrics.within_between`）。
群ごとにスコアを平行移動させても群の中の組は動かず、群をまたぐ組だけが動く。共通のしきい値で群の動作点を
そろえるには、陽性率の違う群のスコアの位置関係を崩す必要があるので、損失による全体 AUROC の低下は
群をまたぐ組に集中すると予想する。群の中の一致率が下がっていれば、群内の判別力そのものが落ちている。

epoch は val の全体 AUROC で選ぶ（Q2 の `auroc` 規則）。差は同じ手順の ERM との seed 平均の差で、
区間は患者単位の bootstrap 95%（200 本）。

入力: `cache/adapter_fair/<procedure>/<loss>_meta_s<seed>/epoch_XX_test.npy`
出力: `results/q2_auroc_decomposition.csv`

    uv run python -m analysis.archive.latent_bias_hypernet.q2_auroc_decomposition
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from analysis.archive.latent_bias_hypernet.data import PACKAGE
from analysis.archive.latent_bias_hypernet.q1_definitions import bootstrap_index
from analysis.archive.latent_bias_hypernet.q2_fairness_loss import SEEDS, SPLITS, demo8
from analysis.archive.latent_bias_hypernet.q2_threshold_equivalence import PROCEDURES, scores
from analysis.conditioning_mechanism.metrics import within_between

LOSSES = ("erm", "eo", "minimax_auc")
PARTS = ("overall", "within", "between")


def decompose(test_scores: list[np.ndarray], y: np.ndarray, g: np.ndarray, index: np.ndarray) -> dict[str, float]:
    """seed 平均の全体 AUROC と、群の中・群をまたぐ組の一致率。"""
    parts = []
    for s in test_scores:
        within, between, share = within_between(s[index], y[index], g[index])
        parts.append((share * within + (1 - share) * between, within, between, share))
    mean = np.mean(parts, axis=0)
    return {"overall": mean[0], "within": mean[1], "between": mean[2], "within_share": mean[3]}


def main() -> None:
    val, test = pd.read_csv(SPLITS / "val.csv"), pd.read_csv(SPLITS / "test.csv")
    y, g = test.target.to_numpy(), demo8(test)
    draws = bootstrap_index(test.image.str.split("/").str[1].to_numpy(), np.random.default_rng(0))[:200]
    rows = []
    for procedure in PROCEDURES:
        runs = {loss: [scores(procedure, loss, seed, val)[1] for seed in SEEDS] for loss in LOSSES}
        boot = {loss: pd.DataFrame([decompose(runs[loss], y, g, i) for i in draws]) for loss in LOSSES}
        point = {loss: decompose(runs[loss], y, g, np.arange(len(y))) for loss in LOSSES}
        for loss in LOSSES:
            row = {"procedure": procedure, "loss": loss, **point[loss]}
            if loss != "erm":
                for part in PARTS:
                    diff = boot[loss][part] - boot["erm"][part]
                    row[f"{part}_diff"] = point[loss][part] - point["erm"][part]
                    row[f"{part}_diff_lo"], row[f"{part}_diff_hi"] = diff.quantile([0.025, 0.975])
            rows.append(row)
        print(procedure, flush=True)
    table = pd.DataFrame(rows)
    table.to_csv(PACKAGE / "results" / "q2_auroc_decomposition.csv", index=False)
    with pd.option_context("display.width", 250):
        print(table.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
