"""CheXpert の属性条件付きモデルが、条件をどれだけ使っているかを測る（HAM・FairVLMed と同じ規則）。

HAM・FairVLMed は `latent_bias_hypernet/condition_use.py` が測っている（`results/<dataset>_e2e/condition_use.csv`）。
CheXpert の race-known の run はその対象外なので、同じ `measure`（test の属性だけを行の間で入れ替えて推論し直す）を
ここから呼ぶ。属性の入れ替えで AUROC が下がらないなら、条件は使われていないか、画像と同じ情報しか運んでいない。

入力: `analysis/archive/race_known_architecture_objective_comparison/runs/<run-id>/`（Spatial LoRA・HyperAdapt × 3 seed）
出力: `analysis/dataset_gain_sources/results/chexpert_condition_use.csv`（GPU で 1 run 約 10 分）

    uv run python -m analysis.dataset_gain_sources.condition_use_chexpert
"""

from __future__ import annotations

import pandas as pd
import rootutils
import torch

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from analysis.archive.latent_bias_hypernet.condition_use import measure  # noqa: E402
from analysis.common.paths import runs_root  # noqa: E402
from analysis.dataset_gain_sources.data import PACKAGE  # noqa: E402

RUNS = [
    "20261002T125005Z-spatial-lora-chexpert-race-known-s42-a21c",
    "20261002T145233Z-spatial-lora-chexpert-race-known-s43-424e",
    "20261002T165551Z-spatial-lora-chexpert-race-known-s44-eb13",
    "20261002T125004Z-hyperadapt-chexpert-race-known-s42-b901",
    "20261002T181025Z-hyperadapt-chexpert-race-known-s43-4ab0",
    "20261002T233012Z-hyperadapt-chexpert-race-known-s44-a7ba",
]


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    root = runs_root("race_known_architecture_objective_comparison")
    rows = []
    for run in RUNS:
        rows.append({"run_id": run, **measure(root / run, device)})
        print(rows[-1], flush=True)
    pd.DataFrame(rows).to_csv(PACKAGE / "results" / "chexpert_condition_use.csv", index=False)


if __name__ == "__main__":
    main()
