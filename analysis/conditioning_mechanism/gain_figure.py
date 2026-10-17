"""3 データセットの上がり幅を、属性が運べる情報の上限（後付けの上がり幅）と並べた図を描く。

条件付きモデルの上がり幅が「出力に属性を後から足しただけで得られる幅」とどう対応するかを 1 枚で見る。
HAM の条件付きモデルはこの上限を超える（学習時の効果がある）。FairVLMed は上限自体が test の標本誤差と同程度、
CheXpert は上限が 0 で、条件付けで得られるものが無い。

入力: `results/gains.csv`・`results/attribute_information.csv`（`gain_sources.py` の出力）
出力: `figures/gain_vs_attribute_ceiling.png`

    uv run python -m analysis.conditioning_mechanism.gain_figure
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rootutils

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from analysis.common.paths import STYLE_SHEET  # noqa: E402
from analysis.conditioning_mechanism.data import PACKAGE  # noqa: E402

DATASETS = {"ham10000": "HAM10000", "fairvlmed": "FairVLMed", "chexpert": "CheXpert"}
BARS = [
    ("meta − resnet", "Attribute-conditioned Spatial LoRA − ResNet", "#2a6f97"),
    ("hyperadapt − resnet", "HyperAdapt − ResNet", "#61a5c2"),
    ("const − resnet", "Constant-condition Spatial LoRA − ResNet", "#a3a3a3"),
    ("ceiling", "ResNet + attributes, late fusion (val-fit)", "#e07a5f"),
]


def bar_values() -> pd.DataFrame:
    """データセット × 棒の値と 95% 区間。後付けは点推定だけ（区間は持たない）。"""
    gains = pd.read_csv(PACKAGE / "results" / "gains.csv").query("split == 'test'")
    info = pd.read_csv(PACKAGE / "results" / "attribute_information.csv")
    ceiling = info[info.quantity == "late_fusion_valfit_gain_resnet"]
    ceiling = ceiling.assign(contrast="ceiling", delta=ceiling.value)
    table = pd.concat([gains, ceiling[["dataset", "contrast", "delta"]]], ignore_index=True)
    return table.set_index(["dataset", "contrast"])


def main() -> None:
    values = bar_values()
    plt.style.use(STYLE_SHEET)
    fig, ax = plt.subplots(figsize=(7.5, 3.6))
    width = 0.2
    for i, (key, label, color) in enumerate(BARS):
        xs, ys, errs = [], [], []
        for j, dataset in enumerate(DATASETS):
            if (dataset, key) not in values.index:
                continue
            row = values.loc[(dataset, key)]
            xs.append(j + (i - 1.5) * width)
            ys.append(row.delta)
            has_ci = not np.isnan(row.get("lo", np.nan))
            errs.append([row.delta - row.lo, row.hi - row.delta] if has_ci else [0.0, 0.0])
        ax.bar(xs, ys, width * 0.92, color=color, label=label)
        ax.errorbar(xs, ys, yerr=np.array(errs).T, fmt="none", ecolor="#333333", elinewidth=1, capsize=2)
    ax.axhline(0, color="#333333", linewidth=0.8)
    ax.set_xticks(range(len(DATASETS)), list(DATASETS.values()))
    ax.set_ylabel("Δ test AUROC")
    ax.legend(loc="upper right", frameon=False, fontsize=8)
    out = PACKAGE / "figures" / "gain_vs_attribute_ceiling.png"
    out.parent.mkdir(exist_ok=True)
    fig.savefig(out)
    print(f"wrote {out.relative_to(PACKAGE)}")


if __name__ == "__main__":
    main()
