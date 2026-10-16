"""HAM10000・FairVLMed の ResNet-50 ERM で、属性間の AUROC 差と、しきい値で測る差（TPR・FPR）を並べる。

方向 5（CheXpert の年齢差はラベルの付け方の違い）を報告するときの補足。ほかのデータセットでも、
順位の差は動作点の差より小さいかを確かめる。新しい学習・推論はしない。

- 予測：各 package の既存 cache（seed 42・43・44 の logit 平均）。run の選択と検算は各 package の comparison.py に任せる。
- しきい値：各 package の群別指標と同じ 0.5（HAM は class weight つき、FairVLMed は陽性率 50%）。
- 区間：HAM は病変単位、FairVLMed は画像（= 患者）単位の bootstrap 1,000 回の 95%。
- 評価集合：HAM は性別・年齢の揃う 1,217 件（package の主比較と同じ）、FairVLMed は test 全 2,000 件。

出力：results/d5/other_datasets.csv
実行例：uv run python -m analysis.archive.direction_poc.d5_other_datasets
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from analysis.fairvlmed_architecture_erm import comparison as fv
from analysis.ham10000_architecture_erm import comparison as ham

# analysis/。この file は analysis/archive/direction_poc/ にある
ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).parent / "results/d5/other_datasets.csv"
BOOT = 1000
# 符号つきの差を取る 2 群（前 − 後）。race は群別 AUROC が最も低い Black を White と比べる
PAIRS = {"age_group_65": ("<65", "≥65"), "sex": ("female", "male"), "gender": ("female", "male"), "race": ("White", "Black")}


def seed_mean(scores: dict) -> np.ndarray:
    """ResNet-50 の 3 seed の logit 平均を返す。"""
    p = np.stack([s for (model, _), s in scores.items() if model == "resnet"])
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p)).mean(axis=0)


def gaps(y: np.ndarray, z: np.ndarray, g: np.ndarray, names: tuple, pair: tuple) -> dict:
    """群ごとの AUROC・TPR・FPR と、pair の 2 群の符号つきの差（pair[0] − pair[1]）を返す（しきい値 logit 0）。

    最大 − 最小は常に正なので、bootstrap 区間が 0 を含むかで差の有無を読めない。符号つきの差にする。
    """
    out = {}
    for k, name in enumerate(names):
        m = g == k
        out[f"auroc {name}"] = roc_auc_score(y[m], z[m])
        out[f"tpr {name}"] = (z[m & (y == 1)] >= 0).mean()
        out[f"fpr {name}"] = (z[m & (y == 0)] >= 0).mean()
    a, b = pair
    for metric in ("auroc", "tpr", "fpr"):
        out[f"{metric} gap"] = out[f"{metric} {a}"] - out[f"{metric} {b}"]
    return out


def summarize(dataset: str, y, z, groups: dict, clusters: np.ndarray) -> list[dict]:
    _, inverse = np.unique(clusters, return_inverse=True)
    members = [np.flatnonzero(inverse == i) for i in range(inverse.max() + 1)]
    rng = np.random.default_rng(0)
    draws = [np.concatenate([members[i] for i in rng.integers(0, len(members), len(members))]) for _ in range(BOOT)]
    rows = []
    for axis, (g, names, pair) in groups.items():
        keep = g >= 0
        point = gaps(y[keep], z[keep], g[keep], names, pair)
        boot = pd.DataFrame([gaps(y[i][keep[i]], z[i][keep[i]], g[i][keep[i]], names, pair) for i in draws])
        row = {"dataset": dataset, "axis": axis, "gap": f"{pair[0]} − {pair[1]}"}
        for name in names:
            m = keep & (g == names.index(name))
            row[f"n_pos {name}"] = int(y[m].sum())
        for key, value in point.items():
            row[key] = value
            if key.endswith("gap"):
                row[f"{key}_lo"], row[f"{key}_hi"] = boot[key].quantile([0.025, 0.975])
        rows.append(row)
    return rows


def main() -> None:
    rows = []
    package = ROOT / "ham10000_architecture_erm"
    _, frame, directories = ham.select_runs(package)
    z = seed_mean(ham.read_scores(package, directories, frame))
    mask = ~(frame["sex_missing"].astype(bool) | frame["age_missing"].astype(bool)).to_numpy()
    ids = ham.group_ids(frame)
    groups = {axis: (ids[axis][mask], ham.AXES[axis][1], PAIRS[axis]) for axis in ("age_group_65", "sex")}
    rows += summarize("HAM10000", frame["target"].to_numpy()[mask], z[mask], groups, frame["lesion_id"].to_numpy()[mask])

    package = ROOT / "fairvlmed_architecture_erm"
    _, frame, directories = fv.select_runs(package)
    z = seed_mean(fv.read_scores(package, directories, frame))
    ids = fv.group_ids(frame)
    groups = {axis: (ids[axis], fv.AXES[axis], PAIRS[axis]) for axis in ("race", "age_group_65", "gender")}
    rows += summarize("FairVLMed", frame["target"].to_numpy(), z, groups, np.arange(len(frame)))

    out = pd.DataFrame(rows)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUT, index=False)
    cols = ["dataset", "axis", "gap", "auroc gap", "auroc gap_lo", "auroc gap_hi", "tpr gap", "tpr gap_lo", "tpr gap_hi", "fpr gap", "fpr gap_lo", "fpr gap_hi"]
    print(out[cols].round(3).to_string(index=False))


if __name__ == "__main__":
    main()
