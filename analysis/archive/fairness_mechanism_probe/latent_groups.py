"""潜在サブグループ：凍結 M01 表現のクラスタと人口統計 8 群で、性能のばらつきの大きさを比べる。

問い：人口統計の 8 群は、モデルの誤りの不均一さのどれだけを捉えているか。表現空間のクラスタの
ほうが大きな性能差を持つなら、属性だけで公平性を定義すると最悪の部分集合を見落とす。

- k-means（train 表現で学習）を test に割り当て、群ごとの AUROC・陽性率を計算する。
- 符号付き margin（陽性は s、陰性は −s）の分散のうち、群分けで説明される割合 η² を比べる。
  クラス内で計算し、陽性率の差が η² に入らないようにする。

出力: `results/latent_groups/{clusters,eta2}_s<seed>.csv`

    uv run python analysis/archive/fairness_mechanism_probe/latent_groups.py --seed 42
"""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd
from features import GROUP_NAMES, PACKAGE, load_split
from sklearn.cluster import MiniBatchKMeans
from sklearn.decomposition import PCA
from sklearn.metrics import roc_auc_score


def eta_squared(value: np.ndarray, label: np.ndarray) -> float:
    """value の全分散のうち、label の群平均で説明される割合。"""
    total = value.var()
    means = pd.Series(value).groupby(label).transform("mean").to_numpy()
    return float(means.var() / total)


def within_class_eta(margin: np.ndarray, target: np.ndarray, label: np.ndarray) -> dict[str, float]:
    return {
        f"eta2_{name}": eta_squared(margin[target == c], label[target == c]) for c, name in ((1, "pos"), (0, "neg"))
    }


def cluster_table(score, target, cluster, frame) -> pd.DataFrame:
    rows = []
    for c in np.unique(cluster):
        m = cluster == c
        n_pos, n_neg = int(target[m].sum()), int((1 - target[m]).sum())
        auc = roc_auc_score(target[m], score[m]) if min(n_pos, n_neg) >= 10 else np.nan
        rows.append(
            dict(
                cluster=c,
                n=int(m.sum()),
                n_pos=n_pos,
                prev=target[m].mean(),
                auc=auc,
                tpr=(score[m][target[m] == 1] > 0).mean() if n_pos else np.nan,
                tnr=(score[m][target[m] == 0] <= 0).mean() if n_neg else np.nan,
                age=frame.age[m].mean(),
                old=(frame.age[m] >= 65).mean(),
                female=(frame.sex[m] == 1).mean(),
                lateral=(frame.frontal_lateral[m] == 1).mean(),
                ap=(frame.ap_pa[m] == 1).mean(),
                white=(frame.race_group[m] == 0).mean(),
                black=(frame.race_group[m] == 2).mean(),
                asian=(frame.race_group[m] == 1).mean(),
            )
        )
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--k", type=int, nargs="+", default=[8, 32, 128])
    args = parser.parse_args()
    train, test = load_split(args.seed, "train"), load_split(args.seed, "test")
    mean, std = train.features.mean(0), train.features.std(0) + 1e-6
    pca = PCA(64, random_state=0).fit((train.features[::4] - mean) / std)
    z_train = pca.transform((train.features - mean) / std)
    z_test = pca.transform((test.features - mean) / std)
    score = test.logits[:, 1] - test.logits[:, 0]
    margin = np.where(test.target == 1, score, -score)
    frame = test.frame
    labels = {
        "demographic_8": test.group,
        "age_decade": (frame.age // 10).clip(1, 9).to_numpy(),
        "view_appa": (frame.frontal_lateral * 2 + frame.ap_pa).to_numpy(),
        "demo8_x_view": test.group * 4 + (frame.frontal_lateral * 2 + frame.ap_pa).to_numpy(),
    }
    out = PACKAGE / "results" / "latent_groups"
    out.mkdir(parents=True, exist_ok=True)
    for k in args.k:
        km = MiniBatchKMeans(k, random_state=args.seed, n_init=3, batch_size=8192).fit(z_train)
        labels[f"kmeans_{k}"] = km.predict(z_test)
        cluster_table(score, test.target, labels[f"kmeans_{k}"], frame).to_csv(
            out / f"clusters_k{k}_s{args.seed}.csv", index=False
        )
    rows = []
    for name, label in labels.items():
        aucs = []
        for c in np.unique(label):
            m = label == c
            if min(test.target[m].sum(), (1 - test.target[m]).sum()) >= 20:
                aucs.append(roc_auc_score(test.target[m], score[m]))
        rows.append(
            dict(
                partition=name,
                n_cells=len(np.unique(label)),
                n_auc_cells=len(aucs),
                auc_min=min(aucs),
                auc_max=max(aucs),
                auc_sd=np.std(aucs),
                **within_class_eta(margin, test.target, label),
            )
        )
    table = pd.DataFrame(rows)
    table.to_csv(out / f"eta2_s{args.seed}.csv", index=False)
    print(table.round(3).to_string())
    print(GROUP_NAMES)


if __name__ == "__main__":
    main()
