"""ISIC 2019 で、施設をまたぐ実際のずれの検証用 split を作る（メラノーマの 2 値分類）。

植え込みに頼らず、「学習施設のモデルを別施設に展開し、展開先で小さなラベル付き監査セットを取る」状況を再現する。

- A：BCN20000（`attribution == 1`）。病変単位で 90% を `isic_train_a`（1 段目の学習）、10% を `isic_val_a`（選択）
- 展開先：HAM10000（`attribution == 2`）。病変単位で 40% を `isic_audit`（監査セット B の母集団）、
  20% を `isic_audit_v`（補正手法と独立な監査器の学習・較正）、40% を `isic_test`
- target：メラノーマなら 1（元の 8 クラスの 0 = MEL）
- 属性：sex（0 = female、1 = male、欠損 -1）、age（`age_approx`）。部位（`anatom_site_general`）は
  属性ではない要因として残す

元の `data/isic2019/splits_fairness/` を全行まとめて使い直す（ここでの split は既存の fairness study とは別物）。
`lesion_id` を持たない画像は 1 枚ずつ独立した病変として扱う。

出力: `cache/context/isic_<split>.csv`（latent_bias_hypernet の data.load がそのまま読める）

    uv run python -m analysis.archive.latent_bias_hypernet.isic_sites
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from analysis.archive.latent_bias_hypernet.data import CACHE, ROOT

SOURCE = ROOT / "data" / "isic2019" / "splits_fairness"
SEED = 0


def lesion_split(frame: pd.DataFrame, fractions: dict[str, float], seed: int) -> pd.Series:
    """病変単位で割り当てる。"""
    lesion = frame.lesion_id.fillna(frame.image)
    uniq = np.sort(lesion.unique())
    order = np.random.default_rng(seed).permutation(uniq)
    edges = np.cumsum([0, *fractions.values()]) * len(order)
    label = {}
    for name, lo, hi in zip(fractions, edges[:-1], edges[1:], strict=True):
        for les in order[int(round(lo)) : int(round(hi))]:
            label[les] = name
    return lesion.map(label)


def main() -> None:
    frame = pd.concat([pd.read_csv(SOURCE / f"{s}.csv") for s in ("train", "val", "test")], ignore_index=True)
    frame["target"] = (frame.target == 0).astype(int)
    frame["age"] = frame.age_approx
    out = CACHE / "context"
    out.mkdir(parents=True, exist_ok=True)
    keep = ["image", "target", "sex", "age", "anatom_site_general", "lesion_id", "attribution"]
    bcn = frame[frame.attribution == 1].copy()
    bcn["split"] = lesion_split(bcn, {"isic_train_a": 0.9, "isic_val_a": 0.1}, SEED)
    ham = frame[frame.attribution == 2].copy()
    ham["split"] = lesion_split(ham, {"isic_audit": 0.4, "isic_audit_v": 0.2, "isic_test": 0.4}, SEED)
    both = pd.concat([bcn, ham])
    for split, part in both.groupby("split"):
        part = part.sort_values("image")[keep].reset_index(drop=True)
        part.to_csv(out / f"{split}.csv", index=False)
        print(split, len(part), "melanoma", round(part.target.mean(), 3), "lesions", part.lesion_id.nunique())


if __name__ == "__main__":
    main()
