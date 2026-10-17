"""属性セルの組で判定の規則が反転する場を、HAM10000 に人工的に作る（ΔW による発見の正の対照）。

`latent_bias_hypernet/rule_planting.py` と同じマーカー（`planting.stamp` の白い正方形）を使い、
反転させる少数側 B だけを属性の組で決める。
- B：女性の四肢（lower / upper extremity）、または 60 歳未満の男性の背中。train の 24%、悪性率 0.151（全体 0.145）
- 多数側 A（B 以外。部位・性別・年齢の欠損行も A）：陽性の 90%、陰性の 10% に付ける（マーカー → 悪性）
- B：陽性の 10%、陰性の 90% に付ける（マーカー → 良性）

B は次の 3 つを満たすように選んだ。
1. 悪性率が全体とほぼ同じ：ΔW のクラスタが B を拾っても、それは「事前確率の違い」では説明できない。
2. 単一の属性でも、評価に使う性別 × 年齢（65 歳）の群でも切り出せない：既定の監査群では偏りが薄まる。
3. 属性入力だけで決まる：属性で条件付けたモデルなら、原理的に B の規則を学べる。

規則はデータの生成過程として train・val・test のすべてに当てる。どの画像に付けるかは image path の hash で決め、
JPEG の再圧縮が手がかりにならないよう、全画像を同じ品質で書き直す。

`--variant lower` は B を「下肢」の 1 属性値だけにした切り分け用の版（train の 21%、悪性率 0.123）。
単純な規則なのでモデルが確実に学べる。このとき ΔW のクラスタが反転を拾えるかで、「物差しが規則に鈍い」のか
「モデルが交互作用を学ばない」のかを分ける。

入力: `data/ham10000/images/`、`data/ham10000/splits/{train,val,test}.csv`
出力: `data/ham10000/images_attr_rule/`（`--variant lower` は `images_attr_rule_lower/`）

    uv run python -m analysis.archive.conditioned_subgroup_discovery.attribute_rule_planting [--variant lower]
"""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd
import rootutils
from joblib import Parallel, delayed
from PIL import Image

ROOT = rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from analysis.archive.latent_bias_hypernet.planting import _uniform, stamp  # noqa: E402

HIGH, LOW = 0.9, 0.1
QUALITY = 95
SOURCE = ROOT / "data" / "ham10000" / "images"
DESTS = {
    "cells": ROOT / "data" / "ham10000" / "images_attr_rule",
    "lower": ROOT / "data" / "ham10000" / "images_attr_rule_lower",
}
FEMALE, MALE = 0, 1
LIMBS = (9, 13)  # lower extremity、upper extremity
BACK = 2
LOWER = 9


def minority(frame: pd.DataFrame, variant: str = "cells") -> np.ndarray:
    """少数側 B に入るか。欠損した属性で B を判定しない（欠損行は A）。"""
    known = (frame.sex_missing == 0) & (frame.localization_missing == 0) & (frame.age_missing == 0)
    if variant == "lower":
        return (known & (frame.localization == LOWER)).to_numpy()
    female_limbs = (frame.sex == FEMALE) & frame.localization.isin(LIMBS)
    young_male_back = (frame.sex == MALE) & (frame.localization == BACK) & (frame.age < 60)
    return (known & (female_limbs | young_male_back)).to_numpy()


def marker_mask(frame: pd.DataFrame, variant: str = "cells") -> np.ndarray:
    """各行にマーカーを付けるか。A では陽性ほど、B では陰性ほど付きやすい。"""
    u = frame.image.map(_uniform).to_numpy()
    agree = (frame.target.to_numpy() == 1) != minority(frame, variant)
    return u < np.where(agree, HIGH, LOW)


def write(image: str, marked: bool, dest) -> None:
    with Image.open(SOURCE / image) as img:
        out = img.convert("RGB")
    (stamp(out) if marked else out).save(dest / image, quality=QUALITY)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", choices=list(DESTS), default="cells")
    variant = parser.parse_args().variant
    dest = DESTS[variant]
    dest.mkdir(exist_ok=True)
    jobs = []
    for split in ("train", "val", "test"):
        frame = pd.read_csv(ROOT / "data" / "ham10000" / "splits" / f"{split}.csv")
        mark = marker_mask(frame, variant)
        table = pd.crosstab([minority(frame, variant), frame.target], mark)
        print(f"{split}：[B か, クラス] × マーカー\n{table.to_string()}")
        jobs += [delayed(write)(image, m, dest) for image, m in zip(frame.image, mark, strict=True)]
    Parallel(n_jobs=16)(jobs)


if __name__ == "__main__":
    main()
