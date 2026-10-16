"""部分群ごとに判定の規則が食い違う場を、HAM10000 に人工的に作る（振り分けの正の対照）。

画像上の小さなマーカー（`planting.stamp` の白い正方形）の意味を、DINOv2 の GMM 8 クラスタ（`gmm8`、画像だけから作った
ラベル不使用のクラスタ）で反転させる。
- 多数側 A（クラスタ 0・1・3・5・6・7）：陽性の 50%、陰性の 10% に付ける（マーカー → 悪性）
- 少数側 B（クラスタ 2・4。train の陽性の 27%）：陽性の 10%、陰性の 50% に付ける（マーカー → 良性）
全体ではマーカーは悪性寄りになるので、共有のモデルは多数側の規則に引かれ、B で誤る。
クラスタで振り分けるモデルが B の規則を守れるかを見る。
`--dose strong` は付けやすい側の割合を 90% に強めた版（`images_rule_strong/`）。
同じ hash を使うので、元の版でマーカーが付いた画像には、この版でも付く。

規則はデータの生成過程そのものとして train・val・test のすべてに当てる（展開先でも同じ規則が成り立つ場）。
どの画像に付けるかは image path の hash で決める（`planting._uniform`）。
JPEG の再圧縮がもう 1 つの手がかりにならないよう、マーカーの有無によらず全画像を同じ品質で書き直す。

入力: `data/ham10000/images/`、`data/ham10000/splits_latent/{train,val,test}.csv`
出力: `data/ham10000/images_rule/`（全画像。マーカー付きは印を描いたもの）、`--dose strong` では `images_rule_strong/`

    uv run python -m analysis.archive.latent_bias_hypernet.rule_planting [--dose strong]
"""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from PIL import Image

from analysis.archive.latent_bias_hypernet.data import ROOT
from analysis.archive.latent_bias_hypernet.planting import _uniform, stamp

MINORITY = (2, 4)
# 版ごとの（付けやすい側の割合, 付けにくい側の割合, 書き出し先）
DOSES = {
    "base": (0.5, 0.1, ROOT / "data" / "ham10000" / "images_rule"),
    "strong": (0.9, 0.1, ROOT / "data" / "ham10000" / "images_rule_strong"),
}
HIGH, LOW, DEST = DOSES["base"]
QUALITY = 95
SOURCE = ROOT / "data" / "ham10000" / "images"


def minority(frame: pd.DataFrame) -> np.ndarray:
    """少数側 B（規則が反転するクラスタ）に入るか。"""
    return frame.gmm8.isin(MINORITY).to_numpy()


def marker_mask(frame: pd.DataFrame, dose: str = "base") -> np.ndarray:
    """各行にマーカーを付けるか。A では陽性ほど、B では陰性ほど付きやすい。"""
    high, low, _ = DOSES[dose]
    u = frame.image.map(_uniform).to_numpy()
    agree = (frame.target.to_numpy() == 1) != minority(frame)
    return u < np.where(agree, high, low)


def write(image: str, marked: bool, dest) -> None:
    with Image.open(SOURCE / image) as img:
        out = img.convert("RGB")
    (stamp(out) if marked else out).save(dest / image, quality=QUALITY)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dose", choices=list(DOSES), default="base")
    dose = parser.parse_args().dose
    dest = DOSES[dose][2]
    dest.mkdir(exist_ok=True)
    frames = {
        s: pd.read_csv(ROOT / "data" / "ham10000" / "splits_latent" / f"{s}.csv") for s in ("train", "val", "test")
    }
    jobs = []
    for split, frame in frames.items():
        mark = marker_mask(frame, dose)
        table = pd.crosstab([minority(frame), frame.target], mark)
        print(f"{split}：[B か, クラス] × マーカー\n{table.to_string()}")
        jobs += [delayed(write)(image, m, dest) for image, m in zip(frame.image, mark, strict=True)]
    Parallel(n_jobs=16)(jobs)


if __name__ == "__main__":
    main()
