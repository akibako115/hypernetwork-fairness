"""検証用に植え込む人工ショートカット（画像上の小さなマーカー）の規則。

発見手法が「渡していない要因」を回収できるかを、正解の分かっている要因で確かめるために使う。
1 段目のモデルが学習する split（train_a と、モデル選択に使う val）では、マーカーをラベルと相関させる。
展開先にあたる split（train_b と test）では、マーカーをラベルと独立に付ける。
こうすると 1 段目のモデルはマーカーを陽性の手がかりとして覚え、展開先のマーカー付き画像で系統的に誤る。

どの画像にマーカーを付けるかは image path の hash で決め、乱数の状態に依存させない。
"""

from __future__ import annotations

import hashlib

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw

# 1 段目が学習する側の規則：陽性の 50%、陰性の 5% に付ける。
CORRELATED = {1: 0.50, 0: 0.05}
# 展開先の規則：ラベルと独立に 20% に付ける。
INDEPENDENT = 0.20
CORRELATED_SPLITS = ("train_a", "val")


def _uniform(image: str) -> float:
    """image path から [0, 1) の一様値を決定的に作る。"""
    digest = hashlib.sha256(image.encode()).digest()
    return int.from_bytes(digest[:8], "little") / 2**64


def marker_mask(frame: pd.DataFrame, split: str) -> np.ndarray:
    """split の各行にマーカーを付けるかを返す。"""
    u = frame.image.map(_uniform).to_numpy()
    if split in CORRELATED_SPLITS:
        rate = np.where(frame.target.to_numpy() == 1, CORRELATED[1], CORRELATED[0])
        return u < rate
    return u < INDEPENDENT


def stamp(image: Image.Image) -> Image.Image:
    """画像の左上寄りに白い正方形を描く。RandomResizedCrop で切れにくいよう端から 15% 内側に置く。"""
    image = image.copy()
    width, height = image.size
    side = max(8, round(0.07 * min(width, height)))
    x0, y0 = round(0.15 * width), round(0.15 * height)
    ImageDraw.Draw(image).rectangle([x0, y0, x0 + side, y0 + side], fill=(255, 255, 255))
    return image


# 1 段目の学習 split で陽性を間引く部分集団。メタデータの属性セルと画像上の要因（レポート由来）を交差させ、
# 属性の掛け合わせで監査すると薄まる部分集団を作る。sex は 0 = 男性、1 = 女性、race_group は
# 0 = White、1 = Asian、2 = Black、3 = Others。
HIDDEN_SUBGROUPS = {
    "pacer": lambda c: c.pacer.to_numpy(dtype=bool),
    # 用量反応：属性への偏り方が違う、画像に写る要因（context_labels.REPORT_FACTORS）
    "picc": lambda c: c.picc.to_numpy(dtype=bool),
    "central_line": lambda c: c.central_line.to_numpy(dtype=bool),
    "sternotomy": lambda c: c.sternotomy.to_numpy(dtype=bool),
    "surgical_clip": lambda c: c.surgical_clip.to_numpy(dtype=bool),
    # 65 歳未満の女性のうち、レポートがデバイス（ライン・チューブ・ペースメーカー等）に触れる画像
    "young_female_device": lambda c: ((c.sex == 1) & (c.age < 65) & c.device).to_numpy(dtype=bool),
}


def hidden_subgroup(context: pd.DataFrame, rule: str) -> np.ndarray:
    """`cache/context/<split>.csv` の行に対して、部分集団に入るかを返す。"""
    return HIDDEN_SUBGROUPS[rule](context)
