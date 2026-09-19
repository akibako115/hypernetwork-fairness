"""共通 split CSV から画像・属性・target の sample を作る Dataset。"""

import os
from collections.abc import Mapping, Sequence

import pandas as pd
from PIL import Image
from torch.utils.data import Dataset

from .attribute_utils import attribute_tensors


class ImageDataset(Dataset):
    """画像・属性 DataFrame を `(image, attributes, target)` sample に変換する。

    入力 DataFrame は `image` と `target` を必須とし、`image` は `img_dir` からの相対パスとして
    そのまま解決する。`attribute_names` は model 入力、`fairness_attribute_names` は
    `evaluation_categorical` を構成する属性列と `{name}_missing` 列を指定する。画像は常に RGB に
    変換してから optional transform を適用する。
    """

    def __init__(
        self,
        df: pd.DataFrame,
        img_dir: str,
        transform=None,
        attribute_names: Mapping[str, Sequence[str]] | None = None,
        fairness_attribute_names: Mapping[str, Sequence[str]] | None = None,
    ):
        """画像・属性 DataFrame と transform を Dataset として保持する。"""
        self.df = df.reset_index(drop=True)
        self.img_dir = img_dir
        self.transform = transform
        self.attribute_names = attribute_names
        self.fairness_attribute_names = fairness_attribute_names

    def __len__(self) -> int:
        """データセットの件数を返す。"""
        return len(self.df)

    def __getitem__(self, idx: int):
        """指定 index の `(image, attributes, target)` を返す。"""
        # 画像データの読み込み
        row = self.df.iloc[idx]
        image_path = os.path.join(self.img_dir, row["image"])
        with Image.open(image_path) as img:
            image = img.convert("RGB")

        # 画像の前処理
        if self.transform:
            image = self.transform(image)

        # ターゲットの取得
        target = int(row["target"])

        # 属性の取得
        attributes = attribute_tensors(row, self.attribute_names)
        fairness_attributes = attribute_tensors(row, self.fairness_attribute_names)
        if "categorical" in fairness_attributes:
            attributes["evaluation_categorical"] = fairness_attributes["categorical"]
            attributes["evaluation_categorical_missing"] = fairness_attributes["categorical_missing"]

        return image, attributes, target
