"""準備済みモデルの受け渡しと、DINOv2 / RAD-DINO 共通の特徴抽出。"""

from collections.abc import Callable
from dataclasses import dataclass

import torch
from PIL import Image
from torch import nn


@dataclass
class LoadedEncoder:
    """凍結モデル、画像1枚の前処理、特徴次元、JSON 化可能な再現情報。"""

    name: str
    module: nn.Module
    transform: Callable[[Image.Image], torch.Tensor]
    embed_dim: int
    metadata: dict


class DinoCLS(nn.Module):
    """DINOv2 / RAD-DINO の最終層 CLS を返す。patch 平均や正規化は加えない。"""

    def __init__(self, backbone: nn.Module) -> None:
        """特徴を取り出す backbone を保持する。"""
        super().__init__()
        self.backbone = backbone

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """(B, 3, H, W) から (B, hidden_size) の CLS を返す。"""
        return self.backbone(pixel_values=pixel_values).last_hidden_state[:, 0]


@dataclass
class DinoProcessor:
    """公式 processor を worker 内で呼ぶ。変換の独自再構成はしない。"""

    processor: object

    def __call__(self, image: Image.Image) -> torch.Tensor:
        """RGB 画像1枚を処理し、バッチ軸だけを外す。"""
        return self.processor(images=image, return_tensors="pt")["pixel_values"][0]
