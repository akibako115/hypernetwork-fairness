"""DINOの特徴定義と公式processorへの受け渡しを、重み取得なしで確認する。"""

from types import SimpleNamespace

import numpy as np
import pytest
import torch
from PIL import Image
from torch import nn
from transformers import BitImageProcessor

from projects.foundation_linear_probe.encoders import DinoCLS, DinoProcessor


def test_dino_returns_cls_without_pooling_or_normalization():
    """patch平均や正規化を混ぜず、最初のトークンを取り出す。"""

    class Backbone(nn.Module):
        """既知のトークン列を返す。"""

        def forward(self, pixel_values):
            """バッチごとに同じトークン列を返す。"""
            return SimpleNamespace(last_hidden_state=torch.arange(12).reshape(1, 4, 3).expand(len(pixel_values), -1, -1))

    result = DinoCLS(Backbone())(torch.zeros(2, 3, 16, 16))
    assert torch.equal(result, torch.tensor([[0, 1, 2], [0, 1, 2]]))


@pytest.mark.parametrize("shape", [(71, 39), (39, 71), (32, 32)])
@pytest.mark.parametrize("mode", ["RGB", "L"])
def test_processor_matches_direct_call(shape, mode):
    """非一様・縦横比の異なる画像でも公式processorの結果を保つ。"""
    processor = BitImageProcessor(size={"shortest_edge": 32}, crop_size={"height": 28, "width": 28})
    pixels = np.arange(shape[0] * shape[1], dtype=np.uint8).reshape(shape)
    image = Image.fromarray(pixels).convert(mode).convert("RGB")
    expected = processor(images=image, return_tensors="pt")["pixel_values"][0]
    assert torch.equal(DinoProcessor(processor)(image), expected)
