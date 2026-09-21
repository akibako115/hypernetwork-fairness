"""CheXpert の画像前処理を定義する。

各 factory は PIL Image を受け取り、ImageNet 正規化済みの
`torch.float32` tensor を返す transform を構築する。
"""

from PIL import ImageOps
from torchvision.transforms import transforms

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
# pad-to-square の余白色。IMAGENET_MEAN を 0-255 スケールに変換して使う
IMAGENET_FILL = tuple(round(value * 255) for value in IMAGENET_MEAN)


def _pad_to_square(image):
    """画像の短辺側を `IMAGENET_FILL` 色で対称パディングし、長辺に合わせた正方形にする。"""
    width, height = image.size
    if width == height:
        return image
    size = max(width, height)
    # 上下左右のパディング幅を算出する（割り切れない端数は右・下側に寄せる）
    pad_left = (size - width) // 2
    pad_top = (size - height) // 2
    pad_right = size - width - pad_left
    pad_bottom = size - height - pad_top
    return ImageOps.expand(
        image,
        border=(pad_left, pad_top, pad_right, pad_bottom),
        fill=IMAGENET_FILL,
    )


########################################################
# 各データセット用のtransforms
########################################################


# 1. CheXpert (CXR-Fairnessと同じ拡張)
def train_transforms_chexpert() -> transforms.Compose:
    """CheXpert 学習用 transform。

    CXR_Fairness（MLforHealth/CXR_Fairness）公開実装の `augment=1` 相当（
    RandomResizedCrop(224, scale=(0.75, 1.0)) -> RandomHorizontalFlip -> RandomRotation(10) ->
    ImageNet normalize）と同一構成にし、先行研究との比較可能性を保つ。

    Args:
        なし

    Returns:
        transforms.Compose: PIL Image を `[3, 224, 224]` の `torch.float32` へ変換する transform
    """
    return transforms.Compose(
        [
            transforms.RandomResizedCrop(224, scale=(0.75, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )


def val_transforms_chexpert() -> transforms.Compose:
    """CheXpert 検証・テスト用 transform。

    `_pad_to_square` でアスペクト比を保ったまま正方形化してから 224x224 にリサイズする。
    CXR_Fairness 公開実装は前処理段階で画像を 224x224 にキャッシュしてから ToTensor +
    normalize するが、本実装では transform 内で同等の入力形状を再現する。

    Args:
        なし

    Returns:
        transforms.Compose: PIL Image を `[3, 224, 224]` の `torch.float32` へ変換する transform
    """
    return transforms.Compose(
        [
            _pad_to_square,
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )
