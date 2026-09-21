"""CheXpert と ISIC 2019 の画像前処理を定義する。

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


# 2. ISIC 2019（dermoscopy）
def train_transforms_isic2019() -> transforms.Compose:
    """ISIC 2019 学習用 transform。

    PAD-UFES-20 と同じ dermoscopy 向けの recipe（Resize(256) -> RandomHorizontalFlip ->
    RandomRotation(15) -> RandomCrop(224) -> ImageNet normalize）を使う。

    CheXpert 側の `_pad_to_square` は使わない。入力は黒縁を除去済みの画像であり、
    そこへ余白を足し直すと除去した意味が無くなる。短辺を 256 に揃えてから切り出す。

    回転の余白も既定の黒ではなく `IMAGENET_FILL` で埋める。黒で埋めると、除去したはずの
    黒縁を augmentation が毎 epoch 描き直すことになる（既定の `fill=0` では黒画素の割合が
    1.6% から 2.8% へ増える）。

    **色を変える augmentation は使わない。**この実験は色恒常性を掛けていない画像を
    意図的に選んでおり、撮影サイトと相関する色かぶりを交絡として残したまま評価する。
    ColorJitter の hue/saturation を入れると、その交絡を augmentation 側で壊してしまう。
    肌色を ITA で後から推定する余地を残す意味でも色は触らない。

    Args:
        なし

    Returns:
        transforms.Compose: PIL Image を ImageNet 正規化済み tensor へ変換する transform
    """
    return transforms.Compose(
        [
            transforms.Resize(256),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15, fill=IMAGENET_FILL),
            transforms.RandomCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )


def val_transforms_isic2019() -> transforms.Compose:
    """ISIC 2019 検証・テスト用 transform。train と同じ Resize(256) 基準で CenterCrop(224) する。

    Args:
        なし

    Returns:
        transforms.Compose: PIL Image を ImageNet 正規化済み tensor へ変換する transform
    """
    return transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )
