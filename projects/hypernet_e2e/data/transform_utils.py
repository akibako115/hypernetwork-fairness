"""CheXpert・HAM10000（MEDFAIR）・ISIC 2019・FairVLMed の画像前処理を定義する。

各 factory は PIL Image を受け取り、ImageNet 正規化済みの
`torch.float32` tensor を返す transform を構築する。
"""

from torchvision.transforms import transforms

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
# 回転の余白色。IMAGENET_MEAN を 0-255 スケールに変換して使う
IMAGENET_FILL = tuple(round(value * 255) for value in IMAGENET_MEAN)


########################################################
# 各データセット用のtransforms
########################################################


# 1. CheXpert・HAM10000（MEDFAIR と同じ前処理）
# MEDFAIR（Zong et al., ICLR 2023、github.com/ys-zong/MEDFAIR）は、2D 画像を前処理の段階で
# `cv2.resize(img, (256, 256))` により縦横比を保たずに 256x256 へ縮めて保存し（notebooks/HAM10000-example.ipynb）、
# 学習時に datasets/utils.py の get_dataset の transform を掛ける。2D データセットはすべて同じ transform を通る
# （CheXpert の前処理 notebook は公開されていない）。ここでは画像を読むたびに `_squash_to_256` で同じ形にしてから、
# MEDFAIR の transform を掛ける。MEDFAIR の HAM10000 の例は cv2 で読んだ BGR のまま渡しているが、ここは RGB で渡す。
# 2026-10-17 までは、CheXpert は CXR_Fairness の構成（train は RandomResizedCrop、評価は pad-to-square → 224）、
# HAM10000 は下の ISIC 2019 の構成（短辺 256、回転の余白は灰色）だった。


def _squash_to_256() -> list:
    """縦横比を保たずに 256x256 へ縮める段を返す。MEDFAIR の `cv2.resize(img, (256, 256))` に合わせる。

    cv2 の INTER_LINEAR はアンチエイリアスを掛けない。PIL の bilinear は掛けるので、PIL Image のまま
    Resize すると、胸部 X 線の文字マーカーのような鋭い縁で 1 画素あたり最大 110 階調ほどずれる。uint8 の
    tensor にして `antialias=False` で縮めると、cv2 との差は全画素で ±1 階調（丸めの差）に収まる。
    後段は MEDFAIR と同じく PIL Image に掛けるので、縮めたあと PIL Image に戻す。
    """
    return [
        transforms.PILToTensor(),
        transforms.Resize((256, 256), antialias=False),
        transforms.ToPILImage(),
    ]


def train_transforms_medfair() -> transforms.Compose:
    """CheXpert・HAM10000 の学習用 transform。MEDFAIR の `augment=True` の構成と同じ。

    256x256 に引き伸ばす（`_squash_to_256`）→ RandomHorizontalFlip → RandomRotation(±15, 余白は黒) →
    RandomCrop(224) → ImageNet normalize。

    Args:
        なし

    Returns:
        transforms.Compose: PIL Image を `[3, 224, 224]` の `torch.float32` へ変換する transform
    """
    return transforms.Compose(
        [
            *_squash_to_256(),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation((-15, 15)),
            transforms.RandomCrop((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )


def val_transforms_medfair() -> transforms.Compose:
    """CheXpert・HAM10000 の検証・テスト用 transform。MEDFAIR の transform_test と同じ。

    256x256 に引き伸ばす（`_squash_to_256`）→ CenterCrop(224) → ImageNet normalize。train と同じ 256x256 の画像から、
    train の切り出し範囲の中央を取る。

    Args:
        なし

    Returns:
        transforms.Compose: PIL Image を `[3, 224, 224]` の `torch.float32` へ変換する transform
    """
    return transforms.Compose(
        [
            *_squash_to_256(),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )


# 2. ISIC 2019（dermoscopy）
def train_transforms_isic2019() -> transforms.Compose:
    """ISIC 2019 学習用 transform。

    PAD-UFES-20 と同じ dermoscopy 向けの recipe（Resize(256) -> RandomHorizontalFlip ->
    RandomRotation(15) -> RandomCrop(224) -> ImageNet normalize）を使う。

    正方形への pad はしない。入力は黒縁を除去済みの画像であり、そこへ余白を足し直すと除去した意味が
    無くなる。短辺を 256 に揃えてから切り出す。

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


# 3. FairVLMed（SLO 眼底）
def train_transforms_fairvlmed() -> transforms.Compose:
    """FairVLMed 学習用 transform。

    評価側は FairCLIP と同じ「短辺 224 → CenterCrop(224)」で、学習側はその切り出し位置と向きを
    揺らす。画像は 512x664 の縦長で視神経乳頭がほぼ中央にあるので、縦方向の切り出しで乳頭は落ちない。

    正方形への pad はしない。背景がほぼ黒（画素の約 4 割が 10 未満）の画像に
    `IMAGENET_FILL` の灰色を足すと、実画像に無い明るい帯を作ってしまう。同じ理由で回転の余白は
    黒（0）で埋める。左右反転は右眼・左眼の入れ替えに相当し、どちらも実在する見え方なので入れる。

    Args:
        なし

    Returns:
        transforms.Compose: PIL Image を `[3, 224, 224]` の `torch.float32` へ変換する transform
    """
    return transforms.Compose(
        [
            transforms.Resize(224, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10, fill=0),
            transforms.RandomCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )


def val_transforms_fairvlmed() -> transforms.Compose:
    """FairVLMed 検証・テスト用 transform。

    FairCLIP が使う CLIP の前処理（bicubic で短辺 224 → CenterCrop(224)）と同じ切り出しにして、
    先行研究との比較可能性を保つ。正規化は他データセットと揃えて ImageNet の値を使う。

    Args:
        なし

    Returns:
        transforms.Compose: PIL Image を `[3, 224, 224]` の `torch.float32` へ変換する transform
    """
    return transforms.Compose(
        [
            transforms.Resize(224, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )
