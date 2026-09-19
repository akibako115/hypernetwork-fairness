import pytest
import torch

from projects.hypernet_two_stage.models.resnet.blocks import BasicBlock
from projects.hypernet_two_stage.models.resnet.network import ResNet, ResNetBackbone


def test_resnet_returns_logits_and_backbone_features() -> None:
    model = ResNet(layers=[1, 1, 1, 1], block="BasicBlock", num_classes=3).eval()
    image = torch.randn(2, 3, 64, 64)

    with torch.no_grad():
        logits = model(image)
        features = model.get_features(image)

    assert logits.shape == (2, 3)
    assert features.shape == (2, 512)


def test_resnet_without_classifier_returns_features() -> None:
    model = ResNet(layers=[1, 1, 1, 1], block=BasicBlock, num_classes=None).eval()

    with torch.no_grad():
        output = model(torch.randn(2, 3, 64, 64))

    assert model.fc is None
    assert output.shape == (2, 512)


def test_backbone_validates_architecture_arguments() -> None:
    with pytest.raises(ValueError, match="layers must have length 4"):
        ResNetBackbone(layers=[1, 1, 1], block="BasicBlock")

    with pytest.raises(ValueError, match="Unknown block name"):
        ResNetBackbone(layers=[1, 1, 1, 1], block="UnknownBlock")
