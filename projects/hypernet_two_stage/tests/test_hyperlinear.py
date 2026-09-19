import pytest
import torch

from projects.hypernet_two_stage.models.spatial_hypernet.modulation import HyperLinearLayer


def test_hyperlinear_starts_as_its_shared_classifier() -> None:
    layer = HyperLinearLayer(condition_dim=3, in_features=5, out_features=2, rank=2)
    features = torch.randn(2, 5)
    condition = torch.randn(2, 3)

    assert torch.count_nonzero(layer.a_generator.weight) == 0
    assert torch.count_nonzero(layer.a_generator.bias) == 0
    assert torch.equal(layer(features, condition), layer.base_linear(features))


def test_hyperlinear_low_rank_path_matches_materialized_delta() -> None:
    layer = HyperLinearLayer(condition_dim=3, in_features=5, out_features=2, rank=2, lora_alpha=2.0)
    layer.a_generator.bias.data.fill_(1.0)
    features = torch.randn(2, 5)
    condition = torch.randn(2, 3)

    expected = layer.base_linear(features) + torch.bmm(layer.weight_delta(condition), features.unsqueeze(-1)).squeeze(-1)

    assert layer.scale == 1.0
    torch.testing.assert_close(layer(features, condition), expected)


def test_hyperlinear_rejects_feature_dimension_mismatch() -> None:
    layer = HyperLinearLayer(condition_dim=3, in_features=5, out_features=2)

    with pytest.raises(ValueError, match="expected 5"):
        layer(torch.randn(2, 4), torch.randn(2, 3))
