import pytest
import torch

from projects.hypernet_e2e.models.spatial_hypernet.metadata_encoder import MetadataEncoder


def test_metadata_encoder_embeds_attributes_and_uses_missing_category() -> None:
    torch.manual_seed(0)
    encoder = MetadataEncoder(
        categorical_cardinalities=[3],
        num_continuous=1,
        embedding_dim=2,
        hidden_dim=4,
        output_dim=5,
        dropout=0,
    ).eval()
    attributes = {
        "categorical": torch.tensor([[0], [1]]),
        "categorical_missing": torch.tensor([[True], [False]]),
        "continuous": torch.tensor([[0.5], [1.5]]),
        "continuous_missing": torch.tensor([[False], [False]]),
    }
    same_missing_value = {**attributes, "categorical": torch.tensor([[2], [1]])}

    with torch.no_grad():
        output = encoder(attributes)
        output_with_different_placeholder = encoder(same_missing_value)

    assert output.shape == (2, 5)
    torch.testing.assert_close(output[0], output_with_different_placeholder[0])


def test_metadata_encoder_rejects_empty_attribute_specification() -> None:
    with pytest.raises(ValueError, match="少なくとも1つ"):
        MetadataEncoder(
            categorical_cardinalities=[],
            num_continuous=0,
            embedding_dim=2,
            hidden_dim=4,
            output_dim=5,
        )
