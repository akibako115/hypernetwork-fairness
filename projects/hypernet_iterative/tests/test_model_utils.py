import torch
import torch.nn as nn

from projects.hypernet_iterative.models.spatial_hypernet.metadata_encoder import MetadataEncoder
from projects.hypernet_iterative.models.utils import compute_embedding_variance, load_compatible_state_dict


def test_load_compatible_state_dict_loads_only_matching_parameters() -> None:
    model = nn.Linear(3, 2)
    weight = torch.full_like(model.weight, 2.0)
    result = load_compatible_state_dict(
        model,
        {
            "weight": weight,
            "bias": torch.ones(3),
            "unknown": torch.ones(1),
        },
    )

    assert torch.equal(model.weight, weight)
    assert result["loaded_keys"] == ["weight"]
    assert result["skipped_keys"] == ["bias", "unknown"]


def test_compute_embedding_variance_uses_eval_mode_and_restores_training_state() -> None:
    encoder = MetadataEncoder(
        categorical_cardinalities=[2],
        num_continuous=0,
        embedding_dim=2,
        hidden_dim=4,
        output_dim=3,
        dropout=0.5,
    )
    encoder.train()
    attributes = {
        "categorical": torch.tensor([[0], [1], [0], [1]]),
        "categorical_missing": torch.zeros(4, 1, dtype=torch.bool),
        "continuous": torch.empty(4, 0),
        "continuous_missing": torch.empty(4, 0, dtype=torch.bool),
    }
    dataloader = [(torch.empty(4, 3, 1, 1), attributes, torch.tensor([0, 1, 0, 1]))]

    variance = compute_embedding_variance(encoder, dataloader)

    assert variance > 0
    assert encoder.training
