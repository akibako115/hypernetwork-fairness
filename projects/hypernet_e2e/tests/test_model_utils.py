import pytest
import torch
import torch.nn as nn

from projects.hypernet_e2e.models.spatial_hypernet.metadata_encoder import MetadataEncoder
from projects.hypernet_e2e.models.utils import compute_embedding_variance, load_compatible_state_dict


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


def _variance_encoder() -> MetadataEncoder:
    return MetadataEncoder(
        categorical_cardinalities=[2],
        num_continuous=0,
        embedding_dim=2,
        hidden_dim=4,
        output_dim=3,
        dropout=0.5,
    )


def _variance_attributes(num_rows: int) -> dict[str, torch.Tensor]:
    return {
        "categorical": (torch.arange(num_rows) % 2).unsqueeze(1),
        "categorical_missing": torch.zeros(num_rows, 1, dtype=torch.bool),
        "continuous": torch.empty(num_rows, 0),
        "continuous_missing": torch.empty(num_rows, 0, dtype=torch.bool),
    }


def test_compute_embedding_variance_uses_eval_mode_and_restores_training_state() -> None:
    encoder = _variance_encoder()
    encoder.train()

    variance = compute_embedding_variance(encoder, _variance_attributes(4))

    assert variance > 0
    assert encoder.training


def test_compute_embedding_variance_is_independent_of_the_chunk_width() -> None:
    """chunk 分割は中間活性を抑えるためだけの都合であり、Var(c) の値を変えてはならない。"""
    encoder = _variance_encoder()
    attributes = _variance_attributes(50)

    assert compute_embedding_variance(encoder, attributes, chunk_size=7) == pytest.approx(compute_embedding_variance(encoder, attributes, chunk_size=1024))


def test_compute_embedding_variance_rejects_attributes_without_rows() -> None:
    encoder = _variance_encoder()

    with pytest.raises(ValueError):
        compute_embedding_variance(encoder, {})
    with pytest.raises(ValueError):
        compute_embedding_variance(encoder, _variance_attributes(0))
