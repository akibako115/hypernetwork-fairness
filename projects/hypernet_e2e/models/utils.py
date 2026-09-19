import torch
import torch.nn as nn


def load_compatible_state_dict(model: nn.Module, state_dict: dict, load_fc: bool = True) -> dict:
    """同名かつ同 shape のキーだけをモデルに部分ロードする。

    Returns:
        dict: loaded_keys / skipped_keys / missing_keys / unexpected_keys を key に持つ結果 dict。
    """
    model_state = model.state_dict()
    filtered = {}
    skipped = []

    for key, value in state_dict.items():
        if not load_fc and key.startswith("fc."):
            continue
        if key not in model_state:
            skipped.append(key)
            continue
        if model_state[key].shape != value.shape:
            skipped.append(key)
            continue
        filtered[key] = value

    incompatible = model.load_state_dict(filtered, strict=False)
    return {
        "loaded_keys": sorted(filtered.keys()),
        "skipped_keys": sorted(skipped),
        "missing_keys": sorted(incompatible.missing_keys),
        "unexpected_keys": sorted(incompatible.unexpected_keys),
    }


@torch.no_grad()
def compute_embedding_variance(
    metadata_encoder: nn.Module,
    dataloader: torch.utils.data.DataLoader,
) -> float:
    """MetadataEncoder 出力の分散を学習データから推定する。

    HyperLinearLayer の初期化に使用する。
    """
    was_training = metadata_encoder.training
    metadata_encoder.eval()
    device = next(metadata_encoder.parameters()).device
    embeddings = []
    # 各batchのmetadata embeddingを集めてvarianceを推定する
    for batch in dataloader:
        _, attributes, _ = batch
        attributes = {k: v.to(device) for k, v in attributes.items()}
        embeddings.append(metadata_encoder(attributes).cpu())

    metadata_encoder.train(was_training)
    all_emb = torch.cat(embeddings, dim=0)
    result = float(all_emb.var(dim=0).mean())
    return result if result > 0 else 1.0
