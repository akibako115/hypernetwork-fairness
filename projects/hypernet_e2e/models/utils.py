from collections.abc import Mapping

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
    attributes: Mapping[str, torch.Tensor],
    chunk_size: int = 16384,
) -> float:
    """MetadataEncoder 出力の分散を train split の属性から推定する。

    HyperLinearLayer と Spatial LoRA の B 生成器の初期化に使う Var(c) を返す。Var(c) は属性列
    だけで決まるため、画像を読む DataLoader ではなく split 全行分の属性テンソルを直接受ける。
    MetadataEncoder は dropout を持つので eval に切り替え、呼び出し前の train/eval 状態へ戻す。

    Args:
        metadata_encoder: 属性辞書を embedding に変換する encoder
        attributes: `categorical` / `categorical_missing` / `continuous` / `continuous_missing`
            のうち encoder が必要とするもの。各値は `[n_rows, n_attributes]` で行数が揃っていること
        chunk_size: encoder に一度に通す行数。全行分の中間活性を同時に確保しないための分割幅

    Returns:
        float: embedding 各次元の分散の平均。分散が求まらない場合は 1.0 を返す
    """
    if not attributes:
        raise ValueError("Var(c) の推定には属性が1つ以上必要")
    num_rows = next(iter(attributes.values())).size(0)
    if num_rows == 0:
        raise ValueError("Var(c) の推定には1行以上の属性が必要")

    was_training = metadata_encoder.training
    metadata_encoder.eval()
    device = next(metadata_encoder.parameters()).device
    embeddings = []
    # 全行を一度に通すと中間活性が n_rows x hidden_dim になるため、固定幅に分割して積む
    for start in range(0, num_rows, chunk_size):
        chunk = {key: value[start : start + chunk_size].to(device) for key, value in attributes.items()}
        embeddings.append(metadata_encoder(chunk).cpu())

    metadata_encoder.train(was_training)
    # 1 行しかない場合 var は nan になる。比較が False になるので既定の 1.0 に落ちる。
    result = float(torch.cat(embeddings, dim=0).var(dim=0).mean())
    return result if result > 0 else 1.0
