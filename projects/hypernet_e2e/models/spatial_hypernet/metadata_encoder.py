from collections.abc import Mapping, Sequence

import torch
import torch.nn as nn


class MetadataEncoder(nn.Module):
    """属性辞書を固定長の埋め込みベクトルに変換する共通エンコーダ。"""

    def __init__(
        self,
        categorical_cardinalities: Sequence[int],
        num_continuous: int,
        embedding_dim: int,
        hidden_dim: int,
        output_dim: int,
        dropout: float = 0.1,
        num_layers: int = 2,
    ):
        """categorical_cardinalities と num_continuous から埋め込み層とMLPを構築する。"""
        super().__init__()
        self.num_continuous = num_continuous
        self.output_dim = output_dim

        if not categorical_cardinalities and num_continuous == 0:
            raise ValueError("少なくとも1つのカテゴリカルまたは連続属性が必要")
        if any(c <= 0 for c in categorical_cardinalities):
            raise ValueError("categorical_cardinalities は正の整数のみ")
        if num_layers < 1:
            raise ValueError(f"num_layers は 1 以上が必要だが {num_layers} が指定された")

        # カテゴリカル属性の埋め込み
        self.categorical_embeddings = nn.ModuleList([nn.Embedding(c + 1, embedding_dim) for c in categorical_cardinalities])

        # 欠損値のインデックス (+1の値を欠損値のインデックスとして使用)
        self.register_buffer(
            "missing_indices",
            torch.tensor(list(categorical_cardinalities), dtype=torch.long),
        )

        # MLPの入力次元
        mlp_input_dim = len(categorical_cardinalities) * embedding_dim
        if num_continuous > 0:
            mlp_input_dim += num_continuous * 2

        # MLPの定義。num_layers は Linear 層の総数（隠れ層は num_layers - 1 個）。
        layers: list[nn.Module] = []
        in_dim = mlp_input_dim
        for _ in range(num_layers - 1):
            layers.append(nn.Linear(in_dim, hidden_dim))
            layers.append(nn.ReLU(inplace=True))
            layers.append(nn.Dropout(dropout))
            in_dim = hidden_dim
        layers.append(nn.Linear(in_dim, output_dim))
        self.mlp = nn.Sequential(*layers)

    def forward(self, attributes: Mapping[str, torch.Tensor]) -> torch.Tensor:
        """
        属性辞書からメタデータ埋め込みを生成する。

        Args:
            attributes: 属性辞書
                - categorical: カテゴリカル属性 (B, num_categorical)
                - categorical_missing: カテゴリカル属性の欠損値フラグ (B, num_categorical)
                - continuous: 連続属性 (B, num_continuous)
                - continuous_missing: 連続属性の欠損値フラグ (B, num_continuous)

        Returns:
            (B, output_dim) のテンソル
        """
        parts: list[torch.Tensor] = []

        if self.categorical_embeddings:
            cat = attributes["categorical"]
            cat_missing = attributes["categorical_missing"].bool()

            # 欠損値を変換
            cat = torch.where(cat_missing, self.missing_indices, cat)

            # カテゴリカル属性の埋め込み
            for i, emb in enumerate(self.categorical_embeddings):
                parts.append(emb(cat[:, i]))

        if self.num_continuous > 0:
            cont = attributes["continuous"]
            cont_missing = attributes["continuous_missing"]
            parts.append(cont)
            parts.append(cont_missing.float())

        return self.mlp(torch.cat(parts, dim=1))
