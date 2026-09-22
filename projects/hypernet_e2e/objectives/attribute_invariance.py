"""GRL による属性不変な backbone 学習 objective を定義する。"""

from collections.abc import Mapping, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from projects.hypernet_e2e.data.attribute_utils import ATTRIBUTE_KINDS, attribute_columns

from .input import ObjectiveInput


class _GradientReverse(torch.autograd.Function):
    """順伝播を恒等写像とし、逆伝播時だけ勾配の符号を反転する。"""

    @staticmethod
    def forward(ctx: torch.autograd.function.FunctionCtx, features: torch.Tensor, scale: float) -> torch.Tensor:
        """features を値を変えずに返し、逆伝播用の係数を保存する。"""
        ctx.scale = scale
        return features.view_as(features)

    @staticmethod
    def backward(ctx: torch.autograd.function.FunctionCtx, gradient: torch.Tensor) -> tuple[torch.Tensor, None]:
        """backbone に渡す勾配だけを反転・スケールする。"""
        return -ctx.scale * gradient, None


class _AttributeAdversary(nn.Module):
    """GRL 後の選択済み表現から属性を予測する内部 module。"""

    def __init__(
        self,
        feature_dim: int,
        categorical_cardinalities: Sequence[int],
        num_continuous: int,
        hidden_dim: int,
        gradient_scale: float,
    ) -> None:
        """属性の型・次元に合わせた予測ヘッドを構築する。"""
        super().__init__()
        if feature_dim < 1 or hidden_dim < 1:
            raise ValueError("feature_dim と hidden_dim は 1 以上である必要がある")
        if any(cardinality < 2 for cardinality in categorical_cardinalities):
            raise ValueError("categorical_cardinalities は 2 以上の整数だけを指定する必要がある")
        if num_continuous < 0:
            raise ValueError("num_continuous は 0 以上である必要がある")
        if not categorical_cardinalities and num_continuous == 0:
            raise ValueError("少なくとも1つの属性予測対象が必要")
        if gradient_scale < 0:
            raise ValueError("gradient_scale は 0 以上である必要がある")
        self.gradient_scale = gradient_scale
        self.trunk = nn.Sequential(nn.Linear(feature_dim, hidden_dim), nn.ReLU(inplace=True))
        self.categorical_heads = nn.ModuleList([nn.Linear(hidden_dim, cardinality) for cardinality in categorical_cardinalities])
        self.continuous_head = nn.Linear(hidden_dim, num_continuous) if num_continuous else None

    def forward(self, features: torch.Tensor, attributes: Mapping[str, torch.Tensor]) -> torch.Tensor:
        """選択済み属性の観測済み列に対する平均損失を返す。"""
        hidden = self.trunk(_GradientReverse.apply(features, self.gradient_scale))
        losses: list[torch.Tensor] = []
        if self.categorical_heads:
            categorical = attributes["categorical"]
            missing = attributes["categorical_missing"].bool()
            if categorical.ndim != 2 or categorical.shape[1] != len(self.categorical_heads) or missing.shape != categorical.shape:
                raise ValueError("categorical / categorical_missing の shape が属性予測器の設定と一致しない")
            for index, head in enumerate(self.categorical_heads):
                observed = ~missing[:, index]
                if observed.any():
                    losses.append(F.cross_entropy(head(hidden[observed]), categorical[observed, index].long()))
        if self.continuous_head is not None:
            continuous = attributes["continuous"]
            missing = attributes["continuous_missing"].bool()
            if continuous.ndim != 2 or continuous.shape[1] != self.continuous_head.out_features or missing.shape != continuous.shape:
                raise ValueError("continuous / continuous_missing の shape が属性予測器の設定と一致しない")
            predicted = self.continuous_head(hidden)
            for index in range(predicted.shape[1]):
                observed = ~missing[:, index]
                if observed.any():
                    losses.append(F.mse_loss(predicted[observed, index], continuous[observed, index]))
        return torch.stack(losses).mean() if losses else features.sum() * 0


class AttributeInvariantTaskLoss(nn.Module):
    """選択した属性に不変な backbone を学習する task loss と GRL 損失を合成する。

    `input_attribute_names` は batch attributes の full tensor の列順、
    `adversarial_attribute_names` はそのうち GRL で予測を妨げる列を表す。両者の型別の対応は
    初期化時に検証し、forward では選択済み列だけを内部属性予測器へ渡す。選択しない属性は
    属性損失にも backbone への逆向き勾配にも入らない。
    """

    requires_features = True

    def __init__(
        self,
        task_loss: nn.Module,
        feature_dim: int,
        input_attribute_names: Mapping[str, Sequence[str]],
        adversarial_attribute_names: Mapping[str, Sequence[str]],
        categorical_cardinalities: Sequence[int] = (),
        num_continuous: int = 0,
        hidden_dim: int = 256,
        gradient_scale: float = 1.0,
        attribute_adversary_weight: float = 1.0,
    ) -> None:
        """task loss、full attribute spec、および選択済み属性予測器を構築する。

        Args:
            task_loss: logits / target / attributes を受ける既存の分類 task loss。
            feature_dim: backbone 表現の次元。1 以上。
            input_attribute_names: full attribute tensor の categorical / continuous 列名と列順。
            adversarial_attribute_names: GRL で予測を妨げる categorical / continuous 列名。
            categorical_cardinalities: full categorical 属性列ごとのカテゴリ数。各要素は 2 以上。
            num_continuous: full continuous 属性列数。0 以上。
            hidden_dim: 内部属性予測器の隠れ次元。1 以上。
            gradient_scale: backbone へ逆向きに渡す勾配の倍率。0 以上。
            attribute_adversary_weight: task loss に加える属性予測損失の重み。0 以上。

        Returns:
            None

        Raises:
            ValueError: attribute spec、選択属性、または属性予測器の次元・係数が不正な場合。
        """
        super().__init__()
        if attribute_adversary_weight < 0:
            raise ValueError("attribute_adversary_weight は 0 以上である必要がある")
        full_categorical = attribute_columns(input_attribute_names, "categorical")
        full_continuous = attribute_columns(input_attribute_names, "continuous")
        cardinalities = list(categorical_cardinalities)
        if len(cardinalities) != len(full_categorical):
            raise ValueError("categorical_cardinalities は input_attribute_names.categorical と同じ長さである必要がある")
        if num_continuous != len(full_continuous):
            raise ValueError("num_continuous は input_attribute_names.continuous と同じ長さである必要がある")

        selected_indices: dict[str, list[int]] = {}
        for kind in ATTRIBUTE_KINDS:
            full_names = attribute_columns(input_attribute_names, kind)
            selected_names = attribute_columns(adversarial_attribute_names, kind)
            if len(set(selected_names)) != len(selected_names):
                raise ValueError(f"adversarial_attribute_names.{kind} に重複した属性名は指定できない")
            other_kind = "continuous" if kind == "categorical" else "categorical"
            wrong_kind = [name for name in selected_names if name in attribute_columns(input_attribute_names, other_kind)]
            if wrong_kind:
                raise ValueError(f"adversarial_attribute_names.{kind} に別種別の属性がある: {wrong_kind}")
            unknown = [name for name in selected_names if name not in full_names]
            if unknown:
                raise ValueError(f"adversarial_attribute_names.{kind} に input_attribute_names にない属性がある: {unknown}")
            selected_indices[kind] = [full_names.index(name) for name in selected_names]
        if not selected_indices["categorical"] and not selected_indices["continuous"]:
            raise ValueError("少なくとも1つの adversarial_attribute_names を指定する必要がある")

        self.register_buffer("_categorical_indices", torch.tensor(selected_indices["categorical"], dtype=torch.long), persistent=False)
        self.register_buffer("_continuous_indices", torch.tensor(selected_indices["continuous"], dtype=torch.long), persistent=False)
        self._full_attribute_counts = {"categorical": len(full_categorical), "continuous": len(full_continuous)}
        self.task_loss = task_loss
        self.attribute_adversary = _AttributeAdversary(
            feature_dim,
            [cardinalities[index] for index in selected_indices["categorical"]],
            len(selected_indices["continuous"]),
            hidden_dim,
            gradient_scale,
        )
        self.attribute_adversary_weight = attribute_adversary_weight

    def forward(self, inputs: ObjectiveInput) -> torch.Tensor:
        """task loss と選択属性の重み付き予測損失を合成して返す。

        Args:
            inputs: logits / target / full attributes と `[B, feature_dim]` の features を持つ入力。

        Returns:
            torch.Tensor: task loss と選択属性の予測損失を足した scalar。

        Raises:
            ValueError: features または full attribute tensor の shape が不正な場合。
        """
        if inputs.features is None:
            raise ValueError("AttributeInvariantTaskLoss には ObjectiveInput.features が必要")
        selected = self._select_attributes(inputs.attributes)
        return self.task_loss(inputs) + self.attribute_adversary_weight * self.attribute_adversary(inputs.features, selected)

    def _select_attributes(self, attributes: Mapping[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """full attribute tensor を検証し、GRL 対象列だけを切り出す。"""
        selected: dict[str, torch.Tensor] = {}
        for kind, indices in (("categorical", self._categorical_indices), ("continuous", self._continuous_indices)):
            expected_width = self._full_attribute_counts[kind]
            if expected_width == 0:
                continue
            if kind not in attributes or f"{kind}_missing" not in attributes:
                raise ValueError(f"{kind} / {kind}_missing は input_attribute_names に従い必要")
            values = attributes[kind]
            missing = attributes[f"{kind}_missing"].bool()
            if values.ndim != 2 or values.shape[1] != expected_width or missing.shape != values.shape:
                raise ValueError(f"{kind} / {kind}_missing の shape が input_attribute_names と一致しない")
            if indices.numel() == 0:
                continue
            selected[kind] = values.index_select(1, indices)
            selected[f"{kind}_missing"] = missing.index_select(1, indices)
        return selected
