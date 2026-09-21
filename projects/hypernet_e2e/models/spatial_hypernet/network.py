"""ResNet backbone に Spatial LoRA と HyperLinear を組み合わせたネットワーク。

backbone の重みは共有のまま、選択した stage の Bottleneck と分類層だけを metadata
condition で変調する。B 生成器の初期化は train split の Var(c) に依存するため、
`initialize_with_attributes` を学習開始前に呼ぶ必要がある。
"""

import logging
from collections.abc import Mapping, Sequence

import torch
import torch.nn as nn

from ..resnet.network import ResNetBackbone
from ..utils import compute_embedding_variance, load_compatible_state_dict
from .blocks import SpatialLoRABottleneckAdapter
from .metadata_encoder import MetadataEncoder
from .modulation import HyperLinearLayer

log = logging.getLogger(__name__)

_ALLOWED_MODULATION_STAGES = {"stage3", "stage4", "fc"}


class SpatialLoRAResNet(nn.Module):
    """stage3/4 Spatial LoRA と HyperLinear を任意に有効化できる ResNet。

    Args:
        num_classes: 分類クラス数。
        backbone: 特徴抽出器として使う `ResNetBackbone`。
        metadata_encoder: attributes から metadata embedding（condition）を生成する `MetadataEncoder`。
        rank: stage3/4 の `SpatialLoRAConv2` 生成器の低ランク次元。
        lora_alpha: stage3/4 の Spatial LoRA のスケール係数（`scale = lora_alpha / rank`）。
        modulation_stages: 変調を適用する箇所。`"stage3"` / `"stage4"` / `"fc"` の部分集合
            指定形式は `[]` / `["fc"]` / `["stage4"]` / `["stage3", "stage4"]` など。
            `"fc"` を含む場合、分類ヘッドは通常の `nn.Linear` ではなく `HyperLinearLayer` になる。
        classifier_rank: `"fc"` 変調時の `HyperLinearLayer` の低ランク次元。
        classifier_lora_alpha: `"fc"` 変調時の `HyperLinearLayer` のスケール係数。
    """

    def __init__(
        self,
        num_classes: int,
        backbone: ResNetBackbone,
        metadata_encoder: MetadataEncoder,
        rank: int = 4,
        lora_alpha: float = 1.0,
        modulation_stages: Sequence[str] = ("stage4",),
        classifier_rank: int = 4,
        classifier_lora_alpha: float = 1.0,
    ):
        """backbone、metadata encoder、Spatial LoRA adapter と分類器を初期化する。"""
        super().__init__()
        unknown_stages = set(modulation_stages) - _ALLOWED_MODULATION_STAGES
        if unknown_stages:
            raise ValueError(f"Unknown modulation stages: {sorted(unknown_stages)}")
        if len(set(modulation_stages)) != len(modulation_stages):
            raise ValueError(f"modulation_stages must not contain duplicates: {list(modulation_stages)}")

        self.backbone = backbone
        self.metadata_encoder = metadata_encoder
        self.rank = rank
        self.lora_alpha = lora_alpha
        self.modulation_stages = list(modulation_stages)
        self.spatial_adapters = nn.ModuleDict()
        if "stage3" in self.modulation_stages:
            self.spatial_adapters["stage3"] = self._make_spatial_adapters(backbone.layer3)
        if "stage4" in self.modulation_stages:
            self.spatial_adapters["stage4"] = self._make_spatial_adapters(backbone.layer4)
        if "fc" in self.modulation_stages:
            self.fc: nn.Module = HyperLinearLayer(
                metadata_encoder.output_dim,
                backbone.feature_dim,
                num_classes,
                rank=classifier_rank,
                lora_alpha=classifier_lora_alpha,
            )
        else:
            self.fc = nn.Linear(backbone.feature_dim, num_classes)

    def forward(self, x: torch.Tensor, attributes: Mapping[str, torch.Tensor]) -> torch.Tensor:
        """画像 (B, 3, H, W) と属性辞書から分類 logits (B, num_classes) を返す。

        Args:
            x: `[B, 3, H, W]` の画像 batch
            attributes: `categorical` / `continuous` とそれぞれの `*_missing`。
                各値は `[B, n_attributes]`

        Returns:
            torch.Tensor: `[B, num_classes]` の logits
        """
        condition = self.metadata_encoder(attributes)
        features = self._forward_backbone(x, condition)
        if isinstance(self.fc, HyperLinearLayer):
            return self.fc(features, condition)
        return self.fc(features)

    def get_features(self, x: torch.Tensor, attributes: Mapping[str, torch.Tensor]) -> torch.Tensor:
        """Spatial LoRA 適用後の flatten 済み特徴量を返す。

        Args:
            x: `[B, 3, H, W]` の画像 batch
            attributes: `categorical` / `continuous` とそれぞれの `*_missing`

        Returns:
            torch.Tensor: `[B, feature_dim]` の特徴量
        """
        condition = self.metadata_encoder(attributes)
        return self._forward_backbone(x, condition)

    def load_base_state_dict(self, state_dict: dict) -> dict[str, list[str]]:
        """ResNet の backbone と分類 head を Spatial LoRA の初期値として読み込む。

        shape の合うキーだけをコピーするため、LoRA generator は初期値のまま残る。

        Args:
            state_dict: ResNet 側の state dict。`fc.*` は共有分類層へ読み替える

        Returns:
            dict[str, list[str]]: loaded_keys / skipped_keys / missing_keys / unexpected_keys
        """
        # ResNet checkpoint の fc を、Spatial LoRA の共有分類層へ対応付ける。
        spatial_state = dict(state_dict)
        if isinstance(self.fc, HyperLinearLayer):
            for parameter_name in ("weight", "bias"):
                source_key = f"fc.{parameter_name}"
                if source_key in state_dict:
                    spatial_state[f"fc.base_linear.{parameter_name}"] = state_dict[source_key]
                    spatial_state.pop(source_key, None)

        # 互換する backbone と base classifier だけをコピーし、LoRA generator は初期値のまま残す。
        return load_compatible_state_dict(self, spatial_state)

    def freeze_base_model(self) -> None:
        """二段階学習で Stage 1 の backbone と共有 classifier を凍結する。

        Args:
            なし

        Returns:
            None
        """
        # Stage 1 から引き継いだ画像 backbone を固定し、Spatial LoRA だけが特徴を補正するようにする。
        for parameter in self.backbone.parameters():
            parameter.requires_grad = False
        self.backbone.eval()

        # 共有 classifier は固定し、fc を変調する場合も属性条件付き LoRA 差分だけを学習する。
        classifier = self.fc.base_linear if isinstance(self.fc, HyperLinearLayer) else self.fc
        for parameter in classifier.parameters():
            parameter.requires_grad = False

    def initialize_with_attributes(self, attributes: Mapping[str, torch.Tensor]) -> None:
        """train split の metadata embedding の分散から B 生成器を初期化する。

        `LitModule.setup("fit")` が duck-typing で呼ぶ固定名フック。B 生成器の既定初期化は
        生成先 layer の fan_in を見ないため、ここで学習データ由来の Var(c) を使って
        base layer と同じ尺度へ揃える。A 生成器はゼロ初期化のまま触らない。
        Var(c) は属性列だけで決まるので、画像を読む train DataLoader は受け取らない。
        推定に使うのは train split 全行の素の分布である。`train_sampling=inverse_frequency` でも
        resample 後の分布ではなく全行の分布から Var(c) を決める（`uniform` の場合は、全行を1周
        していた旧実装と同じ値になる）。

        Args:
            attributes: train split 全行分の model 入力属性。`[n_rows, n_attributes]` の
                `categorical` / `continuous` とそれぞれの `*_missing`

        Returns:
            None
        """
        targets: list[SpatialLoRABottleneckAdapter | HyperLinearLayer] = [adapter for adapters in self.spatial_adapters.values() for adapter in adapters]
        if isinstance(self.fc, HyperLinearLayer):
            targets.append(self.fc)
        if not targets:
            # 変調箇所が無ければ初期化する生成器も無い。
            return

        var_input = compute_embedding_variance(self.metadata_encoder, attributes)
        for target in targets:
            target.initialize_from_variance(var_input)
        log.info(f"Initialized {len(targets)} Spatial LoRA B generators from Var(c)={var_input:.6f}")

    def _forward_backbone(self, x: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
        """選択された Spatial LoRA を適用した backbone 出力を flatten して返す。"""
        x = self.backbone.relu(self.backbone.bn1(self.backbone.conv1(x)))
        x = self.backbone.maxpool(x)
        x = self.backbone.layer1(x)
        x = self.backbone.layer2(x)
        x = self._forward_stage("stage3", self.backbone.layer3, x, condition)
        x = self._forward_stage("stage4", self.backbone.layer4, x, condition)
        x = self.backbone.avgpool(x)
        return torch.flatten(x, 1)

    def _make_spatial_adapters(self, stage: nn.Sequential) -> nn.ModuleList:
        """stage 内の各 Bottleneck に独立した Spatial LoRA adapter を割り当てる。"""
        return nn.ModuleList([SpatialLoRABottleneckAdapter(self.metadata_encoder.output_dim, block, self.rank, self.lora_alpha) for block in stage])

    def _forward_stage(
        self,
        stage_name: str,
        stage: nn.Sequential,
        x: torch.Tensor,
        condition: torch.Tensor,
    ) -> torch.Tensor:
        """選択された stage だけ Spatial LoRA adapter 経由で実行する。"""
        if stage_name not in self.spatial_adapters:
            return stage(x)

        adapters = self.spatial_adapters[stage_name]
        if not isinstance(adapters, nn.ModuleList):
            raise TypeError(f"spatial_adapters[{stage_name!r}] must be nn.ModuleList")
        for adapter in adapters:
            x = adapter(x, condition)
        return x

    def backbone_parameters(self):
        """freeze_backbone 用に backbone（Spatial LoRA adapter・分類ヘッドを除く）の parameter を列挙する。

        Args:
            なし

        Returns:
            Iterator[nn.Parameter]: backbone の parameter
        """
        return self.backbone.backbone_parameters()

    def backbone_stateful_modules(self):
        """freeze_backbone 時に eval モードへ固定すべき backbone の BatchNorm モジュールを列挙する。

        Args:
            なし

        Returns:
            Iterator[nn.BatchNorm2d]: backbone の BatchNorm モジュール
        """
        return self.backbone.backbone_stateful_modules()
