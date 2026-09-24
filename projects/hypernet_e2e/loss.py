"""互換 import path として hypernet_e2e の学習 objective を公開する。

実装は `objectives` package に責務ごとに分割する。この module は既存の Hydra `_target_` と
利用側の import path を維持する facade であり、新しい objective 実装を置かない。
"""

from .objectives import (
    AlternatingAttributeInvariantTaskLoss,
    AttributeInvariantTaskLoss,
    ClassBalancedGroupDROTaskLoss,
    GroupDROTaskLoss,
    ObjectiveInput,
    TaskLoss,
    UniformGroupTaskLoss,
)

__all__ = [
    "AlternatingAttributeInvariantTaskLoss",
    "AttributeInvariantTaskLoss",
    "ClassBalancedGroupDROTaskLoss",
    "GroupDROTaskLoss",
    "ObjectiveInput",
    "TaskLoss",
    "UniformGroupTaskLoss",
]
