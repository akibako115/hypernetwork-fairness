"""hypernet_e2e の学習 objective を責務ごとに公開する。"""

from .attribute_invariance import AlternatingAttributeInvariantTaskLoss, AttributeInvariantTaskLoss
from .group import GlobalClassWeightedGroupDROTaskLoss, SubgroupClassWeightedGroupDROTaskLoss, UniformGroupTaskLoss
from .input import ObjectiveInput
from .task import TaskLoss

__all__ = [
    "AlternatingAttributeInvariantTaskLoss",
    "AttributeInvariantTaskLoss",
    "GlobalClassWeightedGroupDROTaskLoss",
    "SubgroupClassWeightedGroupDROTaskLoss",
    "ObjectiveInput",
    "TaskLoss",
    "UniformGroupTaskLoss",
]
