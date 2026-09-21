"""two-stage の実行時 path を repo root 基準の絶対 path に正規化する。"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from omegaconf import DictConfig, OmegaConf

_MISSING = object()
# この project は model 設定を stage ごとに持つので、checkpoint path も stage ごとに正規化する。
# 存在しない key は読み飛ばされるので、片方の stage しか持たない config でもそのまま使える。
_PATH_KEYS = (
    "paths.project_dir",
    "paths.data_dir",
    "data.data_dir",
    "data.cv_splits_dir",
    "stage1.model.backbone_checkpoint_path",
    "stage2.model.backbone_checkpoint_path",
)


def repository_root() -> Path:
    """この project を含む repository root の絶対 path を返す。"""
    return Path(__file__).resolve().parents[2]


def resolve_repository_path(value: str | Path) -> Path:
    """絶対 path は保ち、相対 path は repository root から解決して返す。"""
    path = Path(value)
    return path if path.is_absolute() else repository_root() / path


def normalize_runtime_paths(config: DictConfig) -> None:
    """既知の runtime path を絶対 path に更新する。

    config に存在しない key と ``null`` の任意入力は変更しない。source config の相対 path を
    portable に保ったまま、DataModule・class weight 解決・run record が同じ保存先を使えるようにする。
    """
    for key in _PATH_KEYS:
        value: Any = OmegaConf.select(config, key, default=_MISSING)
        if value is _MISSING or value is None:
            continue
        if not isinstance(value, (str, Path)):
            raise TypeError(f"{key} は path 文字列または null である必要がある")
        OmegaConf.update(config, key, str(resolve_repository_path(value)), merge=False)
