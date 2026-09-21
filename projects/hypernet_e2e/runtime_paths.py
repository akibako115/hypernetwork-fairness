"""e2e の実行時 path を repo root 基準の絶対 path に正規化する。"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from omegaconf import DictConfig, OmegaConf

_MISSING = object()
_PATH_KEYS = (
    "paths.project_dir",
    "paths.data_dir",
    "data.data_dir",
    "data.cv_splits_dir",
    "model.backbone_checkpoint_path",
)


def repository_root() -> Path:
    """この project を含む repository root の絶対 path を返す。

    Args:
        なし

    Returns:
        Path: repository root の絶対 path
    """
    return Path(__file__).resolve().parents[2]


def resolve_repository_path(value: str | Path) -> Path:
    """絶対 path は保ち、相対 path は repository root から解決して返す。

    Args:
        value: 解決する path

    Returns:
        Path: 絶対 path
    """
    path = Path(value)
    return path if path.is_absolute() else repository_root() / path


def normalize_runtime_paths(config: DictConfig) -> None:
    """既知の runtime path を絶対 path に更新する。

    config に存在しない key と ``null`` の任意入力は変更しない。source config の相対 path を
    portable に保ったまま、DataModule・class weight 解決・run record が同じ保存先を使えるようにする。

    Args:
        config: 書き換え対象の設定。既知の path key を in-place で更新する

    Returns:
        None

    Raises:
        TypeError: path key の値が文字列でも Path でも null でもない場合。
    """
    for key in _PATH_KEYS:
        value: Any = OmegaConf.select(config, key, default=_MISSING)
        if value is _MISSING or value is None:
            continue
        if not isinstance(value, (str, Path)):
            raise TypeError(f"{key} は path 文字列または null である必要がある")
        OmegaConf.update(config, key, str(resolve_repository_path(value)), merge=False)
