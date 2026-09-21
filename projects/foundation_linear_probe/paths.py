"""データとキャッシュの参照先を解決する。"""

from __future__ import annotations

import os
from pathlib import Path

DEFAULT_DATA_FOLDER = Path("/data")


def data_folder() -> Path:
    """データの基点を返す。`DATA_FOLDER` 環境変数があればそれを使う。"""
    return Path(os.environ.get("DATA_FOLDER", DEFAULT_DATA_FOLDER))


def chexpert_root(root: Path | None = None) -> Path:
    """CheXpert の基点を返す。"""
    return Path(root) if root is not None else data_folder() / "chexpert"
