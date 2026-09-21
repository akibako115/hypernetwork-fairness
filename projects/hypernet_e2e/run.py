"""Hydra CLI から e2e の単段学習を起動する。"""

from __future__ import annotations

import hydra
from omegaconf import DictConfig

from projects.hypernet_e2e.training import run_fit


@hydra.main(version_base="1.3", config_path="configs", config_name="train")
def main(config: DictConfig) -> None:
    """Hydra CLI の設定を一回の e2e fit に渡し、run directory を標準出力へ出す。

    Args:
        config: Hydra が合成した設定

    Returns:
        None
    """
    print(run_fit(config))


if __name__ == "__main__":
    main()
