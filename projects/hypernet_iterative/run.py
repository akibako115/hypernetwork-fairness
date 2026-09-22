"""Hydra config から iterative parent workflow を起動する CLI。"""

from __future__ import annotations

import hydra
from omegaconf import DictConfig

from .workflow import run_iterative, run_plan


@hydra.main(version_base="1.3", config_path="configs", config_name="train")
def main(config: DictConfig) -> None:
    """dry-run は起動条件と stage 計画だけを表示し、通常は parent run の path を出力する。

    Args:
        config: Hydra が合成した設定

    Returns:
        None
    """
    if config.get("dry_run", False):
        print(run_plan(config))
        return
    print(run_iterative(config))


if __name__ == "__main__":
    main()
