"""Hydra config から iterative parent workflow を起動する CLI。"""

from __future__ import annotations

import hydra
from omegaconf import DictConfig

from .workflow import plan, run_iterative


@hydra.main(version_base="1.3", config_path="configs", config_name="train")
def main(config: DictConfig) -> None:
    """dry-run は計画だけを表示し、通常は parent run の path を出力する。"""
    if config.get("dry_run", False):
        for stage in plan(config):
            print(f"{stage.kind}: {stage.name}")
        return
    print(run_iterative(config))


if __name__ == "__main__":
    main()
