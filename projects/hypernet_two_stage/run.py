"""Hydra 設定から two-stage workflow を起動する CLI。"""

from __future__ import annotations

import hydra
from omegaconf import DictConfig

from projects.hypernet_two_stage.workflow import run_two_stage


@hydra.main(version_base="1.3", config_path="configs", config_name="train")
def main(config: DictConfig) -> None:
    """二段学習を実行し、作成した parent run directory を標準出力へ表示する。"""
    print(run_two_stage(config))


if __name__ == "__main__":
    main()
