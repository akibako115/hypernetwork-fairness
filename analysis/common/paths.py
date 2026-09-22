"""repo root と、分析が繰り返し読む path を 1 箇所で持つ。

各 script・notebook は先頭で `rootutils.setup_root` を呼んで repo root を import path に
載せ、以降の path はこの module から取る。`parents[2]` のような相対指定を書き散らすと、
file を 1 つ移動しただけで黙って別の場所を読むようになる。

    ROOT = rootutils.setup_root(Path.cwd(), indicator=".project-root", pythonpath=True)   # notebook
    ROOT = rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)     # script
"""

from __future__ import annotations

from pathlib import Path

import rootutils

# root の根拠を `.project-root` 1 つに統一する。`parents[2]` で数えると、この file を
# 1 階層動かしただけで黙って別の場所を指す。
REPOSITORY_ROOT = Path(rootutils.find_root(__file__, indicator=".project-root"))
ANALYSIS_ROOT = REPOSITORY_ROOT / "analysis"
# 図の style は全 package で共有する。色循環と軸設定を package ごとに振らない。
STYLE_SHEET = ANALYSIS_ROOT / "styles" / "fairness.mplstyle"


def runs_root(project: str) -> Path:
    """その project の run が積まれる directory を返す。

    Args:
        project: `hypernet_e2e` / `hypernet_iterative` など `projects/` 直下の名前

    Returns:
        Path: `projects/<project>/runs`
    """
    return REPOSITORY_ROOT / "projects" / project / "runs"


def run_dir(project: str, run_id: str) -> Path:
    """run-id から run directory を返す。

    Args:
        project: `projects/` 直下の名前
        run_id: run ID

    Returns:
        Path: `projects/<project>/runs/<run-id>`
    """
    return runs_root(project) / run_id


def study_dir(study: str) -> Path:
    """仮説 (study) の分析 package を返す。

    Args:
        study: `analysis/` 直下の package 名。学習側に渡した `study` と同じ値

    Returns:
        Path: `analysis/<study>`
    """
    return ANALYSIS_ROOT / study


def split_csv(dataset: str, split: str) -> Path:
    """学習が読んだのと同じ split CSV を返す。

    分析は属性を突き合わせるためにこの CSV を読むが、**書き換えない**。分割は
    `data/` が所有する読み取り専用の固定入力になる。

    Args:
        dataset: `chexpert` など `data/` 直下の名前
        split: `train` / `val` / `test`

    Returns:
        Path: `data/<dataset>/splits/<split>.csv`
    """
    return REPOSITORY_ROOT / "data" / dataset / "splits" / f"{split}.csv"
