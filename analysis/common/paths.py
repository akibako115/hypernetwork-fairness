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


def project_runs_root(project: str) -> Path:
    """学習がその project の run を書き出す directory を返す。

    ここは学習側の正本で、分析は直接読まない。`analysis/common/runs.py import` が
    ここから package の `runs/` へ hardlink で取り込む。

    Args:
        project: `hypernet_e2e` / `hypernet_iterative` など `projects/` 直下の名前

    Returns:
        Path: `projects/<project>/runs`
    """
    return REPOSITORY_ROOT / "projects" / project / "runs"


def runs_root(study: str) -> Path:
    """その分析 package が取り込んだ run の置き場を返す。分析コードはここだけを読む。

    Args:
        study: `analysis/` 直下の package 名

    Returns:
        Path: `analysis/<study>/runs`
    """
    return study_dir(study) / "runs"


def run_dir(study: str, run_id: str) -> Path:
    """package に取り込んだ run directory を返す。

    Args:
        study: `analysis/` 直下の package 名
        run_id: run ID

    Returns:
        Path: `analysis/<study>/runs/<run-id>`
    """
    return runs_root(study) / run_id


def study_dir(study: str) -> Path:
    """仮説 (study) の分析 package を返す。

    Args:
        study: `analysis/` 直下の package 名。学習側に渡した `study` と同じ値

    Returns:
        Path: `analysis/<study>`
    """
    return ANALYSIS_ROOT / study


def local_data_path(recorded: str | Path) -> Path:
    """run が記録した data path を、この repo の `data/` 配下へ読み替える。

    `config.yaml` に残るのは**その run を回した環境の path** で、container で回した run は
    `/workspaces/hypernet-fairness/data/chexpert/splits` のような、local に存在しない絶対 path を
    持つ。`selected_checkpoint` が checkpoint に対してやっているのと同じ読み替えを data 側にも
    かける。判断の根拠を「記録された文字列」ではなく「repo の layout」に寄せる。

    `data` 以降だけを使うので、既に解決できる path（local の絶対 path、repo root からの相対
    path）では同じ Path を返す。`data` を含まない path は repo root からの相対として扱う。

    Args:
        recorded: `config.yaml` の `data.cv_splits_dir` などに記録された path

    Returns:
        Path: `data/<dataset>/...` に読み替えた local の path
    """
    parts = Path(recorded).parts
    if "data" in parts:
        # 同じ名前が複数あるときは最後のものを起点にする。`.../fairness_data/data/chexpert` の
        # ような入れ子でも、実際の dataset directory に近いほうを選ぶ。
        anchor = len(parts) - 1 - parts[::-1].index("data")
        return REPOSITORY_ROOT.joinpath(*parts[anchor:])
    return REPOSITORY_ROOT.joinpath(*parts)


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
