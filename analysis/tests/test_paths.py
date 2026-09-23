"""分析が使う path が、`.project-root` 起点で組み立つことを固定する。"""

from __future__ import annotations

from analysis.common.paths import (
    ANALYSIS_ROOT,
    REPOSITORY_ROOT,
    STYLE_SHEET,
    local_data_path,
    project_runs_root,
    run_dir,
    runs_root,
    split_csv,
    study_dir,
)


def test_repository_root_is_the_directory_that_holds_the_project_root_marker() -> None:
    """root の根拠は `.project-root` 1 つにする。階層を数えて当てない。"""
    assert (REPOSITORY_ROOT / ".project-root").is_file()
    assert (REPOSITORY_ROOT / "pyproject.toml").is_file()


def test_analysis_paths_hang_off_the_repository_root() -> None:
    """package と style は repo root からの固定位置にある。"""
    assert ANALYSIS_ROOT == REPOSITORY_ROOT / "analysis"
    assert STYLE_SHEET.is_file()


def test_run_and_split_paths_follow_the_repository_layout() -> None:
    """run と split CSV の置き場は repo の約束であり、呼ぶ側で組み立てない。

    学習は `projects/<project>/runs/` に書き、分析は package に取り込んだ `analysis/<study>/runs/` を読む。
    """
    assert project_runs_root("hypernet_e2e") == REPOSITORY_ROOT / "projects/hypernet_e2e/runs"
    assert runs_root("iterative_probe") == REPOSITORY_ROOT / "analysis/iterative_probe/runs"
    assert run_dir("iterative_probe", "run-id") == REPOSITORY_ROOT / "analysis/iterative_probe/runs/run-id"
    assert study_dir("iterative_probe") == REPOSITORY_ROOT / "analysis/iterative_probe"
    assert split_csv("chexpert", "test") == REPOSITORY_ROOT / "data/chexpert/splits/test.csv"


def test_no_analysis_directory_name_blocks_import() -> None:
    """study slug は package 名でもあるので、`-` を含めると import できなくなる。"""
    studies = [path.name for path in ANALYSIS_ROOT.iterdir() if path.is_dir() and not path.name.startswith(".")]
    assert [name for name in studies if "-" in name] == []


def test_recorded_data_paths_are_reread_against_this_repository() -> None:
    """container で回した run の `config.yaml` は、local に無い絶対 path を記録している。

    ここが素通りすると `FileNotFoundError` で落ちるのではなく、**別の split を読んで
    静かに間違う**経路が開く（同名の directory が local にあった場合）。
    """
    splits = REPOSITORY_ROOT / "data/chexpert/splits"
    # container の絶対 path、local の絶対 path、repo root からの相対 path。すべて同じ場所を指す。
    assert local_data_path("/workspaces/hypernet-fairness/data/chexpert/splits") == splits
    assert local_data_path(REPOSITORY_ROOT / "data/chexpert/splits") == splits
    assert local_data_path("data/chexpert/splits") == splits


def test_the_last_data_component_anchors_the_reread() -> None:
    """`data` が複数あるときは、dataset に近いほう（最後）を起点にする。"""
    images = REPOSITORY_ROOT / "data/chexpert/images"
    assert local_data_path("/mnt/fast/kohkiakiba/fairness_data/data/chexpert/images") == images


def test_paths_without_a_data_component_are_taken_as_repository_relative() -> None:
    """`data` を含まない記録は、repo root からの相対として読む。当て推量で補わない。"""
    assert local_data_path("chexpert/splits") == REPOSITORY_ROOT / "chexpert/splits"
