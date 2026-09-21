"""probe Notebook の線形分類器を確認する。"""

from pathlib import Path

import nbformat
import numpy as np


def _fit_probe():
    """Notebook の再利用可能な定義セルから fit_probe を取り出す。"""
    notebook_path = Path(__file__).resolve().parents[1] / "notebooks/probe.ipynb"
    notebook = nbformat.read(notebook_path, as_version=4)
    source = next(cell.source for cell in notebook.cells if cell.cell_type == "code" and "def fit_probe" in cell.source)
    namespace: dict[str, object] = {}
    exec(source, namespace)  # noqa: S102 - Notebook のテスト対象セルを読み込む。
    return namespace["fit_probe"]


def test_probe_notebook_fits_a_separating_linear_model() -> None:
    """分離可能な train/val 特徴から正しい順位の確率を得る。"""
    fit_probe = _fit_probe()
    train = (
        np.array(["a", "b", "c", "d"]),
        np.array([0, 0, 1, 1]),
        np.array([[-2.0], [-1.0], [1.0], [2.0]]),
    )
    val = (
        np.array(["e", "f"]),
        np.array([0, 1]),
        np.array([[-1.5], [1.5]]),
    )
    model, selected_c, _ = fit_probe(train, val)

    assert selected_c in (1e-3, 1e-2, 1e-1, 1.0, 10.0)
    assert model.predict_proba(val[2])[:, 1].argmax() == 1
