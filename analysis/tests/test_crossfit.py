"""cross-fit の split と、誤りの向きで作る区分の契約を検証する。"""

import numpy as np
import pandas as pd

from analysis.archive.latent_bias_hypernet.crossfit import K_ITER, error_partition, halve_by_lesion


def test_halves_keep_lesions_together_and_split_each_class() -> None:
    """同じ病変は同じ側に入り、悪性・良性それぞれの病変が半分ずつに分かれる。"""
    lesions = [f"L{i}" for i in range(40) for _ in range(1 + i % 3)]
    frame = pd.DataFrame({"lesion_id": lesions, "target": [int(lesion[1:]) % 4 == 0 for lesion in lesions]})
    first, second = halve_by_lesion(frame)
    assert set(first.lesion_id).isdisjoint(second.lesion_id)
    assert len(first) + len(second) == len(frame)
    for c in (0, 1):
        assert abs(first[first.target == c].lesion_id.nunique() - second[second.target == c].lesion_id.nunique()) <= 1


def test_error_partition_assigns_every_split_without_labels() -> None:
    """区分は train の残差で作り、埋め込みだけから全 split に K_ITER 未満の番号で割り当てる。"""
    rng = np.random.default_rng(0)
    z = {s: rng.normal(size=(n, 4)) for s, n in (("train", 400), ("val", 50), ("test", 60))}
    residual = np.where(z["train"][:, 0] > 0, 0.8, -0.1) + rng.normal(scale=0.05, size=400)
    parts, info = error_partition(z, residual)
    assert {s: len(p) for s, p in parts.items()} == {"train": 400, "val": 50, "test": 60}
    assert all(p.min() >= 0 and p.max() < K_ITER for p in parts.values())
    assert info["k"] == K_ITER
