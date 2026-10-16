"""Q1 の監査の帰無分布：群の入れ替えが単位（患者）とスコアの層を守ることを検証する。"""

import numpy as np

from analysis.archive.latent_bias_hypernet.audit_metrics import block_shuffle, unit_strata


def _patients() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """患者 200 人・1〜3 枚ずつ。群は患者のスコアの高低に沿って決める（スコアで分かれた群の再現）。"""
    rng = np.random.default_rng(0)
    sizes = 1 + np.arange(200) % 3
    cluster = np.repeat(np.arange(200), sizes)
    unit_score = rng.uniform(size=200)
    p = np.repeat(unit_score, sizes)
    group = np.repeat((unit_score * 4).astype(int), sizes)
    return group, cluster, p


def test_shuffle_keeps_patients_together() -> None:
    """同じ患者の画像は、入れ替えた後も同じ群に入る。"""
    group, cluster, p = _patients()
    shuffled = block_shuffle(group, cluster, np.random.default_rng(1), unit_strata(p, cluster))
    for unit in np.unique(cluster):
        assert len(np.unique(shuffled[cluster == unit])) == 1


def test_stratified_shuffle_keeps_each_group_score_layers() -> None:
    """層別の入れ替えでは、各群に入る患者のスコアの層の構成が変わらない。無層別では変わる。"""
    group, cluster, p = _patients()
    strata = unit_strata(p, cluster)
    _, first = np.unique(cluster, return_index=True)

    def layer_counts(g: np.ndarray) -> np.ndarray:
        return np.array([np.bincount(strata[g[first] == k], minlength=10) for k in range(4)])

    stratified = block_shuffle(group, cluster, np.random.default_rng(1), strata)
    free = block_shuffle(group, cluster, np.random.default_rng(1))
    assert np.array_equal(layer_counts(stratified), layer_counts(group))
    assert not np.array_equal(layer_counts(free), layer_counts(group))
