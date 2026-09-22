"""分析側の公平性指標が、学習側と同じ定義であることを golden データで固定する。

`iterative_probe` と `initial_resnet_vs_invariant` は、どちらも群ごとの TPR・FPR から自前で
Eopp0 / Eopp1 / Eodds を組み立てる。群の切り方は package ごとの判断なので実装も package に
閉じるが、**単独属性では学習側と同じ値にならなければならない**。

複製を作った以上、**全部の複製に同じ golden を流す**。片方だけに当てると、もう片方が静かに
別の定義へずれても誰も気付かない。

比較相手をコードではなく `tests/golden/fairness_metrics.v1.json` に置く。理由は
`.agents/skills/migrate/rationale.md`（複製を許した上で、一致を固定データで押さえる）と同じ。
実装を import して突き合わせると、どちらが正しいのか分からないまま両方が動く。
"""

from __future__ import annotations

import json
import math
from typing import Any

import numpy as np
import pandas as pd
import pytest

from analysis.common.paths import REPOSITORY_ROOT
from analysis.initial_resnet_vs_invariant import groups as initial_groups
from analysis.iterative_probe import groups as iterative_groups

# golden を流す実装。package が増えたらここへ足す。
IMPLEMENTATIONS = {
    "iterative_probe": iterative_groups,
    "initial_resnet_vs_invariant": initial_groups,
}

# golden のログキーと、`summarize` が返す列の対応。
SUMMARY_COLUMN = {
    "Eopp0": "Eopp0",
    "Eopp1": "Eopp1",
    "Eodds": "Eodds",
    "auroc_gap": "AUROC gap",
    "bacc_gap": "bACC gap",
    "worst_group_auroc": "worst AUROC",
    "worst_group_bacc": "worst bACC",
}
# 二値前提の実装なので多クラスの case は駆動できない。one-vs-rest の macro 平均は
# groups.py が持たない別の定義で、ここで固定する対象ではない。
UNSUPPORTED = {"multiclass_three_classes"}
# golden の tolerance (1e-9) は、同じ torch float32 の経路で再現する場合の値。分析側は
# numpy / sklearn の float64 で解き直すので、float32 の丸め分だけずれる（実測 ~5e-9）。
# 定義の違いはこの桁では起きないため、実装差を見るには 1e-6 で足りる。
TOLERANCE = 1e-6

GOLDEN = json.loads((REPOSITORY_ROOT / "tests/golden/fairness_metrics.v1.json").read_text(encoding="utf-8"))


def _cases() -> list[dict[str, Any]]:
    """駆動できる golden case を返す。"""
    return [case for case in GOLDEN["cases"] if case["name"] not in UNSUPPORTED]


def _predictions(case: dict[str, Any]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """batch を結合し、cache と同じ形（確率・予測ラベル）にする。"""
    logits = np.concatenate([np.asarray(batch["logits"], dtype=np.float64) for batch in case["batches"]])
    target = np.concatenate([np.asarray(batch["target"], dtype=np.int64) for batch in case["batches"]])
    exponent = np.exp(logits - logits.max(axis=1, keepdims=True))
    probabilities = exponent / exponent.sum(axis=1, keepdims=True)
    return target, probabilities[:, 1], probabilities.argmax(axis=1)


def _codes(case: dict[str, Any], attribute: str) -> np.ndarray:
    """その属性の群符号を、batch を結合した 1 本の列にする。"""
    for kind, names in case["attribute_names"].items():
        if attribute not in names:
            continue
        column = names.index(attribute)
        batches = [np.asarray(batch["attributes"][kind], dtype=np.int64)[:, column] for batch in case["batches"]]
        return np.concatenate(batches)
    raise AssertionError(f"{attribute} は golden の attribute_names に無い")


def _summary(case: dict[str, Any], attribute: str, groups: Any) -> pd.Series:
    """1 属性について、その package の実装で群別指標を出して 1 行へ畳む。"""
    target, probability, prediction = _predictions(case)
    codes = _codes(case, attribute)
    # 負の符号は欠損。`demographics_of` が split CSV の欠損行を落とすのと同じ扱いにする。
    present = codes >= 0
    rows = []
    for code in sorted(set(codes[present].tolist())):
        of_group = codes == code
        rows.append(groups.group_metrics(target[of_group], probability[of_group], prediction[of_group]))
    return groups.summarize(pd.DataFrame(rows))


@pytest.mark.parametrize("package", sorted(IMPLEMENTATIONS))
@pytest.mark.parametrize("case", _cases(), ids=lambda case: case["name"])
def test_group_metrics_reproduce_the_golden_fairness_values(case: dict[str, Any], package: str) -> None:
    """学習中にログされる値と、各 package が出す値が同じ定義であることを固定する。"""
    attributes = {key.split("/")[1] for key in case["expected"]}
    for attribute in sorted(attributes):
        summary = _summary(case, attribute, IMPLEMENTATIONS[package])
        for key, expected in case["expected"].items():
            name = key.split("/")[-1]
            if key.split("/")[1] != attribute or name not in SUMMARY_COLUMN:
                continue
            actual = summary[SUMMARY_COLUMN[name]]
            if expected is None:
                # golden の null は NaN。定義できない指標を 0 として出さないことも固定する。
                assert math.isnan(actual), f"{package} / {case['name']}: {key} は定義できないはず"
                continue
            assert actual == pytest.approx(expected, abs=TOLERANCE), f"{package} / {case['name']}: {key}"


def test_every_supported_case_checks_at_least_one_value() -> None:
    """case を足したのに何も比べていない、という状態にしない。"""
    for case in _cases():
        names = {key.split("/")[-1] for key in case["expected"]}
        assert names & set(SUMMARY_COLUMN), case["name"]
