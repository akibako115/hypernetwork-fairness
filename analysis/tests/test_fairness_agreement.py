"""分析側の公平性指標が、学習側と同じ定義であることを golden データで固定する。

格子（`hypernet_conditioning_grid`）は CheXpert（`grid_metrics.summary`）と HAM10000
（`requested_analysis_ham10000.summary`）で、群ごとの TPR・FPR・AUROC から Eopp0 / Eopp1 / Eodds と
AUROC の gap・最悪を自前で組み立てる。群の切り方はデータごとの判断なので実装もデータごとに分かれるが、
**gap と最悪の定義は学習側と同じ値にならなければならない**。複製が 2 つあるので、両方に同じ golden を流す。

比較相手をコードではなく `tests/golden/fairness_metrics.v1.json` に置く。理由は
`.agents/skills/migrate/rationale.md`（複製を許した上で、一致を固定データで押さえる）と同じ。
実装を import して突き合わせると、どちらが正しいのか分からないまま両方が動く。

golden の case は 1 群に数件しかないので、率を出す最小セル（20 件）を 1 件に下げて流す。最小セルは
評価の方針で、ここで固定したいのは gap と最悪の定義のほうである。

2026-10-17 まで、この golden は archive の `iterative_probe`・`initial_resnet_vs_invariant` の実装にだけ
流していた。archive を消したのに合わせて、いま使っている格子の実装に付け替えた。
"""

from __future__ import annotations

import json
from types import ModuleType
from typing import Any

import numpy as np
import pytest

from analysis.common.paths import REPOSITORY_ROOT
from analysis.hypernet_conditioning_grid import grid_metrics, requested_analysis_ham10000

IMPLEMENTATIONS = {
    "chexpert": grid_metrics,
    "ham10000": requested_analysis_ham10000,
}

# golden のログキーと、`summary` が返すキーの対応。bACC の gap・最悪は格子では計算しない。
SUMMARY_KEY = {
    "Eopp0": "eopp0",
    "Eopp1": "eopp1",
    "Eodds": "eodds",
    "auroc_gap": "auroc_gap",
    "worst_group_auroc": "worst_group_auroc",
}
# 駆動しない case と理由。
UNSUPPORTED = {
    # 二値前提の実装。one-vs-rest の macro 平均は格子が持たない別の定義。
    "multiclass_three_classes": "多クラス",
    # 率を出せる群が 1 つ以下のとき、学習側は gap と最悪を NaN にする。格子は出せた群だけで gap と最悪を取り、
    # 出せた群の数を `groups_tpr`・`groups_fpr` に残す（CheXpert）か、評価不能として止まる（HAM10000）。
    # 実データでは全群が最小セルを満たすので、この違いは結論に効かない。下の別の test で振る舞いを固定する。
    "binary_single_group": "率を出せる群が 1 つ",
    "binary_group_without_negatives": "FPR を出せる群が 1 つ",
}
# golden の tolerance (1e-9) は、同じ torch float32 の経路で再現する場合の値。分析側は
# numpy / sklearn の float64 で解き直すので、float32 の丸め分だけずれる（実測 ~5e-9）。
# 定義の違いはこの桁では起きないため、実装差を見るには 1e-6 で足りる。
TOLERANCE = 1e-6

GOLDEN = json.loads((REPOSITORY_ROOT / "tests/golden/fairness_metrics.v1.json").read_text(encoding="utf-8"))


def _cases(names: set[str] | None = None) -> list[dict[str, Any]]:
    """駆動できる golden case（`names` を渡せばその case）を返す。"""
    if names is not None:
        return [case for case in GOLDEN["cases"] if case["name"] in names]
    return [case for case in GOLDEN["cases"] if case["name"] not in UNSUPPORTED]


def _predictions(case: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    """batch を結合し、cache と同じ形（陽性確率）にする。"""
    logits = np.concatenate([np.asarray(batch["logits"], dtype=np.float64) for batch in case["batches"]])
    target = np.concatenate([np.asarray(batch["target"], dtype=np.int64) for batch in case["batches"]])
    exponent = np.exp(logits - logits.max(axis=1, keepdims=True))
    probabilities = exponent / exponent.sum(axis=1, keepdims=True)
    return target, probabilities[:, 1]


def _group(case: dict[str, Any], attribute: str) -> np.ndarray:
    """その属性の群符号を 0 から詰めた群番号にする。負の符号（欠損）は -1 のまま残す。"""
    for kind, names in case["attribute_names"].items():
        if attribute not in names:
            continue
        column = names.index(attribute)
        batches = [np.asarray(batch["attributes"][kind], dtype=np.int64)[:, column] for batch in case["batches"]]
        codes = np.concatenate(batches)
        present = sorted(set(codes[codes >= 0].tolist()))
        return np.array([present.index(code) if code >= 0 else -1 for code in codes])
    raise AssertionError(f"{attribute} は golden の attribute_names に無い")


def _summary(
    case: dict[str, Any], attribute: str, module: ModuleType, monkeypatch: pytest.MonkeyPatch
) -> dict[str, float]:
    """1 属性について、その実装の `summary` を case の群の数・最小セル 1 件で呼ぶ。"""
    target, probability = _predictions(case)
    group = _group(case, attribute)
    monkeypatch.setattr(module, "MIN_CELL", 1)
    monkeypatch.setattr(module, "GROUP_LABELS", tuple(str(g) for g in range(group.max() + 1)))
    # CheXpert の実装は欠損行を持たない（race-known）ので、欠損行は呼ぶ前に落とす。
    if module is grid_metrics:
        keep = group >= 0
        return module.summary(probability[keep], target[keep], group[keep])
    return module.summary(probability, target, group)


@pytest.mark.parametrize("package", sorted(IMPLEMENTATIONS))
@pytest.mark.parametrize("case", _cases(), ids=lambda case: case["name"])
def test_summary_reproduces_the_golden_fairness_values(
    case: dict[str, Any], package: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """学習中にログされる値と、格子の各実装が出す値が同じ定義であることを固定する。"""
    attributes = {key.split("/")[1] for key in case["expected"]}
    checked = 0
    for attribute in sorted(attributes):
        summary = _summary(case, attribute, IMPLEMENTATIONS[package], monkeypatch)
        for key, expected in case["expected"].items():
            name = key.split("/")[-1]
            if key.split("/")[1] != attribute or name not in SUMMARY_KEY:
                continue
            assert expected is not None, f"{case['name']}: {key} が NaN の case は UNSUPPORTED に入れる"
            assert summary[SUMMARY_KEY[name]] == pytest.approx(expected, abs=TOLERANCE), f"{package} / {key}"
            checked += 1
    assert checked > 0, f"{case['name']} で何も比べていない"


def test_chexpert_takes_gaps_over_the_groups_it_could_evaluate_and_reports_how_many(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """FPR を出せる群が 1 つだと、学習側は Eopp0 を NaN にし、CheXpert の実装は 0 と群の数 1 を返す。"""
    (case,) = _cases({"binary_group_without_negatives"})
    summary = _summary(case, "sex", grid_metrics, monkeypatch)

    assert case["expected"]["val/sex/Eopp0"] is None
    assert summary["eopp0"] == 0.0
    assert summary["groups_fpr"] == 1.0
    assert summary["groups_tpr"] == 2.0


def test_ham10000_refuses_groups_it_cannot_evaluate(monkeypatch: pytest.MonkeyPatch) -> None:
    """率を出せない群があると、HAM10000 の実装は gap を出さずに止まる。"""
    (case,) = _cases({"binary_group_without_negatives"})
    with pytest.raises(ValueError, match="評価不能"):
        _summary(case, "sex", requested_analysis_ham10000, monkeypatch)


def test_every_unsupported_case_exists() -> None:
    """golden から case が消えたのに除外だけが残る、という状態にしない。"""
    names = {case["name"] for case in GOLDEN["cases"]}
    assert set(UNSUPPORTED) <= names
