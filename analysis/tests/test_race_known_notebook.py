"""race-known比較notebookの指標をgoldenと欠損ケースで固定する。"""

import ast
import json

import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

from analysis.archive.race_known_architecture_objective_comparison import comparison
from analysis.common.paths import REPOSITORY_ROOT


def _metric_cell():
    import inspect

    return inspect.getsource(comparison.performance) + "\n" + inspect.getsource(comparison.spread)


def _namespace():
    return {"np": np, "pd": pd, "accuracy_score": accuracy_score, "f1_score": f1_score, "roc_auc_score": roc_auc_score}


@pytest.mark.parametrize(
    "filename",
    [
        "resnet_erm_vs_groupdro_global.ipynb",
        "02_class_weight_scope.ipynb",
        "03_spatial_lora_stage_range.ipynb",
        "04_architecture_comparison.ipynb",
        "05_one_stage_vs_frozen_two_stage.ipynb",
        "06_resnet_baselines_vs_groupdro_architectures.ipynb",
        "07_two_stage_operating_point.ipynb",
    ],
)
def test_notebooks_reference_current_inputs_and_are_valid(filename):
    path = REPOSITORY_ROOT / "analysis/archive/race_known_architecture_objective_comparison" / filename
    notebook = json.loads(path.read_text())
    assert "20261002T124930Z" not in path.read_text()
    for cell in notebook["cells"]:
        if cell["cell_type"] == "code":
            ast.parse("".join(cell["source"]))
            # 利用者が既存notebookをローカルで実行してもテストは失敗させない。
            # 今回新規作成したnotebookのみ、配布用sourceの出力クリアを検算する。
            if filename in {
                "06_resnet_baselines_vs_groupdro_architectures.ipynb",
                "07_two_stage_operating_point.ipynb",
            }:
                assert cell["execution_count"] is None and cell["outputs"] == []
    assert "TODO" in "".join(notebook["cells"][-1]["source"])


def test_notebook_binary_fairness_matches_golden(tmp_path):
    golden = json.loads((REPOSITORY_ROOT / "tests/golden/fairness_metrics.v1.json").read_text())
    case = next(case for case in golden["cases"] if case["name"] == "binary_two_groups")
    logits = np.concatenate([batch["logits"] for batch in case["batches"]])
    target = np.concatenate([batch["target"] for batch in case["batches"]])
    groups = np.concatenate([batch["attributes"]["categorical"] for batch in case["batches"]])[:, 0]
    exponents = np.exp(logits - logits.max(axis=1, keepdims=True))
    namespace = _namespace()
    namespace.update(
        {
            "caches": {
                "ERM": {
                    "target": target,
                    "probabilities": exponents / exponents.sum(axis=1, keepdims=True),
                    "predictions": logits.argmax(axis=1),
                }
            },
            "demographics": pd.DataFrame({"sex": groups}),
            "GROUP_ORDER": {"sex": [0, 1]},
            "RESULTS": tmp_path,
        }
    )
    cached = namespace["caches"]["ERM"]
    rows = [
        comparison.performance(
            target[groups == group], cached["probabilities"][groups == group, 1], cached["predictions"][groups == group]
        )
        for group in [0, 1]
    ]
    row = comparison.fairness_metrics(pd.DataFrame(rows))
    for metric in ["Eopp0", "Eopp1", "Eodds", "auroc_gap", "bacc_gap"]:
        assert row[metric] == pytest.approx(case["expected"][f"val/sex/{metric}"], abs=1e-6)
    for metric in ["auroc", "bacc"]:
        assert row[f"worst_{metric}"] == pytest.approx(case["expected"][f"val/sex/worst_group_{metric}"], abs=1e-6)


@pytest.mark.parametrize("target", [np.array([], dtype=int), np.array([0, 0]), np.array([1, 1])])
def test_undefined_groups_are_not_reported_as_zero(target):
    namespace = _namespace()
    tree = ast.parse(_metric_cell())
    functions = ast.Module(body=[node for node in tree.body if isinstance(node, ast.FunctionDef)], type_ignores=[])
    exec(compile(functions, "notebook-functions", "exec"), namespace)
    metrics = namespace["performance"](target, np.full(len(target), 0.5), np.zeros(len(target), dtype=int))
    assert np.isnan(metrics["auroc"]) and np.isnan(metrics["bacc"])
    worst, best, gap, valid = namespace["spread"]([0.8, np.nan], 2)
    assert np.isnan(worst) and np.isnan(best) and np.isnan(gap) and valid == 1


def test_paired_deltas_align_seeds_and_reject_missing():
    rows = [
        {"condition": condition, "seed": seed, **{metric: offset + seed / 100 for metric in comparison.METRICS}}
        for condition, offset, seeds in [("A", 0, [42, 43, 44]), ("B", 0.1, [44, 42, 43])]
        for seed in seeds
    ]
    frame = pd.DataFrame(rows)
    assert np.allclose(comparison.paired_deltas(frame, [("A", "B")])[comparison.METRICS], 0.1)
    with pytest.raises(ValueError, match="seed"):
        comparison.paired_deltas(frame.iloc[:-1], [("A", "B")])


def test_groups_preserve_encoder_race_and_exclude_missing_age():
    frame = pd.DataFrame(
        {
            "age": [64, 65, 90, np.nan],
            "age_missing": [0, 0, 1, 0],
            "race": [5, 0, 2, 1],
            "race_group": [3, 0, 2, 1],
            "race_group_missing": [0, 0, 0, 0],
        }
    )
    np.testing.assert_array_equal(comparison.group_ids(frame), [3, 4, -1, -1])
    assert frame.race.tolist() == [5, 0, 2, 1]


def test_missing_prediction_cache_is_rejected(tmp_path):
    with pytest.raises(RuntimeError, match="cache"):
        comparison.read_predictions(tmp_path, {("M02", 42): tmp_path / "missing-run"}, pd.DataFrame())


def test_two_stage_parent_requires_same_seed_best_checkpoint(tmp_path):
    best = {"path": "checkpoints/best_val_auroc_008.ckpt", "sha256": "best-sha"}
    parent_record = {"status": "succeeded", "checkpoints": [{"monitor": "val/auroc", "best": best}]}
    record = {"seed": 42, "parent_run": {"run_id": "parent-s42", "backbone_checkpoint": dict(best)}}
    config = {"model": {"freeze_backbone": True}}
    candidates = {("resnet-chexpert-race-known", 42): [(tmp_path / "parent-s42", parent_record, {})]}
    comparison.validate_parent(record, config, candidates)
    record["seed"] = 43
    with pytest.raises(ValueError, match="同seed"):
        comparison.validate_parent(record, config, candidates)
    record["seed"] = 42
    record["parent_run"]["backbone_checkpoint"]["sha256"] = "last-sha"
    with pytest.raises(ValueError, match="best"):
        comparison.validate_parent(record, config, candidates)


@pytest.mark.parametrize("failure", ["missing_epoch", "missing_q", "bad_sum"])
def test_q_diagnostics_reject_incomplete_history(tmp_path, monkeypatch, failure):
    directory = tmp_path / "run"
    (directory / "metrics").mkdir(parents=True)
    metadata = {
        "group_attributes": ["age_group_65", "race_group"],
        "group_cardinalities": [2, 4],
        "groups": [{"id": group, "label": label} for group, label in enumerate(comparison.GROUP_LABELS)],
    }
    (directory / "metrics/group_dro_groups.json").write_text(json.dumps(metadata))
    history = [
        {
            "epoch": epoch,
            **{
                f"train/group_dro/{metric}/g{group}": value
                for metric, value in [("q_end", 0.125), ("q_mean", 0.125), ("loss", 0.2), ("count", 2)]
                for group in range(8)
            },
        }
        for epoch in range(2)
    ]
    if failure == "missing_epoch":
        history.pop()
    elif failure == "missing_q":
        history[0]["train/group_dro/q_end/g0"] = np.nan
    else:
        history[0]["train/group_dro/q_end/g0"] = 0.9
    monkeypatch.setattr(
        comparison,
        "read_config",
        lambda _: {"training_strategy": {"name": "group_dro_global"}, "trainer": {"max_epochs": 2}},
    )
    monkeypatch.setattr(comparison, "read_epoch_metrics", lambda _: history)
    provenance = pd.DataFrame([{"condition": "M02", "seed": 42, "checkpoint": "best_val_auroc_001.ckpt"}])
    with pytest.raises(ValueError):
        comparison.read_q_history({("M02", 42): directory}, provenance)


@pytest.mark.parametrize(
    "filename, pairs",
    [
        ("resnet_erm_vs_groupdro_global.ipynb", [["M01", "M02"]]),
        ("02_class_weight_scope.ipynb", [["M02", "M11"]]),
        ("03_spatial_lora_stage_range.ipynb", [["M04", "M17"]]),
        ("04_architecture_comparison.ipynb", [["M01", "M03"], ["M01", "M16"], ["M01", "M07"]]),
        ("05_one_stage_vs_frozen_two_stage.ipynb", [["M04", "M06"]]),
        ("07_two_stage_operating_point.ipynb", [["M04", "M06"], ["M08", "M10"]]),
        (
            "06_resnet_baselines_vs_groupdro_architectures.ipynb",
            [
                ["M01", "M02"],
                ["M01", "M03"],
                ["M01", "M04"],
                ["M01", "M06"],
                ["M01", "M07"],
                ["M01", "M08"],
                ["M01", "M10"],
                ["M02", "M04"],
                ["M02", "M06"],
                ["M02", "M08"],
                ["M02", "M10"],
                ["M03", "M04"],
                ["M07", "M08"],
                ["M04", "M06"],
                ["M08", "M10"],
            ],
        ),
    ],
)
def test_notebook_executes_independently_with_fixture(tmp_path, monkeypatch, filename, pairs):
    """全セルを小さな入力で通す。出力先はtmpで実結果には混ぜない。"""
    import matplotlib.pyplot as plt

    path = REPOSITORY_ROOT / "analysis/archive/race_known_architecture_objective_comparison" / filename
    notebook = json.loads(path.read_text())
    sources = ["".join(cell["source"]) for cell in notebook["cells"] if cell["cell_type"] == "code"]
    monkeypatch.setattr("analysis.common.paths.study_dir", lambda _: tmp_path)
    namespace = {}
    exec(compile(sources[0], filename, "exec"), namespace)
    assert namespace["PAIRS"] == pairs and namespace["SEEDS"] == [42, 43, 44]
    frame = pd.DataFrame(
        {
            "target": [0, 1] * 8,
            "race_group": np.repeat(list(range(4)) * 2, 2),
            "age": np.repeat([40] * 4 + [70] * 4, 2),
            "age_missing": 0,
            "race_group_missing": False,
            "image": [f"image{i}" for i in range(16)],
        }
    )
    records, directories, caches, q_rows = [], {}, {}, []
    for condition in namespace["CONDITIONS"]:
        for seed in [42, 43, 44]:
            records.append(
                {
                    "condition": condition,
                    "seed": seed,
                    "run_id": f"{condition}-s{seed}",
                    "checkpoint": "best_val_auroc_001.ckpt",
                    "objective": "erm" if condition in {"M01", "M03", "M07", "M16"} else "group_dro_global",
                    "class_weight_scope": "global",
                    "freeze_backbone": condition in {"M06", "M10"},
                    "parent_run": None,
                }
            )
            directories[condition, seed] = tmp_path / "runs" / f"{condition}-s{seed}"
            probability = np.where(frame.target == 1, 0.8, 0.2)
            caches[condition, seed] = {
                "target": frame.target.to_numpy(),
                "probabilities": np.column_stack((1 - probability, probability)),
                "predictions": frame.target.to_numpy(),
            }
            for epoch in range(3):
                for group, label in enumerate(comparison.GROUP_LABELS):
                    q_rows.append(
                        {
                            "condition": condition,
                            "seed": seed,
                            "epoch": epoch,
                            "group_id": group,
                            "group": label,
                            "best_epoch": 1,
                            "q_end": 0.125,
                            "q_mean": 0.125,
                            "loss": 0.2,
                            "count": 2,
                        }
                    )
    provenance, history = pd.DataFrame(records), pd.DataFrame(q_rows)
    namespace.update(
        select_runs=lambda *_: (provenance, frame, directories),
        prediction_command=lambda *_, **__: "",
        read_predictions=lambda *_: caches,
        read_q_history=lambda *_: history[~history.condition.isin({"M01", "M03", "M07", "M16"})],
    )
    for source in sources[1:]:
        exec(compile(source, filename, "exec"), namespace)
    assert len(namespace["summary"]) == len(namespace["CONDITIONS"]) * 3
    assert len(namespace["delta"]) == len(pairs) * 3
    assert (namespace["RESULTS"] / "paired_delta_mean_std.csv").is_file()
    plt.close("all")
