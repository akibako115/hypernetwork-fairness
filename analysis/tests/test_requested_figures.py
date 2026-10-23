"""図の根拠を静かに変える経路：率の復元と同一best/lastのcache選択。"""

import json

import numpy as np
import pandas as pd
import pytest

from analysis.hypernet_conditioning_grid.requested_analysis import best_cache_files, restore_rates, validated_cache


def test_restore_rates_from_imbalanced_confusion_matrix():
    # 100 positive: TP=70, FN=30; 900 negative: FP=90, TN=810.
    tpr, fpr = restore_rates(0.88, 0.8, 0.1)
    assert tpr == pytest.approx(0.7)
    assert fpr == pytest.approx(0.1)
    with pytest.raises(ValueError, match="従属"):
        restore_rates(0.8, 0.8, 0.5)
    with pytest.raises(ValueError, match="範囲|外"):
        restore_rates(0.1, 0.9, 0.1)


def test_best_cache_accepts_identical_last_but_rejects_wrong_split(tmp_path):
    run = tmp_path / "runs" / "run-a"
    run.mkdir(parents=True)
    record = {
        "status": "succeeded",
        "checkpoints": [
            {"monitor": "val/auroc", "best": {"path": "checkpoints/best_val_auroc_029.ckpt", "sha256": "same-content"}}
        ],
    }
    (run / "run.json").write_text(json.dumps(record))
    cache = tmp_path / "cache"
    cache.mkdir()
    path = cache / "run-a_test.npz"
    metadata = {
        "run_id": "run-a",
        "split": "test",
        "checkpoint": "run-a/checkpoints/last.ckpt",
        "checkpoint_sha256": "same-content",
        "split_sha256": "test-split",
    }
    np.savez(path, metadata=json.dumps(metadata))
    table = pd.DataFrame({"run_id": ["run-a"]})
    assert best_cache_files(table, tmp_path, "test") == {("run-a", "best"): path}
    validated_cache(path, run, "test", "test-split")
    with pytest.raises(ValueError, match="不一致"):
        validated_cache(path, run, "test", "other-split")
    metadata["checkpoint_sha256"] = "different-content"
    np.savez(path, metadata=json.dumps(metadata))
    assert best_cache_files(table, tmp_path, "test") == {}
