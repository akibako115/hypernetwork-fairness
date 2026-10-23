"""予測 cache が「古いまま黙って使われる」経路を塞いだことを固定する。

cache は run artifact の派生物で、checkpoint を選び直せば中身が変わる。存在することだけを
見て使い回すと、古い予測から出た表と図がそのまま残る。
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from omegaconf import OmegaConf

from analysis.common import predictions
from analysis.common.predictions import (
    FEATURES_SCHEMA_VERSION,
    SCHEMA_VERSION,
    cache_path,
    evaluation_data_config,
    features_cache_path,
    load_cache,
    load_features,
    rebuild_reason,
    stale_reason,
)

IMAGES = np.array(["a.jpg", "b.jpg", "c.jpg"], dtype=str)


def _cache(path: Path, images: np.ndarray = IMAGES, **metadata: object) -> Path:
    """`write_cache` が書くのと同じ形の `.npz` を作る。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    record = {"schema_version": SCHEMA_VERSION, "checkpoint": "runs/r/checkpoints/best.ckpt", **metadata}
    np.savez_compressed(
        path,
        image=images,
        logits=np.zeros((len(images), 2), dtype=np.float32),
        probabilities=np.full((len(images), 2), 0.5, dtype=np.float32),
        predictions=np.zeros(len(images), dtype=np.int64),
        target=np.zeros(len(images), dtype=np.int64),
        metadata=json.dumps(record, ensure_ascii=False, sort_keys=True),
    )
    return path


def _features_cache(path: Path, images: np.ndarray = IMAGES, **metadata: object) -> Path:
    """`write_cache --features` が書くのと同じ形の `.npz` を作る。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    record = {
        "schema_version": FEATURES_SCHEMA_VERSION,
        "checkpoint": "runs/r/checkpoints/best.ckpt",
        "feature_dim": 4,
        **metadata,
    }
    np.savez_compressed(
        path,
        image=images,
        features=np.zeros((len(images), 4), dtype=np.float16),
        metadata=json.dumps(record, ensure_ascii=False, sort_keys=True),
    )
    return path


def _run(root: Path, checkpoint: str) -> Path:
    """`selected_checkpoint` が 1 つに定まる run directory を作る。"""
    run_dir = root / "runs" / "r"
    (run_dir / "checkpoints").mkdir(parents=True)
    (run_dir / "checkpoints" / checkpoint).touch()
    (run_dir / "run.json").write_text(json.dumps({"status": "succeeded"}), encoding="utf-8")
    return run_dir


@pytest.fixture
def repository(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """cache の metadata が持つ checkpoint path の起点を差し替える。"""
    monkeypatch.setattr(predictions, "REPOSITORY_ROOT", tmp_path)
    return tmp_path


def test_a_cache_matching_the_current_checkpoint_is_reused(repository: Path) -> None:
    """同じ checkpoint から作った cache は作り直さない。推論は高い。"""
    run_dir = _run(repository, "best_val_auroc_007.ckpt")
    path = _cache(repository / "cache" / "r_test.npz", checkpoint="runs/r/checkpoints/best_val_auroc_007.ckpt")

    assert stale_reason(path, run_dir) is None


def test_the_checkpoint_reference_does_not_depend_on_how_the_run_was_addressed(
    repository: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """書く側と照合する側で同じ文字列になる必要がある。相対 path で渡しても同じ値にする。"""
    run_dir = _run(repository, "best_val_auroc_007.ckpt")
    monkeypatch.chdir(repository)

    assert predictions.checkpoint_reference(Path("runs/r")) == predictions.checkpoint_reference(run_dir)


def test_a_cache_keeps_matching_after_the_run_is_imported_into_a_package(repository: Path) -> None:
    """取り込み前の置き場（`projects/<project>/runs/`）を記録した cache も、同じ checkpoint なら使い回す。

    hardlink で取り込んだ run は同じ file を持つ。置き場所が変わっただけで推論をやり直さない。
    """
    run_dir = _run(repository, "best_val_auroc_007.ckpt")
    recorded = "projects/hypernet_e2e/runs/r/checkpoints/best_val_auroc_007.ckpt"
    path = _cache(repository / "cache" / "r_test.npz", checkpoint=recorded)

    assert predictions.checkpoint_reference(run_dir) == "r/checkpoints/best_val_auroc_007.ckpt"
    assert stale_reason(path, run_dir) is None


def test_a_cache_from_another_checkpoint_is_rebuilt_without_asking(repository: Path) -> None:
    """checkpoint を選び直したら、`--overwrite` を思い出せなくても作り直す。"""
    run_dir = _run(repository, "best_val_auroc_009.ckpt")
    path = _cache(repository / "cache" / "r_test.npz", checkpoint="runs/r/checkpoints/best_val_auroc_007.ckpt")

    assert "best_val_auroc_009.ckpt" in (stale_reason(path, run_dir) or "")


def test_a_cache_written_by_an_older_schema_is_rebuilt(repository: Path) -> None:
    """schema を上げたら中身の意味が変わる。書くだけで読まない version にしない。"""
    run_dir = _run(repository, "best_val_auroc_007.ckpt")
    path = _cache(repository / "cache" / "r_test.npz", schema_version=SCHEMA_VERSION - 1)

    assert "schema_version" in (stale_reason(path, run_dir) or "")


def test_a_missing_or_unreadable_cache_is_a_reason_to_rebuild(repository: Path) -> None:
    """壊れた `.npz` で落ちるより、作り直す方が正しい。"""
    run_dir = _run(repository, "best_val_auroc_007.ckpt")
    missing = repository / "cache" / "r_test.npz"
    assert stale_reason(missing, run_dir) == "cache が無い"

    missing.parent.mkdir(parents=True, exist_ok=True)
    missing.write_bytes(b"not an npz")
    assert stale_reason(missing, run_dir) == "cache を読めない"


def test_loading_a_cache_of_an_old_schema_raises(tmp_path: Path) -> None:
    """読む側でも止める。古い cache から作った表が残らないようにする。"""
    path = _cache(tmp_path / "r_test.npz", schema_version=SCHEMA_VERSION - 1)

    with pytest.raises(ValueError, match="schema_version"):
        load_cache(path, IMAGES)


def test_a_cache_whose_rows_do_not_match_the_split_csv_raises(tmp_path: Path) -> None:
    """行がずれても数値は出る。群の割り当てが全部ずれた表が「それらしく」並ぶ。"""
    path = _cache(tmp_path / "r_test.npz", images=np.array(["a.jpg", "c.jpg", "b.jpg"], dtype=str))

    with pytest.raises(ValueError, match="対応していない"):
        load_cache(path, IMAGES)


def test_a_cache_of_a_different_length_raises_instead_of_broadcasting(tmp_path: Path) -> None:
    """長さが違う比較は numpy では真偽が定まらない。先に長さを見る。"""
    path = _cache(tmp_path / "r_test.npz", images=np.array(["a.jpg", "b.jpg"], dtype=str))

    with pytest.raises(ValueError, match="対応していない"):
        load_cache(path, IMAGES)


def test_a_matching_cache_comes_back_with_parsed_metadata(tmp_path: Path) -> None:
    """metadata は JSON 文字列で入っているので、読む側で parse 済みにして返す。"""
    path = _cache(tmp_path / "r_test.npz", run_id="r", split="test")

    cached = load_cache(path, IMAGES)

    assert cached["metadata"]["run_id"] == "r"
    assert cached["probabilities"].shape == (3, 2)


def test_cache_file_names_carry_the_run_and_split(tmp_path: Path) -> None:
    """1 つの run について split ごとに別 cache を持つ。"""
    assert cache_path(tmp_path, "20260921T103036Z-resnet-chexpert-s42-5538", "test").name == (
        "20260921T103036Z-resnet-chexpert-s42-5538_test.npz"
    )


def test_features_are_rebuilt_when_only_the_predictions_were_cached(repository: Path) -> None:
    """予測 cache だけがある run で `--features` を付けたら、推論をやり直す。

    ここを「予測 cache があるから最新」と読むと、**probe が古い run の表現を読む**か、
    そもそも file が無いまま先へ進む。
    """
    run_dir = _run(repository, "best_val_auroc_007.ckpt")
    cache_dir = repository / "cache"
    _cache(cache_path(cache_dir, "r", "test"), checkpoint="runs/r/checkpoints/best_val_auroc_007.ckpt")

    assert rebuild_reason(cache_dir, run_dir, "test", with_features=False) is None
    assert "特徴量" in (rebuild_reason(cache_dir, run_dir, "test", with_features=True) or "")


def test_both_caches_current_means_no_rebuild(repository: Path) -> None:
    """両方とも同じ checkpoint から出ているなら、推論はやり直さない。"""
    run_dir = _run(repository, "best_val_auroc_007.ckpt")
    cache_dir = repository / "cache"
    checkpoint = "runs/r/checkpoints/best_val_auroc_007.ckpt"
    _cache(cache_path(cache_dir, "r", "test"), checkpoint=checkpoint)
    _features_cache(features_cache_path(cache_dir, "r", "test"), checkpoint=checkpoint)

    assert rebuild_reason(cache_dir, run_dir, "test", with_features=True) is None


def test_a_features_cache_from_another_checkpoint_is_rebuilt(repository: Path) -> None:
    """checkpoint を選び直したら表現も変わる。予測 cache と同じ基準で古くなる。"""
    run_dir = _run(repository, "best_val_auroc_009.ckpt")
    path = _features_cache(
        repository / "cache" / "r_test_features.npz", checkpoint="runs/r/checkpoints/best_val_auroc_007.ckpt"
    )

    assert "best_val_auroc_009.ckpt" in (stale_reason(path, run_dir, FEATURES_SCHEMA_VERSION) or "")


def test_a_features_cache_whose_rows_do_not_match_the_split_csv_raises(tmp_path: Path) -> None:
    """表現がずれても probe は学習できてしまう。属性の割り当てが全部ずれた AUROC が出る。"""
    path = _features_cache(tmp_path / "r_test_features.npz", images=np.array(["a.jpg", "c.jpg", "b.jpg"], dtype=str))

    with pytest.raises(ValueError, match="対応していない"):
        load_features(path, IMAGES)


def test_features_come_back_as_float32(tmp_path: Path) -> None:
    """cache は float16 で持つが、`sklearn` に渡す前にここで戻す。"""
    path = _features_cache(tmp_path / "r_test_features.npz")

    loaded = load_features(path, IMAGES)

    assert loaded["features"].dtype == np.float32
    assert loaded["features"].shape == (3, 4)


def test_features_cache_file_names_do_not_collide_with_predictions(tmp_path: Path) -> None:
    """同じ run・同じ split で 2 つの cache を持つ。名前が衝突すると片方が消える。"""
    predictions_path = cache_path(tmp_path, "20260922T063725Z-resnet-chexpert-s43-efcd", "val")
    features_path = features_cache_path(tmp_path, "20260922T063725Z-resnet-chexpert-s43-efcd", "val")

    assert features_path.name == "20260922T063725Z-resnet-chexpert-s43-efcd_val_features.npz"
    assert features_path != predictions_path


def test_named_checkpoint_hash_and_split_invalidate_cache(repository: Path) -> None:
    """best 以外の名前で選んだ checkpoint は、同じ名前のまま中身が変わっても作り直す。"""
    run_dir = _run(repository, "best_val_auroc_007.ckpt")
    path_on_disk = run_dir / "checkpoints" / "swad.ckpt"
    path_on_disk.write_bytes(b"original weights")
    checkpoint = "swad"
    split_dir = repository / "splits"
    split_dir.mkdir()
    split = split_dir / "val.csv"
    split.write_text("image,target\na,0\nb,1\n")
    (run_dir / "config.yaml").write_text(f"data:\n  cv_splits_dir: {split_dir}\n")
    path = _cache(
        repository / "cache" / "explicit.npz",
        checkpoint=predictions.checkpoint_reference(run_dir, checkpoint),
        checkpoint_sha256=predictions._sha256(path_on_disk),
        split_sha256=predictions._sha256(split),
        split="val",
    )
    assert stale_reason(path, run_dir, checkpoint=checkpoint) is None
    split.write_text("image,target\na,1\nb,0\n")
    assert "split SHA" in stale_reason(path, run_dir, checkpoint=checkpoint)
    path_on_disk.write_bytes(b"changed weights at same path")
    assert "checkpoint SHA" in stale_reason(path, run_dir, checkpoint=checkpoint)


def test_sha_is_reused_within_operation_but_new_operation_detects_rewrite(tmp_path, monkeypatch):
    import os

    path = tmp_path / "checkpoint.ckpt"
    path.write_bytes(b"original")
    actual = predictions.hashlib.file_digest
    calls = []

    def digest(stream, algorithm):
        calls.append(stream.name)
        return actual(stream, algorithm)

    monkeypatch.setattr(predictions.hashlib, "file_digest", digest)
    source_hashes = {}
    first = predictions._sha256(path, source_hashes)
    assert predictions._sha256(path, source_hashes) == first and len(calls) == 1
    stat = path.stat()
    path.write_bytes(b"modified")
    os.utime(path, ns=(stat.st_atime_ns, stat.st_mtime_ns))
    assert predictions._sha256(path) != first and len(calls) == 2
    replacement = tmp_path / "replacement"
    replacement.write_bytes(b"replaced")
    replacement.replace(path)
    assert predictions._sha256(path) != first and len(calls) == 3


def test_group_datamodule_config_fills_missing_group_attributes_for_prediction() -> None:
    data = OmegaConf.create(
        {"group_attribute_names": ["sex", "age_group_65"], "group_missing_values": {"sex": 2}, "batch_size": 8}
    )
    filled = evaluation_data_config(data)
    assert dict(filled.group_missing_values) == {"sex": 2, "age_group_65": 0}
    assert "group_missing_values" in data and dict(data.group_missing_values) == {"sex": 2}


def test_plain_datamodule_config_is_left_unchanged() -> None:
    data = OmegaConf.create({"batch_size": 8})
    assert evaluation_data_config(data) is data


def test_a_named_checkpoint_gets_its_own_cache_name(repository: Path) -> None:
    """best の cache 名は run-id だけ、それ以外は名前を付ける。同じ run の best と swad の cache が衝突しない。"""
    run_dir = _run(repository, "best_val_auroc_007.ckpt")

    assert predictions._cache_identity(run_dir, "best") == "r"
    assert predictions._cache_identity(run_dir, "swad") == "r_swad"
