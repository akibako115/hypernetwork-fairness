"""学習済みの条件付きモデルへ、属性を差し替えて test を推論し直す。

2 つの問いを、学習をやり直さずに推論だけで切り分ける。

1. 黒色腫と良性の紛らわしい病変（bkl・bcc）を分ける改善は、行ごとの属性に依存するのか。
   属性を行間で入れ替えても、全行同じ値にしても改善が残るなら、改善は条件付けではなく
   adapter を足したアーキテクチャ側にある。
2. 新施設（external）での劣化は、学習時に一度も現れなかった欠損入力のせいか。
   train/val は sex・age の欠損行を外しているので、sex 欠損の embedding は初期値のまま、
   age 欠損の行は 0 歳を標準化した値（約 -3 SD）と立ったことのない欠損フラグを受け取る。
   train の最頻値・平均で埋めて劣化が戻るなら、原因は欠損入力にある。

属性の差し替えは標準化済みの frame の上で行う。年齢の「平均で埋める」は標準化後の 0 になる。

使い方:
    uv run python analysis/ham10000_architecture_erm/counterfactual.py --device cuda:0 \
        --run-dir analysis/ham10000_architecture_erm/runs/<run-id> ...
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Callable
from pathlib import Path

import numpy as np
import pandas as pd
import rootutils
import torch
from hydra.utils import instantiate
from omegaconf import OmegaConf
from torch.utils.data import DataLoader

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from analysis.common.paths import local_data_path  # noqa: E402
from analysis.common.predictions import checkpoint_reference, load_network  # noqa: E402

PACKAGE = Path(__file__).resolve().parent
SCHEMA_VERSION = 1
# 差し替える属性列。sex・localization が categorical、age が continuous（標準化済み）。
COLUMNS = ("sex", "localization", "age")
# 埋める値。train の最頻値（sex=male、localization=back）と、標準化後の平均 0。
TYPICAL = {"sex": 1, "localization": 2, "age": 0.0}
FACE = 5
EXTERNAL = 0


def _fill(frame: pd.DataFrame, rows: pd.Series, columns: tuple[str, ...], values: dict) -> pd.DataFrame:
    """rows の columns を values で埋め、欠損フラグを下ろす。"""
    out = frame.copy()
    for column in columns:
        out.loc[rows, column] = values[column]
        out.loc[rows, f"{column}_missing"] = 0
    return out


def _shuffle(frame: pd.DataFrame, seed: int) -> pd.DataFrame:
    """sex・age が揃った行の間で、属性の組を丸ごと入れ替える。

    欠損行を混ぜると、欠損入力の問題を既存施設の行へ持ち込むので、学習分布の内側の属性だけを
    入れ替える。組で動かすので、属性の同時分布は保たれ、画像との対応だけが壊れる。
    """
    out = frame.copy()
    rows = np.flatnonzero((frame["sex_missing"] == 0) & (frame["age_missing"] == 0))
    source = np.random.default_rng(seed).permutation(rows)
    for column in COLUMNS:
        for name in (column, f"{column}_missing"):
            out.iloc[rows, out.columns.get_loc(name)] = frame[name].to_numpy()[source]
    return out


def _all_missing(frame: pd.DataFrame) -> pd.DataFrame:
    """全行の全属性を欠損扱いにする（値は配布 CSV と同じく 0 歳・-1 を標準化した値）。"""
    out = frame.copy()
    for column in COLUMNS:
        out[f"{column}_missing"] = 1
    return out


def conditions(frame: pd.DataFrame) -> dict[str, Callable[[], pd.DataFrame]]:
    """条件名から、差し替えた frame を作る関数への対応を返す。"""
    all_rows = pd.Series(True, index=frame.index)
    return {
        "observed": lambda: frame,
        "impute_sex_age": lambda: _fill(
            _fill(frame, frame["sex_missing"] == 1, ("sex",), TYPICAL), frame["age_missing"] == 1, ("age",), TYPICAL
        ),
        # sex と age のどちらの欠損が効くかを分ける。
        "impute_sex": lambda: _fill(frame, frame["sex_missing"] == 1, ("sex",), TYPICAL),
        "impute_age": lambda: _fill(frame, frame["age_missing"] == 1, ("age",), TYPICAL),
        "impute_all": lambda: _fill(
            _fill(
                _fill(frame, frame["sex_missing"] == 1, ("sex",), TYPICAL), frame["age_missing"] == 1, ("age",), TYPICAL
            ),
            frame["localization_missing"] == 1,
            ("localization",),
            TYPICAL,
        ),
        "shuffle_0": lambda: _shuffle(frame, 0),
        "shuffle_1": lambda: _shuffle(frame, 1),
        "constant_typical": lambda: _fill(frame, all_rows, COLUMNS, TYPICAL),
        "constant_face": lambda: _fill(frame, all_rows, COLUMNS, {**TYPICAL, "localization": FACE}),
        "all_missing": lambda: _all_missing(frame),
    }


def cache_path(run_id: str, condition: str) -> Path:
    """run と条件に対応する cache path を返す。"""
    return PACKAGE / "cache" / f"{run_id}_test_cf-{condition}.npz"


def write(run_dir: Path, device: torch.device, overwrite: bool = False) -> None:
    """1 run について、全条件の test 予測を cache へ書く。network と datamodule は 1 度だけ作る。"""
    config = OmegaConf.load(run_dir / "config.yaml")
    if not config.model.get("use_attributes", False):
        raise ValueError(f"{run_dir.name} は属性を取らない（差し替えても予測は変わらない）")
    for key in ("cv_splits_dir", "data_dir"):
        config.data[key] = str(local_data_path(config.data[key]))
    datamodule = instantiate(config.data)
    (frame,) = datamodule.standardized_dataframes("test")
    network = load_network(run_dir, device)
    checkpoint = checkpoint_reference(run_dir)
    for name, make in conditions(frame).items():
        output = cache_path(run_dir.name, name)
        if output.is_file() and not overwrite:
            if json.loads(str(np.load(output)["metadata"])).get("checkpoint") == checkpoint:
                print(f"cached: {output.name}")
                continue
        edited = make()
        # 画像の行順は変えない。属性だけが差し替わる。
        assert (edited["image"].to_numpy() == frame["image"].to_numpy()).all()
        # 逐次で回すので persistent worker は持たない（並列 probe の OOM を避ける）。
        loader = DataLoader(
            datamodule._create_dataset(edited, datamodule.val_transform),
            batch_size=datamodule.batch_size_per_device,
            num_workers=8,
            shuffle=False,
        )
        logits = []
        with torch.inference_mode():
            for image, attributes, _ in loader:
                inputs = {key: value.to(device, non_blocking=True) for key, value in attributes.items()}
                logits.append(network(image.to(device, non_blocking=True), inputs).float().cpu().numpy())
        logits = np.concatenate(logits)
        probabilities = torch.softmax(torch.from_numpy(logits), dim=1).numpy()
        metadata = {
            "schema_version": SCHEMA_VERSION,
            "run_id": run_dir.name,
            "checkpoint": checkpoint,
            "condition": name,
        }
        output.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            output,
            image=frame["image"].to_numpy(dtype=str),
            logits=logits,
            probabilities=probabilities,
            target=frame["target"].to_numpy(dtype=np.int64),
            metadata=json.dumps(metadata, sort_keys=True),
        )
        print(f"wrote: {output.name}")


def load(run_id: str, condition: str, images: np.ndarray) -> np.ndarray:
    """cache を読み、split CSV との行対応を確かめて陽性確率を返す。"""
    with np.load(cache_path(run_id, condition), allow_pickle=False) as cached:
        if not (cached["image"] == images).all():
            raise ValueError(f"{run_id} {condition} の行が split CSV と対応していない")
        return cached["probabilities"][:, 1]


def main() -> None:
    """CLI 入口。"""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run-dir", action="append", required=True, type=Path, dest="run_dirs")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    for run_dir in args.run_dirs:
        write(run_dir.resolve(), torch.device(args.device), args.overwrite)


if __name__ == "__main__":
    main()
