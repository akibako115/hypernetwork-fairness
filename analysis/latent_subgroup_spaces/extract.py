"""1 段・ERM の Spatial LoRA から、潜在サブグループを探す空間を 1 run ずつ書き出す。

学習はしない。1 run について次の 2 種類を `cache/` に書く。

1. `<run-id>_<split>.npz`（val・test）：split CSV の行順に、logit の差（No Finding − 所見あり）、
   分類 head 直前の特徴（2048 次元、float16）、metadata encoder の出力 condition c（16 次元）、
   属性の組み合わせ（セル）の番号。特徴と logit は同じ forward から取る。
2. `<run-id>_cells.npz`：train・val・test に現れる、モデル入力の属性の組み合わせ（セル）ごとの condition と、
   stage ごとの生成 ΔW の Gram 行列。ΔW = (α/r)·A(c)B(c) は低ランクなので、内積は r×r の積で計算し、
   ΔW を実体化しない（`conditioned_subgroup_discovery/delta_probe.py` と同じ計算）。層ごとに元重みの
   二乗ノルムで割ってから足し、stage4 の 1 層が全体を支配しないようにする。

condition と ΔW は属性だけの関数なので、同じセルの行は必ず同じ点に写る。行ごとに ΔW を持たず、
セルの番号で引けば足りる。

使い方（repo root で）:
    uv run python -m analysis.latent_subgroup_spaces.extract --device cuda:0 \
        --run-dir analysis/latent_subgroup_spaces/runs/<run-id> ...
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import rootutils
import torch
from hydra.utils import instantiate
from omegaconf import OmegaConf

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from analysis.archive.race_known_architecture_objective_comparison.spatial_lora_inspection import (  # noqa: E402
    adapter_layers,
)
from analysis.common.paths import local_data_path  # noqa: E402
from analysis.common.predictions import checkpoint_reference, load_network  # noqa: E402
from projects.hypernet_e2e.data.attribute_utils import attribute_tensors  # noqa: E402

PACKAGE = Path(__file__).resolve().parent
SCHEMA_VERSION = 1
SPLITS = ("val", "test")
STAGES = ("stage3", "stage4", "fc")


def load_datamodule(run_dir: Path):
    """run の config から、学習と同じ設定の datamodule と、モデル入力の属性名を返す。"""
    config = OmegaConf.load(run_dir / "config.yaml")
    for key in ("cv_splits_dir", "data_dir"):
        config.data[key] = str(local_data_path(config.data[key]))
    return instantiate(config.data), OmegaConf.to_container(config.model.attribute_names)


def input_columns(attribute_names: dict) -> list[str]:
    """モデル入力を決める列（値と欠損フラグ）。この列が同じ行は同じ condition を受ける。"""
    names = attribute_names["categorical"] + attribute_names["continuous"]
    return names + [f"{name}_missing" for name in names]


def cell_table(frames: dict[str, pd.DataFrame], columns: list[str]) -> pd.DataFrame:
    """全 split に現れるセルと、train での件数・陽性数。

    Args:
        frames: 標準化済みの train・val・test。
        columns: `input_columns` の列。

    Returns:
        DataFrame: セルの鍵の列、`train_count`、`train_positives`。行番号がセルの番号になる。
    """
    every = pd.concat([frame[columns] for frame in frames.values()]).drop_duplicates(ignore_index=True)
    train = (
        frames["train"]
        .groupby(columns, as_index=False)
        .agg(train_count=("target", "size"), train_positives=("target", "sum"))
    )
    table = every.merge(train, on=columns, how="left")
    return table.fillna({"train_count": 0, "train_positives": 0})


def cell_index(frame: pd.DataFrame, cells: pd.DataFrame, columns: list[str]) -> np.ndarray:
    """各行のセル番号。"""
    keyed = cells[columns].reset_index().rename(columns={"index": "cell"})
    merged = frame[columns].merge(keyed, on=columns, how="left", validate="many_to_one")
    if merged["cell"].isna().any():
        raise ValueError("セル表に無い属性の組み合わせがある")
    return merged["cell"].to_numpy(dtype=np.int64)


@torch.inference_mode()
def encode(network, table: pd.DataFrame, attribute_names: dict, device: torch.device) -> torch.Tensor:
    """属性表を condition `[n, 16]` へ写す。"""
    attributes = attribute_tensors(table, attribute_names)
    return network.metadata_encoder({key: value.to(device) for key, value in attributes.items()})


def _factor_gram(layer, condition: torch.Tensor, chunk: int = 512) -> torch.Tensor:
    """1 層の ΔW の Gram 行列 `[n, n]`。⟨A_iB_i, A_jB_j⟩ = Σ (A_iᵀA_j) ⊙ (B_iB_jᵀ)。

    セル数が 3,000 を超えるので、行方向に分けて float64 で積む。
    """
    a, b = layer._factors(condition)
    a = a.double()
    b = b.flatten(2).double()
    n = len(condition)
    gram = torch.empty((n, n), dtype=torch.float64, device=condition.device)
    for start in range(0, n, chunk):
        stop = min(start + chunk, n)
        left = torch.einsum("ior,jos->ijrs", a[start:stop], a)
        right = torch.einsum("ird,jsd->ijrs", b[start:stop], b)
        gram[start:stop] = (left * right).sum((2, 3))
    return layer.scale**2 * gram


@torch.inference_mode()
def delta_grams(network, condition: torch.Tensor) -> dict[str, np.ndarray]:
    """stage ごとに、元重みの二乗ノルムで割った ΔW の Gram を足し合わせる。"""
    grams = {stage: 0.0 for stage in STAGES}
    for name, layer, base_weight in adapter_layers(network):
        stage = name.split(".")[0]
        base_sq = (base_weight.double() ** 2).sum()
        grams[stage] = grams[stage] + _factor_gram(layer, condition) / base_sq
    return {stage: gram.cpu().numpy() for stage, gram in grams.items()}


@torch.inference_mode()
def forward_split(network, datamodule, split: str, device: torch.device) -> tuple[np.ndarray, np.ndarray]:
    """評価 transform・CSV 順で split を推論し、logit の差と head 直前の特徴を返す。"""
    margins, features = [], []
    for image, attributes, _ in datamodule.evaluation_dataloader(split, persistent_workers=False):
        attributes = {key: value.to(device, non_blocking=True) for key, value in attributes.items()}
        condition = network.metadata_encoder(attributes)
        feature = network._forward_backbone(image.to(device, non_blocking=True), condition)
        logits = network.fc(feature, condition)
        margins.append((logits[:, 1] - logits[:, 0]).float().cpu().numpy())
        features.append(feature.half().cpu().numpy())
    return np.concatenate(margins), np.concatenate(features)


def _save(path: Path, metadata: dict, **arrays) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, metadata=json.dumps(metadata, sort_keys=True), **arrays)
    print(f"wrote: {path.relative_to(PACKAGE)}")


def write(run_dir: Path, device: torch.device) -> None:
    """1 run のセル表・Gram と、val・test の行ごとの空間を書く。"""
    datamodule, attribute_names = load_datamodule(run_dir)
    network = load_network(run_dir, device)
    metadata = {"schema_version": SCHEMA_VERSION, "run_id": run_dir.name, "checkpoint": checkpoint_reference(run_dir)}
    columns = input_columns(attribute_names)
    frames = dict(zip(("train", *SPLITS), datamodule.standardized_dataframes("train", *SPLITS), strict=True))
    cells = cell_table(frames, columns)
    condition = encode(network, cells, attribute_names, device)
    grams = delta_grams(network, condition)
    _save(
        PACKAGE / "cache" / f"{run_dir.name}_cells.npz",
        {**metadata, "columns": columns},
        **{f"cell_{key}": cells[key].to_numpy() for key in cells.columns},
        condition=condition.cpu().numpy(),
        **{f"gram_{stage}": gram for stage, gram in grams.items()},
    )
    for split in SPLITS:
        frame = frames[split]
        margin, feature = forward_split(network, datamodule, split, device)
        _save(
            PACKAGE / "cache" / f"{run_dir.name}_{split}.npz",
            {**metadata, "split": split},
            image=frame["image"].to_numpy(dtype=str),
            target=frame["target"].to_numpy(dtype=np.int64),
            margin=margin,
            features=feature,
            condition=encode(network, frame, attribute_names, device).cpu().numpy(),
            cell=cell_index(frame, cells, columns),
        )


def main() -> None:
    """CLI 入口。"""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run-dir", action="append", required=True, type=Path, dest="run_dirs")
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    torch.backends.cudnn.benchmark = True
    for run_dir in args.run_dirs:
        write(run_dir.resolve(), torch.device(args.device))


if __name__ == "__main__":
    main()
