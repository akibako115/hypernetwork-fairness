"""属性条件付き Spatial LoRA の生成 ΔW を属性セルの格子で読み、セルの条件を差し替えて推論し直す。

学習はしない。1 run について次の 2 つを `cache/` に書く。

1. `<run-id>_cells.npz`：train に現れる属性セル（性別 × 部位 × 年齢）ごとの condition と、
   stage ごとの ΔW の Gram 行列（各層の元重みの二乗ノルムで割って足したもの）。
   ΔW = (α/r)·A(c)B(c) は低ランクなので、内積 ⟨A_iB_i, A_jB_j⟩ = Σ (A_iᵀA_j) ⊙ (B_iB_jᵀ) を
   r×r の行列で計算し、ΔW を実体化しない（stage4 の 1 層だけで 1 セル 236 万要素になる）。
2. `<run-id>_swap_<split>.npz`：評価行の全員に、差し替え先のセルの condition を入れたときの logit 差
   （悪性 − 良性）。stage2 までは condition を使わないので、stage2 の出力を 1 回だけ計算して使い回す。

セル内の順位は、セルの全員が同じ condition を受け取る限り、切片の違いでは変わらない。
差し替えで変わるなら、それは ΔW が画像の特徴の使い方（規則）を変えたためである。

使い方:
    uv run python -m analysis.conditioned_subgroup_discovery.delta_probe --device cuda:0 \
        --run-dir analysis/conditioned_subgroup_discovery/runs/<run-id> ...
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
from torch.utils.data import DataLoader

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from analysis.archive.race_known_architecture_objective_comparison.spatial_lora_inspection import (  # noqa: E402
    adapter_layers,
)
from analysis.common.paths import local_data_path  # noqa: E402
from analysis.common.predictions import checkpoint_reference, load_network  # noqa: E402
from projects.hypernet_e2e.data.attribute_utils import attribute_tensors  # noqa: E402

PACKAGE = Path(__file__).resolve().parent
SCHEMA_VERSION = 1
CELL_KEYS = ["sex", "localization", "raw_age"]
STAGES = ("stage3", "stage4", "fc")
# HAM の external（dataset=0）は年齢欠損の符号化で崩れるので評価から外す（dataset_gain_sources と同じ）。
EXTERNAL = 0
# 差し替え先に使うセルの train 件数の下限。件数の少ないセルの ΔW は学習でほとんど動いていない。
MIN_TARGET_COUNT = 10


def load_datamodule(run_dir: Path):
    """run の config から、学習と同じ設定の datamodule を作る。"""
    config = OmegaConf.load(run_dir / "config.yaml")
    for key in ("cv_splits_dir", "data_dir"):
        config.data[key] = str(local_data_path(config.data[key]))
    return instantiate(config.data), OmegaConf.to_container(config.model.attribute_names)


def model_frame(datamodule, split: str) -> pd.DataFrame:
    """学習と同じ標準化を施した split に、生の年齢を `raw_age` として付ける。"""
    (frame,) = datamodule.standardized_dataframes(split)
    raw = datamodule.read_split_dataframe(split)
    if not (frame["image"].to_numpy() == raw["image"].to_numpy()).all():
        raise ValueError("標準化前後で行順が異なる")
    return frame.assign(raw_age=raw["age"].to_numpy())


def known_rows(frame: pd.DataFrame) -> np.ndarray:
    """性別・部位・年齢・施設がすべて既知の行。セルが属性で一意に決まる行だけを使う。"""
    known = (frame.sex_missing == 0) & (frame.localization_missing == 0) & (frame.age_missing == 0)
    return (known & (frame.dataset != EXTERNAL)).to_numpy()


def cell_table(train: pd.DataFrame) -> pd.DataFrame:
    """train に現れる属性セルと、その件数・陽性数・モデル入力の値を返す。

    Args:
        train: `model_frame` の train。

    Returns:
        DataFrame: `sex`・`localization`・`raw_age`（セルの鍵）、`age`（標準化値）、`count`、`positives`。
            モデル入力の欠損フラグはすべて 0。
    """
    rows = train[known_rows(train)]
    table = rows.groupby(CELL_KEYS, as_index=False).agg(
        age=("age", "first"), count=("target", "size"), positives=("target", "sum")
    )
    missing = {f"{name}_missing": 0 for name in ("sex", "localization", "age")}
    return table.assign(**missing)


@torch.inference_mode()
def encode(network, table: pd.DataFrame, attribute_names: dict, device: torch.device) -> torch.Tensor:
    """属性表を metadata condition `[n, condition_dim]` へ写す。"""
    attributes = attribute_tensors(table, attribute_names)
    return network.metadata_encoder({key: value.to(device) for key, value in attributes.items()})


def _factor_gram(layer, condition: torch.Tensor) -> torch.Tensor:
    """1 層の ΔW の Gram 行列 `[n, n]` を低ランク因子から計算する。"""
    a, b = layer._factors(condition)
    a = a.double()
    b = b.flatten(2).double()
    left = torch.einsum("ior,jos->ijrs", a, a)
    right = torch.einsum("ird,jsd->ijrs", b, b)
    return layer.scale**2 * (left * right).sum((2, 3))


@torch.inference_mode()
def delta_grams(network, condition: torch.Tensor) -> dict[str, np.ndarray]:
    """stage ごとに、元重みで規格化した ΔW の Gram 行列を返す。

    層ごとに ‖W_base‖² で割ってから足す。層の大きさの違いで stage4 の 1 層が全体を支配しないように、
    各層の ΔW を「元重みに対する相対的な変化」として同じ重みで数える。

    Returns:
        dict: `stage3`・`stage4`・`fc` → `[n, n]`。
    """
    grams = {stage: 0.0 for stage in STAGES}
    for name, layer, base_weight in adapter_layers(network):
        stage = name.split(".")[0]
        base_sq = (base_weight.double() ** 2).sum()
        grams[stage] = grams[stage] + _factor_gram(layer, condition) / base_sq
    return {stage: gram.cpu().numpy() for stage, gram in grams.items()}


@torch.inference_mode()
def stage2_features(network, datamodule, frame: pd.DataFrame, device: torch.device) -> torch.Tensor:
    """condition を使う前（stage2 の出力）までの特徴 `[n, C, H, W]` を fp16 で device に返す。"""
    # 逐次で回すので persistent worker は持たない（並列 probe の OOM を避ける）。
    loader = DataLoader(
        datamodule._create_dataset(frame, datamodule.val_transform),
        batch_size=datamodule.batch_size_per_device,
        num_workers=8,
        shuffle=False,
    )
    backbone = network.backbone
    chunks = []
    for image, _, _ in loader:
        x = backbone.maxpool(backbone.relu(backbone.bn1(backbone.conv1(image.to(device)))))
        chunks.append(backbone.layer2(backbone.layer1(x)).half())
    return torch.cat(chunks)


@torch.inference_mode()
def logit_margin(network, features: torch.Tensor, condition: torch.Tensor, batch: int = 128) -> np.ndarray:
    """stage2 の特徴に、行ごとの condition `[n, d]` を当てた logit 差（悪性 − 良性）を返す。"""
    margins = []
    for start in range(0, len(features), batch):
        x = features[start : start + batch].float()
        c = condition[start : start + batch]
        for stage, layer_name in (("stage3", "layer3"), ("stage4", "layer4")):
            x = network._forward_stage(stage, getattr(network.backbone, layer_name), x, c)
        logits = network.fc(torch.flatten(network.backbone.avgpool(x), 1), c)
        margins.append((logits[:, 1] - logits[:, 0]).float().cpu().numpy())
    return np.concatenate(margins)


def _save(path: Path, metadata: dict, **arrays) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, metadata=json.dumps(metadata, sort_keys=True), **arrays)
    print(f"wrote: {path.relative_to(PACKAGE)}")


def write_cells(run_dir: Path, network, cells: pd.DataFrame, condition: torch.Tensor, metadata: dict) -> None:
    """セル表・condition・stage ごとの Gram を書く。"""
    grams = delta_grams(network, condition)
    columns = {f"cell_{key}": cells[key].to_numpy() for key in ("sex", "localization", "raw_age", "count", "positives")}
    _save(
        PACKAGE / "cache" / f"{run_dir.name}_cells.npz",
        metadata,
        condition=condition.cpu().numpy(),
        **columns,
        **{f"gram_{stage}": gram for stage, gram in grams.items()},
    )


def write_swap(run_dir: Path, network, datamodule, attribute_names, cells, condition, split, device, metadata):
    """評価行の全員を各差し替え先セルの condition で推論し直した logit 差を書く。"""
    frame = model_frame(datamodule, split)
    frame = frame[known_rows(frame)].reset_index(drop=True)
    features = stage2_features(network, datamodule, frame, device)
    own = logit_margin(network, features, encode(network, frame, attribute_names, device))
    targets = np.flatnonzero(cells["count"].to_numpy() >= MIN_TARGET_COUNT)
    swapped = np.empty((len(targets), len(frame)), dtype=np.float32)
    for index, target in enumerate(targets):
        swapped[index] = logit_margin(network, features, condition[target].expand(len(frame), -1))
    _save(
        PACKAGE / "cache" / f"{run_dir.name}_swap_{split}.npz",
        {**metadata, "split": split},
        image=frame["image"].to_numpy(dtype=str),
        target=frame["target"].to_numpy(dtype=np.int64),
        lesion_id=frame["lesion_id"].to_numpy(dtype=str),
        dx=frame["dx"].to_numpy(dtype=np.int64),
        sex=frame["sex"].to_numpy(),
        localization=frame["localization"].to_numpy(),
        raw_age=frame["raw_age"].to_numpy(),
        own=own,
        target_cells=targets,
        swapped=swapped,
    )


def write(run_dir: Path, device: torch.device, splits: tuple[str, ...]) -> None:
    """1 run の Gram と差し替え推論を cache に書く。"""
    datamodule, attribute_names = load_datamodule(run_dir)
    network = load_network(run_dir, device)
    metadata = {"schema_version": SCHEMA_VERSION, "run_id": run_dir.name, "checkpoint": checkpoint_reference(run_dir)}
    cells = cell_table(model_frame(datamodule, "train"))
    condition = encode(network, cells, attribute_names, device)
    write_cells(run_dir, network, cells, condition, metadata)
    for split in splits:
        write_swap(run_dir, network, datamodule, attribute_names, cells, condition, split, device, metadata)


def main() -> None:
    """CLI 入口。"""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run-dir", action="append", required=True, type=Path, dest="run_dirs")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--splits", nargs="+", default=["val", "test"])
    args = parser.parse_args()
    torch.backends.cudnn.benchmark = True
    for run_dir in args.run_dirs:
        write(run_dir.resolve(), torch.device(args.device), tuple(args.splits))


if __name__ == "__main__":
    main()
