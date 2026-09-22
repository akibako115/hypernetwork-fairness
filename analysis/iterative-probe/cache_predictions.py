r"""選択済み checkpoint の予測を分析用 cache に書き出す。

group ごとの公平性指標は run artifact に無い。`metrics.csv` が持つのは属性ごとの worst / gap まで
であり、群そのものの性能も交差群も残っていない。予測を一度 cache して、群の切り方は分析側で
決める。

iterative の run と `hypernet_e2e` の baseline を同じ形の `.npz` にする。どちらの run も
`config.yaml` に datamodule と network の定義を持つので、project ごとの分岐は checkpoint の
選び方と、network が属性を取るかどうかだけになる。

使い方:
    uv run python analysis/iterative-probe/cache_predictions.py --split test
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Mapping
from pathlib import Path

import numpy as np
import torch
from hydra.utils import instantiate
from omegaconf import OmegaConf

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

CACHE_DIR = Path(__file__).parent / "cache"
DEFAULT_RUN_DIRS = (
    "projects/hypernet_iterative/runs/20260921T103222Z-iterative-s42-d24d",
    "projects/hypernet_iterative/runs/20260921T103225Z-iterative-s42-3166",
    "projects/hypernet_iterative/runs/20260921T103219Z-iterative-s42-e992",
    "projects/hypernet_iterative/runs/20260921T103225Z-iterative-s42-fac5",
    "projects/hypernet_e2e/runs/20260921T103036Z-resnet-chexpert-s42-5538",
)


def selected_checkpoint(run_dir: Path) -> Path:
    """その run が「これが成果物」と記録した checkpoint を返す。

    iterative は `run.json` の `selected_checkpoint` が最終 stage の best global AUROC を指す。
    記録された path は実行環境（container）のものなので、run directory からの相対位置だけを使う。
    baseline は stage を持たないので `checkpoints/best_val_auroc_*.ckpt` を取る。どちらも
    val AUROC で選んだ checkpoint であり、選び方は揃っている。

    Args:
        run_dir: `projects/<project>/runs/<run-id>`

    Returns:
        Path: 評価する checkpoint

    Raises:
        FileNotFoundError: checkpoint がちょうど 1 つに定まらない場合
    """
    record = json.loads((run_dir / "run.json").read_text())
    if "selected_checkpoint" in record:
        parts = Path(record["selected_checkpoint"]["path"]).parts
        return run_dir.joinpath(*parts[parts.index(run_dir.name) + 1 :])
    checkpoints = sorted((run_dir / "checkpoints").glob("best_val_auroc_*.ckpt"))
    if len(checkpoints) != 1:
        raise FileNotFoundError(f"{run_dir} の best_val_auroc checkpoint は 1 つ必要だが、{len(checkpoints)} 件見つかった")
    return checkpoints[0]


def load_network(run_dir: Path, device: torch.device) -> torch.nn.Module:
    """分類 network だけを checkpoint から復元する。

    予測は network の logits だけで決まるので、loss や training objective は復元しない。
    checkpoint の `net.` prefix を持つ重みだけを読む。

    Args:
        run_dir: config と checkpoint を持つ run directory
        device: 推論に使う device

    Returns:
        torch.nn.Module: eval mode で device に置いた network

    Raises:
        TypeError: checkpoint が `state_dict` を持たない場合
    """
    config = OmegaConf.load(run_dir / "config.yaml")
    network = instantiate(config.model.net)
    checkpoint = torch.load(selected_checkpoint(run_dir), map_location="cpu", weights_only=False)
    state_dict = checkpoint.get("state_dict")
    if not isinstance(state_dict, Mapping):
        raise TypeError(f"{run_dir} の checkpoint に state_dict がない")
    network.load_state_dict({key.removeprefix("net."): value for key, value in state_dict.items() if key.startswith("net.")}, strict=True)
    return network.to(device).eval()


def cache_predictions(run_dir: Path, split: str, cache_dir: Path, device: torch.device) -> Path:
    """指定 split を推論し、CSV 行順を保った予測 cache を書き出す。

    `evaluation_dataloader` は split CSV を読み直して固定順で返すので、cache の i 番目は
    split CSV の i 行目に対応する。分析側は split CSV を読み直して属性を突き合わせる。

    Args:
        run_dir: config と checkpoint を持つ run directory
        split: `val` または `test`
        cache_dir: `.npz` を置く directory
        device: 推論に使う device

    Returns:
        Path: 書き出した cache
    """
    config = OmegaConf.load(run_dir / "config.yaml")
    loader = instantiate(config.data).evaluation_dataloader(split)
    network = load_network(run_dir, device)
    # baseline は画像だけ、iterative は metadata で変調するので属性も渡す。
    use_attributes = bool(config.model.get("use_attributes", False))

    logits_parts: list[np.ndarray] = []
    target_parts: list[np.ndarray] = []
    with torch.inference_mode():
        for image, attributes, target in loader:
            image = image.to(device, non_blocking=True)
            if use_attributes:
                logits = network(image, {key: value.to(device, non_blocking=True) for key, value in attributes.items()})
            else:
                logits = network(image)
            logits_parts.append(logits.float().cpu().numpy())
            target_parts.append(target.numpy())

    logits = np.concatenate(logits_parts)
    target = np.concatenate(target_parts).astype(np.int64, copy=False)
    probabilities = torch.softmax(torch.from_numpy(logits), dim=1).numpy()

    cache_dir.mkdir(parents=True, exist_ok=True)
    output = cache_path(cache_dir, run_dir.name, split)
    metadata = {
        "schema_version": 1,
        "run_id": run_dir.name,
        "project": run_dir.parents[1].name,
        "split": split,
        "checkpoint": str(selected_checkpoint(run_dir).relative_to(REPOSITORY_ROOT)),
        "num_examples": len(target),
    }
    np.savez_compressed(
        output,
        image=loader.dataset.df["image"].to_numpy(dtype=str),
        logits=logits,
        probabilities=probabilities,
        predictions=probabilities.argmax(axis=1).astype(np.int64),
        target=target,
        metadata=json.dumps(metadata, ensure_ascii=False, sort_keys=True),
    )
    return output


def cache_path(cache_dir: Path, run_id: str, split: str) -> Path:
    """run と split に対応する cache path を返す。

    Args:
        cache_dir: cache を置く directory
        run_id: run ID
        split: 評価した split 名

    Returns:
        Path: `<cache_dir>/<run_id>_<split>.npz`
    """
    return cache_dir / f"{run_id}_{split}.npz"


def main() -> None:
    """指定した run と split のうち、まだ無い cache だけを作る。

    Args:
        なし

    Returns:
        None
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", action="append", dest="run_dirs", help="repo root からの run directory。複数回指定できる")
    parser.add_argument("--split", default="test", choices=("val", "test"))
    parser.add_argument("--cache-dir", type=Path, default=CACHE_DIR)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    device = torch.device(args.device)
    for run_dir in args.run_dirs or DEFAULT_RUN_DIRS:
        path = REPOSITORY_ROOT / run_dir
        output = cache_path(args.cache_dir, path.name, args.split)
        if output.is_file() and not args.overwrite:
            print(f"cache exists: {output}")
            continue
        print(f"cached: {cache_predictions(path, args.split, args.cache_dir, device)}")


if __name__ == "__main__":
    main()
