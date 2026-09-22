"""保存済み checkpoint の予測を分析用 cache に書き出す。

使い方:
    uv run python analysis/hypernet_e2e/initial-resnet-vs-invariant/cache_predictions.py
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

REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

RUNS_ROOT = REPOSITORY_ROOT / "projects/hypernet_e2e/runs"
CACHE_DIR = Path(__file__).parent / "cache"
DEFAULT_RUN_IDS = (
    "20260921T103036Z-resnet-chexpert-s42-5538",
    "20260921T125711Z-resnet-chexpert-attribute-invariant-s42-9fd3",
)
DEFAULT_SPLITS = ("test",)


def cache_path(cache_dir: Path, run_id: str, split: str) -> Path:
    """run と評価 split に対応する cache path を返す。

    Args:
        cache_dir: cache を置く directory。
        run_id: `projects/hypernet_e2e/runs/` 配下の run ID。
        split: 評価する split 名。

    Returns:
        Path: `<cache_dir>/<run_id>_<split>.npz`。
    """
    return cache_dir / f"{run_id}_{split}.npz"


def best_checkpoint(run_dir: Path) -> Path:
    """run が記録した `best_val_auroc` checkpoint を1つ取得する。

    Args:
        run_dir: 対象 run directory。

    Returns:
        Path: AUROC による最良 checkpoint。

    Raises:
        FileNotFoundError: best checkpoint がちょうど1つでない場合。
    """
    checkpoints = sorted((run_dir / "checkpoints").glob("best_val_auroc_*.ckpt"))
    if len(checkpoints) != 1:
        raise FileNotFoundError(f"{run_dir} の best_val_auroc checkpoint は1つ必要だが、{len(checkpoints)} 件見つかった")
    return checkpoints[0]


def _load_network(run_dir: Path, device: torch.device) -> torch.nn.Module:
    """分類 network だけを checkpoint から復元する。

    属性不変 loss は学習時だけ使い、分類予測は network の logits だけで決まる。そのため loss を
    復元せず、checkpoint の `net.` prefix を持つ重みだけを読む。

    Args:
        run_dir: config と checkpoint を持つ実行 directory。
        device: 推論に使う PyTorch device。

    Returns:
        torch.nn.Module: eval mode で device に置いた分類 network。
    """
    config = OmegaConf.load(run_dir / "config.yaml")
    network = instantiate(config.model.net)
    checkpoint = torch.load(best_checkpoint(run_dir), map_location="cpu", weights_only=False)
    state_dict = checkpoint.get("state_dict")
    if not isinstance(state_dict, Mapping):
        raise TypeError(f"{run_dir} の checkpoint に state_dict がない")
    network_state = {key.removeprefix("net."): value for key, value in state_dict.items() if key.startswith("net.")}
    network.load_state_dict(network_state, strict=True)
    return network.to(device).eval()


def cache_predictions(run_dir: Path, split: str, cache_dir: Path, device: torch.device) -> Path:
    """checkpoint を指定 split に評価し、画像順を保った予測 cache を生成する。

    Args:
        run_dir: config と checkpoint を持つ実行 directory。
        split: `val` または `test` など DataModule が扱う split 名。
        cache_dir: `.npz` を保存する directory。
        device: 推論に使う PyTorch device。

    Returns:
        Path: 書き出した prediction cache。
    """
    config = OmegaConf.load(run_dir / "config.yaml")
    datamodule = instantiate(config.data)
    loader = datamodule.evaluation_dataloader(split)
    frame = loader.dataset.df
    network = _load_network(run_dir, device)

    logits_parts: list[np.ndarray] = []
    target_parts: list[np.ndarray] = []
    with torch.inference_mode():
        for image, _, target in loader:
            logits = network(image.to(device, non_blocking=True))
            logits_parts.append(logits.float().cpu().numpy())
            target_parts.append(target.numpy())

    logits = np.concatenate(logits_parts)
    target = np.concatenate(target_parts).astype(np.int64, copy=False)

    probabilities = torch.softmax(torch.from_numpy(logits), dim=1).numpy()
    output = cache_path(cache_dir, run_dir.name, split)
    cache_dir.mkdir(parents=True, exist_ok=True)
    metadata = {
        "schema_version": 1,
        "run_id": run_dir.name,
        "split": split,
        "checkpoint": str(best_checkpoint(run_dir)),
        "num_examples": len(target),
        "num_classes": int(probabilities.shape[1]),
    }
    np.savez_compressed(
        output,
        image=frame["image"].to_numpy(dtype=str),
        logits=logits,
        probabilities=probabilities,
        predictions=probabilities.argmax(axis=1).astype(np.int64),
        target=target,
        metadata=json.dumps(metadata, ensure_ascii=False, sort_keys=True),
    )
    return output


def main() -> None:
    """CLI 引数の全 run/split について、存在しない予測 cache だけを生成する。

    Args:
        なし。

    Returns:
        None。
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", action="append", dest="run_ids", help="対象 run ID。複数回指定できる")
    parser.add_argument("--split", action="append", dest="splits", choices=("val", "test"), help="評価 split。複数回指定できる")
    parser.add_argument("--runs-root", type=Path, default=RUNS_ROOT)
    parser.add_argument("--cache-dir", type=Path, default=CACHE_DIR)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    device = torch.device(args.device)
    for run_id in args.run_ids or DEFAULT_RUN_IDS:
        for split in args.splits or DEFAULT_SPLITS:
            output = cache_path(args.cache_dir, run_id, split)
            if not args.overwrite and output.is_file():
                print(f"cache exists: {output}")
                continue
            written = cache_predictions(args.runs_root / run_id, split, args.cache_dir, device)
            print(f"cached: {written}")


if __name__ == "__main__":
    main()
