"""選択済み checkpoint の予測を、分析用の `.npz` cache に書き出す。

群ごとの公平性指標は run artifact に無い。`metrics.csv` が持つのは属性ごとの worst / gap
までで、群そのものの性能も交差群も残っていない。予測を一度 cache して、**群の切り方は
分析側で決める**。

cache の i 番目は split CSV の i 行目に対応する（`evaluation_dataloader` が CSV を読み直して
固定順で返す）。分析側はこの対応を前提に属性を突き合わせるので、`load_cache` は image 列で
その対応を検算する。

どの project の run も `config.yaml` に datamodule と network の定義を持つので、分岐は
checkpoint の選び方（`run_artifacts.selected_checkpoint`）と、network が属性を取るかどうか
だけになる。

使い方:
    uv run python analysis/common/predictions.py --study iterative_probe --split test \
      --run-dir projects/hypernet_e2e/runs/<run-id> \
      --run-dir projects/hypernet_iterative/runs/<run-id>
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
import rootutils
import torch
from hydra.utils import instantiate
from omegaconf import OmegaConf

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from analysis.common.paths import REPOSITORY_ROOT, local_data_path, study_dir  # noqa: E402
from analysis.common.run_artifacts import read_config, selected_checkpoint  # noqa: E402

SCHEMA_VERSION = 1


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


def read_metadata(path: Path) -> dict[str, Any]:
    """cache の metadata だけを読む。

    Args:
        path: `cache_path` が返す `.npz`

    Returns:
        dict[str, Any]: 書き出し時の `metadata`
    """
    with np.load(path, allow_pickle=False) as cached:
        return json.loads(str(cached["metadata"]))


def checkpoint_reference(run_dir: Path) -> str:
    """cache に記録する checkpoint を、repo root からの相対 path で返す。

    書く側と照合する側で同じ文字列になる必要があるので 1 箇所で作る。呼ぶ側が相対 path の
    run directory を渡しても同じ値になるよう、ここで絶対 path に直す。

    Args:
        run_dir: config と checkpoint を持つ run directory

    Returns:
        str: `projects/<project>/runs/<run-id>/.../<name>.ckpt`
    """
    return str(selected_checkpoint(run_dir.resolve()).relative_to(REPOSITORY_ROOT))


def stale_reason(path: Path, run_dir: Path) -> str | None:
    """既存 cache を作り直す理由を返す。作り直す必要が無ければ `None`。

    cache は run artifact の派生物であり、checkpoint を選び直せば中身が変わる。存在する
    ことだけを見て使い回すと、**古い予測から出た表と図が黙って残る**。

    Args:
        path: `cache_path` が返す `.npz`
        run_dir: その cache の元になる run directory

    Returns:
        str | None: 作り直す理由。使い回してよければ `None`
    """
    if not path.is_file():
        return "cache が無い"
    try:
        metadata = read_metadata(path)
    except (OSError, ValueError, KeyError):
        return "cache を読めない"
    if metadata.get("schema_version") != SCHEMA_VERSION:
        return f"schema_version が {metadata.get('schema_version')}（現在は {SCHEMA_VERSION}）"
    checkpoint = checkpoint_reference(run_dir)
    if metadata.get("checkpoint") != checkpoint:
        return f"checkpoint が {metadata.get('checkpoint')} から {checkpoint} へ変わった"
    return None


def load_cache(path: Path, images: np.ndarray) -> dict[str, Any]:
    """予測 cache を読み、split CSV との行対応を確かめる。

    行対応の検算を任意にしない。cache の i 番目と split CSV の i 行目がずれていても
    数値は出てしまい、群の割り当てが全部ずれた表が「それらしく」並ぶ。

    Args:
        path: `cache_path` が返す `.npz`
        images: 突き合わせる split CSV の image 列

    Returns:
        dict[str, Any]: `image` / `logits` / `probabilities` / `predictions` / `target` と
            parse 済みの `metadata`

    Raises:
        ValueError: cache の schema が古い場合、または split CSV と行が対応していない場合
    """
    with np.load(path, allow_pickle=False) as cached:
        result = {key: cached[key] for key in cached.files}
    result["metadata"] = json.loads(str(result["metadata"]))
    version = result["metadata"].get("schema_version")
    if version != SCHEMA_VERSION:
        raise ValueError(f"{path} の schema_version は {version}（現在は {SCHEMA_VERSION}）。作り直す")
    if len(result["image"]) != len(images) or not (result["image"] == images).all():
        raise ValueError(f"{path} の行が split CSV と対応していない（群の割り当てが全部ずれる）")
    return result


def load_network(run_dir: Path, device: torch.device) -> torch.nn.Module:
    """分類 network だけを checkpoint から復元する。

    予測は network の logits だけで決まるので、loss も training objective も復元しない
    （属性不変 loss や GroupDRO は学習時だけのもの）。checkpoint の `net.` prefix を持つ
    重みだけを読む。

    この `net.` prefix と `model.use_attributes` は run 契約ではなく、LightningModule の
    構造そのものへの依存になる。projects 側が module の属性名を変えれば、ここは
    `load_state_dict(strict=True)` で落ちる（黙ってずれた重みを読むことはない）。

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
    weights = {key.removeprefix("net."): value for key, value in state_dict.items() if key.startswith("net.")}
    network.load_state_dict(weights, strict=True)
    return network.to(device).eval()


def write_cache(run_dir: Path, split: str, cache_dir: Path, device: torch.device) -> Path:
    """指定 split を推論し、CSV 行順を保った予測 cache を書き出す。

    i 番目が split CSV の i 行目に対応する、という cache の約束はここで作る。行の identity は
    `evaluation_dataloader` と同じ frame（`standardized_dataframes`）から取り、順序が保たれる
    ことは projects 側の `test_evaluation_loader_keeps_the_split_csv_row_order` が固定する。

    Args:
        run_dir: config と checkpoint を持つ run directory
        split: `val` または `test`
        cache_dir: `.npz` を置く directory
        device: 推論に使う device

    Returns:
        Path: 書き出した cache
    """
    config = OmegaConf.load(run_dir / "config.yaml")
    # container で回した run は `/workspaces/...` を記録している。分析は local の `data/` を読む。
    for key in ("cv_splits_dir", "data_dir"):
        if key in config.data:
            config.data[key] = str(local_data_path(config.data[key]))
    datamodule = instantiate(config.data)
    loader = datamodule.evaluation_dataloader(split)
    network = load_network(run_dir, device)
    # 画像だけを取る network と、metadata で変調する network がある。
    use_attributes = bool(config.model.get("use_attributes", False))

    logits_parts: list[np.ndarray] = []
    target_parts: list[np.ndarray] = []
    with torch.inference_mode():
        for image, attributes, target in loader:
            image = image.to(device, non_blocking=True)
            if use_attributes:
                on_device = {key: value.to(device, non_blocking=True) for key, value in attributes.items()}
                logits = network(image, on_device)
            else:
                logits = network(image)
            logits_parts.append(logits.float().cpu().numpy())
            target_parts.append(target.numpy())

    logits = np.concatenate(logits_parts)
    target = np.concatenate(target_parts).astype(np.int64, copy=False)
    probabilities = torch.softmax(torch.from_numpy(logits), dim=1).numpy()

    # 行の identity は datamodule の公開 method から取る。`evaluation_dataloader` 自身が
    # `standardized_dataframes(split)` の frame から作られるので、同じ行・同じ順序になる。
    (frame,) = datamodule.standardized_dataframes(split)
    images = frame["image"].to_numpy(dtype=str)

    cache_dir.mkdir(parents=True, exist_ok=True)
    output = cache_path(cache_dir, run_dir.name, split)
    metadata = {
        "schema_version": SCHEMA_VERSION,
        "run_id": run_dir.name,
        "project": run_dir.parents[1].name,
        "study": read_config(run_dir / "config.yaml").get("study"),
        "split": split,
        "checkpoint": checkpoint_reference(run_dir),
        "num_examples": len(target),
        "num_classes": int(probabilities.shape[1]),
    }
    np.savez_compressed(
        output,
        image=images,
        logits=logits,
        probabilities=probabilities,
        predictions=probabilities.argmax(axis=1).astype(np.int64),
        target=target,
        metadata=json.dumps(metadata, ensure_ascii=False, sort_keys=True),
    )
    return output


def main() -> None:
    """指定した run と split のうち、古くなった cache だけを作り直す。

    Args:
        なし

    Returns:
        None
    """
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--study", required=True, help="cache の置き場を決める analysis/<study>")
    run_help = "repo root からの run directory。複数回指定できる"
    parser.add_argument("--run-dir", action="append", dest="run_dirs", required=True, help=run_help)
    parser.add_argument("--split", default="test", choices=("val", "test"))
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--overwrite", action="store_true", help="古くなくても作り直す")
    args = parser.parse_args()

    cache_dir = study_dir(args.study) / "cache"
    device = torch.device(args.device)
    for relative in args.run_dirs:
        run_dir = REPOSITORY_ROOT / relative
        output = cache_path(cache_dir, run_dir.name, args.split)
        reason = stale_reason(output, run_dir)
        if reason is None and not args.overwrite:
            print(f"cache is current: {output}")
            continue
        print(f"rebuilding ({reason or '--overwrite'}): {output}")
        print(f"cached: {write_cache(run_dir, args.split, cache_dir, device)}")


if __name__ == "__main__":
    main()
