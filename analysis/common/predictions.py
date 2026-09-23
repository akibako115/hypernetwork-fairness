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

`--features` を付けると、同じ forward から backbone 表現（分類 head の直前）も
**別 file** へ書く。属性 probe のように「予測ではなく表現に何が残っているか」を見る分析が
これを読む。

使い方:
    uv run python analysis/common/predictions.py --study iterative_probe --split test \
      --run-dir projects/hypernet_e2e/runs/<run-id> \
      --run-dir projects/hypernet_iterative/runs/<run-id>

    uv run python analysis/common/predictions.py --study initial_resnet_vs_invariant \
      --split val --features --run-dir projects/hypernet_e2e/runs/<run-id>
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
# 特徴量は予測 cache と**別 file** に持つ。1 run 1 split で 90MB あり、読むのは属性 probe
# だけになる。同じ file へ足すと、特徴量を読まない package の cache まで 200 倍になり、
# 片方の schema を上げるたびに全 package の推論をやり直すことになる。
FEATURES_SCHEMA_VERSION = 1


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


def features_cache_path(cache_dir: Path, run_id: str, split: str) -> Path:
    """run と split に対応する特徴量 cache path を返す。

    Args:
        cache_dir: cache を置く directory
        run_id: run ID
        split: 評価した split 名

    Returns:
        Path: `<cache_dir>/<run_id>_<split>_features.npz`
    """
    return cache_dir / f"{run_id}_{split}_features.npz"


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


def stale_reason(path: Path, run_dir: Path, schema_version: int = SCHEMA_VERSION) -> str | None:
    """既存 cache を作り直す理由を返す。作り直す必要が無ければ `None`。

    cache は run artifact の派生物であり、checkpoint を選び直せば中身が変わる。存在する
    ことだけを見て使い回すと、**古い予測から出た表と図が黙って残る**。

    Args:
        path: `cache_path` または `features_cache_path` が返す `.npz`
        run_dir: その cache の元になる run directory
        schema_version: その cache が従うべき schema version

    Returns:
        str | None: 作り直す理由。使い回してよければ `None`
    """
    if not path.is_file():
        return "cache が無い"
    try:
        metadata = read_metadata(path)
    except (OSError, ValueError, KeyError):
        return "cache を読めない"
    if metadata.get("schema_version") != schema_version:
        return f"schema_version が {metadata.get('schema_version')}（現在は {schema_version}）"
    checkpoint = checkpoint_reference(run_dir)
    if metadata.get("checkpoint") != checkpoint:
        return f"checkpoint が {metadata.get('checkpoint')} から {checkpoint} へ変わった"
    return None


def _load_npz(path: Path, images: np.ndarray, schema_version: int) -> dict[str, Any]:
    """cache を読み、schema version と split CSV との行対応を確かめる。

    行対応の検算を任意にしない。cache の i 番目と split CSV の i 行目がずれていても
    数値は出てしまい、群の割り当てが全部ずれた表が「それらしく」並ぶ。

    Args:
        path: 読む `.npz`
        images: 突き合わせる split CSV の image 列
        schema_version: その cache が従うべき schema version

    Returns:
        dict[str, Any]: `.npz` の全 array と parse 済みの `metadata`

    Raises:
        ValueError: cache の schema が古い場合、または split CSV と行が対応していない場合
    """
    with np.load(path, allow_pickle=False) as cached:
        result = {key: cached[key] for key in cached.files}
    result["metadata"] = json.loads(str(result["metadata"]))
    version = result["metadata"].get("schema_version")
    if version != schema_version:
        raise ValueError(f"{path} の schema_version は {version}（現在は {schema_version}）。作り直す")
    if len(result["image"]) != len(images) or not (result["image"] == images).all():
        raise ValueError(f"{path} の行が split CSV と対応していない（群の割り当てが全部ずれる）")
    return result


def load_cache(path: Path, images: np.ndarray) -> dict[str, Any]:
    """予測 cache を読み、split CSV との行対応を確かめる。

    Args:
        path: `cache_path` が返す `.npz`
        images: 突き合わせる split CSV の image 列

    Returns:
        dict[str, Any]: `image` / `logits` / `probabilities` / `predictions` / `target` と
            parse 済みの `metadata`

    Raises:
        ValueError: cache の schema が古い場合、または split CSV と行が対応していない場合
    """
    return _load_npz(path, images, SCHEMA_VERSION)


def load_features(path: Path, images: np.ndarray) -> dict[str, Any]:
    """特徴量 cache を読み、split CSV との行対応を確かめる。

    `features` は float16 で保存してある（1 run 1 split で 90MB あり、float32 だと倍になる）。
    probe は標準化してから学習するので、この丸めは probe の精度には効かない。**読む側で
    float32 に戻す**のは、`sklearn` が float16 をそのまま受け取ると内部で都度変換するため。

    Args:
        path: `features_cache_path` が返す `.npz`
        images: 突き合わせる split CSV の image 列

    Returns:
        dict[str, Any]: `image` / float32 に戻した `features` と parse 済みの `metadata`

    Raises:
        ValueError: cache の schema が古い場合、または split CSV と行が対応していない場合
    """
    result = _load_npz(path, images, FEATURES_SCHEMA_VERSION)
    result["features"] = result["features"].astype(np.float32, copy=False)
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


def _forward_split(
    network: torch.nn.Module,
    loader: Any,
    device: torch.device,
    use_attributes: bool,
    with_features: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    """loader を 1 周して logits・target と、必要なら backbone 表現を集める。

    表現は logits と**同じ forward** から取る。別 forward で取り直すと、推論が 2 倍かかる
    うえに、dropout や BatchNorm の状態がずれた表現を予測と並べて読むことになる。

    Args:
        network: eval mode の network
        loader: `evaluation_dataloader` が返す DataLoader
        device: 推論に使う device
        use_attributes: network が属性を受け取るか
        with_features: 分類 head 直前の表現も集めるか

    Returns:
        tuple[np.ndarray, np.ndarray, np.ndarray | None]: logits・target と、
            `with_features` のときだけ float16 の `[n_rows, feature_dim]`

    Raises:
        TypeError: `with_features` なのに network が `forward_with_features` を持たない場合
    """
    if with_features and not hasattr(network, "forward_with_features"):
        raise TypeError(f"{type(network).__name__} は forward_with_features を実装していない（表現を取れない）")

    logits_parts: list[np.ndarray] = []
    target_parts: list[np.ndarray] = []
    feature_parts: list[np.ndarray] = []
    with torch.inference_mode():
        for image, attributes, target in loader:
            image = image.to(device, non_blocking=True)
            # 画像だけを取る network と、metadata で変調する network がある。
            if use_attributes:
                on_device = {key: value.to(device, non_blocking=True) for key, value in attributes.items()}
                inputs: tuple[Any, ...] = (image, on_device)
            else:
                inputs = (image,)
            if with_features:
                logits, features = network.forward_with_features(*inputs)
                # float16 で積む。probe は標準化してから学習するので丸めは効かず、cache は半分になる。
                feature_parts.append(features.to(torch.float16).cpu().numpy())
            else:
                logits = network(*inputs)
            logits_parts.append(logits.float().cpu().numpy())
            target_parts.append(target.numpy())

    target = np.concatenate(target_parts).astype(np.int64, copy=False)
    features = np.concatenate(feature_parts) if with_features else None
    return np.concatenate(logits_parts), target, features


def write_cache(
    run_dir: Path,
    split: str,
    cache_dir: Path,
    device: torch.device,
    with_features: bool = False,
) -> list[Path]:
    """指定 split を推論し、CSV 行順を保った予測 cache（必要なら特徴量 cache も）を書き出す。

    i 番目が split CSV の i 行目に対応する、という cache の約束はここで作る。行の identity は
    `evaluation_dataloader` と同じ frame（`standardized_dataframes`）から取り、順序が保たれる
    ことは projects 側の `test_evaluation_loader_keeps_the_split_csv_row_order` が固定する。

    Args:
        run_dir: config と checkpoint を持つ run directory
        split: `train` / `val` / `test`。`train` も評価 transform・CSV 順で読む
            （`evaluation_dataloader` が split を読み直すため、train sampling も augmentation も効かない）
        cache_dir: `.npz` を置く directory
        device: 推論に使う device
        with_features: 分類 head 直前の表現も別 file へ書くか

    Returns:
        list[Path]: 書き出した cache。`with_features` のときは予測・特徴量の 2 つ
    """
    config = OmegaConf.load(run_dir / "config.yaml")
    # container で回した run は `/workspaces/...` を記録している。分析は local の `data/` を読む。
    for key in ("cv_splits_dir", "data_dir"):
        if key in config.data:
            config.data[key] = str(local_data_path(config.data[key]))
    datamodule = instantiate(config.data)
    loader = datamodule.evaluation_dataloader(split)
    network = load_network(run_dir, device)
    use_attributes = bool(config.model.get("use_attributes", False))
    logits, target, features = _forward_split(network, loader, device, use_attributes, with_features)
    probabilities = torch.softmax(torch.from_numpy(logits), dim=1).numpy()

    # 行の identity は datamodule の公開 method から取る。`evaluation_dataloader` 自身が
    # `standardized_dataframes(split)` の frame から作られるので、同じ行・同じ順序になる。
    (frame,) = datamodule.standardized_dataframes(split)
    images = frame["image"].to_numpy(dtype=str)

    cache_dir.mkdir(parents=True, exist_ok=True)
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
    output = cache_path(cache_dir, run_dir.name, split)
    np.savez_compressed(
        output,
        image=images,
        logits=logits,
        probabilities=probabilities,
        predictions=probabilities.argmax(axis=1).astype(np.int64),
        target=target,
        metadata=json.dumps(metadata, ensure_ascii=False, sort_keys=True),
    )
    if features is None:
        return [output]
    return [output, _write_features(cache_dir, run_dir, split, images, features, metadata)]


def _write_features(
    cache_dir: Path,
    run_dir: Path,
    split: str,
    images: np.ndarray,
    features: np.ndarray,
    metadata: dict[str, Any],
) -> Path:
    """backbone 表現を、予測 cache と同じ行順で別 file へ書き出す。

    Args:
        cache_dir: `.npz` を置く directory
        run_dir: その表現の元になる run directory
        split: 評価した split
        images: split CSV の image 列（行の identity）
        features: `[n_rows, feature_dim]` の float16 表現
        metadata: 予測 cache の metadata。schema version と表現固有の項目だけ差し替える

    Returns:
        Path: 書き出した特徴量 cache
    """
    output = features_cache_path(cache_dir, run_dir.name, split)
    record = {
        **metadata,
        "schema_version": FEATURES_SCHEMA_VERSION,
        "feature_dim": int(features.shape[1]),
        "feature_dtype": str(features.dtype),
    }
    record.pop("num_classes", None)
    np.savez_compressed(
        output,
        image=images,
        features=features,
        metadata=json.dumps(record, ensure_ascii=False, sort_keys=True),
    )
    return output


def rebuild_reason(cache_dir: Path, run_dir: Path, split: str, with_features: bool) -> str | None:
    """その run・split の cache を作り直す理由を返す。作り直す必要が無ければ `None`。

    特徴量まで要求されているときは、予測 cache が新しくても**特徴量が無ければ作り直す**。
    推論は 1 回で両方を書くので、片方だけを継ぎ足す経路は持たない。

    Args:
        cache_dir: cache を置く directory
        run_dir: config と checkpoint を持つ run directory
        split: `val` または `test`
        with_features: 特徴量 cache も要求されているか

    Returns:
        str | None: 作り直す理由。両方とも使い回してよければ `None`
    """
    reason = stale_reason(cache_path(cache_dir, run_dir.name, split), run_dir)
    if reason is not None:
        return reason
    if not with_features:
        return None
    features = features_cache_path(cache_dir, run_dir.name, split)
    features_reason = stale_reason(features, run_dir, FEATURES_SCHEMA_VERSION)
    return None if features_reason is None else f"特徴量 {features_reason}"


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
    parser.add_argument("--split", default="test", choices=("train", "val", "test"))
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--features", action="store_true", help="分類 head 直前の表現も別 file へ書く")
    parser.add_argument("--overwrite", action="store_true", help="古くなくても作り直す")
    args = parser.parse_args()

    cache_dir = study_dir(args.study) / "cache"
    device = torch.device(args.device)
    for relative in args.run_dirs:
        run_dir = REPOSITORY_ROOT / relative
        reason = rebuild_reason(cache_dir, run_dir, args.split, args.features)
        if reason is None and not args.overwrite:
            print(f"cache is current: {cache_path(cache_dir, run_dir.name, args.split)}")
            continue
        print(f"rebuilding ({reason or '--overwrite'}): {cache_path(cache_dir, run_dir.name, args.split)}")
        for output in write_cache(run_dir, args.split, cache_dir, device, args.features):
            print(f"cached: {output}")


if __name__ == "__main__":
    main()
