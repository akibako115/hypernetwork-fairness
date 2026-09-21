"""参照 stage の metadata embedding から train-fit KMeans cohort artifact を生成する。"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.cluster import KMeans

from ..data.attribute_utils import attribute_tensors
from ..data.datamodule import ImageDataModule
from ..data.splits import SPLITS, validate_split_frame


@dataclass(frozen=True)
class SplitEmbeddings:
    """一つの split の image、target、metadata embedding を保持する。"""

    images: np.ndarray
    targets: np.ndarray
    values: np.ndarray


def load_split_frames(
    datamodule: ImageDataModule,
    attribute_names: Mapping[str, Sequence[str]] | None,
) -> tuple[dict[str, pd.DataFrame], dict[str, pd.DataFrame]]:
    """学習時と同じ CSV 検証・連続属性標準化を通した split frame を返す。

    Args:
        datamodule: split CSV の読み込みと標準化を所有する DataModule
        attribute_names: `categorical` / `continuous` の列名

    Returns:
        tuple[dict[str, pd.DataFrame], dict[str, pd.DataFrame]]: (raw, prepared)。raw は
            CSV そのまま、prepared は連続属性を標準化したもの。どちらも行順は CSV と同じ

    Raises:
        ValueError: いずれかの split で必須列が欠けている場合、または `image` が重複する場合。
    """
    raw = {split: datamodule.read_split_dataframe(split) for split in SPLITS}
    for split, frame in raw.items():
        validate_split_frame(frame, attribute_names, split=split)
    prepared = dict(zip(SPLITS, datamodule.standardized_dataframes(*SPLITS), strict=True))
    return raw, prepared


def extract_embeddings(
    metadata_encoder: torch.nn.Module,
    raw_frames: Mapping[str, pd.DataFrame],
    prepared_frames: Mapping[str, pd.DataFrame],
    *,
    attribute_names: Mapping[str, Sequence[str]] | None,
    batch_size: int,
    device: torch.device,
) -> dict[str, SplitEmbeddings]:
    """各 split を行順どおりに metadata encoder へ通す。

    画像 loader を経由しないので、embedding は data augmentation や sampler に依存しない。

    Args:
        metadata_encoder: 属性辞書を embedding に変換する encoder
        raw_frames: split ごとの CSV そのままの DataFrame。`image` と `target` を取る
        prepared_frames: split ごとの標準化済み DataFrame。属性の取り出しに使う
        attribute_names: `categorical` / `continuous` の列名
        batch_size: encoder に一度に通す行数
        device: encoder を載せる device

    Returns:
        dict[str, SplitEmbeddings]: split ごとの image / target / embedding。行順は CSV と同じ

    Raises:
        ValueError: batch_size が 1 未満の場合、split が空の場合、
            または embedding に NaN / inf が含まれる場合。
    """
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    metadata_encoder.to(device).eval()
    result: dict[str, SplitEmbeddings] = {}
    for split in SPLITS:
        raw, prepared = raw_frames[split], prepared_frames[split]
        attributes = attribute_tensors(prepared, attribute_names)
        batches: list[torch.Tensor] = []
        with torch.inference_mode():
            for start in range(0, len(prepared), batch_size):
                batch = {name: value[start : start + batch_size].to(device) for name, value in attributes.items()}
                batches.append(metadata_encoder(batch).float().cpu())
        if not batches:
            raise ValueError(f"{split} split must not be empty when building a cohort")
        values = torch.cat(batches).numpy().astype("float32")
        if not np.isfinite(values).all():
            raise ValueError("metadata embedding contains NaN or infinity")
        result[split] = SplitEmbeddings(
            images=raw["image"].astype(str).to_numpy(dtype=str),
            targets=raw["target"].to_numpy(dtype="int64"),
            values=values,
        )
    return result


def save_artifact(
    output_dir: Path,
    embeddings: Mapping[str, SplitEmbeddings],
    *,
    clusters: int,
    n_init: int,
    random_state: int,
    reference_checkpoint: Path,
    reference_id: str,
) -> Path:
    """train-fit KMeans、assignment、再現に必要な sidecar を一つの artifact として保存する。

    KMeans は train split だけで fit し、val / test には同じ cluster center を適用する。

    Args:
        output_dir: artifact の出力先。既存 directory は上書きしない
        embeddings: `extract_embeddings` が返した split ごとの embedding
        clusters: KMeans の k。cohort の group 数
        n_init: KMeans の初期化回数
        random_state: KMeans の乱数種
        reference_checkpoint: embedding を生成した checkpoint。path と SHA-256 を記録する
        reference_id: 参照した stage 名

    Returns:
        Path: 後続 stage の DataModule へ渡す `assignments.parquet` の path

    Raises:
        ValueError: clusters が 2 未満、n_init が 1 未満、train の行数が clusters 未満、
            または train assignment が全 group を覆わない場合。
        FileExistsError: output_dir が既に存在する場合。
        FileNotFoundError: reference_checkpoint が存在しない場合。
    """
    if clusters < 2:
        raise ValueError("clusters must be at least 2")
    if n_init < 1:
        raise ValueError("n_init must be positive")
    if len(embeddings["train"].values) < clusters:
        raise ValueError("train sample count must be greater than or equal to clusters")
    if output_dir.exists():
        raise FileExistsError(f"cohort artifact already exists: {output_dir}")
    if not reference_checkpoint.is_file():
        raise FileNotFoundError(reference_checkpoint)

    # train だけで fit し、validation/test には同じ cluster center を予測として適用する。
    kmeans = KMeans(n_clusters=clusters, n_init=n_init, random_state=random_state, algorithm="lloyd").fit(embeddings["train"].values)
    group_ids = {split: kmeans.predict(embeddings[split].values).astype("int64") for split in SPLITS}
    train_groups = set(group_ids["train"])
    if train_groups != set(range(clusters)):
        raise ValueError(f"KMeans train assignments must cover 0..{clusters - 1}, got {sorted(train_groups)}")

    output_dir.mkdir(parents=True)
    embedding_dir = output_dir / "embeddings"
    embedding_dir.mkdir()
    frames: list[pd.DataFrame] = []
    for split in SPLITS:
        data = embeddings[split]
        np.savez_compressed(embedding_dir / f"{split}.npz", image=data.images, target=data.targets, embedding=data.values, split=np.asarray(split))
        frames.append(pd.DataFrame({"split": split, "image": data.images, "group_id": group_ids[split]}))
    assignment_path = output_dir / "assignments.parquet"
    pd.concat(frames, ignore_index=True).astype({"split": "string", "image": "string", "group_id": "int64"}).to_parquet(assignment_path, index=False)
    np.savez_compressed(
        output_dir / "kmeans.npz",
        cluster_centers=kmeans.cluster_centers_.astype("float32"),
        n_features_in=np.asarray(kmeans.n_features_in_, dtype="int64"),
        n_iter=np.asarray(kmeans.n_iter_, dtype="int64"),
        inertia=np.asarray(kmeans.inertia_, dtype="float64"),
        n_init=np.asarray(n_init, dtype="int64"),
        random_state=np.asarray(random_state, dtype="int64"),
        algorithm=np.asarray("lloyd"),
    )
    metadata = {
        "schema_version": 1,
        "name": "metadata_kmeans",
        "reference_id": reference_id,
        "reference_checkpoint": {"path": str(reference_checkpoint.resolve()), "sha256": _sha256_file(reference_checkpoint)},
        "num_groups": clusters,
        "n_init": n_init,
        "random_state": random_state,
        "splits": {split: {"num_rows": len(embeddings[split].images)} for split in SPLITS},
    }
    (output_dir / "cohort.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2, sort_keys=True) + "\n")
    return assignment_path


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
