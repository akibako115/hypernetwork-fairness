"""公平性損失に使う潜在群（生成 ΔW の k-means）と、その対照の群を split CSV に足す。

1 段・ERM の Spatial LoRA（seed 42・43・44）ごとに、train の件数で重み付けした k-means（K = 8）で
セル（モデル入力の属性の組み合わせ）を分け、行ごとの群番号を split CSV の列にする。

- `dw8_s<seed>`：生成 ΔW（stage3・stage4・fc の Gram の和）の kernel PCA 座標の k-means
- `attrk8_s<seed>`：モデル入力の属性（one-hot ＋ 年齢 / 100 ＋ 欠損フラグ）を直接 k-means した対照。
  潜在空間が「属性を同じ数に切る」以上のものを足したかを分ける

群は train だけで決める（k-means の重みは train の件数）。val は checkpoint の選択に、test は評価に使うので、
群づくりには使わない。c・ΔW は属性だけの関数なので、群は「セル → 群番号」の表になり、どの split にも同じ表を当てる。
seed s の ERM から作った群で seed s の学習をする（群の揺れも手順の揺れに含める）。

セルの番号は、run の datamodule が標準化した年齢で引く（`extract.py` と同じ）。dual holdout の train・holdout は
race_known の train の部分集合なので、画像の名前で引き直す。

入力:
- `cache/<run-id>_cells.npz`（`extract.py`）
- `data/chexpert/splits_race_known/{train,val,test}.csv`、`data/chexpert/splits_race_known_dual_holdout/*.csv`
出力:
- `data/chexpert/splits_race_known_latent/{train,val,test}.csv`（ERM・GroupDRO 用）
- `data/chexpert/splits_race_known_dual_holdout_latent/{train,holdout,val,test}.csv`（EO rate 用）
- `results/latent_groups/{cells,counts}.csv`

    uv run python -m analysis.latent_subgroup_spaces.latent_groups
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans

from analysis.latent_subgroup_spaces.extract import cell_index, input_columns, load_datamodule
from analysis.latent_subgroup_spaces.subgroups import PACKAGE, ROOT, kernel_coordinates, load_cells

DATA = ROOT / "data" / "chexpert"
RUNS = {
    42: "20261002T125005Z-spatial-lora-chexpert-race-known-s42-a21c",
    43: "20261002T145233Z-spatial-lora-chexpert-race-known-s43-424e",
    44: "20261002T165551Z-spatial-lora-chexpert-race-known-s44-eb13",
}
K = 8
CATEGORICAL = ("sex", "race", "ethnicity", "frontal_lateral", "ap_pa")


def attribute_cells(cells: pd.DataFrame) -> np.ndarray:
    """セルごとのモデル入力の属性。one-hot ＋ 年齢 / 100 ＋ 欠損フラグ（02 の `attribute_matrix` と同じ尺度）。"""
    onehot = pd.get_dummies(cells[list(CATEGORICAL)].astype(str)).to_numpy(dtype=float)
    missing = cells[[c for c in cells.columns if c.endswith("_missing")]].to_numpy(dtype=float)
    return np.column_stack([cells.age.to_numpy() / 100.0, onehot, missing])


def cell_groups(seed: int, raw_age: np.ndarray) -> pd.DataFrame:
    """1 seed のセル表に `dw8`・`attrk8` を足して返す。`raw_age` は標準化前の年齢（属性の k-means に使う）。"""
    cells, _, grams = load_cells(RUNS[seed])
    weight = cells.train_count.to_numpy()
    coordinates = kernel_coordinates(grams["stage3"] + grams["stage4"] + grams["fc"], weight + 1e-3)
    dw = KMeans(K, n_init=10, random_state=seed).fit(coordinates, sample_weight=weight)
    attributes = attribute_cells(cells.assign(age=raw_age))
    attr = KMeans(K, n_init=10, random_state=seed).fit(attributes, sample_weight=weight)
    return cells.assign(seed=seed, raw_age=raw_age, dw8=dw.predict(coordinates), attrk8=attr.predict(attributes))


def row_groups(seed: int) -> tuple[dict[str, pd.DataFrame], pd.DataFrame]:
    """race_known の train・val・test の行ごとの群番号（画像で引ける表）と、セル表。"""
    datamodule, attribute_names = load_datamodule(PACKAGE / "runs" / RUNS[seed])
    columns = input_columns(attribute_names)
    standardized = dict(
        zip(("train", "val", "test"), datamodule.standardized_dataframes("train", "val", "test"), strict=True)
    )
    cells, _, _ = load_cells(RUNS[seed])
    # 標準化した年齢から元の年齢へ戻す表（同じセルの行は同じ年齢を持つ）。
    raw = pd.concat(
        [pd.read_csv(DATA / "splits_race_known" / f"{split}.csv", usecols=["age"]) for split in standardized],
        ignore_index=True,
    ).age.to_numpy()
    index = np.concatenate([cell_index(frame, cells, columns) for frame in standardized.values()])
    raw_age = pd.Series(raw).groupby(index).first().reindex(range(len(cells))).to_numpy()
    if np.isnan(raw_age).any():
        raise ValueError("年齢を戻せないセルがある")
    table = cell_groups(seed, raw_age)
    rows = {}
    for split, frame in standardized.items():
        cell = cell_index(frame, cells, columns)
        rows[split] = pd.DataFrame(
            {
                "image": frame.image.to_numpy(),
                "cell": cell,
                "dw8": table.dw8.to_numpy()[cell],
                "attrk8": table.attrk8.to_numpy()[cell],
            }
        )
    return rows, table


def _write(dest: Path, name: str, frame: pd.DataFrame) -> None:
    dest.mkdir(parents=True, exist_ok=True)
    # 学習中の run が読む CSV なので、書き終えてから置き換える
    frame.to_csv(dest / f".{name}.csv.tmp", index=False)
    (dest / f".{name}.csv.tmp").replace(dest / f"{name}.csv")


def main() -> None:
    by_image = {}
    tables = []
    for seed in RUNS:
        rows, table = row_groups(seed)
        tables.append(table)
        by_image[seed] = pd.concat(rows.values()).set_index("image")
    sources = {
        "splits_race_known_latent": ("splits_race_known", ("train", "val", "test")),
        "splits_race_known_dual_holdout_latent": (
            "splits_race_known_dual_holdout",
            ("train", "holdout", "val", "test"),
        ),
    }
    counts = []
    for dest_name, (source_name, splits) in sources.items():
        for split in splits:
            frame = pd.read_csv(DATA / source_name / f"{split}.csv")
            for seed, groups in by_image.items():
                matched = groups.reindex(frame.image)
                if matched.isna().any().any():
                    raise ValueError(
                        f"{source_name}/{split}: 群を引けない画像が {int(matched.dw8.isna().sum())} 枚ある"
                    )
                for name in ("dw8", "attrk8"):
                    column = f"{name}_s{seed}"
                    frame[column] = matched[name].to_numpy(dtype=np.int64)
                    frame[f"{column}_missing"] = 0
                    stats = frame.groupby(column).target.agg(n="size", n_pos="sum")
                    counts.append(
                        stats.assign(n_neg=stats.n - stats.n_pos, dir=dest_name, split=split, seed=seed, partition=name)
                        .rename_axis("group")
                        .reset_index()
                    )
            _write(DATA / dest_name, split, frame)
        manifest = (
            json.loads((DATA / source_name / "manifest.json").read_text())
            if (DATA / source_name / "manifest.json").exists()
            else {}
        )
        manifest["latent_groups"] = {
            "source_dir": f"data/chexpert/{source_name}",
            "writer": "analysis/latent_subgroup_spaces/latent_groups.py",
            "runs": RUNS,
            "columns": {
                "dw8_s<seed>": "生成 ΔW の k-means（K=8、train 件数で重み付け）",
                "attrk8_s<seed>": "モデル入力属性の k-means（対照）",
            },
        }
        (DATA / dest_name / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2))
    out = PACKAGE / "results" / "latent_groups"
    shutil.rmtree(out, ignore_errors=True)
    out.mkdir(parents=True)
    pd.concat(tables).to_csv(out / "cells.csv", index=False)
    pd.concat(counts).to_csv(out / "counts.csv", index=False)
    print(f"wrote: {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
