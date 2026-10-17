"""属性条件付きモデルへ、セルの中で一定の属性を入れて test を推論し直す。

問い：属性は「スコアの切片をずらす」だけに使われているのか、「画像の読み方」まで変えているのか。

属性が切片（と傾き）だけに効くなら、logit 差は z = s(a)·g(x) + h(a) の形になる。全員に同じ属性 a を
入れれば、画像の並び順は g(x) の順のままで、a を何にしても同じ集団の AUROC は変わらない。
したがって、**同じ属性セルの行に同じ属性を入れたまま、属性の値だけを変えて**セル内の AUROC が動けば、
属性が画像ごとに違う向きに効いている（交互作用がある）ことになる。

行ごとに入れる属性の作り方（条件ごとに 1 列の logit 差を書く）:

- `observed`：属性をそのまま入れる（参照）
- `age_<t>`：評価セルを決める categorical（HAM は sex、CheXpert は race_group の代表コード）だけ行の値、
  年齢は全行 t 歳、他の categorical は train の最頻値、欠損フラグはすべて 0
- `race_<k>`（CheXpert）/ `sex_<k>`（HAM）：その属性を全行 k にし、年齢は行の年齢群の代表値
  （65 歳未満は 45 歳、以上は 75 歳）、他は `age_<t>` と同じ

どの条件でも、評価セル（HAM は sex × 65 歳、CheXpert は race_group × 65 歳）の中では全行が同じ属性になる。
画像は 1 度だけ読み、同じ batch に全条件の属性を当てて推論する。

入力: `analysis/conditioning_mechanism/runs/<run-id>/`
出力: `analysis/conditioning_mechanism/cache/<run-id>_test_interaction.npz`

    uv run python -m analysis.conditioning_mechanism.interaction_counterfactual --device cuda:0
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

from analysis.common.paths import local_data_path  # noqa: E402
from analysis.common.predictions import checkpoint_reference, load_network  # noqa: E402
from projects.hypernet_e2e.data.attribute_utils import attribute_arrays  # noqa: E402

PACKAGE = Path(__file__).resolve().parent
RUNS = PACKAGE / "runs"
SCHEMA_VERSION = 1
AGE_GRID = (25, 35, 45, 55, 65, 75, 85)
# 年齢群の代表値。race・sex を差し替える条件で、年齢をセルの中で一定にするために使う。
AGE_YOUNG, AGE_OLD, AGE_BOUNDARY = 45, 75, 65
# train の最頻値（data/<dataset>/splits*/train.csv で確認）。評価セルを決めない属性はこれで埋める。
MODE = {
    "ham10000": {"sex": 1, "localization": 2},
    "chexpert": {"sex": 0, "race": 0, "ethnicity": 0, "frontal_lateral": 0, "ap_pa": 0},
}
# CheXpert の race_group → encoder に入れる race のコード。Others（元コード 1・4・5）は最多の 1 で代表させる。
RACE_CODE = {0: 0, 1: 2, 2: 3, 3: 1}
SEX_CODES = (0, 1)


def dataset_of(config) -> str:
    """run の split path からデータセット名を返す。"""
    splits = str(config.data.cv_splits_dir)
    for name in MODE:
        if f"/{name}/" in splits:
            return name
    raise ValueError(f"対応していない split: {splits}")


def _age_scaler(datamodule) -> tuple[float, float]:
    """train の非欠損行から、年齢の標準化に使う mean・std（ddof=0）を返す。学習時と同じ式。"""
    train = datamodule.read_split_dataframe("train")
    ages = train.loc[~train["age_missing"].astype(bool), "age"].astype(float)
    return float(ages.mean()), float(ages.std(ddof=0))


def _constant_frame(
    frame: pd.DataFrame, dataset: str, cell_values: dict[str, np.ndarray], age: np.ndarray, scaler
) -> pd.DataFrame:
    """評価セルを決める属性を cell_values、年齢を age（歳）、残りを最頻値にした frame を返す。"""
    out = frame.copy()
    for column, value in MODE[dataset].items():
        out[column] = value
    for column, values in cell_values.items():
        # 欠損行の値は -1 で、欠損フラグを下ろすと embedding の範囲外になる。欠損行は評価に入らないので最頻値で埋める。
        out[column] = np.where(values < 0, MODE[dataset][column], values)
    mean, std = scaler
    out["age"] = (age - mean) / std
    for column in [*MODE[dataset], "age"]:
        out[f"{column}_missing"] = 0
    return out


def variants(frame: pd.DataFrame, raw_age: np.ndarray, dataset: str, scaler) -> dict[str, pd.DataFrame]:
    """条件名から、属性を差し替えた frame への対応を返す。"""
    n = len(frame)
    group_age = np.where(raw_age >= AGE_BOUNDARY, AGE_OLD, AGE_YOUNG).astype(float)
    if dataset == "ham10000":
        cell = {"sex": frame["sex"].to_numpy()}
        swaps = {f"sex_{k}": {"sex": np.full(n, k)} for k in SEX_CODES}
    else:
        cell = {"race": frame["race_group"].map(RACE_CODE).to_numpy()}
        swaps = {f"race_{group}": {"race": np.full(n, code)} for group, code in RACE_CODE.items()}
    out = {"observed": frame}
    for t in AGE_GRID:
        out[f"age_{t}"] = _constant_frame(frame, dataset, cell, np.full(n, float(t)), scaler)
    for name, values in swaps.items():
        out[name] = _constant_frame(frame, dataset, values, group_age, scaler)
    return out


def cache_path(run_id: str) -> Path:
    """run に対応する cache path を返す。"""
    return PACKAGE / "cache" / f"{run_id}_test_interaction.npz"


def _forward_all(network, loader, inputs: dict[str, dict[str, torch.Tensor]], device) -> np.ndarray:
    """batch ごとに全条件の属性を当てて、logit 差（陽性 − 陰性）を [N, 条件数] で返す。"""
    columns, offset = [], 0
    with torch.inference_mode():
        for image, attributes, _ in loader:
            rows = slice(offset, offset + len(image))
            image = image.to(device, non_blocking=True)
            per_variant = []
            for arrays in inputs.values():
                batch = {key: value.to(device) for key, value in attributes.items()}
                batch.update({key: value[rows].to(device) for key, value in arrays.items()})
                logits = network(image, batch).float()
                per_variant.append((logits[:, 1] - logits[:, 0]).cpu().numpy())
            columns.append(np.stack(per_variant, axis=1))
            offset += len(image)
    return np.concatenate(columns)


def write(run_dir: Path, device: torch.device, overwrite: bool = False) -> None:
    """1 run について、全条件の test の logit 差を cache へ書く。"""
    output = cache_path(run_dir.name)
    checkpoint = checkpoint_reference(run_dir)
    if output.is_file() and not overwrite:
        if json.loads(str(np.load(output)["metadata"])).get("checkpoint") == checkpoint:
            print(f"cached: {output.name}", flush=True)
            return
    config = OmegaConf.load(run_dir / "config.yaml")
    if not config.model.get("use_attributes", False):
        raise ValueError(f"{run_dir.name} は属性を取らない")
    dataset = dataset_of(config)
    for key in ("cv_splits_dir", "data_dir"):
        config.data[key] = str(local_data_path(config.data[key]))
    datamodule = instantiate(config.data)
    (frame,) = datamodule.standardized_dataframes("test")
    raw_age = datamodule.read_split_dataframe("test")["age"].to_numpy(dtype=float)
    edited = variants(frame, raw_age, dataset, _age_scaler(datamodule))
    names = config.data.attribute_names
    inputs = {
        name: {key: torch.as_tensor(value) for key, value in attribute_arrays(table, names).items()}
        for name, table in edited.items()
    }
    # 逐次で回すので persistent worker は持たない（並列 probe の OOM を避ける）。
    loader = DataLoader(
        datamodule._create_dataset(frame, datamodule.val_transform),
        batch_size=datamodule.batch_size_per_device,
        num_workers=8,
        shuffle=False,
    )
    network = load_network(run_dir, device)
    scores = _forward_all(network, loader, inputs, device)
    metadata = {"schema_version": SCHEMA_VERSION, "run_id": run_dir.name, "checkpoint": checkpoint, "dataset": dataset}
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output,
        image=frame["image"].to_numpy(dtype=str),
        target=frame["target"].to_numpy(dtype=np.int64),
        scores=scores,
        variants=np.array(list(edited), dtype=str),
        metadata=json.dumps(metadata, sort_keys=True),
    )
    print(f"wrote: {output.name}", flush=True)


def main() -> None:
    """CLI 入口。--run-id を省くと runs/ の全 run を順に処理する。"""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run-id", action="append", dest="run_ids")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    run_ids = args.run_ids or sorted(path.name for path in RUNS.iterdir() if path.is_dir())
    for run_id in run_ids:
        write(RUNS / run_id, torch.device(args.device), args.overwrite)


if __name__ == "__main__":
    main()
