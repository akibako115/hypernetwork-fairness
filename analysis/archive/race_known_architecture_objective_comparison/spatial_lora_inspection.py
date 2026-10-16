"""2-stage Spatial LoRAの生成ΔW・予測変化・metadata空間を調べる。

08 notebookから使う。2-stageはbackboneと元classifierが同seed M01と同一なので、
ΔW=0の経路はM01そのものになる。この性質を使い、adapterが何をどれだけ変えたかを
(1) 画像なしで決まる生成ΔW・condition、(2) 画像forwardが要る予測変化と反実仮想metadata
に分けて調べる。(2)だけを`cache/`へ書き出し、notebookはGPUを自動起動しない。

入力はpackageの`runs/`とsplit CSV。checkpointはrun.jsonのSHA256と照合する。
lastは「選択epochが早く生成器が未発達なだけか」を見る補助診断に限り、性能比較には使わない。

使い方（repo rootで、利用可能なGPUを明示する）:
    uv run python analysis/archive/race_known_architecture_objective_comparison/spatial_lora_inspection.py \
      --device cuda:0 --checkpoint best --checkpoint last \
      --run-dir analysis/archive/race_known_architecture_objective_comparison/runs/<run-id>
"""

from __future__ import annotations

import argparse
import json
import shlex
from functools import partial
from pathlib import Path

import numpy as np
import pandas as pd
import rootutils
import torch
from hydra.utils import instantiate
from omegaconf import OmegaConf
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from analysis.archive.race_known_architecture_objective_comparison.comparison import digest, group_ids  # noqa: E402
from analysis.common.paths import REPOSITORY_ROOT, local_data_path  # noqa: E402
from analysis.common.run_artifacts import read_run_record  # noqa: E402
from projects.hypernet_e2e.data.attribute_utils import attribute_tensors  # noqa: E402
from projects.hypernet_e2e.models.spatial_hypernet.modulation import HyperLinearLayer  # noqa: E402

PROBE_SCHEMA_VERSION = 1
# encoderへ入るraceは6区分。Othersは件数最多のcode 1で代表させる（4・5は同じ群だが別embedding）。
RACE_CODES = {"White": 0, "Asian": 2, "Black": 3, "Others": 1}
# 年齢の反実仮想は群境界の両側の代表値。中央値ではなく固定値にして、seed間で入力を揃える。
COUNTERFACTUAL_AGES = {"age 50": 50.0, "age 75": 75.0}
STAGE_LAYERS = {"stage1": "layer1", "stage2": "layer2", "stage3": "layer3", "stage4": "layer4"}


# ----- checkpointとnetwork -----


def checkpoint_file(run_dir, kind):
    """best/lastのcheckpointを返し、run.jsonのSHA256と照合する。

    Args:
        run_dir: 取り込み済みrun。
        kind: `best`または`last`。
    Returns:
        Path: 照合済みcheckpoint。
    """
    record = read_run_record(run_dir)
    entry = next(item[kind] for item in record["checkpoints"] if item["monitor"] == "val/auroc")
    path = run_dir / "checkpoints" / Path(entry["path"]).name
    if digest(path) != entry["sha256"]:
        raise ValueError(f"{path}: SHA256がrun.jsonと一致しない")
    return path


def load_spatial_network(run_dir, kind, device):
    """Spatial LoRA networkを指定checkpointから復元する。

    Args:
        run_dir: 取り込み済みrun。
        kind: `best`または`last`。
        device: 置き先。
    Returns:
        SpatialLoRAResNet: eval mode。
    """
    config = OmegaConf.load(run_dir / "config.yaml")
    network = instantiate(config.model.net)
    checkpoint = torch.load(checkpoint_file(run_dir, kind), map_location="cpu", weights_only=False)
    weights = {
        key.removeprefix("net."): value for key, value in checkpoint["state_dict"].items() if key.startswith("net.")
    }
    network.load_state_dict(weights, strict=True)
    return network.to(device).eval()


def adapter_layers(network):
    """生成ΔWを持つ層を、forward順の名前・adapter・元重みで返す。

    Args:
        network: SpatialLoRAResNet。
    Returns:
        list: (`stage3.0.conv2`形式の名前, adapter, 凍結された元重み)。fcは`fc`。
    """
    layers = []
    for stage, layer_name in STAGE_LAYERS.items():
        if stage not in network.spatial_adapters:
            continue
        blocks = getattr(network.backbone, layer_name)
        for index, adapter in enumerate(network.spatial_adapters[stage]):
            for conv_name in adapter.adapted_convs:
                conv = getattr(blocks[index], conv_name)
                layers.append((f"{stage}.{index}.{conv_name}", getattr(adapter, f"{conv_name}_adapter"), conv.weight))
    if isinstance(network.fc, HyperLinearLayer):
        layers.append(("fc", network.fc, network.fc.base_linear.weight))
    return layers


# ----- splitと属性 -----


def model_frame(run_dir, split):
    """学習と同じ標準化を施したsplitと、元CSVの年齢・8群を返す。

    Args:
        run_dir: 属性定義とsplitを記録したrun。
        split: `train`・`val`・`test`。
    Returns:
        tuple: 標準化済みframe（`raw_age`・`group_id`付き）、datamodule。
    """
    config = OmegaConf.load(run_dir / "config.yaml")
    for key in ("cv_splits_dir", "data_dir"):
        config.data[key] = str(local_data_path(config.data[key]))
    datamodule = instantiate(config.data)
    (frame,) = datamodule.standardized_dataframes(split)
    raw = datamodule.read_split_dataframe(split)
    if not (frame["image"].to_numpy() == raw["image"].to_numpy()).all():
        raise ValueError("標準化前後で行順が異なる")
    # 8群は標準化前の年齢で切る。標準化済みageに65の境界を当てると群が全部ずれる。
    frame = frame.assign(raw_age=raw["age"].to_numpy(), group_id=group_ids(raw))
    return frame, datamodule


def age_standardizer(datamodule):
    """生の年齢を学習時と同じ標準化値へ写す関数を返す。

    Args:
        datamodule: 学習と同じ設定のdatamodule。
    Returns:
        Callable: 年齢（歳）→標準化値。
    """
    train = datamodule.read_split_dataframe("train")
    valid = pd.to_numeric(train["age"]).mask(train["age_missing"].astype(bool))
    mean, std = float(valid.mean()), float(valid.std(ddof=0))
    return lambda age: (age - mean) / std


def unique_conditions(frame, attribute_names):
    """encoder入力が同じ行をまとめ、件数と8群・race・年齢群を付ける。

    conditionは属性だけで決まるので、生成ΔWは一意な属性組ごとに1回計算すれば足りる。
    件数を重みにすると、test集団全体での平均・分散と同じになる。

    Args:
        frame: `model_frame`の出力。
        attribute_names: configの`model.attribute_names`。
    Returns:
        DataFrame: 一意な属性組。`count`・`group_id`・`race_group`・`age_group`列を持つ。
    """
    columns = [*attribute_names["categorical"], *attribute_names["continuous"]]
    columns += [f"{name}_missing" for name in columns]
    keys = frame[[*columns, "group_id", "race_group"]].assign(age_group=frame.group_id // 4)
    if (keys.groupby(columns).group_id.nunique() > 1).any():
        raise ValueError("同じencoder入力が別の8群にまたがる")
    table = keys.groupby(columns, as_index=False).agg(
        count=("group_id", "size"), group_id=("group_id", "first"), race_group=("race_group", "first")
    )
    return table.assign(age_group=np.where(table.group_id >= 0, table.group_id // 4, -1))


@torch.inference_mode()
def encode(network, table, attribute_names, device):
    """属性表をmetadata condition cへ写す。

    Args:
        network: SpatialLoRAResNet。
        table: 属性列と`*_missing`列を持つ表。
        attribute_names: configの`model.attribute_names`。
        device: 計算device。
    Returns:
        Tensor: `[n_rows, condition_dim]`。
    """
    attributes = attribute_tensors(table, attribute_names)
    return network.metadata_encoder({key: value.to(device) for key, value in attributes.items()})


# ----- 生成ΔWの分解 -----


def _weighted_means(layer, condition, weights, partitions, chunk):
    """件数重み付きのΔW平均と、各分割の群平均を1周で求める。"""
    total = weights.sum()
    mean = 0.0
    sums = {name: {} for name in partitions}
    for start in range(0, len(condition), chunk):
        delta = layer.weight_delta(condition[start : start + chunk]).flatten(1).double()
        weight = weights[start : start + chunk]
        mean = mean + (weight[:, None] * delta).sum(0) / total
        for name, labels in partitions.items():
            for label in labels[start : start + chunk].unique().tolist():
                mask = labels[start : start + chunk] == label
                sums[name][label] = sums[name].get(label, 0.0) + (weight[mask, None] * delta[mask]).sum(0)
    return mean, sums


@torch.inference_mode()
def delta_decomposition(layer, base_weight, condition, weights, partitions, chunk=64):
    """ΔWの大きさを、全条件共通の平均・群間差・残りに分解する。

    E||ΔW||² = ||μ||² + E||ΔW−μ||²、E||ΔW−μ||² = Σ_g w_g||μ_g−μ||² + 群内。
    共通平均μはmetadataに依らない一律の補正（凍結backboneの通常fine-tuningと同じ働き）で、
    属性ごとの違いは残り2項にだけ現れる。すべて元重みのFrobenius normで割って示す。

    Args:
        layer: `weight_delta`を持つadapter。
        base_weight: 凍結された元重み。
        condition: 一意な属性組のcondition。
        weights: 各属性組の件数。
        partitions: 分割名→各属性組の群ラベル（負値は除外群として平均に含めない）。
        chunk: 一度に実体化する属性組数。
    Returns:
        tuple: 指標dict、属性組ごとの相対norm、分割名→{群: μ_g−μ}。
    """
    weights = torch.as_tensor(weights, dtype=torch.float64, device=condition.device)
    labels = {name: torch.as_tensor(value, device=condition.device) for name, value in partitions.items()}
    mean, sums = _weighted_means(layer, condition, weights, labels, chunk)
    base_sq = base_weight.double().pow(2).sum()
    per_condition, centered = [], []
    for start in range(0, len(condition), chunk):
        delta = layer.weight_delta(condition[start : start + chunk]).flatten(1).double()
        per_condition.append(delta.pow(2).sum(1))
        centered.append((delta - mean).pow(2).sum(1))
    per_condition, centered = torch.cat(per_condition), torch.cat(centered)
    total = weights.sum()
    row = {
        "base_norm": base_sq.sqrt().item(),
        "rms_rel": ((weights * per_condition).sum() / total / base_sq).sqrt().item(),
        "shared_rel": (mean.pow(2).sum() / base_sq).sqrt().item(),
        "varying_rel": ((weights * centered).sum() / total / base_sq).sqrt().item(),
    }
    deviations = {}
    for name, label_sums in sums.items():
        group_weight = {label: weights[labels[name] == label].sum() for label in label_sums}
        kept = sum(group_weight[label] for label in label_sums if label >= 0)
        deviations[name] = {label: label_sums[label] / group_weight[label] - mean for label in label_sums if label >= 0}
        between = sum(group_weight[label] * deviations[name][label].pow(2).sum() for label in deviations[name])
        row[f"between_{name}_rel"] = (between / kept / base_sq).sqrt().item()
    relative = (per_condition / base_sq).sqrt().cpu().numpy()
    return row, relative, deviations


@torch.inference_mode()
def deviation_cosine(deviations):
    """群平均の偏差μ_g−μどうしのcosine類似度を返す。

    Args:
        deviations: `delta_decomposition`が返す1分割分の{群: μ_g−μ}。
    Returns:
        DataFrame: 群×群。1に近い群どうしは同じ向きへ変調されている。
    """
    labels = sorted(deviations)
    stacked = torch.stack([deviations[label] for label in labels])
    normalized = stacked / stacked.norm(dim=1, keepdim=True).clamp_min(1e-30)
    return pd.DataFrame((normalized @ normalized.T).cpu().numpy(), index=labels, columns=labels)


def _attribute_names(run_dir):
    return OmegaConf.to_container(OmegaConf.load(run_dir / "config.yaml").model.attribute_names)


def weight_diagnostics(run_dir, kind, device):
    """1 checkpointの全adapter層について、ΔW分解と8群偏差のcosineを返す。

    test集団の属性組で評価する（画像は読まない）。相対normは元重みのFrobenius normとの比で、
    凍結BNを挟むため出力の変化量そのものではない。出力側の変化は予測probeで別に見る。

    Args:
        run_dir: 2-stage Spatial LoRA run。
        kind: `best`または`last`。
        device: 計算device。
    Returns:
        tuple: 層別の分解表、層×群×群のcosine表。
    """
    names = _attribute_names(run_dir)
    frame, _ = model_frame(run_dir, "test")
    table = unique_conditions(frame, names)
    counts = table["count"].to_numpy()
    partitions = {"group": table.group_id.to_numpy(), "race": table.race_group.to_numpy()}
    partitions["age"] = table.age_group.to_numpy()
    network = load_spatial_network(run_dir, kind, device)
    condition = encode(network, table, names, device)
    rows, cosines = [], []
    for name, layer, base in adapter_layers(network):
        row, relative, deviations = delta_decomposition(layer, base, condition, counts, partitions)
        per_row = np.repeat(relative, counts)
        rows.append(
            {"layer": name, **row, "p05_rel": np.quantile(per_row, 0.05), "p95_rel": np.quantile(per_row, 0.95)}
        )
        cosine = deviation_cosine(deviations["group"]).rename_axis(index="group_a", columns="group_b").stack()
        cosines.append(cosine.rename("cosine").reset_index().assign(layer=name))
    return pd.DataFrame(rows), pd.concat(cosines, ignore_index=True)


# ----- metadata encoderの空間 -----


def weighted_pca(condition, weights):
    """件数重み付きPCAの固有値（降順）と主成分座標を返す。

    Args:
        condition: `[n, d]`のcondition。
        weights: 各行の件数。
    Returns:
        tuple: 固有値、`[n, d]`の座標。
    """
    share = weights / weights.sum()
    centered = condition - (share[:, None] * condition).sum(0)
    values, vectors = np.linalg.eigh((share[:, None] * centered).T @ centered)
    order = np.argsort(values)[::-1]
    return values[order], centered @ vectors[:, order]


def variance_share(condition, weights, labels):
    """属性値ごとの平均で説明できるconditionの分散割合（η²）を返す。

    属性どうしが相関する（例：lateralはAP/PA欠損）ため、属性間で足すと1を超えうる。

    Args:
        condition: `[n, d]`のcondition。
        weights: 各行の件数。
        labels: 各行の属性値。
    Returns:
        float: 群間分散/全分散。
    """
    share = weights / weights.sum()
    mean = (share[:, None] * condition).sum(0)
    total = (share * ((condition - mean) ** 2).sum(1)).sum()
    between = 0.0
    for label in np.unique(labels):
        mask = labels == label
        centroid = (share[mask, None] * condition[mask]).sum(0) / share[mask].sum()
        between += share[mask].sum() * ((centroid - mean) ** 2).sum()
    return between / total


@torch.inference_mode()
def swap_sensitivity(network, table, names, cardinalities, age_step):
    """1属性だけ差し替えたときのcondition移動量を、全体のRMS広がりとの比で返す。

    categoricalは現在と異なる全値への差し替えの平均、年齢は+10歳。件数重み付き平均。
    移動量はencoderの出力空間の距離で、生成ΔWや予測への効き方までは表さない。

    Args:
        network: SpatialLoRAResNet（eval）。
        table: 標準化済みの一意な属性組。
        names: configの`model.attribute_names`。
        cardinalities: categorical属性の値の数。
        age_step: 10歳に相当する標準化後の差。
    Returns:
        dict: 属性名→相対移動量。
    """
    device = next(network.parameters()).device
    weights = table["count"].to_numpy(float)
    base = encode(network, table, names, device)
    spread = np.sqrt((weights * (base - base.mean(0)).pow(2).sum(1).cpu().numpy()).sum() / weights.sum())
    result = {}
    for column, cardinality in zip(names["categorical"], cardinalities, strict=True):
        moved = []
        for value in range(cardinality):
            swapped = table.assign(**{column: value, f"{column}_missing": False})
            distance = (encode(network, swapped, names, device) - base).norm(dim=1).cpu().numpy()
            changed = (table[column].to_numpy() != value) | table[f"{column}_missing"].to_numpy(bool)
            moved.append(np.where(changed, distance, np.nan))
        per_row = np.nanmean(np.stack(moved, axis=1), axis=1)
        result[column] = np.average(per_row, weights=weights) / spread
    older = table.assign(age=table["age"] + age_step)
    distance = (encode(network, older, names, device) - base).norm(dim=1).cpu().numpy()
    result["age +10y"] = np.average(distance, weights=weights) / spread
    return result


def encoder_space(run_dir, kind, device):
    """train集団の属性組のcondition・PCA座標と、属性ごとの分散割合・感度を返す。

    encoderが学習中に見た分布で見るためtrainを使う（件数で重み付け）。

    Args:
        run_dir: 2-stage Spatial LoRA run。
        kind: `best`または`last`。
        device: 計算device。
    Returns:
        tuple: 属性組の表（`c*`・`pc*`列付き）、固有値、属性→η²、属性→感度。
    """
    config = OmegaConf.load(run_dir / "config.yaml")
    names = _attribute_names(run_dir)
    frame, datamodule = model_frame(run_dir, "train")
    table = unique_conditions(frame, names)
    network = load_spatial_network(run_dir, kind, device)
    condition = encode(network, table, names, device).double().cpu().numpy()
    weights = table["count"].to_numpy(float)
    values, coordinates = weighted_pca(condition, weights)
    columns = {f"c{index}": condition[:, index] for index in range(condition.shape[1])}
    table = table.assign(**columns, pc1=coordinates[:, 0], pc2=coordinates[:, 1])
    # 標準化値から歳へ戻す。standardize(a)=(a−mean)/stdなので、10歳分の差からstdが決まる。
    standardize = age_standardizer(datamodule)
    step = standardize(10.0) - standardize(0.0)
    table = table.assign(raw_age=(table["age"] - standardize(0.0)) * 10.0 / step)
    # 欠損は独立した値として数える（AP/PA欠損はほぼlateral画像に一致する）。
    labels = {name: np.where(table[f"{name}_missing"], -1, table[name]) for name in names["categorical"]}
    labels["age decade"] = (np.floor(table["raw_age"].to_numpy() / 10) * 10).astype(int)
    shares = {name: variance_share(condition, weights, value) for name, value in labels.items()}
    cardinalities = list(config.model.net.metadata_encoder.categorical_cardinalities)
    sensitivity = swap_sensitivity(network, table, names, cardinalities, step)
    return table, values, shares, sensitivity


# ----- 画像forwardの診断（cacheへ書く） -----


def probe_cache_path(cache_dir, run_id, kind):
    """予測変化・反実仮想cacheのpathを返す。

    Args:
        cache_dir: packageの`cache/`。
        run_id: run ID。
        kind: `best`または`last`。
    Returns:
        Path: `<run-id>_<kind>_test_spatial_probe.npz`。
    """
    return cache_dir / f"{run_id}_{kind}_test_spatial_probe.npz"


def probe_stale_reason(path, run_dir, kind):
    """cacheを作り直す理由を返す。不要なら`None`。

    Args:
        path: `probe_cache_path`。
        run_dir: 元run。
        kind: `best`または`last`。
    Returns:
        str | None: 理由。
    """
    if not path.is_file():
        return "cacheが無い"
    with np.load(path, allow_pickle=False) as cached:
        metadata = json.loads(str(cached["metadata"]))
    if metadata.get("schema_version") != PROBE_SCHEMA_VERSION:
        return "schema_versionが古い"
    if metadata.get("checkpoint_sha256") != digest(checkpoint_file(run_dir, kind)):
        return "checkpointが変わった"
    return None


def probe_command(package, directories, kinds, device="cuda:0"):
    """欠損したprobe cacheだけを作るコマンドを返す。推論は自動起動しない。

    Args:
        package: 分析package。
        directories: (条件, seed)→run。
        kinds: 必要なcheckpoint種別。
        device: 利用者が明示するGPU。
    Returns:
        str: repo rootで実行するコマンド。全cacheが有効なら空文字。
    """
    missing = [
        path
        for path in directories.values()
        if any(probe_stale_reason(probe_cache_path(package / "cache", path.name, kind), path, kind) for kind in kinds)
    ]
    if not missing:
        return ""
    parts = ["uv", "run", "python", str(Path(__file__).relative_to(REPOSITORY_ROOT)), "--device", device]
    for kind in kinds:
        parts.extend(["--checkpoint", kind])
    for path in missing:
        parts.extend(["--run-dir", str(path.relative_to(REPOSITORY_ROOT))])
    return shlex.join(parts)


def read_probe(package, run_dir, kind, images):
    """probe cacheを検証して読む。

    Args:
        package: 分析package。
        run_dir: 元run。
        kind: `best`または`last`。
        images: test CSVのimage列。
    Returns:
        dict: arrayとmetadata。
    """
    path = probe_cache_path(package / "cache", run_dir.name, kind)
    reason = probe_stale_reason(path, run_dir, kind)
    if reason is not None:
        raise RuntimeError(f"{path.name}: {reason}。notebookが提示するコマンドで生成してください。")
    with np.load(path, allow_pickle=False) as cached:
        result = {key: cached[key] for key in cached.files}
    result["metadata"] = json.loads(str(result["metadata"]))
    if not (result["image"] == images).all():
        raise ValueError(f"{path.name}: 行がtest CSVと対応しない")
    return result


def margin(logits):
    """2クラスlogitsを陽性−陰性のmarginへ変換する。softmax確率はsigmoid(margin)。"""
    return logits[..., 1] - logits[..., 0]


def _auroc(target, score):
    return roc_auc_score(target, score) if 0 < target.sum() < len(target) else np.nan


def prediction_change(probe, target, groups):
    """ΔW=0経路（=同seed M01）から実際の属性での予測への変化を要約する。

    Δmarginを8群平均（群ごとの一律のずれ）と群内の残りに分ける。群平均だけを足した
    「offset-only」予測は群内の順位を変えないので、群内AUROCは親と同じになる。
    actualとoffset-onlyの差が、群内の並べ替え・群内で一様でない補正の寄与になる。

    Args:
        probe: `read_probe`の出力。
        target: test正解。
        groups: 8群ID（−1は群集計から除外）。
    Returns:
        tuple: 全体のdict、群別DataFrame。
    """
    base, actual = margin(probe["logits_base"]), margin(probe["logits_actual"])
    delta = actual - base
    offsets = pd.Series(delta).groupby(groups).transform("mean").to_numpy()
    offset_only = base + offsets
    row = {
        "mean_abs_delta_margin": np.abs(delta).mean(),
        "base_margin_sd": base.std(),
        "mean_abs_delta_prob": np.abs(1 / (1 + np.exp(-actual)) - 1 / (1 + np.exp(-base))).mean(),
        "flip_rate": ((actual > 0) != (base > 0)).mean(),
        "group_offset_share": offsets.var() / delta.var(),
        "feature_shift": probe["feature_shift"].mean(),
        "feature_cosine": probe["feature_cosine"].mean(),
        "auroc_base": _auroc(target, base),
        "auroc_actual": _auroc(target, actual),
        "auroc_offset_only": _auroc(target, offset_only),
    }
    rows = []
    for group in sorted(set(groups) - {-1}):
        mask = groups == group
        rows.append(
            {
                "group_id": group,
                "mean_delta_margin": delta[mask].mean(),
                "sd_delta_margin": delta[mask].std(),
                "spearman_base_actual": spearmanr(base[mask], actual[mask]).statistic,
                "flip_rate": ((actual[mask] > 0) != (base[mask] > 0)).mean(),
                "auroc_base": _auroc(target[mask], base[mask]),
                "auroc_actual": _auroc(target[mask], actual[mask]),
            }
        )
    return row, pd.DataFrame(rows)


def counterfactual_change(probe, target, groups):
    """race・年齢だけを差し替えたときの予測の動きを、実際の変化量と並べて要約する。

    race spreadは4 raceを入れた時のmarginの最大−最小。親からの変化|Δ|より十分小さければ、
    adapterの補正は属性に依らない共通成分が主で、race別の補正は小さい。

    Args:
        probe: `read_probe`の出力。
        target: test正解。
        groups: 8群ID。
    Returns:
        tuple: 全体のdict、variant×群のDataFrame（実際の属性からの平均shiftと群内AUROC）。
    """
    variants = probe["metadata"]["variants"]
    actual, base = margin(probe["logits_actual"]), margin(probe["logits_base"])
    counterfactual = margin(probe["logits_counterfactual"])
    race = counterfactual[:, [variants.index(label) for label in RACE_CODES]]
    age = counterfactual[:, [variants.index(label) for label in COUNTERFACTUAL_AGES]]
    row = {
        "mean_abs_delta_margin": np.abs(actual - base).mean(),
        "race_spread": (race.max(1) - race.min(1)).mean(),
        "age_spread": np.abs(age[:, 1] - age[:, 0]).mean(),
        "race_flip_rate": ((race > 0).any(1) & (race <= 0).any(1)).mean(),
        "age_flip_rate": ((age > 0).any(1) & (age <= 0).any(1)).mean(),
    }
    rows = []
    for index, variant in enumerate(variants):
        for group in sorted(set(groups) - {-1}):
            mask = groups == group
            shift = counterfactual[mask, index] - actual[mask]
            auroc = _auroc(target[mask], counterfactual[mask, index])
            rows.append({"variant": variant, "group_id": group, "mean_shift": shift.mean(), "auroc": auroc})
        row[f"auroc_{variant}"] = _auroc(target, counterfactual[:, index])
    return row, pd.DataFrame(rows)


def counterfactual_attributes(attributes, race_column, standardize_age):
    """raceだけ、または年齢だけを差し替えた属性dictを、variant名→dictで返す。

    他の属性（sex・ethnicity・view）は実際の値のまま残し、1属性の差し替えが出力をどれだけ
    動かすかを見る。年齢は全員を同じ代表値にするので、群内の年齢差も消える。

    Args:
        attributes: 実際の属性dict。
        race_column: `categorical`の中のrace列の位置。
        standardize_age: 生の年齢→標準化値。
    Returns:
        dict: `RACE_CODES`・`COUNTERFACTUAL_AGES`の順のvariant名→属性dict。
    """
    variants = {}
    for label, code in RACE_CODES.items():
        categorical = attributes["categorical"].clone()
        categorical[:, race_column] = code
        missing = attributes["categorical_missing"].clone()
        missing[:, race_column] = False
        variants[label] = {**attributes, "categorical": categorical, "categorical_missing": missing}
    for label, age in COUNTERFACTUAL_AGES.items():
        continuous = torch.full_like(attributes["continuous"], standardize_age(age))
        missing = torch.zeros_like(attributes["continuous_missing"])
        variants[label] = {**attributes, "continuous": continuous, "continuous_missing": missing}
    return variants


def _modulated_forward(network, shared, condition):
    """共有済みの前段特徴から、変調stage以降とclassifierを実行する。"""
    x, start = shared
    for stage in list(STAGE_LAYERS)[start:]:
        x = network._forward_stage(stage, getattr(network.backbone, STAGE_LAYERS[stage]), x, condition)
    features = torch.flatten(network.backbone.avgpool(x), 1)
    if isinstance(network.fc, HyperLinearLayer):
        return network.fc(features, condition), features
    return network.fc(features), features


def _shared_prefix(network, image):
    """condition に依らない stem と未変調stageを1回だけ実行する。"""
    backbone = network.backbone
    x = backbone.maxpool(backbone.relu(backbone.bn1(backbone.conv1(image))))
    start = 0
    for stage, layer_name in STAGE_LAYERS.items():
        if stage in network.spatial_adapters:
            break
        x = getattr(backbone, layer_name)(x)
        start += 1
    return x, start


@torch.inference_mode()
def probe_batch(network, image, attributes, make_variants):
    """1 batchのΔW=0経路・実際の属性・反実仮想属性のlogitsと特徴変化を返す。

    ΔW=0経路は凍結backbone＋元classifierで、2-stageでは同seed M01と同一になる。

    Args:
        network: SpatialLoRAResNet。
        image: 画像batch。
        attributes: 実際の属性dict（device上）。
        make_variants: 属性dict→反実仮想variantのdict。
    Returns:
        dict: numpy配列。
    """
    base_features = network.backbone(image)
    classifier = network.fc.base_linear if isinstance(network.fc, HyperLinearLayer) else network.fc
    shared = _shared_prefix(network, image)
    logits, features = _modulated_forward(network, shared, network.metadata_encoder(attributes))
    counterfactual = []
    for variant in make_variants(attributes).values():
        counterfactual.append(_modulated_forward(network, shared, network.metadata_encoder(variant))[0])
    shift = (features - base_features).norm(dim=1) / base_features.norm(dim=1)
    cosine = torch.nn.functional.cosine_similarity(features, base_features, dim=1)
    return {
        "logits_base": classifier(base_features).float().cpu().numpy(),
        "logits_actual": logits.float().cpu().numpy(),
        "logits_counterfactual": torch.stack(counterfactual, dim=1).float().cpu().numpy(),
        "feature_shift": shift.float().cpu().numpy(),
        "feature_cosine": cosine.float().cpu().numpy(),
    }


def write_probe(run_dir, kinds, cache_dir, device):
    """test全件を1周し、checkpoint種別ごとにprobe cacheを書く。

    Args:
        run_dir: 2-stage Spatial LoRA run。
        kinds: `best`・`last`の列。
        cache_dir: 書き出し先。
        device: 推論device。
    Returns:
        list[Path]: 書き出したcache。
    """
    frame, datamodule = model_frame(run_dir, "test")
    names = OmegaConf.load(run_dir / "config.yaml").model.attribute_names
    if list(names.continuous) != ["age"]:
        raise ValueError("連続属性がage以外を含む")
    make_variants = partial(
        counterfactual_attributes,
        race_column=list(names.categorical).index("race"),
        standardize_age=age_standardizer(datamodule),
    )
    networks = {kind: load_spatial_network(run_dir, kind, device) for kind in kinds}
    parts = {kind: [] for kind in kinds}
    for image, attributes, _ in datamodule.evaluation_dataloader("test"):
        image = image.to(device, non_blocking=True)
        attributes = {key: value.to(device) for key, value in attributes.items() if key != "evaluation_categorical"}
        for kind, network in networks.items():
            parts[kind].append(probe_batch(network, image, attributes, make_variants))
    outputs = []
    for kind in kinds:
        arrays = {key: np.concatenate([part[key] for part in parts[kind]]) for key in parts[kind][0]}
        metadata = {
            "schema_version": PROBE_SCHEMA_VERSION,
            "run_id": run_dir.name,
            "checkpoint": checkpoint_file(run_dir, kind).name,
            "checkpoint_sha256": digest(checkpoint_file(run_dir, kind)),
            "variants": [*RACE_CODES, *COUNTERFACTUAL_AGES],
            "race_codes": RACE_CODES,
            "ages": COUNTERFACTUAL_AGES,
        }
        output = probe_cache_path(cache_dir, run_dir.name, kind)
        np.savez_compressed(output, image=frame["image"].to_numpy(dtype=str), metadata=json.dumps(metadata), **arrays)
        outputs.append(output)
    return outputs


def _parse_arguments():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--run-dir", type=Path, action="append", required=True)
    parser.add_argument("--checkpoint", choices=["best", "last"], action="append", required=True)
    parser.add_argument("--device", required=True, help="推論GPUを明示する（例: cuda:0）")
    return parser.parse_args()


def main():
    """指定runのprobe cacheを書き出す。"""
    arguments = _parse_arguments()
    cache_dir = Path(__file__).resolve().parent / "cache"
    cache_dir.mkdir(exist_ok=True)
    for run_dir in arguments.run_dir:
        for output in write_probe(run_dir.resolve(), arguments.checkpoint, cache_dir, torch.device(arguments.device)):
            print(output)


if __name__ == "__main__":
    main()
