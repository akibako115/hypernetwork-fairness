"""adversary_strength の checkpoint 表現へ post-hoc attribute probe を適用する。

probe は train で fit、val で early stopping、test で評価する。linear probe と、学習時の
adversary と同じ hidden 256 の MLP probe を、各 run・各属性へ独立に適用する。

使い方:
    uv run python analysis/adversary_strength/attribute_probe.py
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
import rootutils
import torch
import torch.nn as nn
from sklearn.metrics import balanced_accuracy_score, mean_absolute_error, r2_score, roc_auc_score

ROOT = rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from analysis.common.paths import split_csv  # noqa: E402
from analysis.common.predictions import features_cache_path, load_features  # noqa: E402

PACKAGE = ROOT / "analysis" / "adversary_strength"
CACHE, RESULTS = PACKAGE / "cache", PACKAGE / "results"
LINEAR, MLP = "linear", "mlp"
PROBES = (MLP,)
RUNS = [
    ("ResNet", 43, "20260922T063725Z-resnet-chexpert-s43-efcd"),
    ("attribute-invariant λ=0.1", 43, "20260922T063725Z-resnet-chexpert-attribute-invariant-s43-546c"),
    ("attribute-invariant λ=1", 43, "20260922T111111Z-resnet-chexpert-attribute-invariant-s43-f5cb"),
    ("attribute-invariant λ=3", 43, "20260922T133206Z-resnet-chexpert-attribute-invariant-s43-2d1b"),
    ("attribute-invariant λ=10", 43, "20260922T155241Z-resnet-chexpert-attribute-invariant-s43-78d1"),
]
MLP_HIDDEN_DIM = 256
MAX_EPOCHS, PATIENCE, BATCH_SIZE = 12, 3, 8192
LEARNING_RATE, WEIGHT_DECAY, PROBE_SEED = 1e-3, 1e-4, 0


@dataclass(frozen=True)
class Target:
    name: str
    kind: str
    adversarial: bool


TARGETS = [
    Target("sex", "categorical", True),
    Target("race", "categorical", True),
    Target("age_group_65", "categorical", True),
    Target("age", "continuous", True),
]


def target_frame(split: str) -> pd.DataFrame:
    """split CSV から probe 対象属性を、この package の定義で作る。"""
    frame = pd.read_csv(split_csv("chexpert", split))
    targets = pd.DataFrame(
        {
            "sex": frame["sex"].map({0: "Male", 1: "Female"}),
            "race": frame["race"].map({0: "White", 2: "Asian", 3: "Black"}),
            "age_group_65": (frame["age"] >= 65).map({False: "<65", True: ">=65"}),
            "ethnicity": frame["ethnicity"].map({0: "Non-Hispanic", 1: "Hispanic"}),
            "frontal_lateral": frame["frontal_lateral"].map({0: "Frontal", 1: "Lateral"}),
            "age": frame["age"],
        }
    )
    missing = pd.DataFrame(
        {
            "sex": frame["sex_missing"].astype(bool),
            "race": frame["race_missing"].astype(bool),
            "age_group_65": frame["age_missing"].astype(bool) | frame["age"].isna(),
            "ethnicity": frame["ethnicity_missing"].astype(bool),
            "frontal_lateral": frame["frontal_lateral_missing"].astype(bool),
            "age": frame["age_missing"].astype(bool) | frame["age"].isna(),
        }
    )
    return targets.mask(missing)


def observed(frame: pd.DataFrame, target: Target) -> tuple[np.ndarray, pd.Series]:
    """属性が観測できる行番号と値を返す。"""
    column = frame[target.name].dropna()
    return column.index.to_numpy(), column


def encode_labels(target: Target, frames: dict[str, pd.DataFrame]) -> tuple[dict[str, np.ndarray], int | None]:
    """categorical target を split 間で共通の整数へ変換する。"""
    observed_values = {role: observed(frame, target)[1] for role, frame in frames.items()}
    if target.kind == "continuous":
        return {role: values.to_numpy(dtype=np.float32) for role, values in observed_values.items()}, None
    names = sorted(set().union(*(set(values.unique()) for values in observed_values.values())))
    codes = {name: index for index, name in enumerate(names)}
    return {
        role: np.array([codes[name] for name in values], dtype=np.int64) for role, values in observed_values.items()
    }, len(codes)


def build_probe(kind: str, feature_dim: int, output_dim: int) -> nn.Module:
    """linear または hidden 256 の MLP を構築する。"""
    torch.manual_seed(PROBE_SEED)
    if kind == LINEAR:
        return nn.Linear(feature_dim, output_dim)
    return nn.Sequential(nn.Linear(feature_dim, MLP_HIDDEN_DIM), nn.ReLU(), nn.Linear(MLP_HIDDEN_DIM, output_dim))


def fit_probe(
    model: nn.Module,
    fit_x: torch.Tensor,
    fit_y: torch.Tensor,
    select_x: torch.Tensor,
    select_y: torch.Tensor,
    loss_fn: nn.Module,
    device: torch.device,
) -> nn.Module:
    """fit split で学習し、select loss が最小の重みを返す。"""
    generator = torch.Generator(device=device).manual_seed(PROBE_SEED)
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    fit_x, fit_y = fit_x.to(device), fit_y.to(device)
    select_x, select_y = select_x.to(device), select_y.to(device)
    best_loss, best_state, waited = float("inf"), None, 0
    for _ in range(MAX_EPOCHS):
        model.train()
        order = torch.randperm(len(fit_x), generator=generator, device=device)
        for start in range(0, len(order), BATCH_SIZE):
            batch = order[start : start + BATCH_SIZE]
            optimizer.zero_grad()
            loss_fn(model(fit_x[batch]), fit_y[batch]).backward()
            optimizer.step()
        model.eval()
        with torch.inference_mode():
            select_loss = loss_fn(model(select_x), select_y).item()
        if select_loss < best_loss - 1e-5:
            best_loss, waited = select_loss, 0
            best_state = {key: value.detach().clone() for key, value in model.state_dict().items()}
        else:
            waited += 1
            if waited >= PATIENCE:
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    return model.eval()


def outputs(model: nn.Module, features: torch.Tensor, device: torch.device) -> np.ndarray:
    """probe output を batch ごとに CPU へ返す。"""
    parts = []
    with torch.inference_mode():
        for start in range(0, len(features), 8192):
            parts.append(model(features[start : start + 8192].to(device)).float().cpu().numpy())
    return np.concatenate(parts)


def classification_scores(
    kind: str, data: dict[str, tuple[torch.Tensor, np.ndarray]], classes: int, device: torch.device
) -> dict[str, float]:
    """categorical target の test balanced accuracy と AUROC を返す。"""
    fit_labels = data["fit"][1]
    counts = np.bincount(fit_labels, minlength=classes).astype(np.float64)
    weight = torch.tensor(len(fit_labels) / (classes * counts), dtype=torch.float32, device=device)
    tensors = {role: (features, torch.from_numpy(labels)) for role, (features, labels) in data.items()}
    model = fit_probe(
        build_probe(kind, data["fit"][0].shape[1], classes),
        *tensors["fit"],
        tensors["select"][0],
        tensors["select"][1],
        nn.CrossEntropyLoss(weight=weight),
        device,
    )
    scores = torch.softmax(torch.from_numpy(outputs(model, data["eval"][0], device)), dim=1).numpy()
    labels = data["eval"][1]
    auroc = (
        roc_auc_score(labels, scores[:, 1])
        if classes == 2
        else roc_auc_score(labels, scores, multi_class="ovr", average="macro")
    )
    return {"balanced accuracy": balanced_accuracy_score(labels, scores.argmax(axis=1)), "AUROC": auroc}


def regression_scores(
    kind: str, data: dict[str, tuple[torch.Tensor, np.ndarray]], device: torch.device
) -> dict[str, float]:
    """continuous age target の test R2 と MAE を返す。"""
    center, scale = float(data["fit"][1].mean()), float(data["fit"][1].std())
    tensors = {
        role: (features, torch.from_numpy(((values - center) / scale).astype(np.float32)).unsqueeze(1))
        for role, (features, values) in data.items()
    }
    model = fit_probe(
        build_probe(kind, data["fit"][0].shape[1], 1),
        *tensors["fit"],
        tensors["select"][0],
        tensors["select"][1],
        nn.MSELoss(),
        device,
    )
    prediction = outputs(model, data["eval"][0], device)[:, 0] * scale + center
    actual = data["eval"][1]
    return {"R2": r2_score(actual, prediction), "MAE (years)": mean_absolute_error(actual, prediction)}


def probe_run(
    run_id: str,
    frames: dict[str, pd.DataFrame],
    splits: dict[str, str],
    device: torch.device,
    probe_kinds: tuple[str, ...],
) -> list[dict[str, Any]]:
    """1 run の全 target × probe × metric を計算する。"""
    features = {}
    for role, split in splits.items():
        images = pd.read_csv(split_csv("chexpert", split))["image"].to_numpy(dtype=str)
        features[role] = load_features(features_cache_path(CACHE, run_id, split), images)["features"]
    moments = (features["fit"].mean(axis=0), features["fit"].std(axis=0) + 1e-6)
    rows = []
    for target in TARGETS:
        indices = {role: observed(frame, target)[0] for role, frame in frames.items()}
        labels, classes = encode_labels(target, frames)
        data = {
            role: (torch.from_numpy((features[role][indices[role]] - moments[0]) / moments[1]), labels[role])
            for role in splits
        }
        for kind in probe_kinds:
            scores = (
                regression_scores(kind, data, device)
                if classes is None
                else classification_scores(kind, data, classes, device)
            )
            for metric, value in scores.items():
                fit_labels = data["fit"][1]
                if classes is None:
                    chance = 0.0 if metric == "R2" else float(np.abs(data["eval"][1] - np.median(fit_labels)).mean())
                else:
                    chance = 0.5 if metric == "AUROC" else 1.0 / classes
                rows.append(
                    {
                        "attribute": target.name,
                        "adversary": target.adversarial,
                        "probe": kind,
                        "n_fit": len(fit_labels),
                        "n_eval": len(data["eval"][1]),
                        "num_classes": classes or 0,
                        "metric": metric,
                        "value": value,
                        "chance": chance,
                    }
                )
    return rows


def main() -> None:
    """train/val/test の特徴量 cache を読み、probe の per-run 表を保存する。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fit-split", default="train", choices=("train", "val", "test"))
    parser.add_argument("--select-split", default="val", choices=("train", "val", "test"))
    parser.add_argument("--eval-split", default="test", choices=("train", "val", "test"))
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--probe", default="linear", choices=(LINEAR, MLP, "both"))
    args = parser.parse_args()
    splits = {"fit": args.fit_split, "select": args.select_split, "eval": args.eval_split}
    if len(set(splits.values())) != 3:
        raise ValueError("fit / select / eval は別の split にする")
    device = torch.device(args.device)
    frames = {role: target_frame(split) for role, split in splits.items()}
    probe_kinds = (LINEAR, MLP) if args.probe == "both" else (args.probe,)
    RESULTS.mkdir(parents=True, exist_ok=True)
    rows = []
    for model, seed, run_id in RUNS:
        rows.extend(
            {"model": model, "seed": seed, "run_id": run_id, **row}
            for row in probe_run(run_id, frames, splits, device, probe_kinds)
        )
        print(f"probed: {run_id}")
    per_run = pd.DataFrame(rows)
    prefix = "attribute_probe" if args.probe == "both" else f"attribute_probe_{args.probe}"
    per_run.to_csv(RESULTS / f"{prefix}_{args.eval_split}.csv", index=False)
    keys = ["model", "attribute", "adversary", "probe", "metric"]
    grouped = per_run.groupby(keys, sort=False)
    summary = pd.DataFrame(
        {
            "mean": grouped["value"].mean(),
            "std": grouped["value"].std(ddof=1),
            "n_seeds": grouped["value"].count(),
            "chance": grouped["chance"].first(),
        }
    ).reset_index()
    summary.to_csv(RESULTS / f"{prefix}_by_method_{args.eval_split}.csv", index=False)
    print(f"{len(RUNS)} runs / {len(per_run)} rows -> {RESULTS}")


if __name__ == "__main__":
    main()
