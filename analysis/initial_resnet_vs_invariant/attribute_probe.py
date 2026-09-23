"""invariant 化で、属性が backbone 表現から読めなくなったのかを post-hoc probe で測る。

`reports/overall_comparison.md` は「性能は下がるのに gap は動かない」で終わっており、
adversary が実際に属性を消せているのか分かっていない。学習ログには属性予測損失が残っていない
（`metrics/fit.json` が持つのは `train/loss` の合計だけ）ので、**checkpoint を後から probe する**。

測るのは分類 head の直前の表現（2048 次元）で、`common/predictions.py --features` が書いた
cache を読む。backbone は凍結されており、ここで学習するのは probe の重みだけになる。

**手順は post-hoc attacker の慣行に合わせる**（Elazar & Goldberg 2018 ほか）。probe は
**train で学習し、val で止めどきを決め、test で報告する**。adversary が消したと主張したい側の
分析なので、**probe は強いほうへ倒す**。probe を val だけで学習すると 2048 次元に対して
22,901 行しか無く、probe が弱いせいで下がった分を「属性が消えた」と読みかねない。

probe を 2 つ置く（この package の判断）:

- **linear** — 表現から属性への線形層 1 枚。表現の線形分離性を測る。
- **mlp** — `hidden 256 + ReLU` の 2 層。学習時の adversary（`_AttributeAdversary`）と同じ構成で、
  **adversary が消し切れたかを、adversary と同じ容量で問い直す**。linear だけだと、
  非線形にしか残っていない情報を「消えた」と読んでしまう。

2 つは層の深さだけが違う同じ手順で学習する（同じ optimizer・同じ epoch の選び方）。
片方を `sklearn`、片方を SGD にすると、probe の差に最適化手続きの差が混ざる。

どちらも **class weight を balanced にする**。race のように 1 群が 7 割を占める属性では、
重みを付けないと多数派を答えるだけの probe になり、balanced accuracy が 1/k 付近に張り付いて
手法の差が見えなくなる。

属性は adversary の対象かどうかで分けて読む。**対象外の属性は対照になる。**

| 属性 | adversary | 読み方 |
| --- | --- | --- |
| sex / race / age / age_group_65 | 対象 | 下がっていれば adversary が効いている |
| ethnicity | 対象外 | ここまで下がるなら、属性固有ではなく表現全体が痩せている |
| frontal_lateral | 対象外 | 画像から直に読める撮影方向。probe 自体が機能していることの確認 |

**この表が言わないこと。** probe の AUROC が下がっても、下流の公平性が改善するとは限らない
（`reports/overall_comparison.md` の 2 節がその実例になる）。ここで分かるのは
「adversary が表現から属性を消せたか」までで、gap が動くかどうかは別の問いになる。
probe が「読んでいる」のか「覚えている」のかの切り分け（control task / selectivity）も
していない。同じ probe を両手法に当てているので手法間の差は読めるが、絶対値は読めない。

入力: `cache/<run-id>_<split>_features.npz`（train / val / test の 3 split）と split CSV
出力: `results/attribute_probe_<eval-split>.csv`、`results/attribute_probe_by_method_<eval-split>.csv`

使い方:
    uv run python analysis/common/predictions.py --study initial_resnet_vs_invariant \
      --split train --features --run-dir projects/hypernet_e2e/runs/<run-id>   # val / test も同様
    uv run python analysis/initial_resnet_vs_invariant/attribute_probe.py
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import rootutils
import torch
import torch.nn as nn
from sklearn.metrics import balanced_accuracy_score, mean_absolute_error, r2_score, roc_auc_score

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from analysis.common.paths import split_csv  # noqa: E402
from analysis.common.predictions import features_cache_path, load_features  # noqa: E402
from analysis.initial_resnet_vs_invariant.groups import RUNS, demographics_of  # noqa: E402

PACKAGE = Path(__file__).parent
CACHE, RESULTS = PACKAGE / "cache", PACKAGE / "results"

LINEAR, MLP = "linear", "mlp"
# 学習時の adversary と同じ容量にする（`_AttributeAdversary` は hidden_dim=256 の 2 層）。
MLP_HIDDEN_DIM = 256
MAX_EPOCHS, PATIENCE, BATCH_SIZE = 40, 5, 512
LEARNING_RATE, WEIGHT_DECAY = 1e-3, 1e-4
# probe の初期値は 6 run で共通にする。run 間の差に probe の引き当てを混ぜない。
PROBE_SEED = 0
# 撮影方向は split CSV の符号。元の文字列は `data/chexpert/df_chexpert_plus_240401.csv` にある。
FRONTAL_LATERAL_LABELS = {0: "Frontal", 1: "Lateral"}


@dataclass(frozen=True)
class Target:
    """probe が当てにいく属性 1 つ分の定義。

    Attributes:
        name: `probe_targets` が返す列名
        kind: `categorical`（分類）か `continuous`（回帰）か
        adversarial: 学習時の adversary がこの属性を消しにいっていたか
    """

    name: str
    kind: str
    adversarial: bool


# adversary が消しにいくのは sex / race（categorical）と age（continuous）の 3 つ。
# age_group_65 はその age から作る派生量で、他の属性と同じ物差し（balanced accuracy）で読むために置く。
TARGETS = [
    Target("sex", "categorical", adversarial=True),
    Target("race", "categorical", adversarial=True),
    Target("age_group_65", "categorical", adversarial=True),
    Target("age", "continuous", adversarial=True),
    Target("ethnicity", "categorical", adversarial=False),
    Target("frontal_lateral", "categorical", adversarial=False),
]


def probe_targets(split: str) -> pd.DataFrame:
    """probe が当てる属性を、1 属性 1 列の表にして返す。

    sex / race / ethnicity / age_group_65 は分類性能側の表と同じ定義を使う（`demographics_of`）。
    **race が White / Asian / Black の 3 群に絞られるのもそこから引き継ぐ**ので、学習時の
    adversary が見ていた 6 カテゴリより易しい問題になる。手法間の比較は同じ問題どうしで行うため
    これで読めるが、絶対値を adversary の損失と並べることはできない。

    index は split CSV の行番号のまま返す。特徴量 cache は CSV と同じ順で並んでいるので、
    この index がそのまま cache の添字になる。

    Args:
        split: `train` / `val` / `test`

    Returns:
        pd.DataFrame: `TARGETS` の列。categorical は表示用ラベル、age は年齢、欠損は NaN
    """
    frame = pd.read_csv(split_csv("chexpert", split))
    targets = demographics_of(split)
    frontal_lateral = frame["frontal_lateral"].map(FRONTAL_LATERAL_LABELS)
    targets["frontal_lateral"] = frontal_lateral.mask(frame["frontal_lateral_missing"].astype(bool))
    targets["age"] = frame["age"].mask(frame["age_missing"].astype(bool) | frame["age"].isna())
    return targets


def label_codes(targets: dict[str, pd.DataFrame], target: Target) -> dict[str, int]:
    """categorical 属性のラベルを、split をまたいで同じ整数に対応づける。

    split ごとに番号を振り直すと、ある split にだけ現れないカテゴリがあったとき**同じ名前が
    別の番号になる**。番号がずれても probe は学習でき、指標も出てしまう。

    Args:
        targets: 役割名 → `probe_targets` の表
        target: 対象の属性

    Returns:
        dict[str, int]: ラベル名 → 0 始まりの整数
    """
    names: set[str] = set()
    for frame in targets.values():
        names.update(frame[target.name].dropna().unique())
    return {name: index for index, name in enumerate(sorted(names))}


def observed(frame: pd.DataFrame, target: Target, codes: dict[str, int] | None) -> tuple[np.ndarray, np.ndarray]:
    """その属性を観測できている行の添字と、対応するラベルを返す。

    属性ごとに母集団が変わる（race は欠損に加えて、絞り込んだ 3 カテゴリ以外も落ちる）。
    落とす場所を 1 つにしておかないと、split ごとに別の母集団を使いかねない。

    Args:
        frame: `probe_targets` が返した表
        target: 対象の属性
        codes: categorical のラベル対応。連続量では `None`

    Returns:
        tuple[np.ndarray, np.ndarray]: split CSV の行番号と、整数ラベルまたは年齢
    """
    column = frame[target.name].dropna()
    index = column.index.to_numpy()
    if codes is None:
        return index, column.to_numpy(dtype=np.float32)
    return index, np.array([codes[name] for name in column], dtype=np.int64)


def balanced_class_weight(labels: np.ndarray, num_classes: int) -> torch.Tensor:
    """`class_weight="balanced"` と同じ重みを返す。

    Args:
        labels: 0 始まりの整数ラベル
        num_classes: クラス数

    Returns:
        torch.Tensor: `[num_classes]` の重み
    """
    counts = np.bincount(labels, minlength=num_classes).astype(np.float64)
    return torch.tensor(len(labels) / (num_classes * counts), dtype=torch.float32)


def standardized(features: np.ndarray, index: np.ndarray, moments: tuple[np.ndarray, np.ndarray]) -> torch.Tensor:
    """選んだ行を、fit split の平均・標準偏差で標準化した tensor にする。

    平均と標準偏差は **fit split だけ**から作る。val / test の統計を混ぜると、表現の尺度を
    通じて評価側を覗くことになる。

    Args:
        features: `[n_rows, feature_dim]` の表現
        index: 使う行の添字
        moments: fit split の平均と標準偏差

    Returns:
        torch.Tensor: `[len(index), feature_dim]` の float32
    """
    mean, scale = moments
    return torch.from_numpy((features[index] - mean) / scale)


def build_probe(probe: str, feature_dim: int, output_dim: int) -> nn.Module:
    """probe の形を作る。`linear` は線形層 1 枚、`mlp` は adversary と同じ 2 層。

    Args:
        probe: `LINEAR` または `MLP`
        feature_dim: 表現の次元
        output_dim: 出力次元。回帰は 1

    Returns:
        nn.Module: 未学習の probe
    """
    torch.manual_seed(PROBE_SEED)
    if probe == LINEAR:
        return nn.Linear(feature_dim, output_dim)
    return nn.Sequential(nn.Linear(feature_dim, MLP_HIDDEN_DIM), nn.ReLU(), nn.Linear(MLP_HIDDEN_DIM, output_dim))


def fit_probe(
    model: nn.Module,
    fit_data: tuple[torch.Tensor, torch.Tensor],
    select_data: tuple[torch.Tensor, torch.Tensor],
    loss_fn: nn.Module,
    device: torch.device,
) -> nn.Module:
    """probe を fit split で学習し、select split の loss が最小だった epoch の重みを返す。

    epoch 数を固定すると、属性ごとに要る学習量が違うぶんが手法の差に混ざる。止めどきは
    backbone の checkpoint 選択に使った val で決め、**test は最後まで触らない**。

    Args:
        model: `build_probe` が返した probe
        fit_data: 学習に使う標準化済み表現とラベル
        select_data: 止めどきを決める標準化済み表現とラベル
        loss_fn: 分類なら重み付き cross entropy、回帰なら MSE
        device: 学習に使う device

    Returns:
        nn.Module: select loss が最小だった時点の probe（eval mode）
    """
    generator = torch.Generator().manual_seed(PROBE_SEED)
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    fit_x, fit_y = fit_data[0].to(device), fit_data[1].to(device)
    select_x, select_y = select_data[0].to(device), select_data[1].to(device)

    best_loss, best_state, waited = float("inf"), None, 0
    for _ in range(MAX_EPOCHS):
        model.train()
        order = torch.randperm(len(fit_x), generator=generator).to(device)
        for start in range(0, len(order), BATCH_SIZE):
            batch = order[start : start + BATCH_SIZE]
            optimizer.zero_grad()
            loss_fn(model(fit_x[batch]), fit_y[batch]).backward()
            optimizer.step()
        model.eval()
        with torch.inference_mode():
            epoch_loss = float(loss_fn(model(select_x), select_y))
        if epoch_loss < best_loss - 1e-5:
            best_loss, waited = epoch_loss, 0
            best_state = {key: value.detach().clone() for key, value in model.state_dict().items()}
        else:
            waited += 1
            if waited >= PATIENCE:
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    return model.eval()


def probe_outputs(model: nn.Module, features: torch.Tensor, device: torch.device) -> np.ndarray:
    """学習済み probe を eval split に通した出力を返す。

    Args:
        model: `fit_probe` が返した probe
        features: 標準化済みの `[n_rows, feature_dim]`
        device: 推論に使う device

    Returns:
        np.ndarray: `[n_rows, output_dim]` の出力（分類なら logits）
    """
    parts = []
    with torch.inference_mode():
        for start in range(0, len(features), 8192):
            parts.append(model(features[start : start + 8192].to(device)).float().cpu().numpy())
    return np.concatenate(parts)


def classification_scores(
    probe: str,
    data: dict[str, tuple[torch.Tensor, np.ndarray]],
    num_classes: int,
    device: torch.device,
) -> dict[str, float]:
    """1 つの (属性, probe) について、eval split の balanced accuracy と AUROC を返す。

    balanced accuracy と AUROC を並べるのは、両者で「読める」の意味が違うため。前者は
    ある動作点での当たり方、後者は順序だけを見る。閾値の都合で片方だけ動くことがある。

    Args:
        probe: `LINEAR` または `MLP`
        data: `fit` / `select` / `eval` ごとの標準化済み表現と整数ラベル
        num_classes: クラス数
        device: 学習・推論に使う device

    Returns:
        dict[str, float]: `balanced accuracy` と `AUROC`（多クラスは ovr macro）
    """
    weight = balanced_class_weight(data["fit"][1], num_classes).to(device)
    tensors = {role: (features, torch.from_numpy(labels)) for role, (features, labels) in data.items()}
    model = build_probe(probe, data["fit"][0].shape[1], num_classes)
    model = fit_probe(model, tensors["fit"], tensors["select"], nn.CrossEntropyLoss(weight=weight), device)

    logits = probe_outputs(model, data["eval"][0], device)
    scores = torch.softmax(torch.from_numpy(logits), dim=1).numpy()
    eval_y = data["eval"][1]
    if num_classes == 2:
        auroc = roc_auc_score(eval_y, scores[:, 1])
    else:
        auroc = roc_auc_score(eval_y, scores, multi_class="ovr", average="macro")
    return {"balanced accuracy": balanced_accuracy_score(eval_y, scores.argmax(axis=1)), "AUROC": auroc}


def regression_scores(
    probe: str,
    data: dict[str, tuple[torch.Tensor, np.ndarray]],
    device: torch.device,
) -> dict[str, float]:
    """連続属性（age）について、eval split の R2 と MAE を返す。

    adversary は age を MSE で消しにいくので、こちらも回帰のまま測る。年齢をそのまま回帰すると
    平均 60 歳へ寄るまでに epoch を使い切るので、**fit split の平均・標準偏差で目盛を揃えて
    学習し、予測を歳へ戻す**（学習側の `standardize_continuous` と同じ扱い）。

    MAE を併記するのは、R2 が「元の分散のどれだけを説明できたか」しか言わず、**何歳ぶん
    当たるのか**が読めないため。

    Args:
        probe: `LINEAR` または `MLP`
        data: `fit` / `select` / `eval` ごとの標準化済み表現と年齢（歳）
        device: 学習・推論に使う device

    Returns:
        dict[str, float]: `R2` と `MAE (years)`
    """
    center, scale = float(data["fit"][1].mean()), float(data["fit"][1].std())
    tensors = {}
    for role, (features, values) in data.items():
        tensors[role] = (features, torch.from_numpy((values - center) / scale).float().unsqueeze(1))
    model = build_probe(probe, data["fit"][0].shape[1], 1)
    model = fit_probe(model, tensors["fit"], tensors["select"], nn.MSELoss(), device)

    predictions = probe_outputs(model, data["eval"][0], device)[:, 0] * scale + center
    eval_y = data["eval"][1]
    return {"R2": r2_score(eval_y, predictions), "MAE (years)": mean_absolute_error(eval_y, predictions)}


def chance_level(target: Target, metric: str, num_classes: int, fit_y: np.ndarray, eval_y: np.ndarray) -> float:
    """その指標で「何も読めていない」ときの値を返す。

    probe の絶対値だけでは、下がったのが「消えたから」なのか「元から読めていないから」なのかが
    分からない。**指標ごとの下限を同じ行に置く。**

    Args:
        target: 対象の属性
        metric: 指標名
        num_classes: 分類のクラス数。回帰では無視する
        fit_y: probe の学習に使ったラベル（回帰の基準値を作る）
        eval_y: 評価側のラベル

    Returns:
        float: 当てずっぽうの値。`MAE (years)` は fit split の中央値を常に答えた場合の MAE
    """
    if target.kind == "continuous":
        return 0.0 if metric == "R2" else float(np.abs(eval_y - np.median(fit_y)).mean())
    return 0.5 if metric == "AUROC" else 1.0 / num_classes


def target_rows(
    target: Target,
    features: dict[str, np.ndarray],
    targets: dict[str, pd.DataFrame],
    device: torch.device,
) -> list[dict[str, Any]]:
    """1 run の 1 属性について、probe × 指標の行を作る。

    Args:
        target: 対象の属性
        features: 役割名 → `[n_rows, feature_dim]` の表現
        targets: 役割名 → `probe_targets` の表
        device: 学習・推論に使う device

    Returns:
        list[dict[str, Any]]: probe と指標の行（手法名・seed はまだ入らない）
    """
    codes = label_codes(targets, target) if target.kind == "categorical" else None
    rows_by_role = {role: observed(frame, target, codes) for role, frame in targets.items()}
    fit_features = features["fit"][rows_by_role["fit"][0]]
    # 定数次元（死んだ ReLU）で 0 除算しないよう、標準偏差に下駄を履かせる。
    moments = (fit_features.mean(axis=0), fit_features.std(axis=0) + 1e-6)
    data = {}
    for role, (index, labels) in rows_by_role.items():
        data[role] = (standardized(features[role], index, moments), labels)

    num_classes = len(codes) if codes is not None else 0
    rows: list[dict[str, Any]] = []
    for probe in (LINEAR, MLP):
        if codes is None:
            scores = regression_scores(probe, data, device)
        else:
            scores = classification_scores(probe, data, num_classes, device)
        for metric, value in scores.items():
            rows.append(
                {
                    "attribute": target.name,
                    "adversary": target.adversarial,
                    "probe": probe,
                    "n_fit": len(data["fit"][1]),
                    "n_eval": len(data["eval"][1]),
                    "num_classes": num_classes,
                    "metric": metric,
                    "value": value,
                    "chance": chance_level(target, metric, num_classes, data["fit"][1], data["eval"][1]),
                }
            )
    return rows


def run_rows(
    model: str,
    seed: int,
    run_id: str,
    features: dict[str, np.ndarray],
    targets: dict[str, pd.DataFrame],
    device: torch.device,
) -> list[dict[str, Any]]:
    """1 run 分を、属性 × probe × 指標の行に展開する。

    Args:
        model: 表に出す手法名
        seed: その run の seed
        run_id: run ID
        features: 役割名 → 表現
        targets: 役割名 → `probe_targets` の表
        device: 学習・推論に使う device

    Returns:
        list[dict[str, Any]]: 手法・seed を付けた行
    """
    rows: list[dict[str, Any]] = []
    for target in TARGETS:
        for row in target_rows(target, features, targets, device):
            rows.append({"model": model, "seed": seed, "run_id": run_id, **row})
    return rows


def across_seeds(frame: pd.DataFrame) -> pd.DataFrame:
    """seed 方向へ平均と標準偏差に畳む。

    SD は標本 SD（`ddof=1`）で、n=3 なので自由度は 2 しかない。分類性能・公平性の表と同じく、
    **揺れの目安であって検定ではない。**

    Args:
        frame: per-seed の probe 表

    Returns:
        pd.DataFrame: `model` / `attribute` / `adversary` / `probe` / `metric` と mean / std / chance
    """
    keys = ["model", "attribute", "adversary", "probe", "metric"]
    grouped = frame.groupby(keys, sort=False)
    summary = pd.DataFrame(
        {
            "mean": grouped["value"].mean(),
            "std": grouped["value"].std(ddof=1),
            "n_seeds": grouped["value"].count(),
            "chance": grouped["chance"].first(),
        }
    )
    return summary.reset_index()


def load_run_features(run_id: str, splits: dict[str, str]) -> dict[str, np.ndarray]:
    """1 run 分の表現を、`fit` / `select` / `eval` の役割名で読む。

    Args:
        run_id: run ID
        splits: 役割名 → split 名

    Returns:
        dict[str, np.ndarray]: 役割ごとの `[n_rows, feature_dim]`
    """
    features = {}
    for role, split in splits.items():
        images = pd.read_csv(split_csv("chexpert", split))["image"].to_numpy(dtype=str)
        features[role] = load_features(features_cache_path(CACHE, run_id, split), images=images)["features"]
    return features


def main() -> None:
    """6 run の属性 probe と、その seed 集約を `results/` へ書く。

    Args:
        なし

    Returns:
        None
    """
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    choices = ("train", "val", "test")
    parser.add_argument("--fit-split", default="train", choices=choices, help="probe の重みを学習する split")
    parser.add_argument("--select-split", default="val", choices=choices, help="probe を止める epoch を決める split")
    parser.add_argument("--eval-split", default="test", choices=choices, help="probe を評価する split")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    splits = {"fit": args.fit_split, "select": args.select_split, "eval": args.eval_split}
    if len(set(splits.values())) != len(splits):
        raise ValueError("fit / select / eval は別の split にする（同じにすると probe の記憶量を測る）")

    device = torch.device(args.device)
    targets = {role: probe_targets(split) for role, split in splits.items()}
    RESULTS.mkdir(exist_ok=True)

    rows: list[dict[str, Any]] = []
    for model, seed, run_id in RUNS:
        features = load_run_features(run_id, splits)
        rows.extend(run_rows(model, seed, run_id, features, targets, device))
        print(f"probed: {run_id}")

    per_seed = pd.DataFrame(rows)
    per_seed.to_csv(RESULTS / f"attribute_probe_{args.eval_split}.csv", index=False)
    across_seeds(per_seed).to_csv(RESULTS / f"attribute_probe_by_method_{args.eval_split}.csv", index=False)
    print(f"{len(RUNS)} runs / {len(per_seed)} rows -> {RESULTS}")


if __name__ == "__main__":
    main()
