"""セル内で一定の属性を入れた推論から、属性が画像の読み方を変えているかを集計する。

`counterfactual.py` の cache を読み、評価セル（HAM10000 は sex × 65 歳、CheXpert は race × 65 歳）ごとに
次を出す。属性が切片だけに効くモデルなら、どれも「差 0・順位相関 1」になる。

- `age_curve.csv`：入力年齢 t ごとのセル内 AUROC（seed 平均）
- `contrasts.csv`：セル内 AUROC の差と、患者（病変）単位の cluster bootstrap 95% 区間
  - `age_matched_minus_flipped`：セルの年齢群の代表値（45・75 歳）を入れたとき − 反対の年齢群の代表値
  - `cell_attr_matched_minus_other`：セルの sex（HAM）・race（CheXpert）を入れたとき − 他の値の平均
  区間は test の標本の揺れだけを表す。seed は 3 本の平均を取り、学習の揺れは区間に入れない
  （bootstrap の各回で同じ行を引いた 3 seed の差を平均する）。
- `rank_change.csv`：入力年齢 45 歳と 75 歳のスコアのセル内 Spearman 順位相関（run ごと）

入力: `cache/<run-id>_test_interaction.npz`、`data/<dataset>/splits*/test.csv`
出力: `results/{age_curve,contrasts,rank_change}.csv`

    uv run python -m analysis.attribute_interaction.summarize
"""

from __future__ import annotations

import json
import re

import numpy as np
import pandas as pd
import rootutils
from scipy.stats import rankdata, spearmanr

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from analysis.attribute_interaction.counterfactual import (  # noqa: E402
    AGE_BOUNDARY,
    AGE_GRID,
    AGE_OLD,
    AGE_YOUNG,
    PACKAGE,
    RACE_CODE,
    SEX_CODES,
)
from analysis.common.paths import REPOSITORY_ROOT as ROOT  # noqa: E402

SPLITS = {"ham10000": ROOT / "data/ham10000/splits/test.csv", "chexpert": ROOT / "data/chexpert/splits_race_known/test.csv"}
CELL_ATTRIBUTE = {"ham10000": "sex", "chexpert": "race_group"}
CELL_LABELS = {
    "sex": {0: "female", 1: "male"},
    "race_group": {0: "White", 1: "Asian", 2: "Black", 3: "Others"},
}
N_BOOTSTRAP = 1000
SEED = 0


def model_of(run_id: str) -> tuple[str, int]:
    """run-id から（モデル名, seed）を返す。"""
    seed = int(re.search(r"-s(\d+)-", run_id).group(1))
    if "-fc-" in run_id:
        return "spatial_lora_fc", seed
    return ("hyperadapt" if "hyperadapt" in run_id else "spatial_lora"), seed


def load_caches() -> dict[tuple[str, str], dict[int, pd.DataFrame]]:
    """(dataset, model) → {seed: 行 × 条件の logit 差} を返す。行は split CSV と image で照合する。"""
    out: dict[tuple[str, str], dict[int, pd.DataFrame]] = {}
    for path in sorted((PACKAGE / "cache").glob("*_test_interaction.npz")):
        with np.load(path, allow_pickle=False) as cached:
            meta = json.loads(str(cached["metadata"]))
            split = pd.read_csv(SPLITS[meta["dataset"]], usecols=["image"])
            if not (cached["image"] == split["image"].to_numpy(dtype=str)).all():
                raise ValueError(f"{path.name} の行が split CSV と対応していない")
            scores = pd.DataFrame(cached["scores"], columns=cached["variants"])
        model, seed = model_of(meta["run_id"])
        out.setdefault((meta["dataset"], model), {})[seed] = scores
    return out


def evaluation_rows(dataset: str) -> pd.DataFrame:
    """評価に使う行と、セル・cluster（患者・病変）の列を返す。属性の揃わない行は外す。"""
    frame = pd.read_csv(SPLITS[dataset])
    attribute = CELL_ATTRIBUTE[dataset]
    complete = (frame["age_missing"] == 0) & (frame[f"{attribute}_missing"] == 0)
    if dataset == "ham10000":
        cluster = frame["lesion_id"]
    else:
        cluster = frame["image"].str.extract(r"(patient\d+)", expand=False)
    rows = pd.DataFrame(
        {
            "row": np.arange(len(frame)),
            "target": frame["target"],
            "attribute": frame[attribute],
            "old": frame["age"] >= AGE_BOUNDARY,
            "cluster": cluster,
        }
    )[complete]
    rows["cell"] = rows["attribute"].map(CELL_LABELS[attribute]) + np.where(rows["old"], " ≥65", " <65")
    return rows


def auroc(target: np.ndarray, score: np.ndarray) -> float:
    """同順位を平均順位で扱う Mann–Whitney の AUROC。"""
    positive = target == 1
    n_pos, n_neg = positive.sum(), (~positive).sum()
    ranks = rankdata(score)
    return (ranks[positive].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)


def contrast_columns(dataset: str, cell: pd.DataFrame) -> dict[str, tuple[str, list[str]]]:
    """対比の名前 → （基準の条件, 引く側の条件の list）。引く側が複数なら平均を引く。"""
    old = bool(cell["old"].iloc[0])
    matched_age, flipped_age = (AGE_OLD, AGE_YOUNG) if old else (AGE_YOUNG, AGE_OLD)
    value = int(cell["attribute"].iloc[0])
    if dataset == "ham10000":
        matched, others = f"sex_{value}", [f"sex_{k}" for k in SEX_CODES if k != value]
    else:
        matched, others = f"race_{value}", [f"race_{k}" for k in RACE_CODE if k != value]
    return {
        "age_matched_minus_flipped": (f"age_{matched_age}", [f"age_{flipped_age}"]),
        "cell_attr_matched_minus_other": (matched, others),
    }


def _cluster_resamples(cell: pd.DataFrame, rng: np.random.Generator) -> list[np.ndarray]:
    """cluster 単位で復元抽出した、セル内の位置 index の list を返す。"""
    positions = pd.Series(np.arange(len(cell))).groupby(cell["cluster"].to_numpy()).apply(np.asarray).to_list()
    draws = []
    for _ in range(N_BOOTSTRAP):
        chosen = rng.integers(0, len(positions), len(positions))
        draws.append(np.concatenate([positions[i] for i in chosen]))
    return draws


def _difference(scores: dict[int, pd.DataFrame], rows: np.ndarray, target: np.ndarray, base: str, others: list[str]) -> float:
    """seed 平均の AUROC(base) − mean AUROC(others)。"""
    values = []
    for table in scores.values():
        reference = auroc(target, table[base].to_numpy()[rows])
        subtract = np.mean([auroc(target, table[name].to_numpy()[rows]) for name in others])
        values.append(reference - subtract)
    return float(np.mean(values))


def contrasts(dataset: str, model: str, scores: dict[int, pd.DataFrame], rows: pd.DataFrame) -> list[dict]:
    """セルごとの対比と bootstrap 区間。全セル平均（セルを等しく重み付け）も `all cells` として足す。"""
    rng = np.random.default_rng(SEED)
    records, replicates = [], {}
    for name, cell in rows.groupby("cell", sort=True):
        draws = _cluster_resamples(cell, rng)
        row_index, target = cell["row"].to_numpy(), cell["target"].to_numpy()
        for contrast, (base, others) in contrast_columns(dataset, cell).items():
            point = _difference(scores, row_index, target, base, others)
            boot = np.array(
                [_difference(scores, row_index[d], target[d], base, others) for d in draws if 0 < target[d].sum() < len(d)]
            )
            replicates.setdefault(contrast, []).append(boot[:N_BOOTSTRAP])
            records.append(_record(dataset, model, name, contrast, point, boot, cell))
    for contrast, boots in replicates.items():
        length = min(len(b) for b in boots)
        pooled = np.mean([b[:length] for b in boots], axis=0)
        point = np.mean([r["delta"] for r in records if r["contrast"] == contrast])
        records.append(_record(dataset, model, "all cells", contrast, point, pooled, rows))
    return records


def _record(dataset, model, cell, contrast, point, boot, members) -> dict:
    low, high = np.percentile(boot, [2.5, 97.5])
    return {
        "dataset": dataset,
        "model": model,
        "cell": cell,
        "contrast": contrast,
        "delta": point,
        "low": low,
        "high": high,
        "n_pos": int(members["target"].sum()),
        "n_neg": int((members["target"] == 0).sum()),
    }


def age_curve(dataset: str, model: str, scores: dict[int, pd.DataFrame], rows: pd.DataFrame) -> list[dict]:
    """入力年齢ごとのセル内 AUROC（seed 平均）。observed（属性をそのまま入れた）も参照として足す。"""
    records = []
    for name, cell in rows.groupby("cell", sort=True):
        index, target = cell["row"].to_numpy(), cell["target"].to_numpy()
        for variant in [*(f"age_{t}" for t in AGE_GRID), "observed"]:
            values = [auroc(target, table[variant].to_numpy()[index]) for table in scores.values()]
            age = np.nan if variant == "observed" else int(variant.split("_")[1])
            records.append({"dataset": dataset, "model": model, "cell": name, "variant": variant, "input_age": age,
                            "auroc": float(np.mean(values)), "auroc_sd_seed": float(np.std(values, ddof=1))})
    return records


def rank_change(dataset: str, model: str, scores: dict[int, pd.DataFrame], rows: pd.DataFrame) -> list[dict]:
    """入力年齢 45 歳と 75 歳のセル内スコアの Spearman 順位相関。切片だけなら 1。"""
    records = []
    for seed, table in scores.items():
        for name, cell in rows.groupby("cell", sort=True):
            index = cell["row"].to_numpy()
            rho = spearmanr(table[f"age_{AGE_YOUNG}"].to_numpy()[index], table[f"age_{AGE_OLD}"].to_numpy()[index])[0]
            records.append({"dataset": dataset, "model": model, "seed": seed, "cell": name, "spearman_45_75": rho})
    return records


def main() -> None:
    """全 cache を集計して results/ に書く。"""
    caches = load_caches()
    curve, contrast, ranks = [], [], []
    for (dataset, model), scores in sorted(caches.items()):
        rows = evaluation_rows(dataset)
        curve += age_curve(dataset, model, scores, rows)
        ranks += rank_change(dataset, model, scores, rows)
        contrast += contrasts(dataset, model, scores, rows)
        print(f"done: {dataset} {model} ({len(scores)} seeds)", flush=True)
    results = PACKAGE / "results"
    results.mkdir(exist_ok=True)
    pd.DataFrame(curve).to_csv(results / "age_curve.csv", index=False)
    pd.DataFrame(contrast).to_csv(results / "contrasts.csv", index=False)
    pd.DataFrame(ranks).to_csv(results / "rank_change.csv", index=False)


if __name__ == "__main__":
    main()
