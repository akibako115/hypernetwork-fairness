"""HAM10000 でだけ属性条件付きモデル（Spatial LoRA・HyperAdapt）が ResNet-50 より上がる理由を、
3 データセットで同じ物差しで測る。

測るもの（すべて既存の予測 cache から。推論しない）:
1. `gains.csv`：val・test の AUROC と、対照との差（test は患者単位の bootstrap 区間）。const（条件を定数にした
   Spatial LoRA）を挟み、「構成の効果」（const − ResNet）と「属性の効果」（meta − const）に分ける
2. `attribute_information.csv`：属性だけでの予測力（train で学習）と、出力に属性を後から足したときの上がり幅
   （val で学習・test 上の cross-fit の 2 通り）。後者が「属性が画像にない情報を運ぶ度合い」の目安
3. `per_attribute.csv`：ResNet の出力に属性を 1 つずつ足したときの上がり幅（test 上の cross-fit）
4. `within_between.csv`：AUROC を、属性の事前確率が近い組・遠い組に分けた一致率
5. `readability.csv`：ResNet の特徴から属性を当てる精度（val で学習、test で評価）。属性が画像から読めるか
6. `ham_pairs.csv`・`ham_age_coupling.csv`：HAM の dx の組ごとの AUROC と、良性内のスコアと年齢の順位相関。
   `meta_fixed` は属性条件付きモデルに test 時だけ全行同じ属性を入れたもの
   （`ham10000_architecture_erm/counterfactual.py`）

入力: 各 package の `cache/<run-id>_{val,test}[_features|_cf-constant_typical].npz`、`data/<dataset>/<splits>/*.csv`
出力: `analysis/dataset_gain_sources/results/*.csv`

    uv run python -m analysis.dataset_gain_sources.gain_sources
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import rootutils
from scipy.stats import spearmanr
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import r2_score, roc_auc_score

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from analysis.dataset_gain_sources.data import (  # noqa: E402
    DATASETS,
    PACKAGE,
    AttributeDesign,
    Dataset,
    cache_files,
    evaluation_rows,
    load_evaluation,
    load_logits,
    patients,
    read_split,
)
from analysis.dataset_gain_sources.metrics import boot_delta, cv_auc, standardize, within_between  # noqa: E402

RESULTS = PACKAGE / "results"
CONTRASTS = [
    ("meta", "resnet"),
    ("hyperadapt", "resnet"),
    ("const", "resnet"),
    ("meta", "const"),
    ("fc", "const"),
    ("random8", "const"),
]
HAM_DX = {1: "bcc", 2: "bkl", 4: "mel", 5: "nv"}


def seed_mean_auc(y: np.ndarray, scores: list[np.ndarray]) -> float:
    return float(np.mean([roc_auc_score(y, s) for s in scores]))


def gains(dataset: Dataset) -> pd.DataFrame:
    """val・test の AUROC の差。test だけ bootstrap 区間を付ける。

    val は checkpoint の選択に使った split なので参考値。
    """
    rows = []
    for split in ("val", "test"):
        frame, scores = load_evaluation(dataset, split)
        y, groups = frame.target.to_numpy(), patients(dataset, frame)
        for a, b in CONTRASTS:
            if a not in scores or b not in scores:
                continue
            row = {"dataset": dataset.name, "split": split, "contrast": f"{a} − {b}"}
            if split == "test":
                row.update(boot_delta(scores[a], scores[b], y, groups))
            else:
                row["delta"] = seed_mean_auc(y, scores[a]) - seed_mean_auc(y, scores[b])
            rows.append(row)
        for model, s in scores.items():
            rows.append({"dataset": dataset.name, "split": split, "contrast": model, "delta": seed_mean_auc(y, s)})
    return pd.DataFrame(rows)


def late_fusion_valfit(val: tuple, test: tuple, design: AttributeDesign) -> float:
    """val でロジットと属性の線形結合を学習し、test の AUROC を返す（seed 平均）。test の事前確率を使わない正直な版。"""
    (val_frame, val_scores), (test_frame, test_scores) = val, test
    aucs = []
    for sv, st in zip(val_scores, test_scores, strict=True):
        mean, sd = sv.mean(), sv.std()
        model = LogisticRegression(max_iter=3000)
        model.fit(np.c_[(sv - mean) / sd, design(val_frame)], val_frame.target)
        test_x = np.c_[(st - mean) / sd, design(test_frame)]
        aucs.append(roc_auc_score(test_frame.target, model.decision_function(test_x)))
    return float(np.mean(aucs))


def attribute_information(dataset: Dataset) -> pd.DataFrame:
    """属性だけの予測力と、出力に属性を後から足したときの上がり幅。"""
    train = read_split(dataset, "train")
    design = AttributeDesign(dataset, train)
    prior = LogisticRegression(max_iter=3000).fit(design(train), train.target)
    val, test = load_evaluation(dataset, "val"), load_evaluation(dataset, "test")
    y, groups = test[0].target.to_numpy(), patients(dataset, test[0])
    rows = [
        {"quantity": "attr_only_val", "value": roc_auc_score(val[0].target, prior.decision_function(design(val[0])))},
        {"quantity": "attr_only_test", "value": roc_auc_score(y, prior.decision_function(design(test[0])))},
    ]
    for model in ("resnet", "const", "meta", "hyperadapt"):
        if model not in test[1]:
            continue
        base = np.mean([cv_auc(standardize(s)[:, None], y, groups) for s in test[1][model]])
        fused = np.mean([cv_auc(np.c_[standardize(s), design(test[0])], y, groups) for s in test[1][model]])
        rows.append({"quantity": f"late_fusion_crossfit_gain_{model}", "value": fused - base})
        if model in val[1]:
            valfit = late_fusion_valfit((val[0], val[1][model]), (test[0], test[1][model]), design)
            rows.append({"quantity": f"late_fusion_valfit_auc_{model}", "value": valfit})
            gain = valfit - seed_mean_auc(y, test[1][model])
            rows.append({"quantity": f"late_fusion_valfit_gain_{model}", "value": gain})
    return pd.DataFrame(rows).assign(dataset=dataset.name)


def per_attribute(dataset: Dataset) -> pd.DataFrame:
    """ResNet の出力に属性を 1 つずつ足したときの上がり幅（test 上の cross-fit、seed 平均）。"""
    train = read_split(dataset, "train")
    frame, scores = load_evaluation(dataset, "test")
    y, groups = frame.target.to_numpy(), patients(dataset, frame)
    rows = []
    for attribute in (*dataset.categorical, *dataset.continuous):
        if attribute in dataset.categorical:
            extra = pd.get_dummies(frame[attribute].astype(int).astype(str)).to_numpy(float)
        else:
            standardized = (frame[attribute] - train[attribute].mean()) / train[attribute].std()
            extra = np.c_[standardized, frame[f"{attribute}_missing"]]
        gain = []
        for s in scores["resnet"]:
            base = standardize(s)[:, None]
            gain.append(cv_auc(np.hstack([base, extra]), y, groups) - cv_auc(base, y, groups))
        rows.append({"dataset": dataset.name, "attribute": attribute, "gain": float(np.mean(gain))})
    return pd.DataFrame(rows)


def prior_cells(dataset: Dataset, frame: pd.DataFrame, bins: int = 10) -> np.ndarray:
    """属性だけのモデル（train で学習）の判別値の十分位。属性の組そのものだと同じセルの組が 1〜2% しかなく、
    セル内の一致率が推定できないため、事前確率の近さでまとめる。"""
    train = read_split(dataset, "train")
    design = AttributeDesign(dataset, train)
    prior = LogisticRegression(max_iter=3000).fit(design(train), train.target).decision_function(design(frame))
    return pd.qcut(pd.Series(prior).rank(method="first"), bins, labels=False).to_numpy()


def within_between_table(dataset: Dataset) -> pd.DataFrame:
    frame, scores = load_evaluation(dataset, "test")
    y, cell = frame.target.to_numpy(), prior_cells(dataset, frame)
    rows = []
    for model, seeds in scores.items():
        parts = np.array([within_between(s, y, cell) for s in seeds])
        rows.append(
            {
                "dataset": dataset.name,
                "model": model,
                "within": parts[:, 0].mean(),
                "between": parts[:, 1].mean(),
                "within_share": parts[0, 2],
            }
        )
    return pd.DataFrame(rows)


def readability(dataset: Dataset, max_rows: int = 5000) -> pd.DataFrame:
    """ResNet-50（seed 42）の特徴から属性を当てる。val で学習、test で評価。

    特徴は val で PCA 256 次元に落とし、val は最大 `max_rows` 行に間引く（CheXpert の 2 万行で多クラスの
    ロジスティック回帰が重いため。3 データセットで同じ手順にそろえる）。カテゴリは test に 5 件以上ある値の
    one-vs-rest AUROC の平均、年齢は R²。値が低いほど、属性は画像にない情報になりうる。
    """
    paths = {split: cache_files(dataset, split, "_features")["resnet"][42] for split in ("val", "test")}
    frames = {split: read_split(dataset, split) for split in paths}
    feats = {split: np.load(path, allow_pickle=True)["features"] for split, path in paths.items()}
    keep = evaluation_rows(dataset, frames["test"])
    test, test_x = frames["test"][keep], feats["test"][keep]
    rows = np.random.default_rng(0).permutation(len(frames["val"]))[:max_rows]
    val, val_x = frames["val"].iloc[rows], feats["val"][rows]
    pca = PCA(256, random_state=0).fit(val_x)
    val_x, test_x = pca.transform(val_x), pca.transform(test_x)
    out = []
    for attribute in dataset.categorical:
        model = LogisticRegression(C=0.05, max_iter=3000).fit(val_x, val[attribute].astype(int))
        proba, classes = model.predict_proba(test_x), list(model.classes_)
        truth = test[attribute].astype(int).to_numpy()
        present = [c for c in classes if 5 <= (truth == c).sum() <= len(truth) - 5]
        value = np.mean([roc_auc_score(truth == c, proba[:, classes.index(c)]) for c in present])
        out.append({"dataset": dataset.name, "attribute": attribute, "metric": "macro_ovr_auroc", "value": value})
    for attribute in dataset.continuous:
        fit_rows = (val[f"{attribute}_missing"] == 0).to_numpy()
        eval_rows = (test[f"{attribute}_missing"] == 0).to_numpy()
        model = Ridge(alpha=100).fit(val_x[fit_rows], val[attribute][fit_rows])
        value = r2_score(test[attribute][eval_rows], model.predict(test_x[eval_rows]))
        out.append({"dataset": dataset.name, "attribute": attribute, "metric": "r2", "value": value})
    return pd.DataFrame(out)


def ham_scores(split: str) -> tuple[pd.DataFrame, dict[str, list[np.ndarray]]]:
    """HAM の評価行のロジット。test には test 時だけ属性を固定した `meta_fixed`・`hyper_fixed` を足す。"""
    dataset = DATASETS["ham10000"]
    frame, scores = load_evaluation(dataset, split)
    if split == "test":
        full = read_split(dataset, "test")
        keep = evaluation_rows(dataset, full)
        fixed = load_logits(dataset, "test", full.image.to_numpy(), "_cf-constant_typical")
        scores["meta_fixed"] = [s[keep] for s in fixed["meta"]]
        scores["hyper_fixed"] = [s[keep] for s in fixed["hyperadapt"]]
    return frame, scores


def ham_pairs() -> pd.DataFrame:
    """HAM の dx の組（黒色腫 vs 母斑・bkl・bcc）ごとの AUROC。test には主要な対比の bootstrap 区間を付ける。"""
    rows = []
    for split in ("val", "test"):
        frame, scores = ham_scores(split)
        for other in (5, 2, 1):
            rows.extend(ham_pair_rows(split, frame, scores, other))
    return pd.DataFrame(rows)


def ham_pair_rows(split: str, frame: pd.DataFrame, scores: dict, other: int) -> list[dict]:
    pair = frame.dx.isin([4, other]).to_numpy()
    y, groups = (frame.dx[pair] == 4).to_numpy().astype(int), frame.lesion_id.to_numpy()[pair]
    sub = {model: [s[pair] for s in seeds] for model, seeds in scores.items()}
    base = {"split": split, "pair": f"mel vs {HAM_DX[other]}", "n_other": int((frame.dx == other).sum())}
    rows = [{**base, "contrast": model, "delta": seed_mean_auc(y, s)} for model, s in sub.items()]
    if split == "test":
        for a, b in [("const", "resnet"), ("meta_fixed", "const"), ("meta", "meta_fixed"), ("hyper_fixed", "resnet")]:
            rows.append({**base, "contrast": f"{a} − {b}", **boot_delta(sub[a], sub[b], y, groups)})
    return rows


def ham_age_coupling() -> pd.DataFrame:
    """良性・各 dx の中で、スコアと年齢の順位相関（seed 平均）。

    クラスの中でスコアが年齢とともに上がるなら、model は年齢（老いた皮膚らしさ）を悪性の代わりの手がかりに
    している。属性で学習した model の `meta_fixed` でこれが下がれば、年齢を条件が肩代わりしている。
    """
    rows = []
    for split in ("val", "test"):
        frame, scores = ham_scores(split)
        strata = {"benign": frame.target == 0, "nv": frame.dx == 5, "bkl": frame.dx == 2, "mel": frame.dx == 4}
        for name, member in strata.items():
            member = member.to_numpy()
            for model, seeds in scores.items():
                rho = [spearmanr(s[member], frame.age[member]).statistic for s in seeds]
                row = {"split": split, "stratum": name, "model": model, "rho": np.mean(rho), "rho_sd": np.std(rho)}
                rows.append(row)
    return pd.DataFrame(rows)


def main() -> None:
    RESULTS.mkdir(parents=True, exist_ok=True)
    tables = {
        "gains": gains,
        "attribute_information": attribute_information,
        "per_attribute": per_attribute,
        "within_between": within_between_table,
        "readability": readability,
    }
    for name, function in tables.items():
        table = pd.concat([function(dataset) for dataset in DATASETS.values()], ignore_index=True)
        table.to_csv(RESULTS / f"{name}.csv", index=False)
        print(f"wrote {name}.csv", flush=True)
    ham_pairs().to_csv(RESULTS / "ham_pairs.csv", index=False)
    ham_age_coupling().to_csv(RESULTS / "ham_age_coupling.csv", index=False)
    print("wrote ham_pairs.csv, ham_age_coupling.csv", flush=True)


if __name__ == "__main__":
    main()
