"""潜在的な偏りが属性群にどれだけ重なるかと、潜在空間の補正による属性群の改善の関係を調べる。

画像に写り属性には無い要因 S（ペースメーカー・PICC・中心静脈ライン・胸骨切開・手術クリップ）の陽性を、
1 段目の学習で 10% に間引いた。属性群 g ごとに次を test で求める。
- 重なり：P(S | g) / P(S)（g に S がどれだけ濃く含まれるか）
- 損害：AUROC_g(植え込み・補正前) − AUROC_g(植え込みなし・補正前)。同じ test 画像なので対応が取れる
- 回復：[AUROC_g(植え込み・補正後) − AUROC_g(植え込み・補正前)]
        − [AUROC_g(植え込みなし・補正後) − AUROC_g(植え込みなし・補正前)]
  植え込みなしでも補正で上がる分を差し引き、植え込みの損害を戻した分だけを残す
補正後の値は seed 0〜2 の平均。

出力: `results/dose_response/groups.csv`（群ごと）と `results/dose_response/subgroup.csv`（S そのもの）

    uv run python -m analysis.archive.latent_bias_hypernet.dose_response
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from analysis.archive.latent_bias_hypernet.data import CACHE, PACKAGE, load
from analysis.archive.latent_bias_hypernet.planting import hidden_subgroup

PLANTS = {
    "pacer": "a_pacersub_s42",
    "picc": "a_piccsub_s42",
    "central_line": "a_central_linesub_s42",
    "sternotomy": "a_sternotomysub_s42",
    "surgical_clip": "a_surgical_clipsub_s42",
    # 強い植え込み：同じ要因の陽性をすべて陰性に反転した 1 段目（キーは「要因:flip」）
    "pacer:flip": "a_pacerflip_s42",
    "central_line:flip": "a_central_lineflip_s42",
    "picc:flip": "a_piccflip_s42",
}
# 対照：要因と無関係に陽性をランダムに除いた 1 段目と、seed だけを変えた 1 段目。植え込みの「損害」と、
# 学習ごとの揺れ・平均への回帰で生じる見かけの損害・回復を分けるために使う
NULLS = {
    "random600_d0": "a_rand600d0_s42",
    "random600_d1": "a_rand600d1_s42",
    "random1400_d0": "a_rand1400d0_s42",
    "seed43": "a_s43",
}
CLEAN = "a_s42"
CONDITIONS = ("global", "demo32", "gmm32", "router_meta", "router_he")
SEEDS = (0, 1, 2)
MIN_CLASS = 20


def coarse_groups(c: pd.DataFrame) -> dict[str, np.ndarray]:
    sex = c.sex.to_numpy().astype(int)
    old = (c.age.to_numpy() >= 65).astype(int)
    race = c.race_group.to_numpy().astype(int)
    return {"sex": sex, "age": old, "race": race, "sex×age": sex * 2 + old, "age×race": old * 4 + race}


def test_score(stage1: str, cond: str | None, seed: int, base: np.ndarray) -> np.ndarray:
    if cond is None:
        return base
    return np.load(CACHE / "assign" / f"im_{stage1}_{cond}_s{seed}_test_score.npy")


def group_auroc(y, s, g) -> dict[int, float]:
    out = {}
    for k in np.unique(g):
        m = g == k
        if y[m].sum() >= MIN_CLASS and (1 - y[m]).sum() >= MIN_CLASS:
            out[k] = roc_auc_score(y[m], s[m])
    return out


def mean_group_auroc(stage1, cond, y, base, g) -> dict[int, float]:
    seeds = SEEDS if cond else (0,)
    per = [group_auroc(y, test_score(stage1, cond, seed, base), g) for seed in seeds]
    return {k: float(np.mean([p[k] for p in per])) for k in per[0]}


def main() -> None:
    clean = load(CLEAN, "test")
    y = clean.target
    groups = coarse_groups(clean.context)
    rows, srows = [], []
    clean_auc = {
        (gname, cond): mean_group_auroc(CLEAN, cond, y, clean.base, g)
        for gname, g in groups.items()
        for cond in (None, *CONDITIONS)
    }

    def ready(stage1: str) -> bool:
        return (CACHE / "assign" / f"im_{stage1}_router_he_s2_test_score.npy").exists()

    available = {k: v for k, v in {**PLANTS, **NULLS}.items() if ready(v)}
    for factor, stage1 in available.items():
        T = load(stage1, "test")
        if not np.array_equal(T.target, y):
            raise ValueError("test の行順が植え込みなしと一致しない")
        is_null = factor not in PLANTS
        # 対照には要因が無いので、重なりは比べる相手のいない 1 として扱う
        S = np.ones(len(y), dtype=bool) if is_null else hidden_subgroup(T.context, factor.split(":")[0])
        for gname, g in groups.items():
            plant_base = mean_group_auroc(stage1, None, y, T.base, g)
            for k, auc_pb in plant_base.items():
                m = g == k
                row = {
                    "factor": factor,
                    "null": is_null,
                    "grouping": gname,
                    "group": int(k),
                    "n": int(m.sum()),
                    "share_S": float(S[m].mean()),
                    "enrichment": float(S[m].mean() / S.mean()),
                    "damage": auc_pb - clean_auc[(gname, None)][k],
                }
                for cond in CONDITIONS:
                    plant_corr = mean_group_auroc(stage1, cond, y, T.base, g)[k]
                    clean_gain = clean_auc[(gname, cond)][k] - clean_auc[(gname, None)][k]
                    row[f"recovery_{cond}"] = (plant_corr - auc_pb) - clean_gain
                rows.append(row)
        if is_null:
            continue
        srow = {"factor": factor, "n_S": int(S.sum()), "pos_S": int(y[S].sum())}
        srow["auroc_S_clean"] = roc_auc_score(y[S], clean.base[S])
        srow["auroc_S_plant"] = roc_auc_score(y[S], T.base[S])
        for cond in CONDITIONS:
            srow[f"auroc_S_{cond}"] = float(
                np.mean([roc_auc_score(y[S], test_score(stage1, cond, s, T.base)[S]) for s in SEEDS])
            )
        srows.append(srow)
    out = PACKAGE / "results" / "dose_response"
    out.mkdir(parents=True, exist_ok=True)
    table = pd.DataFrame(rows)
    table.to_csv(out / "groups.csv", index=False)
    pd.DataFrame(srows).to_csv(out / "subgroup.csv", index=False)
    with pd.option_context("display.width", 250, "display.max_columns", 30):
        print(pd.DataFrame(srows).round(3).to_string(index=False))
        fine = table[table.grouping == "age×race"]
        print(fine.round(3).to_string(index=False))
        for kind, part in (("植え込み", table[~table.null]), ("対照", table[table.null])):
            if part.empty:
                continue
            print(f"--- {kind}: |損害| の平均 {part.damage.abs().mean():.4f}、SD {part.damage.std():.4f}")
            for cond in CONDITIONS:
                r = np.corrcoef(part.damage, -part[f"recovery_{cond}"])[0, 1]
                print(f"{cond} corr(損害, −回復) = {r:.3f}")
        plants = table[~table.null]
        print("植え込み corr(重なり, 損害) =", round(np.corrcoef(plants.enrichment, plants.damage)[0, 1], 3))
    standardized(table).to_csv(out / "beyond_null.csv", index=False)


def standardized(table: pd.DataFrame) -> pd.DataFrame:
    """対照の群ごとの損害・回復の分布に対して、植え込みの損害と回復を標準化する。

    損害の z = (損害 − 対照の平均) / 対照の SD。z ≤ −2 の群（揺れを超えて悪くなった群）について、
    各補正の回復から対照の回復の平均を引いた「正味の回復」を返す。
    """
    key = ["grouping", "group"]
    nulls = table[table.null]
    if nulls.factor.nunique() < 2:
        return pd.DataFrame()
    rec = [f"recovery_{c}" for c in CONDITIONS]
    ref = nulls.groupby(key)[["damage", *rec]].agg(["mean", "std"])
    rows = []
    for _, r in table[~table.null].iterrows():
        g = ref.loc[(r.grouping, r.group)]
        z = (r.damage - g[("damage", "mean")]) / (g[("damage", "std")] + 1e-9)
        row = {"factor": r.factor, "grouping": r.grouping, "group": r.group, "n": r.n}
        row.update(enrichment=r.enrichment, damage=r.damage, z=z)
        for c in rec:
            row[f"net_{c}"] = r[c] - g[(c, "mean")]
        rows.append(row)
    out = pd.DataFrame(rows)
    hit = out[out.z <= -2]
    with pd.option_context("display.width", 250, "display.max_columns", 30):
        print(f"--- 対照（{nulls.factor.nunique()} 本）の揺れを超えて悪くなった群: {len(hit)} / {len(out)}")
        print(hit.round(3).to_string(index=False))
        if len(hit):
            print("正味の回復の平均:", hit[[f"net_{c}" for c in rec]].mean().round(4).to_dict())
    return out


if __name__ == "__main__":
    main()
