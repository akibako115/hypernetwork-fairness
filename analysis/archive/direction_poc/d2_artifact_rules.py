"""方向 2：規則の食い違い（部分群ごとに判定規則が違う場）が HAM10000 の実データにあるかを、アーチファクトで測る。

hypernet の振り分けが効くには、部分群ごとに別の判定規則が要る場が要る
（`latent_bias_hypernet/reports/fairness_methods.md` 5 節の正の対照）。実データで規則が食い違いそうなのは、
属性に無いアーチファクト（暗い四隅・毛・ペンの印・気泡や反射）が、施設や部位によって違う意味を持つ場合である。
ここでは 3 段で確かめる。

1. `flags`：画像処理のヒューリスティクスでアーチファクトの得点を出す（公開注釈は使わない。ダウンロードしないため）。
   - 暗い四隅：四隅の小領域の明るさの中央値が、中央の明るさに比べて十分暗い隅の数
   - 毛：グレースケールの黒帽子変換（細く暗い構造）の強い画素の割合
   - ペン・インク：青〜紫の色相で彩度の高い画素の割合（病変の中心を外した外周だけで数える）
   - 気泡・反射：明るく彩度の低い画素の割合
   閾値は全画像の得点の分布から決め、`results/d2/artifact_prevalence.csv` に頻度・施設別頻度・悪性率を書く。
2. `heterogeneity`：`latent_bias_hypernet/heterogeneity.py` と同じ頭の比べ合い（DINOv2 の PCA 32 の上で
   global・intercept・interaction・MLP）を、群＝アーチファクトの有無、アーチファクト × 施設で回す。
3. `meaning`：アーチファクトの「意味」が群で変わるかを直接測る。得点（DINOv2 の global 頭の logit、または
   ERM の logit）を固定し、label ~ score + artifact の係数を群ごとに推定して、群間で符号が変わるかを病変単位の
   bootstrap で見る。物差しの妥当性は、植え込みの正の対照（マーカーの意味が GMM 8 のクラスタ 2・4 で反転）で
   同じ測り方が反転を検出できるかで確かめる。

出力: `cache/d2/artifact_scores.csv`、`results/d2/{artifact_prevalence,heterogeneity,meaning,meaning_planted}.csv`、
`results/d2/meaning{,_planted}_summary.csv`（`meaning_summary`：群ごとの係数を畳んだ表と尤度比検定）

    OMP_NUM_THREADS=8 uv run python -m analysis.archive.direction_poc.d2_artifact_rules flags
    OMP_NUM_THREADS=8 uv run python -m analysis.archive.direction_poc.d2_artifact_rules prevalence
    OMP_NUM_THREADS=8 uv run python -m analysis.archive.direction_poc.d2_artifact_rules heterogeneity
    OMP_NUM_THREADS=8 uv run python -m analysis.archive.direction_poc.d2_artifact_rules meaning
    OMP_NUM_THREADS=8 uv run python -m analysis.archive.direction_poc.d2_artifact_rules meaning_summary
"""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd
import rootutils
from joblib import Parallel, delayed
from PIL import Image
from scipy import ndimage

ROOT = rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

PACKAGE = ROOT / "analysis" / "archive" / "direction_poc"
CACHE = PACKAGE / "cache" / "d2"
RESULTS = PACKAGE / "results" / "d2"
IMAGES = ROOT / "data" / "ham10000" / "images"
SPLITS = ROOT / "data" / "ham10000" / "splits_latent"
SITE_NAMES = {0: "rosendahl", 1: "vidir_modern", 2: "vidir_molemax", 3: "vienna_dias"}


def image_scores(name: str) -> dict:
    """1 枚の画像からアーチファクトの得点を出す。300×225 に縮めて計算する。"""
    with Image.open(IMAGES / name) as img:
        rgb = np.asarray(img.convert("RGB").resize((300, 225)), dtype=np.float32) / 255.0
    hsv = np.asarray(Image.fromarray((rgb * 255).astype(np.uint8)).convert("HSV"), dtype=np.float32) / 255.0
    gray = rgb.mean(axis=2)
    h, w = gray.shape
    c = 24
    corners = [gray[:c, :c], gray[:c, -c:], gray[-c:, :c], gray[-c:, -c:]]
    center = np.median(gray[h // 4 : 3 * h // 4, w // 4 : 3 * w // 4])
    corner_med = np.array([np.median(k) for k in corners])
    # 毛：閉包（暗く細い構造を埋める）との差。病変そのものの暗さは太いので、構造要素より細いものだけが残る
    closed = ndimage.grey_closing(gray, size=(9, 9))
    blackhat = closed - gray
    # 外周（中央の楕円の外）だけでインクを数える。病変の青い色素を拾わないため
    yy, xx = np.mgrid[:h, :w]
    outer = ((yy - h / 2) / (h * 0.42)) ** 2 + ((xx - w / 2) / (w * 0.42)) ** 2 > 1
    hue, sat, val = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    ink = (hue > 0.55) & (hue < 0.85) & (sat > 0.25) & (val > 0.15)
    # 青いペンの印は彩度が高い。機器の色かぶりで全体が紫がかった画像を拾わないよう、彩度を強く取った版も持つ
    ink_strong = (hue > 0.55) & (hue < 0.75) & (sat > 0.5) & (val > 0.2)
    bright = (val > 0.92) & (sat < 0.12)
    return {
        "image": name,
        "corner_min": float(corner_med.min()),
        "corner_ratio_2nd": float(np.sort(corner_med)[1] / (center + 1e-3)),
        "n_dark_corners": int((corner_med < 0.35 * center).sum()),
        "hair_frac": float((blackhat > 0.08).mean()),
        # 色素ネットワーク（病変の中の細い暗い網目）を拾わないよう、外周だけで濃い細線を数える版
        "hair_frac_outer": float((blackhat[outer] > 0.12).mean()),
        "ink_frac_outer": float(ink[outer].mean()),
        "ink_strong_outer": float(ink_strong[outer].mean()),
        "bright_frac": float(bright.mean()),
        "center_gray": float(center),
    }


def flags() -> None:
    frames = []
    for split in ("train", "val", "test"):
        frames.append(pd.read_csv(SPLITS / f"{split}.csv").assign(split=split))
    meta = pd.concat(frames, ignore_index=True)
    rows = Parallel(n_jobs=8, batch_size=64)(delayed(image_scores)(n) for n in meta.image)
    CACHE.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(CACHE / "artifact_scores.csv", index=False)
    print(pd.DataFrame(rows).describe(percentiles=[0.5, 0.75, 0.9, 0.95, 0.99]).round(4).to_string())


# 閾値は得点の分布と目視（各フラグの上位・閾値付近の画像）で決めた。
# - 暗い四隅：中央の 35% より暗い隅が 2 つ以上（全画像の 6.5%）。上位・閾値付近とも、ほぼすべて円形のビネット
# - 毛：外周の濃い細線が 2% 超（9.9%）。閾値付近で 10 枚中 8 枚ほどが毛。外周に限らない版は色素ネットワークを拾った
# - ペンの印：外周の高彩度の青が 0.3% 超（40 枚）。彩度の弱い版は機器の色かぶり（全体が紫がかった画像）を拾った
# - 明るい背景：明るく彩度の低い画素が 2% 超（4.7%）。気泡を狙ったが、実際には露出の高い白い背景を拾う
ARTIFACTS = {
    "暗い四隅": ("n_dark_corners", 2),
    "毛": ("hair_frac_outer", 0.02),
    "ペンの印": ("ink_strong_outer", 0.003),
    "明るい背景": ("bright_frac", 0.02),
}


def frames() -> pd.DataFrame:
    """train・val・test の split CSV に、アーチファクトの有無の列を足して縦に並べる。"""
    scores = pd.read_csv(CACHE / "artifact_scores.csv")
    parts = [pd.read_csv(SPLITS / f"{s}.csv").assign(split=s) for s in ("train", "val", "test")]
    meta = pd.concat(parts, ignore_index=True).merge(scores, on="image", how="left", validate="one_to_one")
    for name, (col, thr) in ARTIFACTS.items():
        meta[name] = (meta[col] >= thr) if col == "n_dark_corners" else (meta[col] > thr)
    meta["site_name"] = np.where(meta.site_missing == 1, "external", meta.site.map(SITE_NAMES))
    return meta


def prevalence() -> pd.DataFrame:
    """フラグごとの頻度、施設別の頻度、フラグの有無での悪性率。"""
    meta = frames()
    rows = []
    for name in ARTIFACTS:
        flag = meta[name]
        row = {"artifact": name, "n": int(flag.sum()), "rate": flag.mean()}
        row["malignant_with"] = meta.target[flag].mean()
        row["malignant_without"] = meta.target[~flag].mean()
        for site, sub in meta.groupby("site_name"):
            row[f"rate_{site}"] = sub[name].mean()
        rows.append(row)
    out = pd.DataFrame(rows)
    examples = {name: meta.loc[meta[name]].nlargest(6, ARTIFACTS[name][0]).image.tolist() for name in ARTIFACTS}
    out["top_examples"] = [" ".join(examples[n]) for n in out.artifact]
    return out


def ham_partitions(f: pd.DataFrame, t: pd.DataFrame) -> dict:
    """群の切り方。アーチファクト単独（2 群）、3 つの組み合わせ（8 群）、アーチファクト × 施設（8 群）。"""
    from analysis.archive.latent_bias_hypernet.heterogeneity import random_groups

    def flag(frame, name):
        return frame[name].to_numpy().astype(int)

    parts = {"施設": (f.site.to_numpy(), t.site.to_numpy(), 4)}
    for name in ("暗い四隅", "毛", "明るい背景"):
        parts[name] = (flag(f, name), flag(t, name), 2)
    combo_f = flag(f, "暗い四隅") * 4 + flag(f, "毛") * 2 + flag(f, "明るい背景")
    combo_t = flag(t, "暗い四隅") * 4 + flag(t, "毛") * 2 + flag(t, "明るい背景")
    parts["3 つの組み合わせ"] = (combo_f, combo_t, 8)
    for name in ("暗い四隅", "毛"):
        parts[f"{name} × 施設"] = (flag(f, name) * 4 + f.site.to_numpy(), flag(t, name) * 4 + t.site.to_numpy(), 8)
    uf, ut = f.lesion_id.to_numpy(str), t.lesion_id.to_numpy(str)
    parts["ランダム 2"] = (random_groups(uf, 0) % 2, random_groups(ut, 1) % 2, 2)
    parts["ランダム 8"] = (random_groups(uf, 0), random_groups(ut, 1), 8)
    return parts


def heterogeneity() -> None:
    """`latent_bias_hypernet/heterogeneity.py` と同じ頭の比べ合い。施設が分かる test の行だけで評価する。"""
    from analysis.archive.latent_bias_hypernet.heterogeneity import run_setting, sex_age

    emb = ROOT / "analysis" / "archive" / "latent_bias_hypernet" / "cache" / "embed" / "dinov2"
    meta = frames()
    f = meta[meta.split == "train"].reset_index(drop=True)
    t = meta[meta.split == "test"].reset_index(drop=True)
    ef_npz, et_npz = np.load(emb / "ham_train.npz"), np.load(emb / "ham_test.npz")
    assert np.array_equal(ef_npz["image"], f.image.to_numpy()) and np.array_equal(et_npz["image"], t.image.to_numpy())
    known = t.site_missing.to_numpy() == 0
    parts = {k: (gf, gt[known], K) for k, (gf, gt, K) in ham_partitions(f, t).items()}
    out = run_setting(
        "HAM10000（DINOv2）",
        ef_npz["embedding"].astype(np.float32),
        f.target.to_numpy(),
        f.lesion_id.to_numpy(str),
        et_npz["embedding"].astype(np.float32)[known],
        t.target.to_numpy()[known],
        t.lesion_id.to_numpy(str)[known],
        sex_age(t)[known],
        parts,
    )
    # 2 群の切り方は、同じ群の数のランダム区分で差し引いた値も持つ（頭の数が増える効果をそろえる）
    rand2 = out.loc[out.partition == "ランダム 2"].iloc[0]
    out["rule_gain_vs_random_sameK"] = np.where(out.K == 2, out.rule_gain - rand2.rule_gain, out.rule_gain_vs_random)
    RESULTS.mkdir(parents=True, exist_ok=True)
    out.to_csv(RESULTS / "heterogeneity.csv", index=False)
    cols = ["partition", "K", "auroc_global", "rule_gain", "rule_gain_lo", "rule_gain_hi", "rule_gain_vs_random_sameK"]
    cols += ["auroc_interaction", "auroc_mlp", "vs_mlp", "vs_mlp_lo", "vs_mlp_hi"]
    print(out[cols].round(4).to_string(index=False))


def coef_by_group(y, score, artifact, group, units, boot: int = 200) -> pd.DataFrame:
    """群ごとに label ~ score + artifact を当て、artifact の係数と病変単位 bootstrap の 95% 区間を返す。

    score を固定するので、係数は「モデルが既に使っている分を超えて、その群でアーチファクトが持つ意味」になる。
    群ごとに符号が変わるなら、共有の規則では吸収できない食い違いがある。係数そのものは因果効果を意味しない。
    """
    from sklearn.linear_model import LogisticRegression

    def fit(idx):
        x = np.column_stack([score[idx], artifact[idx]])
        if artifact[idx].min() == artifact[idx].max() or y[idx].min() == y[idx].max():
            return np.nan
        return LogisticRegression(C=1e4, max_iter=2000).fit(x, y[idx]).coef_[0, 1]

    rng = np.random.default_rng(0)
    rows = []
    for g in np.unique(group):
        idx = np.flatnonzero(group == g)
        uniq = np.unique(units[idx])
        by_unit = pd.Series(idx).groupby(units[idx]).apply(np.array).to_dict()
        draws = []
        for _ in range(boot):
            pick = np.concatenate([by_unit[u] for u in rng.choice(uniq, len(uniq))])
            draws.append(fit(pick))
        lo, hi = np.nanpercentile(draws, [2.5, 97.5])
        n_art = int(artifact[idx].sum())
        n_art_pos = int((artifact[idx] & (y[idx] == 1)).sum())
        rows.append(
            {
                "group": g,
                "n": len(idx),
                "n_artifact": n_art,
                "n_artifact_pos": n_art_pos,
                "coef": fit(idx),
                "lo": lo,
                "hi": hi,
            }
        )
    return pd.DataFrame(rows)


def heldout_scores(kind: str) -> pd.DataFrame:
    """1 段目（ResNet-50 ERM、cross-fit の stage1 で学習）が見ていない行の logit。3 seed の平均。

    行は stage2 の train・val・test（植え込みの版も同じ行）。`kind` は "real" か "rule"。
    """
    import glob

    lbh = ROOT / "analysis" / "archive" / "latent_bias_hypernet" / "cache"
    pattern = "resnet-ham10000-cf-stage1" if kind == "real" else "resnet-ham10000-rule-strong-cf-stage1"
    cf = ROOT / "data" / "ham10000" / "splits_crossfit" / "stage2"
    out = []
    for split in ("train", "val", "test"):
        frame = pd.read_csv(cf / f"{split}.csv").assign(split=split)
        files = sorted(glob.glob(str(lbh / f"*-{pattern}-s4[234]-*_cf_{split}.npz")))
        assert len(files) == 3, files
        logits = []
        for path in files:
            npz = np.load(path)
            assert np.array_equal(npz["target"], frame.target.to_numpy())
            logits.append(npz["score"])
        out.append(frame.assign(score=np.mean(logits, axis=0)))
    return pd.concat(out, ignore_index=True)


def meaning() -> None:
    """アーチファクトの意味が群で変わるか。正の対照（植え込みの反転）で物差しを確かめてから実データを測る。"""
    from analysis.archive.latent_bias_hypernet.heterogeneity import sex_age
    from analysis.archive.latent_bias_hypernet.rule_planting import marker_mask, minority

    RESULTS.mkdir(parents=True, exist_ok=True)
    # 正の対照：マーカーの意味が少数側 B（GMM 8 のクラスタ 2・4）で反転する。ヒューリスティクスの誤りを模して、
    # マーカーの有無を一定の割合で入れ替えた版でも同じ検出ができるかを見る
    rule = heldout_scores("rule")
    marked = marker_mask(rule, "strong")
    units = rule.lesion_id.to_numpy(str)
    rng = np.random.default_rng(0)
    planted = []
    for noise in (0.0, 0.1, 0.2):
        flip = rng.random(len(marked)) < noise
        noisy = marked ^ flip
        for gname, group in (("A / B", minority(rule).astype(int)), ("GMM 8", rule.gmm8.to_numpy())):
            res = coef_by_group(rule.target.to_numpy(), rule.score.to_numpy(), noisy, group, units)
            planted.append(res.assign(noise=noise, partition=gname))
    planted = pd.concat(planted, ignore_index=True)
    planted.to_csv(RESULTS / "meaning_planted.csv", index=False)
    print(planted.round(3).to_string(index=False))

    real = heldout_scores("real").merge(frames()[["image", *ARTIFACTS]], on="image", how="left", validate="one_to_one")
    real["site_name"] = np.where(real.site_missing == 1, "external", real.site.map(SITE_NAMES))
    loc_top = real.localization.value_counts().index[:6]
    real["loc_group"] = np.where(real.localization.isin(loc_top), real.localization, -1)
    groupings = {
        "全体": np.zeros(len(real), dtype=int),
        "施設": real.site_name.to_numpy(),
        "部位（上位 6 ＋その他）": real.loc_group.to_numpy(),
        "性別 × 年齢": sex_age(real),
        "GMM 8": real.gmm8.to_numpy(),
    }
    rows = []
    for name in ("暗い四隅", "毛", "明るい背景", "ペンの印"):
        for gname, group in groupings.items():
            res = coef_by_group(
                real.target.to_numpy(),
                real.score.to_numpy(),
                real[name].to_numpy(),
                group,
                real.lesion_id.to_numpy(str),
            )
            rows.append(res.assign(artifact=name, partition=gname))
    out = pd.concat(rows, ignore_index=True)
    out["group"] = out.group.astype(str)
    out.to_csv(RESULTS / "meaning.csv", index=False)
    print(out.round(3).to_string(index=False))


MIN_CELL = 5  # 係数を読む群の条件：アーチファクトありの陽性・陰性がそれぞれこの数以上


def lr_test(y, score, artifact, group) -> tuple[float, int]:
    """artifact の係数が群で同じ、という帰無仮説の尤度比検定（群ごとの切片・score の傾きは両方のモデルに入れる）。

    病変内の画像の相関を無視しているので、p は小さめに出る。群の比べ合いの目安にだけ使う。
    """
    from scipy.stats import chi2
    from sklearn.linear_model import LogisticRegression

    levels = np.unique(group)
    onehot = (group[:, None] == levels[None, :]).astype(float)
    base = np.column_stack([onehot[:, 1:], onehot * score[:, None]])
    x0 = np.column_stack([base, artifact])
    x1 = np.column_stack([base, onehot * artifact[:, None]])

    def loglik(x):
        p = LogisticRegression(C=1e4, max_iter=5000).fit(x, y).predict_proba(x)[:, 1].clip(1e-9, 1 - 1e-9)
        return float(np.sum(y * np.log(p) + (1 - y) * np.log(1 - p)))

    stat = 2 * (loglik(x1) - loglik(x0))
    df = len(levels) - 1
    return float(chi2.sf(max(stat, 0.0), df)), df


def readable_groups(table: pd.DataFrame) -> pd.Series:
    """係数を読める群か。アーチファクトありの陽性・陰性が少ない群は、完全分離で係数が ±5 以上に飛ぶので外す。"""
    n_neg = table.n_artifact - table.n_artifact_pos
    return (table.n_artifact_pos >= MIN_CELL) & (n_neg >= MIN_CELL)


def meaning_summary() -> None:
    """群ごとの係数を、アーチファクト × 区分ごとに 1 行へ畳む。符号の反転（区間が 0 を挟まず逆向きの群がある）と、
    読める群だけでの尤度比検定。正の対照も同じ形で畳む。"""
    from analysis.archive.latent_bias_hypernet.heterogeneity import sex_age
    from analysis.archive.latent_bias_hypernet.rule_planting import marker_mask, minority

    def fold(table, keys):
        rows = []
        for key, sub in table.groupby(keys):
            ok = sub[readable_groups(sub)]
            row = dict(zip(keys, key, strict=True))
            row["n_groups_readable"] = len(ok)
            row["coef_min"], row["coef_max"] = ok.coef.min(), ok.coef.max()
            row["n_sig_pos"], row["n_sig_neg"] = int((ok.lo > 0).sum()), int((ok.hi < 0).sum())
            row["reversal"] = row["n_sig_pos"] > 0 and row["n_sig_neg"] > 0
            row["readable"] = " ".join(ok.group.astype(str))
            rows.append(row)
        return pd.DataFrame(rows)

    planted = fold(pd.read_csv(RESULTS / "meaning_planted.csv"), ["noise", "partition"])
    real_tab = pd.read_csv(RESULTS / "meaning.csv")
    real_tab = real_tab[real_tab.partition != "全体"]
    real = fold(real_tab, ["artifact", "partition"])

    rule = heldout_scores("rule")
    marked = marker_mask(rule, "strong")
    planted["lr_p"] = np.nan
    for i, r in planted.iterrows():
        flip = np.random.default_rng(int(r.noise * 10)).random(len(marked)) < r.noise
        group = minority(rule).astype(int) if r.partition == "A / B" else rule.gmm8.to_numpy()
        planted.loc[i, "lr_p"] = lr_test(
            rule.target.to_numpy(), rule.score.to_numpy(), (marked ^ flip).astype(float), group
        )[0]
    data = heldout_scores("real").merge(frames()[["image", *ARTIFACTS]], on="image", how="left", validate="one_to_one")
    data["site_name"] = np.where(data.site_missing == 1, "external", data.site.map(SITE_NAMES))
    loc_top = data.localization.value_counts().index[:6]
    columns = {
        "施設": data.site_name.astype(str),
        "部位（上位 6 ＋その他）": pd.Series(np.where(data.localization.isin(loc_top), data.localization, -1)).astype(
            str
        ),
        "性別 × 年齢": pd.Series(sex_age(data)).astype(str),
        "GMM 8": data.gmm8.astype(str),
    }
    real["lr_p"] = np.nan
    for i, r in real.iterrows():
        keep = columns[r.partition].isin(r.readable.split()).to_numpy()
        if r.n_groups_readable < 2:
            continue
        real.loc[i, "lr_p"] = lr_test(
            data.target.to_numpy()[keep],
            data.score.to_numpy()[keep],
            data[r.artifact].to_numpy()[keep].astype(float),
            columns[r.partition].to_numpy()[keep],
        )[0]
    planted.to_csv(RESULTS / "meaning_planted_summary.csv", index=False)
    real.to_csv(RESULTS / "meaning_summary.csv", index=False)
    print(planted.round(3).to_string(index=False))
    print(real.round(3).to_string(index=False))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("step", choices=["flags", "prevalence", "heterogeneity", "meaning", "meaning_summary"])
    args = parser.parse_args()
    if args.step == "prevalence":
        RESULTS.mkdir(parents=True, exist_ok=True)
        out = prevalence()
        out.to_csv(RESULTS / "artifact_prevalence.csv", index=False)
        print(out.drop(columns="top_examples").round(3).to_string(index=False))
        return
    steps = {"flags": flags, "heterogeneity": heterogeneity, "meaning": meaning, "meaning_summary": meaning_summary}
    steps[args.step]()


if __name__ == "__main__":
    main()
