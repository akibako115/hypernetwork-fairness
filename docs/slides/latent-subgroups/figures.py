"""slide-latent-subgroups.pptx の図を生成する。

入力: analysis/archive/latent_bias_hypernet/results/ の集計 CSV（runs/ に取り込んだ succeeded の run から作ったもの）。
      LHCF のラベル漏れ（E2E 側の NMI）は、レポートの表の値を写している。
出力: figures/*.png（図のラベル・凡例・タイトルは英語）
実行例: uv run python docs/slides/latent-subgroups/figures.py
"""

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
RES = ROOT / "analysis/archive/latent_bias_hypernet/results"
HAM = RES / "ham10000_e2e"
OUT = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).parent / "figures"
OUT.mkdir(exist_ok=True)

plt.rcParams.update(
    {
        "font.family": "Noto Sans CJK JP",
        "font.size": 9,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.titlesize": 10,
        "axes.titleweight": "bold",
        "legend.frameon": False,
    }
)
THEME, ACCENT, GREY, LIGHT = "#1F3B57", "#C0504D", "#8C99A6", "#C9D3DC"
SIZE = (4.6, 3.3)


def save(fig, name: str) -> None:
    fig.tight_layout()
    fig.savefig(OUT / f"{name}.png", dpi=250)
    plt.close(fig)


def forest(ax, labels, delta, lo, hi, colors) -> None:
    y = np.arange(len(labels))[::-1]
    for yi, d, a, b, c in zip(y, delta, lo, hi, colors, strict=True):
        ax.plot([a, b], [yi, yi], color=c, lw=1.6)
        ax.plot(d, yi, "o", color=c, ms=5)
    ax.axvline(0, color="#555555", lw=0.8, ls="--")
    ax.set_yticks(y, labels)
    ax.spines["left"].set_visible(False)
    ax.tick_params(axis="y", length=0)


# Q1: 同じ 1 段目を属性セル・潜在スライスで切り、3 つの指標で測る（audit_metrics.csv）
Q1_ROWS = {
    "chexpert": [
        ("demo8", "Age × race (8)", "attr"),
        ("demo32", "Sex × age × race (21)", "attr"),
        ("random32", "Random (32)", "ctrl"),
        ("gmm32", "GMM on embeddings (27)", "latent"),
        ("tree32", "Tree on residuals (29)", "latent"),
        ("auditor32", "Residual auditor (32)", "latent"),
        ("context", "Known context (17)", "known"),
    ],
    "isic": [
        ("性別", "Sex (2)", "attr"),
        ("年齢（65 歳で 2 区分）", "Age ≥ 65 (2)", "attr"),
        ("性別 × 年齢 4 区分（8）", "Sex × age (8)", "attr"),
        ("部位", "Body site (5)", "known"),
        ("ランダム（16）", "Random (16)", "ctrl"),
        ("GMM 16（DINOv2）", "GMM on embeddings (11)", "latent"),
        ("監査器 16 分位", "Residual auditor (16)", "latent"),
    ],
}
Q1_TITLES = {"chexpert": "CheXpert", "isic": "ISIC 2019, BCN20000 → HAM10000"}


def q1_table(dataset: str) -> pd.DataFrame:
    """族ごとの値。監査器は 3 seed の平均で、有意は 3 seed 中 2 つ以上で p < 0.05 のとき。"""
    m = pd.read_csv(RES / "audit_metrics.csv")
    m = m[m.dataset == dataset]
    rows = []
    for key, label, kind in Q1_ROWS[dataset]:
        sub = m[m.family.str.startswith(key)] if "auditor" in key or "監査器" in key else m[m.family == key]
        row = {"label": label, "kind": kind, "oe": sub.logoe_excess_sd.mean(), "oe_sig": sub.logoe_above_null.mean() > 0.5}
        for k in ("worst_auroc", "tpr_gap"):
            row |= {k: sub[k].mean(), f"{k}_null": sub[f"{k}_null"].mean(), f"{k}_sig": (sub[f"{k}_p"] < 0.05).mean() > 0.5}
        rows.append(row)
    return pd.DataFrame(rows)


def fig_q1_metrics(dataset: str) -> None:
    t = q1_table(dataset)
    base = {"attr": ACCENT, "ctrl": GREY, "latent": THEME, "known": "#B08A3E"}
    y = np.arange(len(t))[::-1]
    fig, axes = plt.subplots(1, 3, figsize=(9.0, 2.9), sharey=True)
    ax = axes[0]
    for yi, r in zip(y, t.itertuples(), strict=True):
        c = base[r.kind]
        ax.barh(yi, r.oe, height=0.6, color=c if r.oe_sig else "white", edgecolor=c, lw=1.2)
        ax.text(r.oe + 0.01, yi, f"{r.oe:.2f}", va="center", fontsize=7.5)
    ax.set_xlim(0, t.oe.max() * 1.25 + 0.02)
    ax.set_xlabel("Spread of log O/E (SD, null-subtracted)")
    ax.set_title("Calibration bias", fontsize=9.5)
    for ax, k, title, xlabel in [
        (axes[1], "worst_auroc", "Ranking bias", "Worst-group AUROC"),
        (axes[2], "tpr_gap", "Threshold bias", "TPR gap (max − min)"),
    ]:
        for yi, r in zip(y, t.itertuples(), strict=True):
            c = base[r.kind]
            v, n, sig = getattr(r, k), getattr(r, f"{k}_null"), getattr(r, f"{k}_sig")
            ax.plot([n, v], [yi, yi], color=c, lw=1.0, alpha=0.6)
            ax.plot(n, yi, "|", color="#333333", ms=9, mew=1.4)
            ax.plot(v, yi, "o", ms=6, color=c, mfc=c if sig else "white", mew=1.3)
        ax.set_xlabel(xlabel)
        ax.set_title(title, fontsize=9.5)
    axes[0].set_yticks(y, t.label)
    for tick, kind in zip(axes[0].get_yticklabels(), t.kind, strict=True):
        tick.set_color(base[kind])
    for ax in axes:
        ax.tick_params(axis="y", length=0)
        ax.spines["left"].set_visible(False)
    handles = [
        plt.Line2D([], [], marker="o", ls="", color=THEME, label="Beyond the random-group null (p < 0.05)"),
        plt.Line2D([], [], marker="o", ls="", color=THEME, mfc="white", label="Within the null"),
        plt.Line2D([], [], marker="|", ls="", color="#333333", ms=9, mew=1.4, label="Null mean (groups shuffled)"),
    ]
    kinds = [("attr", "Demographic attributes"), ("known", "Known non-demographic factor"), ("latent", "Latent slices"), ("ctrl", "Random groups")]
    handles += [plt.Line2D([], [], marker="s", ls="", color=base[k], label=lab) for k, lab in kinds if k in set(t.kind)]
    fig.legend(handles=handles, loc="lower center", ncol=4, fontsize=7.5, bbox_to_anchor=(0.5, -0.01))
    fig.suptitle(f"{Q1_TITLES[dataset]}: which bias each way of grouping reveals", fontsize=10, fontweight="bold")
    fig.tight_layout(rect=(0, 0.13, 1, 1))
    fig.savefig(OUT / f"q1_metrics_{dataset}.png", dpi=250)
    plt.close(fig)


# Q2-a: 監査セットの大きさと偏り（ペースメーカー、[h; e] で振り分け）
def fig_q2a_audit_set() -> None:
    b = pd.read_csv(RES / "bgrid/pacer.csv")
    b = b[b.method == "router_he"]
    sizes = ["1000", "5000", "20000", "all"]
    fig, ax = plt.subplots(figsize=SIZE)
    for bias, name, c in [(0.0, "Unbiased audit set", THEME), (0.5, "Half of the training bias", GREY), (1.0, "Same bias as training", ACCENT)]:
        m = b[b.bias == bias].groupby("size").S_pos_auroc.mean().reindex(sizes)
        ax.plot(range(4), m.values, "o-", color=c, label=name, lw=1.6, ms=4)
    ax.axhline(0.720, color="#555555", ls="--", lw=0.8)
    ax.text(3.05, 0.722, "before correction", fontsize=7.5, color="#555555", ha="right", va="bottom")
    ax.set_xticks(range(4), ["1k", "5k", "20k", "all (78k)"])
    ax.set_xlabel("Audit set size (images)")
    ax.set_ylabel("AUROC, pacemaker positives vs. negatives")
    ax.set_title("Post-hoc fix of a planted hidden subgroup")
    ax.legend(fontsize=7.5, loc="upper left")
    save(fig, "q2a_audit_set")


LATENT = {
    "ランダム 8 区分": "Random 8 partitions",
    "DINOv2 GMM 8": "DINOv2 GMM 8",
    "ラベル直交 GMM 8": "Label-orthogonal GMM 8",
    "DINOv2 埋め込み（連続）": "DINOv2 embedding",
}
REFS = {"撮影施設（上限）": "Acquisition site", "Spatial LoRA（属性で条件付け）": "Attributes (sex, age, body site)"}


# Q2-b: 条件だけ替えた E2E（定数との差）
def fig_q2b_conditions() -> None:
    c = pd.read_csv(HAM / "contrast.csv")
    c = c[c.metric == "auroc_all"].set_index("condition")
    names = {**LATENT, **REFS}
    sub = c.loc[list(names)]
    colors = [THEME] * len(LATENT) + [GREY] * len(REFS)
    fig, ax = plt.subplots(figsize=SIZE)
    forest(ax, list(names.values()), sub.delta, sub.lo, sub.hi, colors)
    ax.set_xlabel("Δ test AUROC vs. constant condition (95% CI)")
    fig.suptitle("HAM10000: conditioning variable", fontsize=10, fontweight="bold")
    save(fig, "q2b_conditions")


# Q2-b: 条件を画像の間で入れ替えたときの AUROC の低下
def fig_q2b_shuffle() -> None:
    names = {**LATENT, **REFS}
    fig, ax = plt.subplots(figsize=SIZE)
    x = np.arange(len(names))
    for i, (ds, label, col) in enumerate([("ham10000_e2e", "HAM10000", THEME), ("fairvlmed_e2e", "FairVLMed", GREY)]):
        u = pd.read_csv(RES / ds / "condition_use.csv")
        u["drop"] = u.auroc_true - u.auroc_perm
        m = u.groupby("condition")["drop"].mean().reindex(list(names))
        ax.bar(x + (i - 0.5) * 0.38, m.values, width=0.38, color=col, label=label)
    ax.set_xticks(x, ["Random 8", "DINOv2\nGMM 8", "Label-orth.\nGMM 8", "DINOv2\nembedding", "Acq.\nsite", "Attributes"], fontsize=7.5)
    ax.set_ylabel("AUROC drop when the condition\nis shuffled across images")
    ax.set_title("Does the model use its condition?")
    ax.axvspan(-0.5, 3.5, color="#EEF2F5", zorder=0)
    ax.text(1.5, 0.064, "latent subgroups", ha="center", fontsize=7.5, color="#555555")
    ax.legend(fontsize=7.5, loc="upper left", bbox_to_anchor=(0.02, 0.86))
    ax.set_ylim(0, 0.075)
    save(fig, "q2b_shuffle")


# Q2-b: 属性をどこに入れるか
def fig_q2b_where() -> None:
    s = pd.read_csv(HAM / "summary.csv")
    conds = [
        ("定数（構成だけ）", "Constant condition", LIGHT),
        ("Spatial LoRA（属性で条件付け）", "Attributes → backbone + classifier", GREY),
        ("属性で分類層だけ条件付け", "Attributes → classifier only", THEME),
    ]
    metrics = [("auroc_known", "Known sites"), ("auroc_external", "External site\n(attributes missing)")]
    fig, ax = plt.subplots(figsize=SIZE)
    x = np.arange(len(metrics))
    for i, (cond, label, col) in enumerate(conds):
        g = s[s.condition == cond]
        ax.bar(x + (i - 1) * 0.26, [g[m].mean() for m, _ in metrics], yerr=[g[m].std() for m, _ in metrics], width=0.26, color=col, label=label, capsize=2, error_kw={"lw": 0.8})
    ax.set_xticks(x, [m[1] for m in metrics])
    ax.set_ylim(0.75, 0.97)
    ax.set_ylabel("Test AUROC (mean ± SD, 3 seeds)")
    ax.set_title("HAM10000: where should attributes enter?")
    ax.legend(fontsize=7.5, loc="upper right")
    save(fig, "q2b_where")


# Q2-c: 公平性損失（性別 × 年齢の最悪群、ERM との差）
def fig_q2c_methods() -> None:
    c = pd.read_csv(HAM / "fairness_contrast.csv")
    c = c[(c.metric == "sexage_min") & (c.select == "overall")].set_index("condition")
    sa = "（群：性別 × 年齢）"
    names = {
        f"GroupDRO（q 学習率 0.1）{sa}": "GroupDRO",
        f"群の再重み付け{sa}": "Group reweighting",
        f"群 × クラス均等抽出{sa}": "Group × class balanced sampling",
        f"FairDRO{sa}": "FairDRO",
        f"minimax AUC{sa}": "Minimax AUC",
        f"EO 制約{sa}": "Equalized-odds constraint",
        "属性の敵対的除去（DANN）": "Adversarial attribute removal",
        f"LHCF の最悪コホート損失{sa}": "Worst-cohort loss",
        f"FIS{sa}": "FIS",
        f"FaMI{sa}": "FaMI",
        "SWAD": "SWAD (no groups)",
        "SWAD ＋ 最悪コホート損失（群：LHCF の隠れコホート）": "SWAD + worst-cohort loss (LHCF cohorts)",
    }
    sub = c.loc[list(names)]
    colors = [THEME] * 10 + [GREY, ACCENT]
    fig, ax = plt.subplots(figsize=(4.6, 3.5))
    forest(ax, list(names.values()), sub.delta, sub.lo, sub.hi, colors)
    ax.tick_params(axis="y", labelsize=7.5)
    ax.set_xlabel("Δ worst sex × age AUROC vs. ERM (95% CI)")
    fig.suptitle("Group-aware losses (HAM10000, ResNet-50)", fontsize=10, fontweight="bold")
    save(fig, "q2c_methods")


# Q2-c: SWAD ＋ 最悪コホート損失の群を替える
def fig_q2c_lhcf() -> None:
    f = pd.read_csv(HAM / "fairness_summary.csv")
    f = f[f.select == "overall"]
    base = "SWAD ＋ 最悪コホート損失（群：{}）"
    groups = [
        ("LHCF の隠れコホート", "LHCF\nhidden cohorts", ACCENT),
        ("クラス", "Class\n(benign / malignant)", THEME),
        ("性別 × 年齢", "Sex × age", THEME),
        ("大きさだけ揃えたランダムなコホート", "Size-matched\nrandom cohorts", GREY),
    ]
    fig, ax = plt.subplots(figsize=SIZE)
    for i, (g, _label, col) in enumerate(groups):
        v = f[f.condition == base.format(g)].sexage_min
        ax.bar(i, v.mean(), yerr=v.std(), color=col, width=0.6, capsize=2, error_kw={"lw": 0.8})
    for cond, label in [("SWAD", "SWAD alone"), ("ERM", "ERM")]:
        m = f[f.condition == cond].sexage_min.mean()
        ax.axhline(m, color="#555555", ls="--", lw=0.8)
        ax.text(4.25, m, label, fontsize=7.5, color="#555555", ha="right", va="center")
    ax.set_xlim(-0.5, 4.3)
    ax.set_xticks(range(4), [g[1] for g in groups], fontsize=7.5)
    ax.set_ylim(0.80, 0.92)
    ax.set_ylabel("Worst sex × age group AUROC")
    ax.set_title("SWAD + worst-cohort loss: which groups?")
    save(fig, "q2c_lhcf")


# Q2: ラベルの漏れ（コホートとラベルの NMI）。E2E 側は fairness_methods.md 6 節の値
def fig_q2_leak() -> None:
    p = pd.read_csv(HAM / "crossfit_partitions.csv")
    p = p[(p.field == "real") & (p.name == "lhcf")]
    two = [p[p.split == s].nmi_label.mean() for s in ("train", "test")]
    e2e = [0.229, 0.086]
    fig, ax = plt.subplots(figsize=SIZE)
    x = np.arange(2)
    ax.bar(x - 0.19, [e2e[0], two[0]], width=0.38, color=ACCENT, label="Train")
    ax.bar(x + 0.19, [e2e[1], two[1]], width=0.38, color=GREY, label="Test")
    ax.set_xticks(x, ["Cohorts from the model's\nown training embeddings", "Cohorts from\nheld-out embeddings"])
    ax.set_ylabel("NMI between cohort and label")
    ax.set_title("Cohorts as a condition leak the label")
    ax.legend(fontsize=8)
    save(fig, "q2_leak")


# Q3-a: 規則の反転を植え込んだ場での少数側 B
def fig_q3a_rule() -> None:
    s = pd.read_csv(HAM / "rule_summary_strong.csv")
    ceil = pd.read_csv(HAM / "rule_ceiling_strong.csv").groupby("reference").auroc_B.mean()
    conds = [("const", "Constant", GREY), ("random8", "Random 8\npartitions", GREY), ("gmm8", "GMM 8\n(boundary)", ACCENT)]
    fig, ax = plt.subplots(figsize=SIZE)
    for i, (key, _label, col) in enumerate(conds):
        v = s[s.exp.str.endswith(key)].auroc_B
        ax.bar(i, v.mean(), yerr=v.std(), color=col, width=0.6, capsize=2, error_kw={"lw": 0.8})
        ax.text(i, 0.505, f"{v.mean():.3f}", ha="center", va="bottom", fontsize=8, color="white")
    for ref, label in [("規則を知る", "knows the rule"), ("規則を知らない", "ignores the rule")]:
        ax.axhline(ceil[ref], color="#555555", ls="--", lw=0.8)
        ax.text(3.35, ceil[ref], f"reference:\n{label}", fontsize=7, color="#555555", ha="right", va="center")
    ax.set_xlim(-0.5, 3.4)
    ax.set_xticks(range(3), [c[1] for c in conds])
    ax.set_ylim(0.5, 1.0)
    ax.set_ylabel("AUROC on minority B (mean ± SD)")
    ax.set_title("Conditioning variable of the hypernet")
    save(fig, "q3a_rule")


# Q3-b: 1 段（E2E）と 2 段
def fig_q3b_stages() -> None:
    s = pd.read_csv(HAM / "crossfit_rule_summary.csv")
    rows = {
        "E2E": ["E2E：（条件：定数）", "E2E：（条件：ランダム 8 区分）", "E2E：（条件：GMM 8、反転の境界を含む）"],
        "2stage": ["2 段目：定数", "2 段目：ランダム 8 区分", "2 段目：GMM 8（反転の境界を含む）"],
    }
    fig, ax = plt.subplots(figsize=SIZE)
    x = np.arange(2)
    for i, (label, col) in enumerate([("Constant", LIGHT), ("Random 8", GREY), ("GMM 8 (boundary)", ACCENT)]):
        vals = [s[s.condition == rows[k][i]].auroc_B for k in rows]
        ax.bar(x + (i - 1) * 0.26, [v.mean() for v in vals], yerr=[v.std() for v in vals], width=0.26, color=col, label=label, capsize=2, error_kw={"lw": 0.8})
    st1 = s[s.condition.str.startswith("1 段目")].auroc_B.mean()
    ax.axhline(st1, color="#555555", ls="--", lw=0.8)
    ax.text(1.85, st1, "stage 1\nalone", fontsize=7, color="#555555", ha="right", va="center")
    ax.set_xlim(-0.5, 1.9)
    ax.set_xticks(x, ["One-stage\n(end-to-end)", "Two-stage\n(frozen stage 1)"])
    ax.set_ylim(0.6, 1.0)
    ax.set_ylabel("AUROC on minority B (mean ± SD)")
    ax.set_title("Planted rule reversal: one vs. two stages")
    ax.legend(fontsize=7.5, loc="upper right", ncol=3, bbox_to_anchor=(1.0, 1.0))
    save(fig, "q3b_stages")


# Q3-c: 誤りから区分を作り直す
def fig_q3c_iter() -> None:
    c = pd.read_csv(HAM / "crossfit_rule_contrast.csv")
    c = c[(c.metric == "auroc_B") & (c.reference == "2 段目：定数")].set_index("condition")
    names = {f"2 段目：誤りの向きで作り直した区分（{i} 周目）": f"Error-based, iteration {i}" for i in (1, 2, 3)}
    names |= {"2 段目：ランダム 8 区分": "Random 8 partitions", "2 段目：GMM 8（反転の境界を含む）": "GMM 8 (contains the boundary)"}
    sub = c.loc[list(names)]
    fig, ax = plt.subplots(figsize=SIZE)
    forest(ax, list(names.values()), sub.delta, sub.lo, sub.hi, [THEME] * 3 + [GREY, ACCENT])
    ax.set_xlabel("Δ AUROC on minority B vs. constant (95% CI)")
    fig.suptitle("Partitions rebuilt from errors (two-stage)", fontsize=10, fontweight="bold")
    save(fig, "q3c_iter")


# 付録：発見の指標と是正の効果
def fig_app_discovery() -> None:
    d = pd.read_csv(HAM / "discovery_remedy.csv").groupby("grouping")[["auc_gap", "d_auroc_external"]].mean().dropna()
    labels = {
        "score_q8": "Score octiles",
        "h_km8": "ERM-feature k-means",
        "he_km8": "[ERM; DINOv2] k-means",
        "h_lo_km8": "ERM-feature k-means\n(label/score removed)",
        "site": "Acquisition site",
        "dino_gmm8": "DINOv2 GMM 8",
        "dino_gmm8_lo": "Label-orth. GMM 8",
        "sexage": "Sex × age",
        "random8": "Random 8",
    }
    fig, ax = plt.subplots(figsize=(4.8, 3.3))
    ax.scatter(d.auc_gap, d.d_auroc_external, color=THEME, s=22)
    offsets = {"dino_gmm8": (4, -9), "site": (4, 3), "random8": (-6, -10), "sexage": (4, 3), "dino_gmm8_lo": (4, 3), "h_lo_km8": (4, -2), "he_km8": (4, 3), "h_km8": (4, 3), "score_q8": (-4, 5)}
    for k, r in d.iterrows():
        if k in ("dino_gmm8", "dino_gmm8_lo", "sexage", "random8", "site"):
            continue
        dx, dy = offsets.get(k, (4, 3))
        ax.annotate(labels[k], (r.auc_gap, r.d_auroc_external), xytext=(dx, dy), textcoords="offset points", fontsize=6.5, ha="right" if dx < 0 else "left")
    ax.annotate("sex × age, site, DINOv2 GMM 8,\nlabel-orth. GMM 8, random 8", (0.07, -0.006), xytext=(0.13, -0.022), fontsize=6.5, arrowprops={"arrowstyle": "-", "color": "#888888", "lw": 0.6})
    ax.axhline(0, color="#555555", lw=0.8, ls="--")
    ax.set_xlabel("Discovery: AUROC gap, overall − within group (val)")
    ax.set_ylabel("Remedy: Δ external AUROC")
    ax.set_title("Better at discovery, worse at remedy")
    ax.text(0.98, 0.95, "Spearman ρ = −0.85 (n = 9)", transform=ax.transAxes, ha="right", fontsize=8)
    save(fig, "app_discovery")


# Q1 の物差しの説明：群内の順位（AUROC）が同じでも、予測の水準がずれうる（数値は説明用の例）
def fig_q1_oe_illustration() -> None:
    fig, ax = plt.subplots(figsize=SIZE)
    groups = ["Group X", "Group Y"]
    expected, observed = [20, 10], [20, 20]
    x = np.arange(2)
    ax.bar(x - 0.19, expected, width=0.38, color=GREY, label="Expected positives (sum of predicted prob.)")
    ax.bar(x + 0.19, observed, width=0.38, color=THEME, label="Observed positives")
    for xi, e, o in zip(x, expected, observed, strict=True):
        ax.text(xi, max(e, o) + 1.2, f"log O/E = {np.log(o / e):+.2f}\nwithin-group AUROC = 0.85", ha="center", fontsize=8)
    ax.set_xticks(x, groups)
    ax.set_ylim(0, 31)
    ax.set_ylabel("Number of positives (100 images per group)")
    ax.set_title("Illustration: same AUROC, different calibration")
    ax.legend(fontsize=7.5, loc="upper left")
    save(fig, "q1_oe_illustration")


if __name__ == "__main__":
    fig_q1_metrics("chexpert")
    fig_q1_metrics("isic")
    for fn in [
        fig_q1_oe_illustration,
        fig_q2a_audit_set,
        fig_q2b_conditions,
        fig_q2b_shuffle,
        fig_q2b_where,
        fig_q2c_methods,
        fig_q2c_lhcf,
        fig_q2_leak,
        fig_q3a_rule,
        fig_q3b_stages,
        fig_q3c_iter,
        fig_app_discovery,
    ]:
        fn()
    print(f"wrote figures to {OUT}")
