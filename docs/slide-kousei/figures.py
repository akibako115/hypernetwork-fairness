"""slide-kousei.pptx の GroupDRO スライド用の模式図を生成する。"""

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams.update({"font.family": "Noto Sans CJK JP", "font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
out = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).parent / "figures"
out.mkdir(exist_ok=True)
POS, NEG = "#C0504D", "#A7B4BF"

# ① global な class weight: サンプル損失 ℓ が全員同じでも、群損失は陽性率で決まる
p = {"群 X": 0.5, "群 Y": 0.1}
p_all = 0.2
w_pos = (1 - p_all) / p_all  # = 4
fig, ax = plt.subplots(figsize=(4.4, 3.4), dpi=200)
names = list(p)
pos = [p[g] * w_pos for g in names]
neg = [1 - p[g] for g in names]
labels = [f"{g}\n陽性率 {p[g]:.0%}" for g in names]
ax.bar(labels, neg, color=NEG, width=0.5, label="陰性（重み 1）")
ax.bar(labels, pos, bottom=neg, color=POS, width=0.5, label=f"陽性（重み {w_pos:.0f}）")
for i, g in enumerate(names):
    ax.text(i, pos[i] + neg[i] + 0.05, f"{pos[i] + neg[i]:.1f} ℓ", ha="center", fontsize=11)
ax.set_ylabel("群の損失（サンプル損失 ℓ の倍数）")
ax.set_ylim(0, 3.0)
ax.legend(frameon=False, fontsize=9, loc="upper right")
ax.set_title(f"全サンプルの損失が同じ ℓ でも（全体の陽性率 {p_all:.0%}）", fontsize=9.5)
fig.tight_layout()
fig.savefig(out / "global-class-weight.png")

# ② 群ごとの class weight: 重み付き BCE の最適 logit は log w+_g だけずれる
w = {"群 X": (1 - 0.5) / 0.5, "群 Y": (1 - 0.1) / 0.1}  # 1, 9
x = np.linspace(-5, 7, 400)
pdf = lambda m: np.exp(-((x - m) ** 2) / 2)
fig, axes = plt.subplots(2, 1, figsize=(4.4, 3.4), dpi=200, sharex=True)
for ax, g in zip(axes, w):
    s = np.log(w[g])
    ax.fill_between(x, pdf(-1.5 + s), color=NEG, alpha=0.6, label="陰性")
    ax.fill_between(x, pdf(1.5 + s), color=POS, alpha=0.6, label="陽性")
    ax.axvline(0, color="black", ls="--", lw=1)
    ax.set_yticks([])
    ax.spines["left"].set_visible(False)
    ax.set_title(f"{g}：w+ = {w[g]:.0f}、logit のずれ log w+ = {s:.1f}", fontsize=9, loc="left")
axes[0].legend(frameon=False, fontsize=8.5, loc="upper right")
axes[1].set_xlabel("logit（破線：全群共通のしきい値）")
fig.tight_layout()
fig.savefig(out / "per-group-class-weight.png")
