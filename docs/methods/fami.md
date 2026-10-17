# FaMI（Sadri et al.）：原論文・実装・使い方

2026-10-16 作成。FaMI の本文を読み直し、この repo の実装（HEAD `0f2af99`）と突き合わせた結果をまとめる。
FaMI の run を設計・起動・評価するセッションは、先にこれを読む。

## 先に押さえること

- **FaMI は「モデルの出力 v と群 u の相互情報量 I(u; v)」を罰則として CE に足す。** I を直接は推定せず、その勾配だけを
  **score function**（確率密度の対数の勾配 ∇_v ln p(v)。密度が増える向きを指す）の差から作る。この差を
  **SFD**（score function difference）λ(u, v) = ∇_v ln p(u|v) と呼ぶ。
- **公式実装は無い。** SFD の推定は引用先（Babaie-Zadeh & Jutten）の多項式推定に委ねられており、細部はこちらで決めた。
- **11 群・バッチ 128 では、λ の推定がほぼ雑音。** 1 群あたり平均 12 件で、多項式の係数 7 個を解く。群と v が独立な人工データでも
  |λ| の平均が約 13 になる（2 群なら 0.44、11 群でもバッチ 1024 なら 0.45）。論文の設定は 3 群・バッチ 64（1 群 21 件前後）。
- **原論文は勾配クリップ（最大ノルム 1.0）を使う。この repo には無い。** 雑音の大きい λ がそのまま勾配に入る。
- **LHCF のコホートで使うと、分類そのものとぶつかる（推測）。** FaMI は正解クラスで条件付けずに v を群から独立にする。
  この repo のコホートは train ではほぼ良性 / 悪性で割れている（[worst-cohort.md](worst-cohort.md)）ので、
  v をコホートから独立にすることは、v をラベルから独立にすることに近い。
- **既存の FaMI run 6 本（2026-10-10）は使わない。** 10-15 の修正より前の版（ガウス KDE、1 次元の logit 差）。

## 原論文と公式実装

- Sadri, DeSilvio, Viswanath. *Mutual Information Regularization for Fairness-aware Deep Imaging Representations.*
  MICCAI 2025（[論文ページ](https://papers.miccai.org/miccai-2025/0625-Paper1404.html)）
- 公式実装：なし
- LHCF（arXiv 2605.29827）の文献 [34] は FairDomain（Tian et al., ECCV 2024）を指すが、本文の説明と名前はこの論文のもの。引用の誤りと判断した。
- SFD の出典：Babaie-Zadeh, Jutten, Nayebi. *Differential of the mutual information.* IEEE SPL 2004（[20]）と、
  Babaie-Zadeh & Jutten. *A general approach for mutual information minimization and its application to blind source separation.*
  Signal Processing 2005（[21]）

### 原論文の条件（本文で確認済み）

| 項目 | 原論文 | 出典 |
| --- | --- | --- |
| 目的 | L(θ) = L_CE(θ) + μ I(u; v(θ))。v = f_θ(x) はモデルの出力、u は属性 | 3 節、式 (1) |
| I の勾配 | ∇_θ L_MI = J_vᵀ ∇_v I(u; v)（J_v は v の θ についてのヤコビ行列）。∇_v I = E[λ(u, v)]、λ = ∇_v ln p(u\|v) | 式 (2)(3) |
| 更新 | バッチ b で ∇_θ L_MI = (1/\|b\|) Σ_i J_{v_i}ᵀ λ(u_i, v_i)。θ ← θ − η(∇L_CE + μ ∇L_MI)。p_i = softmax(v_i) | Algorithm 1 |
| SFD の推定 | 多項式カーネルによる推定（[20][21]）。基底・次数・正則化の記載なし | 3 節 |
| μ | 0.5（先行研究に基づき、本研究でも確認） | 4.2 節 |
| データ | C1：緑内障の OCT（3,300 件中 train 2,100・test 1,200、race 3 群は各群同数）。C2：直腸癌の MRI 186 件、sex 2 群 | 4.1 節、Table 1 |
| モデル | 3D の DenseNet-121 と ViT | 4.2 節 |
| 最適化 | SGD、lr 1e-4、weight decay 1e-2、150 epoch、batch 64、**勾配クリップ（最大ノルム 1.0）**。train で 10-fold の cross-test | 4.2 節 |
| バッチの大きさ | 「SFD の推定はバッチの大きさと埋め込みの次元に依るかもしれないが、安定していた」 | 4.2 節 |
| checkpoint | 記載なし | — |
| 指標 | AUC、EOM（群間の TPR の均等さ）、EOdd。Wilcoxon（Bonferroni 補正） | 4.3 節 |

## この repo の実装

`projects/hypernet_e2e/objectives/cohort.py:159–259`。`training_strategy=fami`（`configs/training_strategy/fami.yaml`、`mi_weight` が μ）。

- **v**：logits そのもの（`[B, 2]`）。
- **λ の作り方**（`FaMITaskLoss.score_difference`、L234–252）：u が離散なので、ベイズの定理で
  λ = ∇_v ln p(u|v) = ∇_v ln p(v|u) − ∇_v ln p(v) と分ける（p(u) は v によらない）。
  バッチ内の群ごとの v から ∇ ln p(v|u) を、バッチ全体の v から ∇ ln p(v) を推定して引く。v は detach する。
- **score の推定**（`polynomial_log_density_gradient`、L182–199）：score を基底 k(v) = [1, v_j, v_j², v_j³]（各次元、交差項なし、
  2 次元なら 7 個）の線形結合で近似し、E[k kᵀ] w = E[∂k/∂v] を標本平均で解く。正則化なし、擬似逆行列、倍精度。
  基底は `_polynomial_basis`（L159–179）。
- **小さい群**：基底の数（1 + 3 × クラス数 = 7）より件数の少ない群は λ = 0。
- **損失**（L254–259）：CE + μ · mean_i(λ_i · v_i)。λ を detach しているので、罰則の勾配は (1/B) Σ_i J_{v_i}ᵀ λ_i で、Algorithm 1 と同じ。
- **クラス重み**：HAM10000 の preset は `weighting: inverse` なので、CE に逆頻度の重みが掛かる（罰則には掛からない）。
- **記録されるもの**：通常の `metrics.csv` と checkpoint に加え、`metrics/objective_diagnostics.jsonl`（1 行 = 1 epoch）に
  罰則 `penalty`（μ を掛ける前）、群ごとの ‖λ‖ の平均 `lambda_norm`、λ を推定できた群 `lambda_active` を残す。
  依存が無くても ‖λ‖ は推定の揺らぎで 0 にならないので、群の間の差で読む。2026-10-17 から `ObjectiveDiagnosticsRecorder` が既定の callback に入り、全 preset で書き出す。

λ の大きさ（人工データ。2 次元の v、群ごとの件数は一様に割り当て、200 バッチの平均。2026-10-16 に再実行）：

| 群の数 | バッチ | 群と v が独立 | 群 0 だけ v を 1 ずらす |
| --- | --- | --- | --- |
| 2 | 128 | 0.44 | 0.67 |
| 4 | 128 | 0.93 | 1.04 |
| 11 | 128 | 13.47 | 4.66 |
| 11 | 1024 | 0.45 | 0.53 |

11 群・バッチ 128 では、依存の無いときのほうが |λ| が大きい。推定の雑音が信号を上回っている。

### 原論文との違い

| 項目 | 原論文 | この repo |
| --- | --- | --- |
| 群 | race 3 群（各群同数）か sex 2 群 | 性別 × 年齢の 4 群か LHCF の 11 コホート |
| 1 群あたりの件数（バッチ平均） | 約 21（3 群・batch 64） | 32（4 群）か約 12（11 群）。最小のコホートは約 6 件で、λ = 0 になりやすい |
| SFD の推定の細部 | 記載なし | 各次元 3 次・交差項なし・正則化なし・擬似逆行列。条件付き − 周辺で分解。小さい群は 0 |
| 最適化 | SGD、150 epoch、batch 64、勾配クリップ 1.0 | AdamW lr 1e-4、weight decay 0.01、30 epoch、batch 128、クリップなし |
| モデル・データ | 3D DenseNet / ViT、OCT・MRI | ResNet-50、HAM10000 |
| クラス重み | なし | 逆頻度 |

## 起動

| preset（`configs/experiment/`） | 群 |
| --- | --- |
| `resnet_ham10000_fg_fami_lhcf` | LHCF の 11 コホート |
| `resnet_ham10000_fg_fami_sex_age` | 性別 × 年齢の 4 群 |

```bash
uv run python -m projects.hypernet_e2e.run experiment=resnet_ham10000_fg_fami_sex_age seed=42 study=<slug>
# 原論文の勾配クリップを足す（trainer config にキーが無いので + を付ける）
uv run python -m projects.hypernet_e2e.run experiment=resnet_ham10000_fg_fami_sex_age seed=42 study=<slug> \
  +trainer.gradient_clip_val=1.0
```

提案：回すなら 4 群（性別 × 年齢）と勾配クリップから始める。11 コホートでは上の表のとおり λ が雑音になるので、
バッチを大きくできない限り結果を FaMI の効果として読みにくい。

## checkpoint の選び方と予測の使い方

- 原論文に記載がない。preset は `_fairness_ham10000.yaml` により、best val AUROC（`best_val_auroc_*.ckpt`）、
  val の性別 × 年齢の最悪群 AUROC（`best_val_worst_sexage_*.ckpt`）、`last.ckpt` を残す。
  他手法と同じ基準で選び、どれを使ったかを run の記録に書く。LHCF のコホートで学習した run を性別 × 年齢で選ぶと、
  属性ラベルを選択に使うことになる。
- 予測は `analysis/common/predictions.py`。best val AUROC 以外は `--checkpoint` は名前で渡す（既定 `best` は run.json に記録された val AUROC の best、`xxx` は `checkpoints/xxx.ckpt`。例 `best_val_auroc_009`、`last`、`swad`）。

## 2026-10-16 の照合結果

- **一致を確かめたこと**：λ = ∇_v ln p(u|v) = ∇ ln p(v|u) − ∇ ln p(v) の分解。罰則 λ.detach()·v の勾配が Algorithm 1 の
  (1/|b|) Σ J_vᵀ λ と一致すること。多項式の score 推定の符号（ガウス分布の score を再現する。
  `tests/test_cohort_objectives.py`）。μ = 0.5。
- **確かめていないこと**：基底が Babaie-Zadeh & Jutten (2005) の式 (29)–(32) に対応するか（コードのコメントは (29)–(32)、
  [reviews/20261015_lhcf_paper_alignment.md](../reviews/20261015_lhcf_paper_alignment.md) は (29)–(35) と書き、食い違う）。
- **論文と違う点**：勾配クリップが無い。11 群・バッチ 128 では λ の推定が雑音（上の表）。
- **実装の履歴と既存 run**：10-15 までは v を 1 次元の logit 差にし、score をガウス KDE（Silverman の幅）で推定していた。
  10-15 に今の版へ直し、10-16 に `671750e` で commit した。既存の 6 本（`resnet-ham10000-fg-fami-{lhcf,sex-age}`、
  seed 42〜44、10-10）は前の版で、使わない。今の版で回した run はまだ無い。
