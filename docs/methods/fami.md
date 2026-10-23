# FaMI

モデルの出力と敏感属性の相互情報量を罰則として CE に足し、その勾配だけを推定して学習する手法。
Sadri et al., MICCAI 2025。この repo では strategy `fami`。

## 1. 手法のロジック

### 解く問題

普通の学習（ERM）は予測に役立つなら属性（人種・性別など）に結びついた手がかりも使うので、予測が属性によって変わりうる。
FaMI は「予測 v が属性 u と独立なら公平」という定義（u ⊥ v）をとり、v と u の**相互情報量** I(u; v)
（v を知ると u についてどれだけ分かるか。独立なら 0）を下げる。

### 目的関数

v = f_θ(x) をモデルの出力（logits）、μ ≥ 0 を罰則の重みとする。

```
L(θ) = L_CE(θ) + μ · I(u; v(θ))                                  （式 1）
```

I(u; v) は分布が分からないと直接計算できない。FaMI は I の値を推定せず、勾配だけを作る。
連鎖律で θ の勾配を v の勾配に戻し、v の勾配には「相互情報量の微分の定理」（Babaie-Zadeh et al. 2004）を使う。

```
∇_θ I = J_vᵀ · ∇_v I(u; v)            J_v：v の θ についてのヤコビ行列        （式 2）
∇_v I(u; v) = E[λ(u, v)],  λ(u, v) = ∇_v ln p(u | v)                           （式 3）
```

λ は **SFD**（score function difference）と呼ばれる。score function は確率密度の対数の勾配 ∇ ln p で、
密度が増える向きを指す。λ は「v をどちらへ動かすと u が当てやすくなるか」を表し、その逆向きに v を動かすと
v から u の情報が減る。λ は多項式カーネルによる score の推定（Babaie-Zadeh & Jutten 2005）で、
密度そのものを推定せずに求める。

### 学習の手順（原論文 Algorithm 1）

ミニバッチ b ごとに次を行う。

1. 各標本の出力 v_i = f_θ(x_i) と予測 p_i = softmax(v_i) を出す。
2. バッチの v と u から λ(u_i, v_i) を推定する。
3. CE の勾配 ∇L_CE と、罰則の勾配 ∇L_MI = (1/|b|) Σ_i J_{v_i}ᵀ λ(u_i, v_i) を出す。
4. θ ← θ − η (∇L_CE + μ ∇L_MI) で更新する。

λ の推定はバッチの中の標本だけで行うので、1 群あたりの件数が少ないと推定が不安定になる。

## 2. 原論文の設定

- Sadri, DeSilvio, Viswanath. *Mutual Information Regularization for Fairness-aware Deep Imaging Representations.*
  MICCAI 2025, LNCS 15973, pp. 399–409（[論文ページ](https://papers.miccai.org/miccai-2025/0625-Paper1404.html)、
  [PDF](https://papers.miccai.org/miccai-2025/paper/1404_paper.pdf)）
- 公式実装：なし（論文ページの code は N/A）
- SFD の推定の出典：Babaie-Zadeh, Jutten, Nayebi. *Differential of the mutual information.* IEEE SPL 2004（文献 [20]）と、
  Babaie-Zadeh & Jutten. *A general approach for mutual information minimization and its application to blind source
  separation.* Signal Processing 2005（文献 [21]）
- LHCF（arXiv:2605.29827）は FaMI を比較手法に挙げるが、その文献 [34] は FairDomain（Tian et al., ECCV 2024）を指す。
  本文の名前と説明はこの論文のものなので、引用の誤りと判断した。

### データセットとタスク

どちらも train（クラスごとに 70%）と test に分け、クラスの比を保つ。

| データセット | 入力 | 予測ラベル（クラス数） | 敏感属性（群数） |
| --- | --- | --- | --- |
| C1：緑内障（Harvard の公開データ、3,300 件。train 2,100・test 1,200） | 3D OCT（RNFL 厚さマップと B-scan） | glaucoma / non-glaucoma（2 クラス） | race：Asian / Black / White（3 群、train は各 700 件で同数） |
| C2：直腸癌（4 施設、186 件。train 130・test 56） | 治療前の T2 強調 MRI（腫瘍周りを 79 × 79 × 40 voxel に切り出し） | 病理学的完全奏効 pCR / non-pCR（2 クラス） | sex（2 群） |

### ハイパーパラメータ

| 項目 | 値 |
| --- | --- |
| モデル | 3D の DenseNet-121（FaMI-DN）と ViT（FaMI-ViT） |
| optimizer | SGD、lr 1e-4、weight decay 1e-2 |
| batch・epoch | batch 64、150 epoch |
| 勾配クリップ | 最大ノルム 1.0 |
| 学習の分け方 | train の中で 10-fold の cross-test |
| データ拡張 | 記載なし（未確認） |
| μ | 0.5（先行研究に基づき、本研究でも確認したと書く。探索範囲は記載なし） |
| SFD の推定 | 多項式カーネル（[20][21]）。基底・次数・正則化は記載なし |
| バッチの大きさ | 「SFD の推定はバッチの大きさと埋め込みの次元に依るかもしれないが、安定していた」 |

### モデルの選び方と評価

checkpoint・ハイパーパラメータの選び方は記載なし。評価は AUC、EOM（群間の TPR の均等さ）、EOdd（群間の TPR と FPR の差）で、
train と test の両方を報告する。手法の比較は Wilcoxon 検定（Bonferroni 補正、p = 0.005）。

## 3. この repo の実装

### ファイルと設定

| 役割 | 場所 |
| --- | --- |
| 損失 | [objectives/cohort.py](../../projects/hypernet_e2e/objectives/cohort.py) の `FaMITaskLoss`（λ は `score_difference`） |
| score の推定 | 同じ file の `polynomial_log_density_gradient` と `_polynomial_basis` |
| strategy | [configs/training_strategy/fami.yaml](../../projects/hypernet_e2e/configs/training_strategy/fami.yaml)（`mi_weight` が μ） |
| クラス重み | [training.py](../../projects/hypernet_e2e/training.py) の `_resolve_inverse_class_weights` |
| 診断値の記録 | [callbacks/objective_diagnostics.py](../../projects/hypernet_e2e/callbacks/objective_diagnostics.py) の `ObjectiveDiagnosticsRecorder` |
| テスト | [tests/test_cohort_objectives.py](../../projects/hypernet_e2e/tests/test_cohort_objectives.py)（score の推定がガウス分布の score を再現すること） |
| 代表的な preset | `resnet_ham10000_fg_fami_sex_age`（性別 × 年齢の 4 群）、`resnet_ham10000_fg_fami_lhcf`（LHCF の 11 コホート、[worst-cohort.md](worst-cohort.md)） |

### 処理の流れ

**起動時（fit の前）。** HAM10000 の preset は `weighting: inverse` なので、`_resolve_inverse_class_weights` が
全群共通のクラス重み（平均 1 に正規化した逆頻度）を `class_weight` に書き込む。重みは CE だけに掛かり、罰則には掛からない。

**学習の各ステップ（`FaMITaskLoss.forward`）。**

1. v を logits そのもの（`[B, 2]`）とし、クラス重み付き CE のバッチ平均を出す。
2. `score_difference`：u が離散なので、ベイズの定理で λ = ∇_v ln p(u|v) = ∇_v ln p(v|u) − ∇_v ln p(v) と分ける
   （p(u) は v によらない）。バッチ全体の v から ∇ ln p(v) を、群ごとの v から ∇ ln p(v|u) を推定して引く。v は detach する。
3. `polynomial_log_density_gradient`：score を基底 k(v) = [1, v_j, v_j², v_j³]（各次元、交差項なし。2 次元なら 7 個）の
   線形結合で近似し、E[k kᵀ] w = E[∂k/∂v] を標本平均で解く。正則化なし、擬似逆行列、倍精度。
4. 基底の数（1 + 3 × クラス数 = 7）より件数の少ない群は λ = 0 にする。
5. `CE + μ · mean_i(λ_i · v_i)` を返す。λ を detach しているので、罰則の勾配は (1/B) Σ_i J_{v_i}ᵀ λ_i になり、Algorithm 1 と一致する。

**設計上の選択。** 公式実装が無いので、p(u|v) の score を条件付きと周辺の差で求める点、基底を 3 次・交差項なしにした点、
小さい群を 0 にする点はこちらで決めた。1 群あたりの件数が基底の数に近いと、群と v が独立でも |λ| が大きく出る
（推定の雑音が信号を上回る）。11 コホート・batch 128 では 1 群平均 12 件前後で、この状況に近い。

**記録と checkpoint。**

- `ObjectiveDiagnosticsRecorder` が epoch ごとに `metrics/objective_diagnostics.jsonl` へ、罰則（`penalty`、μ を掛ける前）、
  群ごとの ‖λ‖ の平均（`lambda_norm`）、λ を推定できた群（`lambda_active`）を書く。依存が無くても ‖λ‖ は推定の揺らぎで
  0 にならないので、大きさそのものより群の間の差で読む。
- checkpoint は best val AUROC（`best_val_auroc_*.ckpt`）、`last.ckpt`、val の性別 × 年齢の最悪群 AUROC で選んだ
  `best_val_worst_sexage_*.ckpt`（`_fairness_ham10000.yaml`）を残す。原論文に選び方の記載がないので、他手法と同じ基準で選び、
  どれを使ったかを記録する。予測は `analysis/common/predictions.py` に `--checkpoint <名前>` で渡す。

### 原論文との違い

| 項目 | 原論文 | この repo |
| --- | --- | --- |
| データ・タスク | 3D OCT・MRI、2 クラス | HAM10000 の皮膚画像、良性 / 悪性の 2 クラス |
| 群 | race 3 群（各群同数）か sex 2 群 | 性別 × 年齢の 4 群か LHCF の 11 コホート |
| 1 群あたりの件数（バッチ平均） | 約 21（3 群・batch 64） | 32（4 群）か約 12（11 群）。小さいコホートは λ = 0 になりやすい |
| SFD の推定の細部 | 記載なし | 各次元 3 次・交差項なし・正則化なし・擬似逆行列。条件付き − 周辺で分解。小さい群は 0 |
| モデル | 3D DenseNet-121 / ViT | ResNet-50（ImageNet 事前学習） |
| 最適化 | SGD lr 1e-4・weight decay 1e-2・150 epoch・batch 64 | AdamW lr 1e-4・weight decay 0.01・30 epoch・batch 128 |
| 勾配クリップ | 最大ノルム 1.0 | なし（既定）。雑音の大きい λ がそのまま勾配に入る |
| クラス重み | なし | 逆頻度（CE だけ） |
| データ拡張 | 記載なし | あり（`train_transforms_medfair`） |

原論文に近づけるなら、まず勾配クリップを足す（下の起動例）。λ の推定を安定させるには群を少なくする（4 群）か
バッチを大きくする（`data.batch_size`）。LHCF のコホートは train でクラスとの重なりが大きいので（[worst-cohort.md](worst-cohort.md)）、
正解クラスで条件付けない FaMI でコホートから独立にすると、ラベルから独立にすることに近くなるおそれがある（推測）。

### 起動例

```bash
uv run python -m projects.hypernet_e2e.run experiment=resnet_ham10000_fg_fami_sex_age study=<slug> seed=42
# 原論文の勾配クリップを足す（trainer config にキーが無いので + を付ける）
uv run python -m projects.hypernet_e2e.run experiment=resnet_ham10000_fg_fami_sex_age study=<slug> seed=42 +trainer.gradient_clip_val=1.0
```
