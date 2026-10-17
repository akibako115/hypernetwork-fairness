# 最悪コホート損失 ℓ_wst（LHCF）：原論文・実装・使い方

2026-10-16 作成。LHCF の本文を読み直し、この repo の実装（HEAD `0f2af99`）とコホートの作り方を突き合わせた結果をまとめる。
ℓ_wst や LHCF のコホートを群に使う run を設計・起動・評価するセッションは、先にこれを読む。

## 先に押さえること

- **LHCF の提案は「群の作り方」で、損失は単純。** 画像の埋め込みを GMM でクラスタに分け、そのクラスタを属性の代わりに
  既存の公平性手法へ入れる。このクラスタを**コホート**（隠れコホート）と呼ぶ。属性ラベルを使わず、見た目だけで作った患者の部分集合のこと。
  ℓ_wst は本文が例に挙げる公平性損失の 1 つで、「平均損失が最大のコホートの損失」を CE に足す。
- **λ・最適化・epoch 数・checkpoint の選び方は本文に無く、コードも未公開。** この repo の λ = 1 はこちらで決めた値。
- **この repo のコホートは、train ではほぼラベルで割れている。** 11 個のうち 6 個は train の悪性率が 0.5% 以下、2 個は 87% 以上
  （下の表）。最悪コホートは悪性の多いコホートになりやすく、ℓ_wst はクラス重み付けに近い働きをするおそれがある（推測）。
  これを確かめる対照として「群 = クラス」の preset（`*_swadwst_label`）がある。
- **最悪の取り方はバッチの中。** 論文の R_k はコホート全体の平均損失だが、実装はバッチに現れたコホートの中で最大を取る。
  ミニバッチで学習する以上避けられない近似。
- 既存の run は 30 本（2026-10-10〜11、10 構成 × seed 42〜44）。損失のコードはそれ以降変わっていない（下の照合結果）。

## 原論文と公式実装

- Masroor, Nguyen, Wells, Carneiro. *Fairness Beyond Demographics: Optimizing Performance Across Appearance-Based
  Hidden Cohorts in Medical Imaging.* [arXiv:2605.29827](https://arxiv.org/abs/2605.29827)（2026-05）
- 公式実装：なし（本文に「Code will be available upon paper acceptance」）

### 原論文の条件（本文で確認済み）

| 項目 | 原論文 | 出典 |
| --- | --- | --- |
| 目的 | (1/N) Σ ℓ_clss + λ ℓ_fair。ℓ_wst = max_k R_k、R_k はコホート C_k の平均分類損失。ほかに ℓ_gap = max_k R_k − min_k R_k | 2.1 節、式 (2)(3) |
| λ | λ ≥ 0 のハイパーパラメータ。値の記載なし | 2.1 節 |
| コホートの作り方 | ERM（か自己教師あり）で学習した encoder の埋め込みに全共分散の GMM を EM で当てる。K は BIC 最小。各標本は責務（その成分に属する事後確率）最大の成分に割り当てる | 2.1 節、式 (1) |
| 埋め込みと K | ERM の ResNet18 で、HAM10000 は K = 7。MedCLIP は 6、CLIP は 10、DINOv2 は 5 | 3.1 節 Table 1、3.2 節 |
| 書かれていないこと | 次元削減の有無、K の探索範囲（式の上では 1〜N）、val・test への割り当て方、step (ii) の初期値 | — |
| 他手法への組み込み | 「LHCF はモデルにも損失にも依らない」。SWAD・FIS・FEBS・FairCLIP-OT・FaMI・FairDi の属性の入る所にコホートを入れる。各手法での入れ方、ℓ_wst と ℓ_gap のどちらを使ったかは未記載 | 2.1 節、2.2 節 |
| データ | HAM10000（9,948 枚、良性 / 悪性）、Fitzpatrick17K、CMMD。80/10/10（CMMD は 70/10/20） | 2.2 節 |
| 評価する群 | Gender、Age_binary（≤60 / >60）、Age_multi4、Gender × Age_binary、Gender × Age_multi4 | 2.2 節 |
| モデル・最適化・checkpoint | 記載なし（Fig. 1 の FairDi は ResNet18） | — |
| 指標 | AUC、AUC Gap、Worst-case AUC、ES-AUC、PSD（Mean・Max）。Friedman 検定 + Nemenyi | 2.2 節 |
| 理論 | 属性群がコホートの和集合なら、属性群の損失は max_k R_k 以下（Lemma 1） | 2.1 節 |

## この repo の実装

**損失**：`projects/hypernet_e2e/objectives/cohort.py:26–55` の `WorstCohortTaskLoss`。

- 標本ごとの CE のバッチ平均に、λ × 「バッチに現れたコホートの平均損失の最大」を足す（L49–55）。
  コホート平均は `objectives/group.py:61` の `_group_losses`。
- `training_strategy=worst_cohort`（`configs/training_strategy/worst_cohort.yaml`）で選ぶ。`fairness_weight` が λ。
- クラス重み：strategy は `class_weight: null` だが、HAM10000 の preset は `weighting: inverse`（`_base_ham10000.yaml`）なので、
  起動時に `training.py:229` が train の逆頻度（陽性 ≈ 2.54、陰性 ≈ 0.62）を入れる。平均 CE と R_k の両方に掛かる。
- **記録されるもの**：通常の `metrics.csv`（train loss、val の全体・群別指標）と checkpoint に加え、
  `metrics/objective_diagnostics.jsonl`（1 行 = 1 epoch）に `group_loss`（群の平均損失、居ない群は 0）・`group_present`・
  `worst_group`（最悪の群の one-hot）を残す。`epoch_mean` の `worst_group` が各コホートが最悪になった割合、
  `group_loss / group_present` が居た batch での平均損失。2026-10-17 から `ObjectiveDiagnosticsRecorder` が既定の callback に入り、全 preset で書き出す。既存の 30 本には残っていない。

**コホートの供給**：群は split CSV の列から `GroupImageDataModule` が作る。LHCF のコホートは
`configs/data/ham10000_lhcf_groups.yaml`（`cv_splits_dir: data/ham10000/splits_lhcf`、列 `lhcf`、11 群）。
列は `analysis/archive/latent_bias_hypernet/lhcf_cohorts.py` が次の手順で足した。

1. ResNet-50 ERM（`resnet-ham10000-fg-erm` の seed 42）の train・val・test の埋め込み（2048 次元、cache から読む）
2. train で標準化と PCA 32 次元を当てる
3. 全共分散の GMM を K = 2〜12（`n_init=5`）で当て、BIC 最小を採る → K = 11（BIC 1,373,660。K = 12 は 1,375,348）
4. train・val・test を同じ変換と GMM の `predict`（責務最大）で割り当てる。ラベルも属性も使わない
5. 対照の列も足す：`lhcf_random`（split ごとに `lhcf` を並べ替えた、大きさだけ同じ群）、`label_group`（クラスそのもの）

結果は `analysis/archive/latent_bias_hypernet/results/ham10000_e2e/lhcf_bic.csv` と `lhcf_cohorts.csv`。
コホートごとの件数と悪性率（train）：

| コホート | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 件数 | 911 | 539 | 1197 | 622 | 571 | 747 | 803 | 749 | 414 | 665 | 1239 |
| 悪性率 | 0.000 | 0.494 | 0.000 | 0.873 | 0.037 | 0.001 | 0.005 | 0.000 | 0.944 | 0.000 | 0.000 |

val・test ではラベルとの対応が弱まる（例：コホート 0 は val で 9/148 が悪性）。埋め込みを作った ERM が train のラベルを
覚えているためと読める（推測）。ほかの群の data config は `ham10000_sex_age_groups.yaml`（性別 × 65 歳の 4 群）、
`ham10000_lhcf_random_groups.yaml`、`ham10000_label_groups.yaml`、2 段学習用の `ham10000_part_groups.yaml`
（`crossfit.py` の `lhcf` 区分。1 段目の stage2 埋め込みに同じ手順を当てたもの）。

### 原論文との違い

| 項目 | 原論文 | この repo |
| --- | --- | --- |
| コホートの埋め込み | ERM の ResNet18（512 次元） | ERM の ResNet-50（2048 次元）→ 標準化 → PCA 32。全共分散 GMM を 2048 次元に当てられないため |
| K | BIC で 7 | BIC（2〜12）で 11。K = 13 は BIC が上がり、14 以上は当てられない（[記録](../reviews/20261015_lhcf_paper_alignment.md)） |
| 最悪の取り方 | R_k はコホート全体の平均（式の上では） | バッチに現れたコホートの中の最大 |
| λ | 記載なし | 1 |
| クラス重み | 記載なし | 逆頻度 |
| 分割 | 80/10/10 | train・val は lesion_id 単位、test は ISIC2018 Task3 の公式 test |
| 評価の年齢の境目 | 60 歳 | 65 歳（`age_group_65`） |
| 最適化 | 記載なし | AdamW lr 1e-4、weight decay 0.01、batch 128、30 epoch、データ拡張あり |
| SWAD + LHCF | 組み込み方の記載なし | SWAD + ℓ_wst と解釈した（推測。SWAD 自体は群を使わない） |

## 起動

| preset（`configs/experiment/`） | 内容 |
| --- | --- |
| `resnet_ham10000_fg_wst_lhcf` | ResNet-50、ℓ_wst、群は LHCF のコホート |
| `resnet_ham10000_fg_wst_sex_age` | 同上、群は性別 × 年齢（属性で学習する対照） |
| `resnet_ham10000_fg_swadwst_{lhcf,sex_age,lhcfrand,label}` | SWAD + ℓ_wst。群をコホート・属性・ランダム・クラスで替える |
| `spatial_lora_ham10000_fg_swadwst_lhcf` | Spatial LoRA（条件は定数）、SWAD + ℓ_wst |
| `spatial_lora_ham10000_fg_lhcfcond_swadwst_lhcf` | 同上、コホートで条件付けもする |
| `spatial_lora_ham10000_cf_{const,lhcf}_wst` | 2 段学習の 2 段目。群は `crossfit.py` の `lhcf` 区分 |

```bash
uv run python -m projects.hypernet_e2e.run experiment=resnet_ham10000_fg_wst_lhcf seed=42 study=<slug>
# λ を変える
uv run python -m projects.hypernet_e2e.run experiment=resnet_ham10000_fg_wst_lhcf seed=42 study=<slug> \
  model.loss_fn.fairness_weight=0.5
```

コホートを作り直すときは `lhcf_cohorts.py` を回す。`splits_lhcf` を上書きするので、既存 run との対応が切れる点に注意する。

## checkpoint の選び方と予測の使い方

- 原論文には選び方の記載がない。preset は `_fairness_ham10000.yaml` により、best val AUROC（`best_val_auroc_*.ckpt`）、
  val の性別 × 年齢の最悪群 AUROC（`best_val_worst_sexage_*.ckpt`）、`last.ckpt` を残す。SWAD の preset は `swad.ckpt` を評価する。
- **LHCF の前提（学習に属性ラベルを使わない）に合わせるなら、best val AUROC か `swad.ckpt` を使う。** 性別 × 年齢で選ぶと、
  checkpoint の選択で属性ラベルを使うことになる。どちらで比べるかは比較の目的に合わせて決め、run の記録に書く。
- コホートの最悪 AUROC で選ぶことはできない。val ではコホート 2 と 10 に悪性が 1 件も無く、AUROC が定まらない。
- 予測は `analysis/common/predictions.py`。best 以外を使うときは `--checkpoint` は名前で渡す（既定 `best` は run.json に記録された val AUROC の best、`xxx` は `checkpoints/xxx.ckpt`。例 `best_val_auroc_009`、`last`、`swad`）。

## 2026-10-16 の照合結果

- **一致を確かめたこと**：損失の形（CE + λ max_k R_k、式 (2)(3)）。コホートの作り方（全共分散 GMM、BIC、責務最大の割り当て）。
- **論文に無く、こちらで決めたこと**：λ = 1、PCA 32、K の探索範囲 2〜12、val・test への割り当て方、クラス重み、
  バッチ内での最大。「SWAD + LHCF」を SWAD + ℓ_wst とした解釈。
- **既存 run**：30 本はすべて 2026-10-10〜11 に、`b53c4f8` の上の未 commit の作業ツリーで回したもの。`WorstCohortTaskLoss` と
  `_group_losses` は、初めて commit された `6a09df9`（10-13）から HEAD まで変わっていない。run 時点のコードも同じと読むが、
  git では確かめられない（推測）。コホート（`splits_lhcf`）も 10-10 から変えていない。
- コードの docstring は LHCF を「MICCAI 2026」と書くが、arXiv 版の本文では採択を確認できない。
