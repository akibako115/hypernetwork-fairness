# 最悪コホート損失 ℓ_wst（LHCF）

平均の分類損失に、平均損失が最も大きいコホートの損失を λ 倍して足す手法。LHCF（Masroor et al., arXiv 2026）の式 (3)。
LHCF の提案の中心は群の作り方（画像の埋め込みを GMM で分けたコホートを属性の代わりに使う）で、ℓ_wst はその上で使う
公平性損失の 1 つ。この repo では strategy `worst_cohort`。群は data config で、LHCF のコホートにも属性にも替えられる。

## 1. 手法のロジック

### 解く問題

普通の学習（ERM）は全サンプルの平均損失を下げるので、少数の部分集団の成績が悪くても放置されやすい。
公平性の手法の多くは性別・年齢などの属性で群を作り、群の間の差を縮める。LHCF はこの前提の 2 つの弱点を挙げる。

- 属性は、モデルが実際に間違える部分集団とあまり一致しないことがある（hidden stratification）。
- 属性を組み合わせると群の数が増え、1 群あたりの件数が減る。

LHCF は、属性の代わりに画像の見た目から作った部分集団（隠れコホート）を群として、既存の公平性手法に入れる。
学習に属性ラベルは使わない。

### 手順 (i)：コホートを作る

1. ERM か自己教師あり学習で学習済みの encoder で、train の画像を埋め込みに変える。
2. 埋め込みに全共分散の GMM（Gaussian Mixture Model）を EM で当てる。
3. 成分の数 K は BIC（Bayesian Information Criterion：対数尤度にパラメータ数の罰則を足した量）が最小のものを選ぶ。
4. 各画像を、責務（その成分に属する事後確率）が最大の成分に割り当てる。これがコホート c_i になる。

### 手順 (ii)：コホートで公平性を最適化する

```
min_θ  (1/N) Σ_i ℓ_clss(y_i, f_θ(x_i))  +  λ · ℓ_fair

ℓ_wst = max_k R_k(θ)              R_k：コホート C_k に属するサンプルの平均分類損失
ℓ_gap = max_k R_k − min_k R_k     （もう 1 つの例）
λ ≥ 0：ハイパーパラメータ
```

ℓ_wst の勾配は、その時点で最悪のコホートのサンプルにだけ追加でかかる。GroupDRO が群の重み q を少しずつ動かして
最悪群を追うのに対し、ℓ_wst は毎回の最大を直接取り、平均損失に足す。

### 理論（Lemma 1）

コホートが互いに重ならず、属性の群（例：女性）がいくつかのコホートの和集合になっているなら、その群の平均損失は
max_k R_k 以下になる。したがって ℓ_wst を下げると、どの属性の群の損失の上限も下がる。

## 2. 原論文の設定

- Masroor, Nguyen, Wells, Carneiro. *Fairness Beyond Demographics: Optimizing Performance Across Appearance-Based Hidden
  Cohorts in Medical Imaging.* [arXiv:2605.29827](https://arxiv.org/abs/2605.29827)（2026-05、v1）
- 公式実装：なし（本文に「Code will be available upon paper acceptance」）。採択先は本文で確認できない。

### データセットとタスク

どれも **2 値分類**。評価に使う属性の群（visible cohort）は学習には使わない。

| データセット | 入力 | 予測ラベル（クラス数） | 評価する属性の群（群数） | 分割 |
| --- | --- | --- | --- | --- |
| HAM10000（9,948 枚） | ダーモスコピー画像 | 良性 / 悪性（2 クラス） | Gender（2）、Age_binary ≤60 / >60（2）、Age_multi4（4）、Gender × Age_binary（4）、Gender × Age_multi4（8） | 80/10/10 |
| Fitzpatrick17K | 皮膚の臨床画像 | 良性（非腫瘍性 + 良性）/ 悪性（2 クラス） | Fitzpatrick の肌の型 0〜5（6） | 80/10/10 |
| CMMD（5,152 枚） | マンモグラフィ | 非がん / がん（2 クラス） | 年齢 ≤55 / >55（2） | 70/10/20 |

HAM10000 では、ERM の ResNet18 の埋め込みで BIC が K = 7 を選んだ。MedCLIP では 6、CLIP では 10、DINOv2 では 5。
Fitzpatrick17K と CMMD の K は本文に無い。

### ハイパーパラメータ

| 項目 | 値 |
| --- | --- |
| コホートの埋め込み | ResNet18（ERM）、CLIP、MedCLIP、DINOv2 を比べる。ResNet18 と MedCLIP が良い |
| GMM | 全共分散、EM。K は BIC 最小（式の上では 1〜N）。感度分析で K ∈ {3, 5, 9} を試し、K = 7 が最良 |
| λ | λ ≥ 0 のハイパーパラメータ。値は本文に無い |
| 埋め込みの次元削減・val と test への割り当て方 | 本文に無い |
| 組み込む手法 | ERM、SWAD、FIS、FaMI、FEBS、FairCLIP-OT、FairDi の「属性が入る所」にコホートを入れる。手法ごとの入れ方と、ℓ_wst・ℓ_gap のどちらを使ったかは本文に無い |
| モデル・optimizer・lr・batch・epoch・データ拡張 | 本文に無い（Fig. 1 の FairDi は ResNet18） |

### モデルの選び方と評価

checkpoint とハイパーパラメータの選び方は本文に無い。評価は属性の群で行い、AUC、AUC Gap、Worst-case（Min）AUC、
ES-AUC、PSD（Mean・Max）を使う。手法の比較は Friedman 検定と Nemenyi の事後検定（p < 0.05）。
コホートの質は Brier score と Average Purity（コホートと属性の一致度）で測る。

## 3. この repo の実装

### ファイルと設定

| 役割 | 場所 |
| --- | --- |
| 損失 | [objectives/cohort.py](../../projects/hypernet_e2e/objectives/cohort.py) の `WorstCohortTaskLoss`。群平均は [objectives/group.py](../../projects/hypernet_e2e/objectives/group.py) の `_group_losses` |
| strategy | [configs/training_strategy/worst_cohort.yaml](../../projects/hypernet_e2e/configs/training_strategy/worst_cohort.yaml)（`fairness_weight` が λ、既定 1） |
| クラス重みの計算 | [training.py](../../projects/hypernet_e2e/training.py) の `_resolve_inverse_class_weights` |
| 群の定義 | [ham10000_lhcf_groups.yaml](../../projects/hypernet_e2e/configs/data/ham10000_lhcf_groups.yaml)（LHCF のコホート、11 群）、`ham10000_sex_age_groups.yaml`（性別 × 65 歳の 4 群）、対照の `ham10000_lhcf_random_groups.yaml`・`ham10000_label_groups.yaml`、2 段学習用の `ham10000_part_groups.yaml` |
| 診断値の記録 | [callbacks/objective_diagnostics.py](../../projects/hypernet_e2e/callbacks/objective_diagnostics.py) の `ObjectiveDiagnosticsRecorder` |
| 代表的な preset | `resnet_ham10000_fg_wst_lhcf`（群はコホート）、`resnet_ham10000_fg_wst_sex_age`（群は属性）、`resnet_ham10000_fg_swadwst_lhcf`（SWAD + ℓ_wst） |

### 処理の流れ

**起動時（fit の前）。** HAM10000 の preset は `weighting: inverse`（`_base_ham10000.yaml`）なので、
`_resolve_inverse_class_weights` が train のラベルから全群共通のクラス重み（平均 1。HAM10000 の train では陰性 0.29、
陽性 1.71）を `class_weight` に書く。strategy の `class_weight: null` はこれで上書きされる。
data は `GroupImageDataModule` で、split CSV の列（`lhcf` など）から群の番号を作る。train は自然抽出（`train_sampling: uniform`）。

**学習の各ステップ（`WorstCohortTaskLoss.forward`）。**

1. `_weighted_ce`：サンプルごとの CE にクラス重みを掛ける。
2. `_group_losses`：バッチに現れた群ごとに、1 の和を群の件数で割る。バッチに居ない群は除く。
3. バッチに現れた群の平均損失の最大を取る。
4. 1 のバッチ平均 + λ × 3 を返し、これでモデルを更新する。クラス重みは平均と R_k の両方に掛かる。

論文の R_k はコホート全体の平均損失だが、実装はバッチに現れたコホートの中での最大を取る。ミニバッチで学習する以上、
これが素直な近似になる。状態（q などの重み）は持たないので、checkpoint に損失側の状態は無い。

**LHCF のコホート。** `data/ham10000/splits_lhcf` の列 `lhcf` に入っている。作り方は次のとおり。

1. ResNet-50 の ERM（`resnet-ham10000-fg-erm`、seed 42）の埋め込み（2048 次元）を train・val・test について取る。
2. train で標準化と PCA 32 次元を当てる。全共分散 GMM を 2048 次元のまま当てるのは難しいため。
3. 全共分散 GMM を K = 2〜12（`n_init=5`）で当て、BIC 最小の K = 11 を採る。
4. train・val・test を同じ変換と GMM の `predict`（責務最大）で割り当てる。ラベルも属性も使わない。
5. 対照の列も足す。`lhcf_random` は split ごとに `lhcf` を並べ替えた、大きさだけ同じ群。`label_group` はクラスそのもの。

作ったスクリプト `lhcf_cohorts.py` は分析 archive とともに repo から消えた（`snapshot/agent-era-20261016` ブランチに残る）。
作り直すと `splits_lhcf` が上書きされ、既存 run との対応が切れる。

このコホートは train ではほぼラベルで割れている。11 個のうち 6 個は train の悪性率が 0.5% 以下、2 個は 87% 以上。
そのため最悪コホートは悪性の多いコホートになりやすく、ℓ_wst がクラス重み付けに近い働きをするおそれがある。
この見方を確かめる対照が、群をクラスにした preset `resnet_ham10000_fg_swadwst_label`。

**記録と checkpoint。**

- `ObjectiveDiagnosticsRecorder` が `metrics/objective_diagnostics.jsonl`（1 行 = 1 epoch）に `group_loss`（群の平均損失、
  居ない群は 0）・`group_present`・`worst_group`（最悪の群の one-hot）を書く。`epoch_mean` の `worst_group` が各群が最悪に
  なった割合、`group_loss / group_present` が居たバッチでの平均損失。
- checkpoint は、既定の `best_val_auroc_*.ckpt`（val AUROC が最大の 1 つ）と `last.ckpt` に加え、`_fairness_ham10000.yaml` が
  `best_val_worst_sexage_*.ckpt`（val の性別 × 年齢の最悪群 AUROC が最大）を残す。SWAD の preset は `swad.ckpt` を評価する。
- 学習に属性ラベルを使わない LHCF の前提に合わせるなら、`best_val_auroc_*` か `swad.ckpt` を使う。`best_val_worst_sexage_*` は
  checkpoint の選択で属性を使うことになる。コホートの最悪 AUROC では選べない（val に悪性が 1 件も無いコホートがある）。
- 予測は `analysis/common/predictions.py` に `--checkpoint <名前>` で渡す（例 `best_val_auroc_009`、`last`、`swad`）。

### 原論文との違い

| 項目 | 原論文 | この repo |
| --- | --- | --- |
| コホートの埋め込み | ERM の ResNet18 | ERM の ResNet-50 → 標準化 → PCA 32 次元 |
| K | BIC で 7（HAM10000、ResNet18） | BIC（2〜12）で 11 |
| 最悪の取り方 | R_k はコホート全体の平均 | バッチに現れたコホートの中の最大 |
| λ | 記載なし | 1 |
| クラス重み | 記載なし | 逆頻度（平均 1） |
| 分割 | 80/10/10 | train・val は lesion_id 単位、test は ISIC2018 Task3 の公式 test |
| 評価の年齢の境目 | 60 歳 | 65 歳（`age_group_65`） |
| 最適化 | 記載なし | AdamW lr 1e-4、weight decay 0.01、batch 128、30 epoch、データ拡張あり |
| SWAD + LHCF | 組み込み方の記載なし | SWAD + ℓ_wst と解釈した（SWAD 自体は群を使わない） |
| 2 段学習 | なし | `spatial_lora_ham10000_cf_{const,lhcf}_wst` は 2 段目に ℓ_wst を使う独自の構成 |

論文に揃える余地があるのは埋め込み（ResNet18 にする）と評価の年齢の境目（60 歳）くらいで、λ・最適化・選び方は論文に
無いので揃えようがない。λ は `model.loss_fn.fairness_weight` で変える。

### 起動例

```bash
uv run python -m projects.hypernet_e2e.run experiment=resnet_ham10000_fg_wst_lhcf study=<slug> seed=42
uv run python -m projects.hypernet_e2e.run experiment=resnet_ham10000_fg_wst_lhcf study=<slug> seed=42 model.loss_fn.fairness_weight=0.5
```
