# EO rate

群ごとの TPR・FPR の差を「率の制約」として課し、制約つき最適化を 2 人のゲームとして解く手法。
Cotter et al. の proxy-Lagrangian（JMLR 2019）と、制約側を別データで測る 2 データセット方式（ICML 2019）による。
この repo では strategy `equalized_odds_rate`。2 データセット方式は `*_dual_holdout` の preset で使う。

## 1. 手法のロジック

### 解く問題

普通の学習（ERM）は平均の誤りだけを下げるので、群によって陽性の見落とし（TPR の低さ）や偽陽性（FPR の高さ）が
偏っても止める仕組みがない。Equalized Odds（EO）は、群 g ごとの TPR_g = P(ŷ=1 | y=1, g) と
FPR_g = P(ŷ=1 | y=0, g) が群の間で等しいことを求める。この手法は EO を許容値 ε つきの制約として書き、
制約を守る範囲で誤りを最小にする。

```
min_θ  g_0(θ)   s.t.  g_i(θ) ≤ 0   (i = 1, …, m)
g_0：目的関数（誤り率）
g_i：率の一次結合 − 許容値。例 FPR_g(θ) − FPR_全体(θ) − ε
```

率は「陽性と予測した件数の割合」なので、指示関数 1(s > 0)（s はモデルの score）の平均になる。
指示関数の勾配はほとんどの点で 0 なので、g_i をそのまま勾配法で下げることはできない。

### proxy-Lagrangian

制約の重み λ（m+1 個の非負の値で合計 1、先頭が目的関数の重み）を持ち、θ と λ が別々の目的を持つゲームにする。

```
θ 側が最小化：  L_θ(θ, λ) = λ_1 · g_0(θ) + Σ_i λ_{i+1} · g̃_i(θ)
λ 側が最大化：  L_λ(θ, λ) = Σ_i λ_{i+1} · g_i(θ)
g̃_i：g_i の微分できる上界（代理制約、proxy）
```

θ の更新には勾配が必要なので代理制約 g̃_i を使う。λ の更新には制約の値だけあればよいので、元の 0-1 の率で測った
g_i を使う。両方に代理を使う普通の方法では、代理の上で制約を満たす解に落ち着き、元の率では制約がきつくなりすぎる。
proxy-Lagrangian は元の率での制約を狙う点が違う。

代理には hinge を使う。率に正の係数がかかる項は 1(s > 0) の上界 U = max(0, 1 + s) で、負の係数がかかる項は
下界 D = 1 − max(0, 1 − s) で置き換える。こうすると全体が g_i の上界になる。

### λ の更新（swap regret）

ゲームが零和でないので、λ 側は普通の勾配上昇ではなく swap regret を小さくする更新を使う。
(m+1)×(m+1) の行列 M（各列が合計 1 の分布）を持ち、λ は M の定常分布（Mλ = λ を満たす分布）とする。

```
M ← M ⊙ exp(η_λ · Δ_λ λᵀ)        Δ_λ = (0, g_1, …, g_m)：L_λ の λ についての勾配
M の各列を合計 1 に正規化
```

違反している制約（g_i > 0）の行が増え、その制約の重みが上がる。M は全要素 1/(m+1) で初期化する。

### 学習の手順（JMLR Algorithm 3、ICML Algorithm 3）

ミニバッチごとに次を行う。θ と λ は同時に 1 歩ずつ動く。

1. λ = M の定常分布。
2. L_θ の確率的勾配で θ を 1 歩更新する。
3. 元の率で測った制約の値 g_i から M を更新する。

### 2 データセット方式（ICML 2019）

λ 側が θ と同じ train データで制約を測ると、θ が train に過学習しただけで制約が満たされたように見え、test では
制約が破れる。そこで train を 2 つに分け、θ は一方（S_train）で L_θ を、λ は他方（S_val）で L_λ を計算する。
ハイパーパラメータを validation で選ぶのと同じ考え方で、制約の汎化が良くなる代わりに誤り率は少し上がる。

### 出力

理論の保証は、学習中の iterate θ^(1), …, θ^(T) を確率的に混ぜた分類器に対して与えられる。shrinking は、
iterate の上の分布を線形計画で選び直し、高々 m+1 個の iterate の混合にする手順である。実用上は、iterate を 1 つだけ選ぶ
（best iterate）ことでもほぼ同じ結果になると JMLR 版は報告している。

## 2. 原論文の設定

- Cotter, Jiang, Gupta, Wang, Narayan, You, Sridharan. *Optimization with Non-Differentiable Constraints with Applications
  to Fairness, Recall, Churn, and Other Goals.* JMLR 20, 2019（[jmlr.org/papers/v20/18-616](https://jmlr.org/papers/v20/18-616.html)）
- Cotter, Gupta, Jiang, Srebro, Sridharan, Wang, Woodworth, You. *Training Well-Generalizing Classifiers for Fairness Metrics
  and Other Data-Dependent Constraints.* ICML 2019（[arXiv:1807.00028](https://arxiv.org/abs/1807.00028)）
- 公式実装：[TensorFlow Constrained Optimization（TFCO）](https://github.com/google-research/tensorflow_constrained_optimization)
  （`train/proxy_lagrangian_optimizer.py`、`rates/loss.py` の `HingeLoss`）

### データセットとタスク

どちらの論文も表形式データの **2 値分類**。制約はすべて「各群の率を全体の率と比べる片側の制約」で、かける率は 1 種類。
群は重なってよい（人種の群と性別の群に同じ人が入る）。

| 論文 | データセット | 入力 | 予測ラベル（クラス数） | 群（群数） | 制約（本数） |
| --- | --- | --- | --- | --- | --- |
| JMLR | Bank Marketing | 表形式 60 特徴 | 商品に加入するか（2 クラス） | 年齢の 5 分位（5） | 陽性予測率の統計的パリティ、許容 2%（5） |
| JMLR | Adult | 表形式 122 特徴 | 年収 5 万ドル超か（2 クラス） | Black・White・Female・Male（4） | TPR_g ≥ 0.95 · TPR_全体（4） |
| JMLR | COMPAS | 表形式 31 特徴 | 再犯するか（2 クラス） | 同じ 4 群（4） | TPR_g ≤ TPR_全体 + 0.05（4） |
| JMLR | Business Entity Resolution（Google 内部） | 表形式 37 特徴 | 同じ事業者か（2 クラス） | 国 16・チェーンか否か | 最小 recall（18）と精度の差（1） |
| ICML | Communities and Crime | 表形式 140 特徴 | 犯罪率が上位 30% か（2 クラス） | 人種 4 種の割合をそれぞれ中央値で二分（8） | FPR_g ≤ FPR_全体（8、ε = 0） |
| ICML | Adult・COMPAS・Business Entity Resolution | 同上 | 同上 | 同上 | Adult は陽性予測率の 80% ルール、COMPAS は上と同じ |

JMLR にはほかに Google 内部の 3 データ（Thresholding・Map Intent・Filtering）がある。ICML には過学習しやすい合成データ
（recall ≥ 97% の制約）もある。

### ハイパーパラメータ

| 項目 | JMLR 2019 | ICML 2019 |
| --- | --- | --- |
| モデル | 線形、2 層の ReLU NN（隠れ unit 10〜32）、lattice の ensemble | 線形、隠れ層 1 層の ReLU NN（50 unit）、lattice |
| optimizer | θ・λ とも ADAM。lr は θ 用と λ 用を別々に、0.001 の前後を 10 倍刻みで validation で選ぶ | θ・λ とも ADAM の既定値（ハイパーパラメータ探索を避けるため） |
| 目的関数と代理制約 | どちらも hinge 上界 | どちらも hinge 上界 |
| M の初期値 | 全要素 1/(m+1)（Algorithm 3） | 同じ（Algorithm 3） |
| 学習の長さ | 制約なしで性能が良い反復回数に、先に固定する | 未確認 |
| batch size・正則化・データ拡張 | 本文に記載なし | 本文に記載なし |
| 2 データセットの分け方 | — | train を半分に分け、S_train と S_val にする。100 回の乱数分割の平均 |
| 許容値 ε | 上の表のとおり制約ごとに決める | 同じ |

TFCO の既定は論文と 2 点違う。M の初期値は目的関数側にほぼ全部の重みを置き、各制約の重みを 1e-3/(m+1) にする。
また、M の要素を `minimum_multiplier_radius / (m+1)`（既定 1e-3）以上に保つ。hinge は margin 1 で、上の U・D と同じ形。

### モデルの選び方と評価

- JMLR：ハイパーパラメータは validation で選ぶ。validation 損失の順位と最大違反の順位のうち悪いほうが最小のものを採り、
  同順位は validation 損失で決める（式 18）。出力は 4 種を比べる。iterate 全体の一様混合（T-stoch）、shrinking で
  m+1 個に絞った混合（m-stoch）、最後の iterate（Last）、式 18 を train で当てて選んだ iterate（Best）。
  best iterate は多くの場合混合と同程度で、last は 7 データのうち 4 つで best より test が悪かった。
- ICML：100 個の等間隔の iterate を残し、shrinking で確率的分類器を作る。shrinking の線形計画は、目的関数を S_train で、
  制約を S_val で測る。
- 評価は test の 0-1 誤り率と、制約の最大違反。

## 3. この repo の実装

### ファイルと設定

| 役割 | 場所 |
| --- | --- |
| 損失（λ の更新を含む） | [objectives/equalized_odds.py](../../projects/hypernet_e2e/objectives/equalized_odds.py) の `EqualizedOddsRateTaskLoss`・`proxy_bounds`。親 class は [objectives/conditional.py](../../projects/hypernet_e2e/objectives/conditional.py) の `ConditionalObjective`（`_cells`・`_task`） |
| strategy | [configs/training_strategy/equalized_odds_rate.yaml](../../projects/hypernet_e2e/configs/training_strategy/equalized_odds_rate.yaml) |
| 起動時の準備 | [training.py](../../projects/hypernet_e2e/training.py) の `_resolve_conditional_population`・`_resolve_inverse_class_weights` |
| λ の更新の時点と held-out | [module.py](../../projects/hypernet_e2e/module.py) の `optimizer_step`・`on_train_start`・`_measure_on_held_out` |
| セル均等抽出と held-out | [data/group_class_sampler.py](../../projects/hypernet_e2e/data/group_class_sampler.py) の `GroupClassImageDataModule`（`held_out_batches`）。split は [data_pipeline/chexpert_dual_holdout_splits.py](../../data_pipeline/chexpert_dual_holdout_splits.py)・[ham10000_dual_holdout_splits.py](../../data_pipeline/ham10000_dual_holdout_splits.py) |
| 診断値の記録 | [callbacks/objective_diagnostics.py](../../projects/hypernet_e2e/callbacks/objective_diagnostics.py) の `ObjectiveDiagnosticsRecorder` |
| 代表的な preset | `resnet_chexpert_age_race_eo_rate_dual_holdout`（全ペア・TPR と FPR）、`resnet_chexpert_age_race_eo_fpr_overall_dual_holdout`（原論文の形）、`resnet_ham10000_sex_age_eo_rate_dual_holdout` |

`EqualizedOddsRateTaskLoss` の引数：

| 引数 | 既定 | 意味 |
| --- | --- | --- |
| `constraint_form` | `pairwise` | `pairwise`：全有向群ペアで rate_g − rate_h ≤ ε（最大 − 最小 ≤ ε と同じ）。`overall`：各群を全体と比べる片側の制約 |
| `rates` | `[tpr, fpr]` | 制約をかける率 |
| `epsilon_pos` / `epsilon_neg` | 0.1 / 0.1 | TPR / FPR の許容値 |
| `dual_step_size` | 0.01 | M の更新の η_λ |
| `tau` | 1.0 | hinge の幅。U = max(0, 1 + s/τ)、D = 1 − max(0, 1 − s/τ) |
| `initial_task_weight` | null | null なら M を一様に置く。値を渡すと、λ の初期値の先頭（目的関数の重み）をその値にする |

### 処理の流れ

**起動時（fit の前）。**

- `_resolve_inverse_class_weights`：`weighting=inverse` なら train のラベルから全群共通のクラス重み（平均 1）を作る。
- `_resolve_conditional_population`：train の群 × クラスの件数 `[G, 2]` を数え、`population_cell_counts` に書く。
  空のセルがあれば止める。単一 device・勾配の蓄積なし・bf16 か float32 も要求する。fp16 の GradScaler は optimizer step を
  飛ばすことがあり、そのとき λ だけが進むため。data がセル均等抽出の `GroupClassImageDataModule` でなければ止める。
- 損失の `__init__`：制約の一覧を作る。`pairwise` は |rates| × G(G−1) 本（8 群・両方の率で 112 本）、`overall` は
  |rates| × G 本。`overall` の「全体の率」は、train の群構成で重み付けした群の率の和として係数行列にしておく。
  M は全列を同じ分布（一様か `initial_task_weight`）にして log で持つ。その分布がそのまま初期の λ になる。
- `on_train_start`：data に held-out があれば、held-out からセル均等のバッチを尽きずに返す流れを作る。
- 途中からの再開は `on_load_checkpoint` が拒否する。

**学習の各ステップ。**

1. `EqualizedOddsRateTaskLoss.forward`（θ の更新前）
   - `_cells` がバッチを群 × クラスのセルに分ける。空のセルがあれば止める。全セルが必要なので、train は
     `GroupClassBatchSampler` で群 × クラスを均等に引く（8 群・batch 128 なら 1 セル 8 件）。
   - s = logit_1 − logit_0 とし、セルごとの元の率 `hard[g, c]` = mean(s > 0) を測る。c = 0 のセルが FPR、c = 1 が TPR。
   - 制約ごとに違反（元の率で測った制約の値 − ε）と代理制約（U・D で置き換えた値 − ε）を計算する。
   - λ = `_stationary`（M のべき乗法。1000 回で収束しなければ止める）。
   - 目的関数 `_task` は、セル均等に引いたバッチを train の母集団の比率に重み付けし直した、クラス重みつきの CE。
   - 返り値 λ_1 · task + Σ λ_{i+1} · proxy_i で θ を更新する。違反と λ は `_pending` に保留する。
2. Lightning の optimizer step で θ を更新する（AdamW）。
3. `module.optimizer_step`：held-out があれば `_measure_on_held_out` で held-out の 1 バッチを流し、保留中の違反を
   held-out で測った値に置き換える。測るのは更新後の θ で、net は eval モードにする（BatchNorm の統計を held-out で
   動かさないため）。
4. `commit_update`：log M に η_λ · outer([0, 違反], λ) を足し、列ごとに正規化する（`log_softmax`）。

先に forward で違反を保留し、θ の更新後に M を進めるのは、θ の 1 歩に古い λ を使う Algorithm 3 の同時更新に合わせるため。
held-out の split は train の 80% / 20% で、CheXpert は患者単位、HAM10000 は病変単位で分ける。val・test は元の split と同じ。

**設計上の選択。**

- `overall` の TPR は TPR_全体 − TPR_g ≤ ε（低すぎない）の向き。原論文の COMPAS は TPR_g ≤ TPR_全体 + 0.05（高すぎない）
  だが、ここでは陽性を見落とす群を抑えたいので逆にした。FPR は FPR_g − FPR_全体 ≤ ε で、Communities と同じ向き。
- `pairwise` は評価指標（群の TPR・FPR の最大 − 最小）に直接対応させた独自の形。
- `initial_task_weight: 0.9` は pairwise の preset が入れている。ImageNet 初期化から 1 段で学習すると、一様な λ では目的関数の
  重みが 1/(m+1)（112 本なら 1/113）しかなく、全予測を片側へ寄せる自明解に落ちたため。overall FPR の preset は null（一様）。
- 元の率のしきい値は s > 0、つまり確率 0.5 超で、評価の固定しきい値 0.5 と同じ。

**記録と checkpoint。**

- `ObjectiveDiagnosticsRecorder` が `metrics/objective_diagnostics.jsonl`（1 行 = 1 epoch）に書く。`epoch_mean` は train バッチの
  平均で、`lambda`（先頭が目的関数の重み）、`hard_g`（違反、正なら制約を破っている）、`proxy_g`、`rates`（`[G, 2]`、
  列 0 が FPR、列 1 が TPR）、`task`、`active_fraction`（|s| < τ の割合）を持つ。`held_out_epoch_mean` は λ の更新に使った
  held-out バッチの平均。キーだけのものは epoch 最後の 1 バッチの値。`rates` はバッチの率の平均で、母集団の率ではない。
- `_dual_holdout_chexpert.yaml`・`_dual_holdout_ham10000.yaml` は `save_top_k: -1`・`save_weights_only: true` で全 epoch の重みを
  残す。混合分類器は作らず、epoch を 1 つ分析側で選ぶ。予測は `analysis/common/predictions.py` に
  `--checkpoint best_val_auroc_<epoch>` で epoch を渡す。年齢 2 群の preset（`resnet_chexpert_age_equalized_odds_rate_race_known`）は
  `ConditionalFairnessRecorder` が epoch ごとの val の候補を `fairness_candidates.json` に残す。

### 原論文との違い

| 項目 | 原論文 | この repo |
| --- | --- | --- |
| 制約の形 | 各群 vs 全体の片側、率は 1 種類 | 既定は全有向ペア・TPR と FPR の両方。`overall` も選べる |
| TPR の向き（overall） | COMPAS は高すぎないこと | 低すぎないこと |
| ε | 0〜0.05 | 0.1（overall FPR の preset は 0.05） |
| 目的関数 | hinge | 母集団の比率に戻したクラス重みつき CE |
| 代理制約 | hinge（margin 1） | 同じ（τ = 1） |
| M の初期値 | 一様（TFCO の既定は目的関数寄り） | 一様。pairwise の preset は目的関数の重み 0.9 |
| M の下限 | TFCO は 1e-3/(m+1) | なし |
| optimizer | θ・λ とも ADAM | θ は AdamW（lr 1e-4、weight decay 0.01）。λ は素の指数勾配、η = 0.01 固定 |
| モデルとデータ | 線形・小さな NN・lattice、表形式 | ResNet-50・Spatial LoRA、CheXpert・HAM10000 の画像 |
| 抽出 | 未確認 | 群 × クラスのセル均等 |
| 2 データセットの分け方 | train を半分に分ける | train を 80% / 20% に分け、θ を更新した後に eval モードで測る |
| 学習の長さ | 制約なしで良い反復回数 | 30 epoch |
| 出力 | shrinking の混合（ICML）、Best iterate も比較（JMLR） | 全 epoch を残し、1 つを分析側で選ぶ |

原論文に揃えるなら、`constraint_form=overall`・`rates=[fpr]`・`initial_task_weight=null` にし（preset
`*_eo_fpr_overall_dual_holdout`）、`epsilon_neg=0` にする。学習の長さは同じ構成の ERM の best val epoch を目安に
`trainer.max_epochs` で縮める。

### 起動例

```bash
uv run python -m projects.hypernet_e2e.run experiment=resnet_chexpert_age_race_eo_fpr_overall_dual_holdout study=<slug> seed=42 \
  trainer.max_epochs=10
uv run python -m projects.hypernet_e2e.run experiment=resnet_ham10000_sex_age_eo_rate_dual_holdout study=<slug> seed=42
```
