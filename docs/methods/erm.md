# ERM とクラス不均衡の補正

群を使わない普通の学習（ERM）と、陽性・陰性の数の偏りを補正する 4 つのつまみ（inverse class weight、クラスの逆頻度抽出、
群 × クラスの均等抽出、抽出補正 ERM）をまとめる。ERM 自体に単一の原論文は無く、つまみごとに依拠する文献が違う。
この repo では `training_strategy=erm` を基本に、`weighting`・`data.train_sampling`・`data=*_cells`・`training_strategy=sampling_corrected_erm` で選ぶ。

## 1. 手法のロジック

### ERM

ERM（empirical risk minimization）は、train の全サンプルの平均損失を最小化する。

```
min_θ  (1/N) Σ_i ℓ(f_θ(x_i), y_i)          N：train の件数、ℓ：cross-entropy（CE）
```

どのサンプルも同じ重みで効くので、件数の多いクラスや群ほど学習を強く引っ張る。

### クラス不均衡で何が困るか

陽性が 2 割しかないデータでは、全部を陰性と答えても平均損失はそれなりに小さくなる。モデルは陰性側に寄り、
しきい値 0.5 では陽性の取りこぼし（TPR の低さ）が目立つ。これを補うには、少ないクラスのサンプルを損失の中で重くするか、
抽出の回数を増やす。どちらも「1 サンプルあたりの実効重み」を変える操作で、次の 4 つのつまみがある。

- **inverse class weight**：クラス c のサンプルの損失に w_c = N / (C · n_c) を掛ける。C はクラス数、n_c はクラス c の件数。
  各クラスの重みの合計が N / C で等しくなる。
- **クラスの逆頻度抽出**：サンプルを確率 ∝ 1/n_c で復元抽出する。ミニバッチの中でクラスが**期待値として**釣り合う。
  損失には重みを掛けない。
- **群 × クラスの均等抽出**：群 g（属性で決める）とクラス c の組を「セル」と呼ぶ。全セルから同じ数ずつ抽出するので、
  各群の中で陽性と陰性が 1:1 になる。
- **抽出補正 ERM**：群 × クラスの均等抽出で作ったバッチに、抽出確率の比 ρ = P(g,c) / Q(g,c) を掛けて、母集団の重み付き CE
  に戻す。P は train での割合、Q は抽出での割合（2G 個のセルに均等なら 1/(2G)、G は群数）。期待値では
  「ERM + inverse class weight」と同じ目的になる。均等抽出を前提にする手法と同じバッチで ERM を比べるための対照で、新しい手法ではない。

### 1 サンプルあたりの実効重みで比べる

各つまみを「train のサンプル 1 件が、期待値で目的関数にどれだけ効くか」で並べると、互いの関係が分かる。
n_c はクラス c、n_g は群 g、n_gc はセル (g, c) の件数。比例定数は省く。

| 条件 | 実効重み | 群の中の陽性:陰性 |
| --- | --- | --- |
| ERM、重みなし | 1 | train のまま |
| ERM + inverse class weight | 1/n_c | train の比に、全体で共通の補正を掛けたもの |
| クラスの逆頻度抽出 | 1/n_c | 同上 |
| 抽出補正 ERM（inverse class weight つき） | 1/n_c | 同上 |
| 群 × クラスの均等抽出 | 1/n_gc | **全群で 1:1** |
| uniform group（[uniform-group.md](uniform-group.md)） | w_c / n_g | ERM + inverse class weight と同じ |
| uniform group + 群ごとのクラス重み w_gc = n_g / (C · n_gc) | 1/n_gc | 全群で 1:1（群 × クラスの均等抽出と同じ） |

上の 4 行は、抽出か損失かの違いはあっても期待値では同じ目的になる。違うのはバッチごとのばらつきで、抽出は
同じサンプルを何度も引き、損失の重みは少数クラスのサンプル 1 件の勾配を大きくする。

群 × クラスの均等抽出は、群ごとに学習時の陽性率を 50% に変える。群を見分けられるモデルでは、最適な logit が群ごとに
log(n_{g,0} / n_{g,1}) だけずれる（陽性率の低い群ほどスコアが持ち上がる）。しきい値を固定した群別 TPR・FPR は、
この切片のずれの影響を受ける。

### 重み付けと正則化

損失の重み付けや抽出の偏りは、正則化の弱い大きなモデルでは学習が進むほど効かなくなる（Byrd & Lipton）。
大きなモデルは train の全サンプルを当てられるので、どのサンプルの損失も 0 に近づき、重みの差が学習に残りにくい。
重み付けを効かせるには、強い正則化か早めの打ち切りが要る。GroupDRO の原論文も同じ前提に立つ（[group-dro.md](group-dro.md)）。

## 2. 原論文の設定

### inverse class weight

まとまった原論文は無い。式は scikit-learn の `class_weight="balanced"` と同じ `n_samples / (n_classes * np.bincount(y))`
（[compute_class_weight](https://scikit-learn.org/stable/modules/generated/sklearn.utils.class_weight.compute_class_weight.html)）。

### クラスの逆頻度抽出（RWY）と群 × クラスの均等抽出（RWG）

- Idrissi, Arjovsky, Pezeshki, Lopez-Paz. *Simple data balancing achieves competitive worst-group-accuracy.* CLeaR 2022
  （[arXiv:2110.14503](https://arxiv.org/abs/2110.14503)）。4 節で RWY・RWG と、抽出しない間引き版 SUBY・SUBG を定義する。
- 公式実装：[facebookresearch/BalancingGroups](https://github.com/facebookresearch/BalancingGroups)（`datasets.py` の `count_groups`・`get_loaders`）

RWY は「ミニバッチがクラスで期待値として釣り合うよう、各サンプルの抽出確率を変える」、RWG は同じことを群で行う。
群は gDRO と同じく G = Y × A（予測ラベル × 属性）なので、RWG はこの文書の「群 × クラスの均等抽出」に当たる。
公式実装は抽出の重みを RWY で `N / n_y`、RWG で `N / n_(y,a)` とし、`WeightedRandomSampler(weights, len(weights))`
（既定の復元抽出、1 epoch = N 回）で引く。損失には重みを掛けない。

#### データセットとタスク

| データセット | 入力 | 予測ラベル（クラス数） | 属性（群数） |
| --- | --- | --- | --- |
| Waterbirds | 画像（鳥を背景に合成） | 水鳥 / 陸鳥（2 クラス） | 背景が水 / 陸（4 群） |
| CelebA | 顔画像 | 金髪 / 金髪でない（2 クラス） | 性別（4 群） |
| MultiNLI | 文の組 | 矛盾 / 含意 / 中立（3 クラス） | 2 文目の否定語の有無（6 群） |
| CivilComments | 文章（コメント） | 有害 / 有害でない（2 クラス） | 8 種の属性への言及の有無をまとめた 2 値（4 群） |

評価は test の最悪群 accuracy（群ごとの accuracy の最小値）と平均 accuracy。

#### ハイパーパラメータ

| 項目 | 値 |
| --- | --- |
| モデル | 画像は ImageNet 事前学習の ResNet50、文章は BERT |
| optimizer | 画像は SGD（momentum の値は本文に無く未確認）、文章は AdamW |
| lr | {1e-5, 1e-4, 1e-3} から探索 |
| weight decay | {1e-4, 1e-3, 1e-2, 1e-1, 1} から探索 |
| batch size | 画像 {2, 4, …, 128}、文章 {2, 4, …, 32} から探索 |
| epoch | Waterbirds 360、CelebA 60、MultiNLI・CivilComments 7 |
| 探索 | 手法・データセットごとに 50 通りの無作為な組み合わせを、各 5 回（seed 違い）回す |
| データ拡張 | 本文に記載なし。公式実装は Waterbirds が Resize + CenterCrop、CelebA が CenterCrop + Resize だけで、拡張は無い |

#### モデルの選び方

lr などのハイパーパラメータと epoch を合わせて、**val の最悪群 accuracy が最大のもの**を選ぶ（5 節）。属性を使わない
RWY でも、選ぶときには val の属性ラベルを使う。val の平均 accuracy で選ぶと全手法で最悪群の成績が下がる（表 3）。
RWY・RWG は学習の早い epoch で最悪群 accuracy が最良になり、その後下がる（5.4 節）ので、epoch の選び方が結果を左右する。

### 群 × クラスの均等抽出のもう一つの出所（Sagawa et al. の upweighting）

Sagawa et al. ICLR 2020 の 4 節の upweighting（UW）も、群 = 予測ラベル × 属性を等確率で抽出するので、RWG と同じ操作になる。
設定は [uniform-group.md](uniform-group.md) の 2 節に書く。

### 抽出補正 ERM

対応する文献は無い。抽出確率の比で重みを掛け直すのは重要度重み付け（importance weighting）の基本形で、
この repo では均等抽出を前提にする条件付き公平性の目的関数と比べるために置いている。

### 重み付けの効き方

Byrd & Lipton. *What is the effect of importance weighting in deep learning?* ICML 2019
（[arXiv:1812.03372](https://arxiv.org/abs/1812.03372)）。重要度重み付けは学習の初めにはモデルを変えるが、epoch を重ねると
効果が薄れる。ℓ2 正則化と batch normalization で一部戻り、dropout では戻らない（要旨で確認。本文の実験条件は未確認）。

## 3. この repo の実装

### ファイルと設定

| 役割 | 場所 |
| --- | --- |
| 損失（ERM） | [objectives/task.py](../../projects/hypernet_e2e/objectives/task.py) の `TaskLoss`（`loss.py` から再公開） |
| 損失（抽出補正 ERM） | [objectives/sampling_corrected.py](../../projects/hypernet_e2e/objectives/sampling_corrected.py) の `SamplingCorrectedTaskLoss`、親 class は [objectives/conditional.py](../../projects/hypernet_e2e/objectives/conditional.py) の `ConditionalObjective`（`_task`） |
| strategy | [configs/training_strategy/erm.yaml](../../projects/hypernet_e2e/configs/training_strategy/erm.yaml)、[sampling_corrected_erm.yaml](../../projects/hypernet_e2e/configs/training_strategy/sampling_corrected_erm.yaml) |
| クラス重みの計算 | [training.py](../../projects/hypernet_e2e/training.py) の `_resolve_inverse_class_weights`・`_scoped_inverse_weights`・`_inverse_frequency_weights`・`_group_inverse_frequency_weights` |
| クラスの逆頻度抽出 | [data/datamodule.py](../../projects/hypernet_e2e/data/datamodule.py) の `ImageDataModule._train_sampler`（`train_sampling=inverse_frequency`） |
| 群 × クラスの均等抽出 | [data/group_class_sampler.py](../../projects/hypernet_e2e/data/group_class_sampler.py) の `GroupClassBatchSampler`・`GroupClassImageDataModule`。data config は `*_cells.yaml`（例 [ham10000_sex_age_cells.yaml](../../projects/hypernet_e2e/configs/data/ham10000_sex_age_cells.yaml)） |
| 抽出補正 ERM の起動前検査 | training.py の `_resolve_conditional_population`・`_prepare_age_continuation` |
| 抽出補正 ERM の候補記録 | [callbacks/conditional_fairness.py](../../projects/hypernet_e2e/callbacks/conditional_fairness.py) の `ConditionalFairnessRecorder` |
| 代表的な preset | `resnet_ham10000_fg_erm`（自然抽出 + inverse）、`resnet_chexpert_inverse_weighted_sampling`（クラスの逆頻度抽出）、`resnet_ham10000_fg_resample_sex_age`（群 × クラスの均等抽出）、`resnet_chexpert_from_resnet_age_sampling_corrected_erm_race_known`（抽出補正 ERM） |

**既定の ERM は「自然抽出 + 全群共通の inverse class weight」**。`_base_chexpert`・`_base_ham10000` が `weighting: inverse`
を持ち、これを継ぐ preset はすべてそうなる。`_base_fairvlmed`（陽性率がほぼ 50%）と `_base_isic2019` は `weighting: none`。

### 処理の流れ

**起動時（fit の前）。** `training.py` の `run_fit` が次を行う。`dry_run=true` の `run_plan` も同じ順で通す。

- `_resolve_inverse_class_weights`：`weighting=inverse` のときだけ、`train.csv` のラベルからクラス重みを計算して
  `model.loss_fn.class_weight` に書き込む。strategy の `class_weight_scope` で形が決まる。
  - `global`（既定）：w_c = N / (C · n_c) を平均 1 に正規化した `[num_classes]`。
  - `subgroup`：w_gc = n_g / (C · n_gc) の `[num_groups, num_classes]`。平均 1 には正規化しない。空のセルがあると止まる。
    `TaskLoss` は 2 次元の重みを受けないので、ERM では使えない（uniform group で使う）。
- `ImageDataModule.__init__`：`train_sampling` は `uniform`（shuffle だけ）か `inverse_frequency` のどちらか。
  `GroupClassImageDataModule` は `group_class_balanced` だけを受ける。
- 抽出補正 ERM のときだけ `_resolve_conditional_population` が、single device・`accumulate_grad_batches=1`・
  bf16 か fp32・data が `GroupClassImageDataModule` であることを確かめ、train のセルの件数 `[G, 2]` を
  `population_cell_counts` に書く。`weighting=none` なら `class_weight` を null にする。
- 継続学習の preset（`continuation.require_parent: true`）は `_prepare_age_continuation` が、親の checkpoint が同じ seed の
  ERM run のものかを、親 run の `config.yaml` で確かめる。

**学習の各ステップ。**

1. 抽出：
   - `uniform`：普通の shuffle。
   - `inverse_frequency`：`_train_sampler` が重み 1/n_c の `WeightedRandomSampler`（復元抽出、1 epoch = N 回）を作る。
   - `*_cells`：`GroupClassBatchSampler.__iter__` が、各セルから**ちょうど** B/(2G) 件を復元抽出し、バッチの中で並べ替える。
     1 epoch は ⌈N/B⌉ バッチで、最後のバッチも満杯。乱数は `sampling_seed + epoch`（`sampling_seed` の既定は `${seed}`）で、
     epoch は `LitModule.on_train_epoch_start` が `set_epoch` で渡す。B は 2G の倍数、ラベルは 2 値、train の全セルが
     1 件以上という条件がある。val・test は自然分布のまま。
2. 損失：
   - ERM（`TaskLoss.forward`）：`F.cross_entropy(weight=w)`。バッチの中の Σ w_{y_i} · CE_i を Σ w_{y_i} で割る。
   - 抽出補正 ERM（`ConditionalObjective._task`）：サンプルごとの CE に ρ = 2G · P(g,c) と w_c を掛けてバッチ平均し、
     固定の定数 Z = Σ_{g,c} P(g,c) · w_c で割る。

**設計の理由。**

- 抽出で補正する preset は `weighting: none` を明記している。base の `weighting: inverse` を受けたままだと、
  抽出とクラス重みが二重に掛かるため。override を自分で組むときも同じ点に気をつける。
- 群 × クラスの均等抽出を確率的な `WeightedRandomSampler` でなく件数固定にしたのは、条件付き公平性の目的関数が
  毎バッチに全セルを要求するため。
- 抽出補正 ERM がバッチの重みの合計でなく固定の Z で割るのは、セル均等のバッチから母集団の重み付き CE を
  偏りなく推定するため（`tests/test_conditional_objectives.py` の `test_population_corrected_ce_fixed_normalizer`）。

**記録と checkpoint。**

- ERM に固有の診断値は無い。解決したクラス重みは `run.json` の解決済み config と、dry-run の表の「クラス重み」欄に出る。
- checkpoint は既定で `best_val_auroc_{epoch}`（val AUROC 最大）と `last`。HAM10000 の公平性比較の preset
  （`_fairness_ham10000` を継ぐもの）は、val の性別 × 年齢の最悪群 AUROC 最大の `best_val_worst_sexage_{epoch}` も残す。
  ERM は比較の基準なので、比べる相手と同じ基準の checkpoint を使う。
- 抽出補正 ERM の継続学習 preset は `ConditionalFairnessRecorder` が毎 epoch の `checkpoints/candidate_XXX.ckpt` と
  `metrics/fairness_candidates.json` を残し、学習の終わりに `metrics/fairness_selection.json` で 1 つ選ぶ。選び方は、
  親より val の全体 AUROC と年齢群内 AUROC の最小が 0.005 以上下がらない epoch のうち、陽性の群 × 陰性の群の組の
  AUROC の最小が最大のもの。このコールバックは年齢 2 群専用。セルごとの重複抽出の割合は
  `train/conditional/sampler_duplicate_fraction/cell*` に出る。
- 予測は `analysis/common/predictions.py`。既定 `best` 以外は `--checkpoint` に名前で渡す（例 `last`、`best_val_auroc_009`）。

### 原論文との違い

| 項目 | 原論文（Idrissi et al. の RWY・RWG） | この repo |
| --- | --- | --- |
| 群 | 予測ラベル × 属性 | 群 × クラスの均等抽出は属性（または潜在クラスタ）× ラベルのセルで、同じ形 |
| 群 × クラスの抽出 | 期待値で均等（バッチごとに件数が揺れる） | 各バッチでちょうど B/(2G) 件。seed + epoch で再現できる |
| クラス重みと抽出 | どちらか一方（抽出だけ） | ERM の既定はクラス重み。抽出の preset は `weighting: none` |
| optimizer・正則化 | SGD、lr・weight decay・batch を探索 | AdamW lr 1e-4、weight decay 0.01、30 epoch。探索なし |
| データ拡張 | なし | あり（`train_transforms_medfair`） |
| モデルの選び方 | val の最悪群 accuracy（epoch も含めて選ぶ） | best val AUROC と last。HAM10000 の比較 preset は val の性別 × 年齢の最悪群 AUROC 最大も残す |
| 評価 | 最悪群 accuracy | しきい値 0.5 の群別 TPR・FPR（[research-premises.md](../research-premises.md)） |

[research-premises.md](../research-premises.md) は、属性 × ラベルのセルで均等にする重み付けを使わないとしている
（群ごとに切片がずれるため）。`resnet_ham10000_fg_resample_*` は、この方針とは別条件の対照として読む。
原論文に揃えるなら、epoch を短くするか weight decay を探索し、val の最悪群の成績で epoch を選ぶ。

### 起動例

```bash
# 自然抽出 + inverse class weight（既定）
uv run python -m projects.hypernet_e2e.run experiment=resnet_ham10000_fg_erm study=<slug> seed=42
# 抽出補正 ERM は、同じ seed の親 ERM の checkpoint が要る
uv run python -m projects.hypernet_e2e.run experiment=resnet_chexpert_from_resnet_age_sampling_corrected_erm_race_known \
  study=<slug> seed=42 model.backbone_checkpoint_path=<親 ERM の run>/checkpoints/<ckpt>
```

起動前に `dry_run=true` で「クラス重み」と `train_sampling` の組み合わせを確かめる。
