# Uniform group

群ごとの平均損失を、どの群も同じ重みで足して学習する手法。単一の原論文は無く、Sagawa et al. ICLR 2020 の
upweighting（UW）と Idrissi et al. CLeaR 2022 の RWG が同じ考え方をとる。この repo では strategy `uniform_group`
（自然抽出のバッチで計算）と `sampling_corrected_uniform_group`（群 × クラスの均等抽出のバッチから補正して計算）の 2 つ。

## 1. 手法のロジック

### 解く問題

ERM は全サンプルの平均損失を下げるので、件数の少ない群は平均への寄与が小さく、成績が悪くても放置されやすい。
uniform group は、群 g = 1, …, G の平均損失を 1/G ずつ足したものを下げる。件数によらずどの群も同じだけ効くので、
小さい群が相対的に重くなる。

```
min_θ  (1/G) Σ_g L_g(θ)          L_g(θ)：群 g に属するサンプルの平均損失
```

サンプル 1 件あたりで見ると、群 g のサンプルに 1/(G · n_g) を掛けた重み付き ERM と同じ（n_g は群 g の train の件数）。
「群の頻度が均等な分布」での損失を最小化することに当たる。

GroupDRO（[group-dro.md](group-dro.md)）との違いは重みが固定であること。GroupDRO は損失の大きい群へ重み q を動かすが、
uniform group の重みは学習の間ずっと 1/G のまま。群によって学習のしやすさが違うと、train の損失は群ごとにそろわない。

### 群の作り方で何が変わるか

- **群 = 予測ラベル × 属性**（原論文の形）：各群に 1 つのクラスしか入らないので、群を等重みにすると、各属性の中で
  陽性と陰性も等重みになる。
- **群 = 属性だけ**（または潜在クラスタ）：群の中のクラス比は train のまま残る。クラスの偏りを補うにはクラス重み w_c を
  群の損失の中で掛ける：`L_g = (1/n_g) Σ_{i∈g} w_{y_i} · ℓ_i`。

後者でクラス重みを全群共通にすると、群 g が目的関数に占める重みの合計は (1/G) · Σ_c π(c|g) · w_c になる
（π(c|g) は群 g の中でのクラス c の割合）。陽性の重みが大きいので、陽性率の高い群ほど重く効き、群は厳密には等重みにならない。
群ごとのクラス重み w_gc = n_g / (C · n_gc)（C はクラス数、n_gc は群 g・クラス c の件数）にすると合計は全群で 1/G にそろい、
各群の中で陽性:陰性も 1:1 になる。これは「予測ラベル × 属性」を群にした場合と同じ目的で、群 × クラスの均等抽出とも
期待値で一致する。各条件の 1 サンプルあたりの実効重みは [erm.md](erm.md) の 1 節の表にまとめた。

### ミニバッチでの推定

L_g は train 全体での平均なので、ミニバッチからの推定の仕方が 2 通りある。

1. **自然抽出のバッチから**：バッチに入った群 g のサンプルの重み付き平均で L_g を推定し、群の間で平均する。
   小さい群は 1 件も入らないバッチがあり、そのバッチでは残りの群だけで平均する。
2. **群 × クラスの均等抽出のバッチから**：各セル (g, c) から同じ数を引き、セルの平均損失 m_gc を出す。
   抽出でクラス比が 1:1 に変わっているので、train の群内のクラス比で戻す：`L_g = Σ_c w_c · π(c|g) · m_gc`。
   全セルが毎バッチ入る代わりに、小さいセルは同じサンプルを何度も引く。

どちらも期待値では同じ目的になる。違いはバッチごとのばらつきと、どの抽出を前提にするかにある。

## 2. 原論文の設定

### Sagawa et al. の upweighting（UW）

- Sagawa, Koh, Hashimoto, Liang. *Distributionally Robust Neural Networks for Group Shifts: On the Importance of
  Regularization for Worst-Case Generalization.* ICLR 2020（[arXiv:1911.08731](https://arxiv.org/abs/1911.08731)）の 4 節
  「Comparison between DRO and importance weighting」
- 公式実装：[kohpangwei/group_DRO](https://github.com/kohpangwei/group_DRO)。`--reweight_groups` を付け、`--robust` を付けない

4 節は、群の重み w_g を train での群の頻度の逆数 1/P̂(g) にした重み付き ERM（式 (6)）を UW と呼ぶ。実際の学習では損失に
重みを掛けず、**各群を等確率で抽出する**。損失の重み付けより抽出のほうが効くという Buda et al.（2018）の結果に従った。
公式実装 `data/dro_dataset.py` の `get_loader` は、重み `len(self) / group_counts` の
`WeightedRandomSampler(weights, len(self), replacement=True)` を train だけに使う。コード中のコメントは、これを
「各 (y, c) 群が等しい重みを持つ reweighted ERM」と説明している。

データセットとタスクは GroupDRO と同じ（Waterbirds・CelebA が 2 クラスで 4 群、MultiNLI が 3 クラスで 6 群。
群は予測ラベル × 属性）。表は [group-dro.md](group-dro.md) の 2 節にある。評価は群ごとの accuracy の最小値（最悪群 accuracy）と平均 accuracy。

#### ハイパーパラメータ

モデル・optimizer は GroupDRO と共通で、画像は ImageNet 事前学習の ResNet50、SGD（momentum 0.9）、固定 lr、batch 128、
データ拡張なし（付録 C.2。詳しくは [group-dro.md](group-dro.md)）。UW に関わるのは次の点。

| 項目 | 値 |
| --- | --- |
| ℓ2 の強さ λ | ERM・DRO と同じ grid を探索する。Waterbirds {1e-4, 0.1, 1.0}、CelebA {1e-4, 0.01, 0.1}（付録 C.2） |
| lr | λ ごとに決めた値。λ = 1e-4 で Waterbirds 1e-3・CelebA 1e-4、最も強い λ（Waterbirds 1.0・CelebA 0.1）で 1e-5、中間の λ で 1e-4。中間の値について本文は「Waterbirds and CelebA at λ = 0.1」と書いており、CelebA の 0.01 を指すのかは読み取れない |
| epoch | grid に含めて選ぶ。学習させた総 epoch 数は 4 節の本文に無く未確認 |
| 抽出 | 群を等確率に復元抽出、1 epoch = N 回（公式実装） |
| 文章（MultiNLI） | BERT base、AdamW、lr 2e-5 から線形に減衰、ℓ2 なし、batch 32 |

#### モデルの選び方

ℓ2 の強さと epoch を grid search し、**val の最悪群 accuracy が最大のモデル**を報告する（4 節、表 3）。
UW は Waterbirds・CelebA では ERM より大きく良く DRO よりわずかに悪いが、MultiNLI では ERM より悪い。
少数の群を強調しすぎ、その群の train accuracy がごく低くなった、と論文は説明している。

### Idrissi et al. の RWG

RWG は「ミニバッチが群で期待値として釣り合うよう、各サンプルの抽出確率を変える」方法で、群は予測ラベル × 属性。
UW と同じ操作になる。データセット・ハイパーパラメータ・選び方は [erm.md](erm.md) の 2 節に書いた。

## 3. この repo の実装

### ファイルと設定

| 役割 | 場所 |
| --- | --- |
| 損失（自然抽出） | [objectives/group.py](../../projects/hypernet_e2e/objectives/group.py) の `UniformGroupTaskLoss`（`_per_sample_loss`・`_group_losses` は GroupDRO と共有） |
| 損失（均等抽出 + 補正） | [objectives/sampling_corrected.py](../../projects/hypernet_e2e/objectives/sampling_corrected.py) の `SamplingCorrectedGroupTaskLoss`、親 class は [objectives/conditional.py](../../projects/hypernet_e2e/objectives/conditional.py) の `ConditionalObjective` |
| strategy | [configs/training_strategy/uniform_group.yaml](../../projects/hypernet_e2e/configs/training_strategy/uniform_group.yaml)、[sampling_corrected_uniform_group.yaml](../../projects/hypernet_e2e/configs/training_strategy/sampling_corrected_uniform_group.yaml) |
| クラス重みの計算 | [training.py](../../projects/hypernet_e2e/training.py) の `_resolve_inverse_class_weights`・`_scoped_inverse_weights` |
| 群の定義 | data config の `*_groups.yaml`（例 [ham10000_sex_age_groups.yaml](../../projects/hypernet_e2e/configs/data/ham10000_sex_age_groups.yaml)：性別 × 65 歳未満/以上 = 4 群。`ham10000_gmm8_groups`・`ham10000_he8_groups` は潜在クラスタ 8 群） |
| 均等抽出 | [data/group_class_sampler.py](../../projects/hypernet_e2e/data/group_class_sampler.py) の `GroupClassBatchSampler`（data config `chexpert_age_known_cells`）。仕組みは [erm.md](erm.md) の 3 節 |
| 記録 | [callbacks/objective_diagnostics.py](../../projects/hypernet_e2e/callbacks/objective_diagnostics.py) の `ObjectiveDiagnosticsRecorder`、[callbacks/conditional_fairness.py](../../projects/hypernet_e2e/callbacks/conditional_fairness.py) の `ConditionalFairnessRecorder` |
| 代表的な preset | `resnet_ham10000_fg_ug_sex_age`、`resnet_ham10000_fg_ug_gmm8`、`resnet_chexpert_from_resnet_age_sampling_corrected_uniform_group_race_known` |

### 処理の流れ

**起動時（fit の前）。**

- `_resolve_inverse_class_weights`：`weighting=inverse` なら `model.loss_fn.class_weight` にクラス重みを書く。strategy の config は
  `class_weight: null` だが、`_base_ham10000`・`_base_chexpert` が `weighting: inverse` なので、preset では全群共通の重み
  （w_c = N / (C · n_c) を平均 1 に正規化した `[2]`）が入る。
  `+training_strategy.class_weight_scope=subgroup` を足すと群ごとの重み w_gc = n_g / (C · n_gc)（`[G, 2]`、平均 1 に正規化しない）
  になる。空のセルがあると止まる。この override を使う preset は無い。
- `sampling_corrected_uniform_group` は、`_resolve_conditional_population` が train のセルの件数 `[G, 2]` を
  `population_cell_counts` に書き、single device・`accumulate_grad_batches=1`・bf16 か fp32・セル均等抽出の DataModule を
  確かめる。継続学習の preset なので、`_prepare_age_continuation` が同じ seed の親 ERM の checkpoint を求める。
  `class_weight_scope` は `global` 固定（`ConditionalObjective` が `[2]` の重みしか受けない）。

**学習の各ステップ（`uniform_group`、`UniformGroupTaskLoss.forward`）。**

1. `attributes[group_key]` から群番号を取る。
2. `_per_sample_loss`：サンプルごとの CE にクラス重みを掛ける（`[2]` ならラベルで、`[G, 2]` なら群とラベルで引く）。
   重みが無ければ CE のまま。
3. `_group_losses`：群ごとに、2 の和を群の件数で割る（群内の重みの合計では割らない）。バッチに 1 件もない群は外す。
4. バッチに現れた群の損失を単純平均して返す。現れない群があれば、残りの群が 1/(現れた群の数) ずつになる。

**学習の各ステップ（`sampling_corrected_uniform_group`、`SamplingCorrectedGroupTaskLoss.forward`）。**

1. `ConditionalObjective._cells`：logits が `[B, 2]`、ラベルが 2 値で、全セルがバッチにあることを確かめる。
   セル均等抽出なので常に満たす。
2. CE を fp32 で計算し、セルごとに平均して m_gc を出す。
3. 群のリスク `risk_g = Σ_c w_c · π(c|g) · m_gc`。π(c|g) は `population_cell_counts` から出す train の群内クラス比で、
   バッチの中の比（1:1）ではない。
4. G 群のリスクを単純平均して返す。群の重みは 1/G で固定で、双対変数などの状態は持たない。

**2 つの版がある理由。** 期待値では同じ目的なので、主に使うのは任意の群数で動く `uniform_group`。
`sampling_corrected_uniform_group` は、FairDRO・minimax AUC・EO rate など群 × クラスの均等抽出を前提にする目的関数と、
同じ抽出・同じ親 ERM からの継続学習で比べるための対照として置いている。

**記録と checkpoint。**

- `uniform_group` に固有の診断値は無い。`UniformGroupTaskLoss` は `batch_diagnostics` を持たないので、
  `ObjectiveDiagnosticsRecorder` も `GroupDRODiagnostics`（q を持つ目的関数だけ）も何も書かない。
- `sampling_corrected_uniform_group` は群のリスク `risk`（`[G]`）を `ObjectiveDiagnosticsRecorder` が
  `metrics/objective_diagnostics.jsonl` に epoch ごとに書く。`ConditionalFairnessRecorder` の候補 checkpoint と
  epoch の選び方は抽出補正 ERM と同じ（[erm.md](erm.md) の 3 節）。このコールバックは年齢 2 群専用。
- `uniform_group` の HAM10000 preset は `best_val_auroc_{epoch}`・`last` に加え、`_fairness_ham10000` の
  `best_val_worst_sexage_{epoch}`（val の性別 × 年齢の最悪群 AUROC 最大）を残す。この群は学習に使った群と関係なく
  常に性別 × 年齢で、GMM8・HE8 の run では学習に使った群での選び方にならない。全 epoch を残すには
  `callbacks.model_checkpoint.save_top_k=-1 callbacks.model_checkpoint.save_weights_only=true` を足す。

### 原論文との違い

| 項目 | 原論文（Sagawa の UW・Idrissi の RWG） | この repo |
| --- | --- | --- |
| 群 | 予測ラベル × 属性 | 属性だけ、または潜在クラスタ。陽性と陰性が同じ群に入る |
| 等重みの作り方 | 抽出（群を等確率に復元抽出） | `uniform_group` は損失の重み。補正版はセル均等抽出 + 群内クラス比での戻し |
| クラス重み | なし（群がラベルを含むので要らない） | 全群共通の inverse（群は厳密には等重みにならない） |
| optimizer・正則化 | SGD、ℓ2 の grid | AdamW lr 1e-4、weight decay 0.01、30 epoch。探索なし |
| データ拡張 | なし | あり（`train_transforms_medfair`） |
| モデルの選び方 | val の最悪群 accuracy が最大の epoch | best val AUROC と last。HAM10000 は val の性別 × 年齢の最悪群 AUROC 最大も残す |
| 評価 | 最悪群 accuracy | しきい値 0.5 の群別 TPR・FPR（[research-premises.md](../research-premises.md)） |

原論文の「予測ラベル × 属性」の群に揃えるなら、`+training_strategy.class_weight_scope=subgroup` で群ごとのクラス重みにする
（期待値で群 × クラスの均等抽出と同じ）。ただし [research-premises.md](../research-premises.md) は群ごとのクラス重みを使わない
としている。正則化と選び方を揃えるなら、epoch を短くするか weight decay を上げ、学習に使った群の val 成績で epoch を選ぶ。

### 起動例

```bash
uv run python -m projects.hypernet_e2e.run experiment=resnet_ham10000_fg_ug_sex_age study=<slug> seed=42
# 補正版は、同じ seed の親 ERM の checkpoint が要る
uv run python -m projects.hypernet_e2e.run \
  experiment=resnet_chexpert_from_resnet_age_sampling_corrected_uniform_group_race_known \
  study=<slug> seed=42 model.backbone_checkpoint_path=<親 ERM の run>/checkpoints/<ckpt>
```

起動前に `dry_run=true` で class_weight の形（`[2]` か `[G, 2]`）を確かめる。
