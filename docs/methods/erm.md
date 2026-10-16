# ERM とクラス不均衡の補正：原論文・実装・使い方

2026-10-16 作成。ERM（群を使わない普通の学習）と、クラスの偏りを補正する 5 つのつまみを、コードと参照文献で突き合わせた。
ERM の run、または再抽出を使う run を設計・起動・評価するセッションは、先にこれを読む。

## 先に押さえること

- **既定の ERM は「自然抽出 + 全群共通の inverse class weight」**。CheXpert・HAM10000 の base preset が
  `weighting: inverse` を持つ。FairVLMed・ISIC2019 の base は `weighting: none`（確認済み）。
- **補正のつまみは「1 サンプルあたりの実効重み」で比べると分かる**（下の表）。クラスの逆頻度抽出は、期待値では
  inverse class weight と同じ。セル均等抽出は、群ごとのクラス重みを掛けた uniform group と同じになる。
- **抽出で補正する preset は `weighting: none` を明記している**。抽出とクラス重みを二重に掛けないため。
  自分で override を組むときも同じ点に気をつける。
- **セル均等抽出（`data=*_cells`）は群ごとに陽性:陰性を 1:1 にする**。群ごとに学習時の陽性率が変わり、
  スコアの切片が群ごとにずれる。research-premises は「属性 × ラベルのセルで均等にする重み付けは使わない」としている。
  HAM10000 の `fg_resample_*` は、この方針と別条件の対照として読む。
- **`sampling_corrected_erm` は期待値では「ERM + inverse class weight」と同じ**。セル均等抽出のバッチから母集団の
  損失を推定し直すだけ。条件付き公平性 objective と同じ抽出で比べるための対照で、新しい手法ではない。

## 原論文と公式実装

ERM 側のつまみには、まとまった原論文が無い。次の標準的な文献と実装に従う。

| つまみ | 参照 | この repo との関係 |
| --- | --- | --- |
| inverse class weight | scikit-learn `class_weight="balanced"`：`n_samples / (n_classes * bincount(y))`（[doc](https://scikit-learn.org/stable/modules/generated/sklearn.utils.class_weight.compute_class_weight.html)） | 同じ式。この repo は平均 1 に正規化する |
| クラスの逆頻度抽出（RWY） | Idrissi, Arjovsky, Pezeshki, Lopez-Paz. *Simple data balancing achieves competitive worst-group-accuracy.* CLeaR 2022（[arXiv:2110.14503](https://arxiv.org/abs/2110.14503)、公式 [facebookresearch/BalancingGroups](https://github.com/facebookresearch/BalancingGroups)） | 同じ（復元抽出、1 epoch = N 回） |
| 群 × クラスの均等抽出（RWG、UW） | 同上の RWG。Sagawa et al. ICLR 2020 の 4 節 upweighting（UW）と公式 `--reweight_groups`（[group-dro.md](group-dro.md)） | 群の定義（属性 × ラベル）は同じ。抽出の作り方が少し違う |
| 重み付けの効き方 | Byrd & Lipton. *What is the effect of importance weighting in deep learning?* ICML 2019 | 正則化が弱いと、重み付けの効果は学習が進むと薄れる（Sagawa 4 節・Idrissi 5.5 節も同じ趣旨を引用） |

### 原論文の条件（Idrissi et al. の本文と公式実装で確認済み）

| 項目 | 原論文・公式実装 | 出典 |
| --- | --- | --- |
| 群 | G = Y × A（ラベル × 属性のセル） | 3 節（gDRO の定義） |
| RWY・RWG | 各サンプルの抽出確率を変え、ミニバッチがクラス（RWY）・群（RWG）で**期待値として**釣り合うようにする。重みは `N / n_y`、`N / n_(y,a)` | 4 節、`datasets.py`（`count_groups`） |
| sampler | `WeightedRandomSampler(weights, len(weights))`（復元抽出、1 epoch = N 回） | `datasets.py`（`get_loaders`）、5.5 節 |
| データ | Waterbirds、CelebA、MultiNLI、CivilComments | 5 節 |
| モデルと optimizer | 画像は ImageNet 事前学習の ResNet50 + SGD、文章は BERT + AdamW | 5 節 |
| 探索 | lr {1e-5, 1e-4, 1e-3}、weight decay {1e-4〜1}、batch {2〜128}。50 通りを 5 seed ずつ | 5 節 |
| データ拡張 | Waterbirds の transform は Resize + CenterCrop だけ（拡張なし）。他の dataset は未確認 | `datasets.py` |
| checkpoint | **val の最悪群 accuracy** で lr などと epoch をまとめて選ぶ。平均 accuracy で選ぶと全手法で悪化する | 5 節、5.1 節・表 3 |
| 結果 | RWG・SUBG は gDRO に近い（平均の差は 2 点前後）。RWY・RWG・gDRO は早い epoch で最良になり、その後下がる | 表 2、5.4 節 |

## この repo の実装

| つまみ | 選び方 | 実装 | 中身 |
| --- | --- | --- | --- |
| ERM | `training_strategy=erm`（base preset の既定） | `objectives/task.py:29-60` `TaskLoss` | `F.cross_entropy(weight=w)`。バッチ内の重みの合計で割る |
| inverse class weight | `weighting=inverse` | `training.py:229-248, 284-296` | train split から w_c = N/(C·n_c) を出し、平均 1 に正規化して `model.loss_fn.class_weight` に書く |
| 群ごとのクラス重み | `training_strategy.class_weight_scope=subgroup` | `training.py:251-266` | w_gc = n_g/(C·n_gc)。群の中で陽性と陰性の重みの合計を等しくする。平均 1 には正規化しない。preset で使うのは `group_dro_subgroup` だけ |
| クラスの逆頻度抽出 | `data.train_sampling=inverse_frequency` | `data/datamodule.py:164-181` | 重み 1/n_c の `WeightedRandomSampler`、復元抽出、1 epoch = N 回 |
| 群 × クラスの均等抽出 | `data=*_cells`（`GroupClassImageDataModule`） | `data/group_class_sampler.py:14-111` | 各バッチに各セルから**ちょうど** B/(2G) 件を復元抽出。seed は `sampling_seed + epoch`（`sampling_seed` の既定は `${seed}`）。最後のバッチも満杯で、1 epoch = ⌈N/B⌉ バッチ。B は 2G の倍数が必須 |
| 抽出補正 ERM | `training_strategy=sampling_corrected_erm` | `objectives/sampling_corrected.py:10-23`、`objectives/conditional.py:63-67` | 各サンプルに ρ = P(g,c)/Q(g,c) = 2G·P(g,c) と w_c を掛け、固定の Z = Σ P(g,c)·w_c で割る |

**1 サンプルあたりの実効重み**（式から導き、合成データで数値一致を確認済み）。n_c はクラス、n_g は群、n_gc はセルの件数。

| 条件 | 実効重み | 群の中の陽性:陰性 |
| --- | --- | --- |
| ERM、重みなし | 1 | 母集団どおり |
| ERM + inverse class weight | 1/n_c | 群の中の比 × 全体の補正 |
| クラスの逆頻度抽出 | 1/n_c（期待値） | 同上 |
| 抽出補正 ERM（inverse） | 1/n_c（期待値） | 同上 |
| 群 × クラスの均等抽出 | 1/n_gc | **全群で 1:1** |

最後の行は「uniform group + 群ごとのクラス重み」と一致する（[uniform-group.md](uniform-group.md)）。
抽出補正 ERM は、F.cross_entropy がバッチの重みの合計で割るのに対し、固定の Z で割る。母集団の重み付き CE の
不偏推定になるのはこちら（`tests/test_conditional_objectives.py` の `test_population_corrected_ce_fixed_normalizer`）。

**記録されるもの**：ERM 固有の診断値は無い。クラス重みの解決値は `run.json` の解決済み config と dry-run の
「クラス重み」欄に出る。抽出補正 ERM の継続学習 preset では `ConditionalFairnessRecorder` が
`metrics/fairness_*.json` と毎 epoch の `checkpoints/candidate_XXX.ckpt` を残す。

### 原論文との違い

| 項目 | 原論文（Idrissi et al. の RWY・RWG） | この repo |
| --- | --- | --- |
| 群 × クラス抽出のばらつき | セルの件数はバッチごとに確率的（期待値で均等） | 各バッチでちょうど B/(2G) 件。epoch ごとに seed を変えて再現できる |
| クラス重みと抽出の併用 | どちらか一方 | ERM の既定はクラス重み。抽出の preset は `weighting: none` |
| optimizer・正則化 | SGD、lr・weight decay・batch を探索 | AdamW lr 1e-4、weight decay 0.01、30 epoch。探索なし |
| データ拡張 | Waterbirds はなし | あり |
| checkpoint | val 最悪群 accuracy（epoch も含めて選ぶ） | best val AUROC と last。HAM10000 の `_fairness_ham10000` はさらに val の性別 × 年齢の最悪群 AUROC 最大を残す |
| 評価 | 最悪群 accuracy | 固定しきい値 0.5 の群別 TPR・FPR（research-premises） |

## 起動

| つまみ | preset | run 数（inventory、2026-10-16 時点） |
| --- | --- | --- |
| ERM + inverse | `resnet_chexpert`、`resnet_ham10000`、`*_fg_erm` など、`_base_chexpert`・`_base_ham10000` を継ぐ大半の preset | ERM 全体で 198（CheXpert 34、HAM10000 137、FairVLMed 27） |
| クラスの逆頻度抽出 | `resnet_chexpert_inverse_weighted_sampling`、`spatial_lora_chexpert_inverse_weighted_sampling` | 0 |
| 群 × クラスの均等抽出 | `resnet_ham10000_fg_resample_sex_age`、`resnet_ham10000_fg_resample_gmm8` | 6（ResNet、HAM10000） |
| 抽出補正 ERM | `resnet_chexpert_from_resnet_age_sampling_corrected_erm_race_known`（年齢 2 群、親 ERM からの継続学習） | 0 |

inventory には weighting の列が無い。上の内訳は preset の既定からの推測。

```bash
# 自然抽出 + inverse class weight（既定）
uv run python -m projects.hypernet_e2e.run experiment=resnet_ham10000_fg_erm study=<slug> seed=42
# 抽出補正 ERM は親 ERM の checkpoint（同じ seed）が必須
uv run python -m projects.hypernet_e2e.run experiment=resnet_chexpert_from_resnet_age_sampling_corrected_erm_race_known \
  study=<slug> model.backbone_checkpoint_path=<親 ERM run>/checkpoints/<ckpt>
```

抽出補正 ERM は single device・`accumulate_grad_batches=1`・bf16/fp32 でしか起動しない（`training.py:149-167`）。
起動前に `dry_run=true` で「クラス重み」と `train_sampling` の組み合わせを確かめる。

## checkpoint の選び方と予測の使い方

- ERM は比較の基準なので、**比べる相手と同じ基準で選ぶ**。HAM10000 の公平性比較では、best val AUROC と
  `best_val_worst_sexage_*`（val の性別 × 年齢の最悪群 AUROC）の 2 つが残る。どちらを主に使うかは study で決める。
- 抽出補正 ERM は `metrics/fairness_selection.json` が選んだ epoch を使う。親より全体 AUROC と最悪年齢群 AUROC が
  0.005 以上下がらない epoch のうち、群の組（陽性の群 × 陰性の群）の AUROC の最小が最大のもの
  （`callbacks/conditional_fairness.py:33-69`）。
- 予測は `analysis/common/predictions.py`。best 以外を使うときは `--checkpoint` を渡す。

## 2026-10-16 の照合結果

- ERM + inverse class weight：正しい。w_c ∝ N/(C·n_c) を平均 1 に正規化し、`F.cross_entropy(weight=)` がバッチの重みの合計で割る。
- 群ごとのクラス重み：正しい。w_gc = n_g/(C·n_gc)。
- クラスの逆頻度抽出：正しい。重み 1/n_c、復元抽出、N 回。preset は `weighting: none` で二重補正を避けている。
- 群 × クラスの均等抽出：正しい。各バッチに各セル B/(2G) 件ちょうど、seed + epoch で再現。公式の `WeightedRandomSampler`
  （バッチごとに件数が揺れる）とはこの点だけ違う。epoch の切り替えは `module.py:239-241` が sampler に伝える。
- 抽出補正 ERM：正しい。セル均等抽出の下で、母集団の inverse 重み付き CE の不偏推定になる。
