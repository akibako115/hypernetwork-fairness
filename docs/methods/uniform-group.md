# Uniform group（群を等重みにする学習）：原論文・実装・使い方

2026-10-16 作成。群の平均損失を等重みで足す 2 つの実装を、コードと参照文献（Sagawa et al. 2020 の upweighting、
Idrissi et al. 2022 の RWG）で突き合わせた。uniform group の run を設計・起動・評価するセッションは、先にこれを読む。

## 先に押さえること

- **目的は「群ごとの平均損失を 1/G ずつ足したもの」**。どの群も同じ重みで効くので、小さい群が相対的に上げ重みされる。
  重みは固定で、GroupDRO のように損失の大きい群へ寄っていくことはない。
- **実装は 2 つあり、期待値では同じ目的を最小化する**（合成データで数値一致を確認済み）。違うのは抽出とばらつき。
  - `uniform_group`：自然抽出のバッチで、群ごとにクラス重み付き CE を平均し、バッチに現れた群で等重み平均。
  - `sampling_corrected_uniform_group`：セル均等抽出のバッチで、セル平均 CE に train の群内クラス比 π(c|g) を掛けて
    群のリスクに戻し、1/G で平均。
- **この repo の群は属性だけ（または潜在クラスタ）で、群の中の陽性率は変えない**。原論文の upweighting の群は
  属性 × ラベルのセルで、群ごとに陽性:陰性を 1:1 にする。この repo でそれに当たるのは uniform group ではなく
  ERM + セル均等抽出（`fg_resample_*`、[erm.md](erm.md)）。
- **クラス重みは各群の損失の中で正規化されない**。陽性の重みが大きいので、陽性率の高い群ほど群損失の値が大きく、
  その群の勾配も大きく効く（GroupDRO の「q が向く群」と同じ構造。[group-dro.md](group-dro.md)）。
- **checkpoint は val の最悪群の成績で選ぶ**のが原論文の手順。HAM10000 の preset は best val AUROC に加え、
  val の性別 × 年齢の最悪群 AUROC 最大も残す。

## 原論文と公式実装

uniform group そのものの原論文は無い。参照は次の 2 つ。

- Sagawa, Koh, Hashimoto, Liang. ICLR 2020（[arXiv:1911.08731](https://arxiv.org/abs/1911.08731)）の 4 節
  「Comparison between DRO and importance weighting」の upweighting（UW）。公式 [kohpangwei/group_DRO](https://github.com/kohpangwei/group_DRO)
  の `--reweight_groups`（`--robust` なし）。
- Idrissi, Arjovsky, Pezeshki, Lopez-Paz. CLeaR 2022（[arXiv:2110.14503](https://arxiv.org/abs/2110.14503)）の RWG・SUBG。
  公式 [facebookresearch/BalancingGroups](https://github.com/facebookresearch/BalancingGroups)。条件は [erm.md](erm.md) の表。

### 原論文の条件（Sagawa et al. の UW。本文と公式実装で確認済み）

| 項目 | 原論文・公式実装 | 出典 |
| --- | --- | --- |
| 目的 | 群の頻度の逆数 w_g = 1/P̂(g) で重み付けした ERM。群の頻度が均等な test 分布を狙う | 4 節 式 (6) |
| 群 | 属性 × ラベル（m = \|A\| × \|Y\|） | 2 節 |
| 実装 | 損失は重み付けせず、**各群を等確率で抽出**する（`WeightedRandomSampler`、重み N/n_g、復元抽出、1 epoch = N 回）。公式コードのコメントは「各 (y, c) 群が等重みの reweighted ERM」 | 4 節、`data/dro_dataset.py`（`get_loader`） |
| 抽出を選んだ理由 | 損失の重み付けより抽出のほうが効く、という Buda et al. 2018 の結果に従う | 4 節 |
| データ・モデル | Waterbirds、CelebA、MultiNLI。画像は ImageNet 事前学習の ResNet50、SGD momentum 0.9、batch 128、拡張なし | 4 節、付録 C.2（[group-dro.md](group-dro.md)） |
| 正則化 | DRO と同じ ℓ2 の grid | 4 節 |
| checkpoint | **val 最悪群 accuracy が最大の epoch**（early stopping） | 4 節、表 3 |
| 結果 | Waterbirds・CelebA では ERM より大きく良く、DRO よりわずかに悪い。MultiNLI では ERM より悪い（少数群を過度に強調） | 表 3 |

## この repo の実装

| class | strategy | data | 抽出 | 群のリスク |
| --- | --- | --- | --- | --- |
| `UniformGroupTaskLoss`（`objectives/group.py:85-103`） | `uniform_group` | `*_groups`（`GroupImageDataModule`） | 自然抽出 | バッチの群 g の中で w_c·CE を平均 |
| `SamplingCorrectedGroupTaskLoss`（`objectives/sampling_corrected.py:26-48`） | `sampling_corrected_uniform_group` | `*_cells`（`GroupClassImageDataModule`） | セル均等（各セル B/(2G) 件） | Σ_c w_c·π(c\|g)·（セル平均 CE） |

**`UniformGroupTaskLoss`**
- 群ごとに per-sample の CE に w_c（全群共通、または `class_weight_scope=subgroup` なら w_gc）を掛け、群内で単純平均する
  （`group.py:55-72`）。バッチに現れた群だけで平均を取る。現れない群があると、残りの群が 1/(現れた群の数) ずつになる。
- クラス重みは `weighting=inverse` のとき `training.py:229-248` が入れる。strategy の config では `class_weight: null` だが、
  HAM10000 の base が `weighting: inverse` なので、preset では全群共通の inverse 重みが入る（dry-run で確認：[0.29, 1.71]）。
- 実効重みはサンプル 1 件あたり w_c/(G·n_g)。ERM + inverse（w_c/N）に比べ、群の件数に反比例して上げ重みされる。
  群の中の陽性:陰性の比は ERM + inverse と同じ。

**`SamplingCorrectedGroupTaskLoss`**
- π(c|g) は train の件数（`population_cell_counts`、`training.py:149-167` が train.csv から入れる）から出す。
  セル均等のバッチでもクラス比を母集団に戻すので、群の中の陽性:陰性の比は `uniform_group` と同じになる。
- 群の重みは 1/G で固定。2026-10-16（commit 0f2af99）に GroupDRO 版（`mode=dro`）を削除し、双対状態も持たなくなった。
- バッチに全セルが必要（`conditional.py:50-61`）。セル均等抽出なので常に満たす。2 クラス専用。
- 小さいセルは同じ画像を何度も引く。`ConditionalFairnessRecorder` は sampler の重複率を
  `train/conditional/sampler_duplicate_fraction/cell*` に出す。

**2 つの違い**

| 項目 | `uniform_group` | `sampling_corrected_uniform_group` |
| --- | --- | --- |
| 期待値で最小化するもの | (1/G) Σ_g Σ_c w_c·π(c\|g)·CE_gc | 同じ |
| クラス比の出どころ | バッチの中の比（揺れる） | train 全体の比（固定） |
| 小さい群・セル | バッチに入らない・数件しか入らないことがある（推測。件数は未計測） | 毎バッチ B/(2G) 件。代わりに重複抽出が多い |
| 対応する群・preset | 任意の G（HAM10000 の性別 × 年齢 4 群、潜在クラスタ 8 群） | CheXpert の年齢 2 群、親 ERM からの継続学習だけ（recorder が 2 群専用） |
| 何のための条件か | 公平性手法の比較の 1 本 | 条件付き公平性 objective と同じ抽出で比べる対照 |

**記録されるもの**：`uniform_group` に固有の診断値は無い（`group_dro_diagnostics` は q を持つ objective にしか動かない）。
`SamplingCorrectedGroupTaskLoss` は群のリスク `risk`（`[num_groups]`）を `metrics/objective_diagnostics.jsonl` に epoch ごとに残す。
2026-10-17 から `ObjectiveDiagnosticsRecorder` が既定の callback に入り、全 preset で書き出す。それより前の run には残っていない。

### 原論文との違い

| 項目 | 原論文（Sagawa UW・Idrissi RWG） | この repo |
| --- | --- | --- |
| 群 | 属性 × ラベル | 属性だけ、または潜在クラスタ。陽性と陰性が同じ群に入る |
| 等重みの作り方 | 抽出（群を等確率で引く） | `uniform_group` は損失の重み。補正版はセル均等抽出 + 群内クラス比の補正 |
| クラス重み | なし（群 = セルなので不要） | 全群共通の inverse |
| optimizer・正則化 | SGD、ℓ2 の grid | AdamW lr 1e-4、weight decay 0.01、30 epoch |
| データ拡張 | なし | あり |
| checkpoint | val 最悪群 accuracy | best val AUROC、last、HAM10000 は val 性別 × 年齢の最悪群 AUROC 最大 |

## 起動

| preset | 群 | run 数（inventory、2026-10-16 時点） |
| --- | --- | --- |
| `resnet_ham10000_fg_ug_sex_age` | 性別 × 年齢（<65 / ≥65）の 4 群 | 3 |
| `resnet_ham10000_fg_ug_gmm8` | DINOv2 の GMM 8 クラスタ | 3 |
| `resnet_ham10000_fg_ug_he8` | ERM 特徴 + DINOv2 の k-means 8 | 0 |
| `resnet_chexpert_from_resnet_age_sampling_corrected_uniform_group_race_known` | CheXpert の年齢 2 群（race 既知の subset） | 0 |

```bash
uv run python -m projects.hypernet_e2e.run experiment=resnet_ham10000_fg_ug_sex_age study=<slug> seed=42
# 原論文の群（属性 × ラベル）に近づけるなら、群ごとのクラス重みを使う（期待値でセル均等抽出と同じ）
uv run python -m projects.hypernet_e2e.run experiment=resnet_ham10000_fg_ug_sex_age study=<slug> \
  +training_strategy.class_weight_scope=subgroup
# 補正版は親 ERM の checkpoint（同じ seed）が必須
uv run python -m projects.hypernet_e2e.run \
  experiment=resnet_chexpert_from_resnet_age_sampling_corrected_uniform_group_race_known \
  study=<slug> model.backbone_checkpoint_path=<親 ERM run>/checkpoints/<ckpt>
```

`class_weight_scope=subgroup` の override は、群 × クラスのセルに空きがあると止まる（`training.py:262-263`）。
起動前に `dry_run=true` で class_weight の形（`[G, 2]` か `[2]`）を確かめる。

## checkpoint の選び方と予測の使い方

- 原論文どおりなら **val の最悪群の成績が最大の epoch**。HAM10000 の preset では `best_val_worst_sexage_*` がそれに近い
  （ただし群は常に性別 × 年齢で、GMM8・HE8 で学習した run でも学習に使った群ではない）。
- 全 epoch は残らない（best val AUROC、last、上の 1 つだけ）。学習に使った群で選び直したいときは
  `callbacks.model_checkpoint.save_top_k=-1 callbacks.model_checkpoint.save_weights_only=true` を足して回す
  （GroupDRO の strategy と同じ設定。dry-run で合成できることを確認済み）。
- 補正版は `metrics/fairness_selection.json` の epoch を使う（[erm.md](erm.md) の抽出補正 ERM と同じ規則）。
- 予測は `analysis/common/predictions.py`。best val AUROC 以外は `--checkpoint` を渡す。

## 2026-10-16 の照合結果

- `UniformGroupTaskLoss`：正しい。バッチに現れた群の平均損失を等重みで平均する。
- `SamplingCorrectedGroupTaskLoss`：正しい。群のリスクは Σ_c w_c·π(c|g)·（セル平均 CE）で、セル均等抽出の下で母集団の
  群リスクの不偏推定になる。群の重みは 1/G 固定。以前の `mode=dro`（GroupDRO 版）は commit 0f2af99 で削除済み。
- 追加で確かめたこと：同じセル別 CE の下で、`uniform_group`（自然分布の全件）と補正版（セル均等のバッチ）の損失は
  数値で一致した。また、セル均等抽出 + 重みなし CE は、`uniform_group` + 群ごとのクラス重み n_g/(2·n_gc) と一致した。
