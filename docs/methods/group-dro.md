# GroupDRO

群ごとの損失のうち最も大きいものを下げるように学習する手法。Sagawa et al., ICLR 2020。
この repo では strategy `group_dro_global`（原典の形）と `group_dro_subgroup`（q の更新だけ群ごとのクラス重みを使う版）の 2 つ。

## 1. 手法のロジック

### 解く問題

普通の学習（ERM）は全サンプルの平均損失を下げる。少数の群（例：高齢 × Black）は平均への寄与が小さいので、
その群の成績が悪くても平均はあまり悪くならず、放置されやすい。GroupDRO は、あらかじめ決めた群 g = 1, …, m の
**最悪の群の損失**を下げる。

```
min_θ  max_g  L_g(θ)          L_g(θ)：群 g に属するサンプルの平均損失
```

max は微分しにくいので、群の重み q（m 個の非負の値で合計 1）を導入して書き換える。

```
min_θ  max_{q ∈ Δ_m}  Σ_g q_g · L_g(θ)
```

q を自由に選べるなら、最悪の群に重み 1 を置くのが最大になるので、上の式と同じ問題になる。

### 学習の手順（オンライン版、原論文 Algorithm 1）

θ（モデル）と q を交互に 1 歩ずつ動かす。ミニバッチごとに次の 3 つを行う。

1. バッチの中で群ごとの平均損失 ℓ_g を計算する。
2. q を指数的に更新して正規化する：`q_g ← q_g · exp(η_q · ℓ_g)`、`q ← q / Σ q`。
   損失の大きい群ほど重みが増える。η_q は q の動きの速さ。
3. 更新後の q で重み付けした損失 `Σ_g q_g · ℓ_g` で θ を勾配更新する。

q は学習の途中で「いま損失の大きい群」を追いかける重みで、学習が進んで損失が逆転すれば別の群に移る。

### 正則化が前提であること

原論文の主張の中心は副題（*On the Importance of Regularization for Worst-Case Generalization*）にある。
大きなモデルは train の全サンプルをほぼ完全に当てられるので、正則化が弱いと全群の train 損失が 0 に近づき、
q がどの群に寄っても θ の更新が ERM と変わらなくなる。強い ℓ2 正則化か早めの打ち切り（early stopping）で
train 損失が 0 にならないようにして、初めて最悪群の test 成績が ERM より良くなる。

### group adjustment

少数の群ほど train と test の差（汎化ギャップ）が大きいと見込み、群損失に `C / √n_g`（n_g は群の train 件数）を
足してから q を更新する。C はハイパーパラメータ。

## 2. 原論文の設定

- Sagawa, Koh, Hashimoto, Liang. *Distributionally Robust Neural Networks for Group Shifts: On the Importance of
  Regularization for Worst-Case Generalization.* ICLR 2020（[arXiv:1911.08731](https://arxiv.org/abs/1911.08731)）
- 公式実装：[kohpangwei/group_DRO](https://github.com/kohpangwei/group_DRO)（`loss.py` の `compute_robust_loss`、`run_expt.py`）

### データセットとタスク

群はどれも **予測ラベル × 属性** の組（m = |Y| × |A|）で、陽性と陰性は別の群になる。

| データセット | 入力 | 予測ラベル（クラス数） | 属性（見かけの相関を持つ変数） | 群数 |
| --- | --- | --- | --- | --- |
| Waterbirds（CUB の鳥を Places の背景に合成） | 画像 | 水鳥 / 陸鳥（2 クラス） | 背景（水 / 陸） | 4 |
| CelebA | 顔画像 | 金髪 / 黒髪（2 クラス） | 性別 | 4 |
| MultiNLI | 文の組 | 含意 / 中立 / 矛盾（3 クラス） | 否定語の有無 | 6 |

評価は群ごとの accuracy の最小値（最悪群 accuracy）と平均 accuracy。

### ハイパーパラメータ

画像（ResNet50、torchvision の ImageNet 事前学習済み）：

| 項目 | 値 |
| --- | --- |
| optimizer | SGD（momentum 0.9）、固定 lr（スケジューラなし） |
| batch size | 128 |
| データ拡張 | なし |
| 標準の設定（3.1 節） | ℓ2 λ = 1e-4。lr は Waterbirds 1e-3、CelebA 1e-4。Waterbirds 300 epoch、CelebA 50 epoch |
| 強い ℓ2（3.2 節） | λ は Waterbirds 1.0、CelebA 0.1。lr 1e-5 |
| early stopping（3.2 節） | 1 epoch で打ち切る |
| group adjustment（3.3 節） | C ∈ {0, 1, 2, 3, 4, 5} から選ぶ |
| η_q | 本文に記載なし。公式実装 `--robust_step_size` の既定は 0.01 |

文章（MultiNLI、BERT base uncased）：AdamW、lr 2e-5 から線形に減衰、ℓ2 なし、batch 32。標準 20 epoch、early stopping 3 epoch。

抽出：Algorithm 1 は群を一様に引く形で書かれている。公式実装の例のコマンドは `--reweight_groups`
（群の頻度の逆数で重みを付けた `WeightedRandomSampler`、train だけ）を付ける。

### モデルの選び方

ベンチマーク（4 節、表 3）では、ℓ2 の強さ・epoch 数・C を grid search し、**val の最悪群 accuracy が最大のモデル**を
報告する（付録 C.2「best early stopping epoch (as measured by robust validation accuracy)」）。

## 3. この repo の実装

### ファイルと設定

| 役割 | 場所 |
| --- | --- |
| 損失（q の更新を含む） | [objectives/group.py](../../projects/hypernet_e2e/objectives/group.py) の `_ClassWeightedGroupDROTaskLoss` と、その子 class `GlobalClassWeightedGroupDROTaskLoss`・`SubgroupClassWeightedGroupDROTaskLoss` |
| strategy | [configs/training_strategy/group_dro_global.yaml](../../projects/hypernet_e2e/configs/training_strategy/group_dro_global.yaml)、[group_dro_subgroup.yaml](../../projects/hypernet_e2e/configs/training_strategy/group_dro_subgroup.yaml) |
| クラス重みの計算 | [training.py](../../projects/hypernet_e2e/training.py) の `_resolve_inverse_class_weights`・`_scoped_inverse_weights` |
| 群の定義 | data config（例 [chexpert_age_race_known_groups.yaml](../../projects/hypernet_e2e/configs/data/chexpert_age_race_known_groups.yaml)：65 歳未満/以上 × race 4 区分 = 8 群） |
| q の記録 | [callbacks/group_dro_diagnostics.py](../../projects/hypernet_e2e/callbacks/group_dro_diagnostics.py) の `GroupDRODiagnostics` |
| 代表的な preset | `resnet_chexpert_age_race_group_dro_race_known`（global）、`resnet_chexpert_age_race_group_dro_subgroup_race_known`（subgroup）、`resnet_ham10000_sex_age_group_dro` |

### 処理の流れ

**起動時（fit の前）。** `training.py` が次の 2 つを行う。

- `_validate_group_dro_weighting`：strategy 名が `group_dro_` で始まるのに `weighting=inverse` でなければ止める。
  どちらの版もクラス重みを必須にしているため。
- `_resolve_inverse_class_weights`：`train.csv` のラベルからクラス重みを計算し、config に書き込む。
  - `class_weight`（モデルの更新用、両版共通）：w_c = N / (C · n_c) を平均 1 に正規化した `[num_classes]`。
  - `q_class_weight`（q の更新用、`group_dro_subgroup` だけ）：w_gc = n_g / (C · n_gc) の `[num_groups, num_classes]`。
    群の中で陽性の重みの合計と陰性の重みの合計が等しくなる。平均 1 には正規化しない。
    どちらを計算するかは strategy の `class_weight_scope`・`q_class_weight_scope` で決まる。

**学習の各ステップ（`_ClassWeightedGroupDROTaskLoss.forward`）。**

1. `_per_sample_loss`：サンプルごとの CE にクラス重みを掛ける。
2. `_group_losses`：群ごとに、1 の損失の和を群の件数で割る（群内の重みの合計では割らない）。
   バッチに 1 件もない群の損失は 0 にする。
3. `_exponentiated_update`：`q ← q · exp(step_size · ℓ_g)` を全群で正規化する。損失 0 の群は係数 1 のままだが、
   他の群が増えて正規化されるので相対的に下がる。q は buffer `adv_probs`（初期値 1/m）として持ち、checkpoint に保存される。
4. 更新後の q と、`class_weight` で計算した群損失の内積 `Σ q_g · ℓ_g` を返し、これでモデルを更新する。

`group_dro_subgroup` は、3 の q の更新にだけ `q_class_weight` で計算した群損失を使う。4 は両版とも同じ。
ここまでの順序（先に q を更新し、更新後の q で損失を重み付けする）・バッチにない群の扱い・η の既定 0.01 は
公式の `compute_robust_loss` と同じ。

**2 つの版がある理由。** 全群共通のクラス重みでは、陽性 1 件の重みが陰性の数倍（CheXpert で約 8.7 倍）になるので、
予測の質が同じでも陽性率の高い群ほど群損失が大きくなる。q は最悪の成績の群ではなく陽性率の高い群に寄る
（CheXpert の 8 群では 65 歳未満の群）。`q_class_weight` で計算した群損失は「陽性の平均 CE と陰性の平均 CE の平均」に
なり、陽性率によらないので、q の更新にだけこれを使う。モデルの更新まで群ごとの重みにすると、陽性と陰性の重みの比が
群ごとに違うので、群を見分けられるモデルでは最適な logit が群ごとに log(n_{g,0} / n_{g,1}) だけずれ、しきい値 0.5 の
群別 TPR・FPR に「群ごとにしきい値を変える」のと同じ効果が混ざる（式からの導出で、run では未測定）。そのため
モデルの更新は全群共通の重みのままにしている。

**記録と checkpoint。**

- `GroupDRODiagnostics` が epoch ごとの q（`train/group_dro/q/g{id}`）、モデルの更新に使った群損失（`loss`）、
  q の更新に使った群損失（`q_loss`）を `metrics.csv` に、群番号とラベルの対応を `metrics/group_dro_groups.json` に書く。
  複数 GPU では q を GPU 間で揃える処理が無いので、このコールバックが `world_size != 1` で止める。
- 両 strategy とも `save_top_k: -1`・`save_weights_only: true` で全 epoch の重みを残す。原論文のように val の最悪群の
  成績で epoch を選ぶのは分析側で行う（属性だけの群で何を「最悪群の成績」とするかは未決）。予測は
  `analysis/common/predictions.py` に `--checkpoint <名前>` で epoch を渡す。

### 原論文との違い

| 項目 | 原論文 | この repo |
| --- | --- | --- |
| 群 | 予測ラベル × 属性 | 属性だけ（陽性と陰性が同じ群損失に入る） |
| クラス重み | なし | あり（全群共通の inverse。subgroup 版は q の更新に群ごとの重み） |
| 抽出 | 群を均等に（`--reweight_groups`） | 自然抽出（`train_sampling: uniform`）。batch 128 では少数群が 1 件も入らないバッチがある |
| 正則化 | 強い ℓ2（0.1〜1.0）か 1 epoch の early stopping | AdamW の weight decay 0.01、30 epoch |
| optimizer | SGD momentum 0.9、固定 lr | AdamW lr 1e-4 |
| データ拡張 | なし | あり（`train_transforms_medfair`） |
| group adjustment | C を探索 | なし（C = 0 相当） |
| モデルの選び方 | val の最悪群 accuracy が最大の epoch | 全 epoch を残し、分析側で選ぶ |
| 評価 | 最悪群 accuracy | しきい値 0.5 の群別 TPR・FPR（[research-premises.md](../research-premises.md)） |

正則化の弱さは原論文の主張に直接かかわる。原論文に揃えるなら、`trainer.max_epochs` を ERM の best val epoch
前後まで縮めるか、`model.optimizer.weight_decay` を上げる（AdamW の decoupled weight decay は ℓ2 罰則と同じではない）。
群を「予測ラベル × 属性」にするには、群 × クラスのセルを群とする data config を足す必要がある。

### 起動例

```bash
uv run python -m projects.hypernet_e2e.run experiment=resnet_chexpert_age_race_group_dro_subgroup_race_known study=<slug> seed=42
```
