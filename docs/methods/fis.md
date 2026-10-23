# Fair Identity Scaling（FIS）

標本ごとの CE に「損失の大きい標本」と「損失の分布が全体から離れた群」を重くする重みを掛ける手法。
Luo et al.（FairVision、arXiv 2023）。この repo では strategy `fis`。

## 1. 手法のロジック

### 解く問題

普通の学習（ERM）は全標本の平均損失を下げるので、少数の群の標本は平均への寄与が小さく、成績が悪くても放置されやすい。
群ごとに重みを付ける方法（GroupDRO など）は群の平均だけを見るので、群の中の外れ値に引きずられて、
群の大半の標本を必要以上に重く（または軽く）することがある。FIS は、標本単位の重み s^I と群単位の重み s^G を混ぜて
この 2 つの問題を同時に扱う。

### 目的関数

ミニバッチ B^t（学習の step t のバッチ）の標本 i の損失を ℓ_i、群を a_i、群の数を |A| とする。

```
L_FIS = (1/|B^t|) Σ_i ((1 − c) · s^I_i + c · s^G_i) · ℓ_i                     （式 3）

s^I_i = exp(ℓ_i) / Σ_{j ∈ B^t} exp(ℓ_j)                                         （式 4）
s^G_i = exp(OT(ℓ(B^t), ℓ(B^t_{a_i}))) / Σ_{a=1..|A|} exp(OT(ℓ(B^t), ℓ(B^t_a)))   （式 5）
```

- **s^I（個別の重み）**：バッチ内の損失の softmax。まだうまく当てられていない標本ほど重い。バッチの和は 1。
- **s^G（群の重み）**：群 a の train から別に引いたミニバッチ B^t_a の損失の分布が、B^t の損失の分布からどれだけ
  離れているかの softmax。離れた群ほど重い。群について正規化するので、群の和が 1。
- **OT（最適輸送）**：一方の分布を他方へ動かすのに要る最小のコスト。分布どうしの距離として使う。
- **c（混合比）**：0 なら個別の重みだけ、1 なら群の重みだけ。

### 学習の手順

1 step で次を行う。

1. 学習バッチ B^t を forward し、標本ごとの損失 ℓ_i を出す。
2. 各群 a について、その群の train から B^t_a を引いて forward し、損失を出す。
3. B^t の損失と各 B^t_a の損失の OT 距離から s^G を、B^t の損失から s^I を作る。
4. 式 (3) の重み付き損失でモデルを更新する。

### 重みの大きさについて

式 (4)(5) のとおりに正規化すると、s^I_i は 1/|B^t| 前後、s^G は 1/|A| 前後の値になる。バッチ全体での重みの和は
個別の項が (1 − c)、群の項が c·|B^t|/|A| 前後なので、c = 0.5 のとき群の項の割合は |B^t| / (|B^t| + |A|) になる。
バッチが群の数より十分大きいと、群の項がほぼ全部を占め、個別の項はほとんど効かない。

## 2. 原論文の設定

- Luo, Khan, Tian, Shi, Dou, Elze, Fang, Wang. *FairVision: Equitable Deep Learning for Eye Disease Screening via
  Fair Identity Scaling.* [arXiv:2310.02492](https://arxiv.org/abs/2310.02492)（v3、2024-04）
- 公式実装：[Harvard-Ophthalmology-AI-Lab/FairVision](https://github.com/Harvard-Ophthalmology-AI-Lab/FairVision)
  （`src/modules.py` の `Fair_Loss_Scaler`、`scripts/train_{amd,dr,glaucoma}_fair_fis.py`、`scripts/train_amd_{vit,3d}_fis.sh`）

### データセットとタスク

Harvard-FairVision。疾患ごとに 10,000 人（1 人 1 件）で、train 6,000・val 1,000・test 3,000。
入力は 2D の SLO 眼底画像と 3D の OCT B-scan の 2 種類で、別々に学習する。群は 1 属性ずつ使う。

| データセット | 入力 | 予測ラベル（クラス数） | 敏感属性（群数） |
| --- | --- | --- | --- |
| AMD | SLO 眼底 / OCT B-scan | normal / early / intermediate / late（4 クラス。公式実装の既定 `--num_classes 4` で CE） | race（3）、gender（2）、ethnicity（2） |
| DR | 同上 | 視力を脅かさない / 脅かす DR（2 クラス） | 同上 |
| Glaucoma | 同上 | normal / glaucoma（2 クラス。公式実装は 1 出力の BCE） | 同上 |

### ハイパーパラメータ

| 項目 | 値 |
| --- | --- |
| モデル | ResNet50、DenseNet121、EfficientNet、ViT-B、Swin-B（2D）。3D は 3D ResNet（公式 script は resnet18 の Conv3d） |
| optimizer | AdamW（全モデル）。公式実装の CNN の分岐は betas (0.0, 0.1) |
| CNN | lr 1e-4、10 epoch、batch 6 |
| Swin-B | lr 1e-4、50 epoch、batch 64 |
| ViT-B | MAE の設定。base lr 5e-4、50 epoch、batch 64、layer decay 0.55、weight decay 0.01、drop path 0.1 |
| 3D | lr 5e-5、50 epoch、batch 2 |
| weight decay（CNN） | 本文に記載なし。公式 script の `--weight_decay` の既定は 0.05（CNN の起動 script は未確認） |
| データ拡張 | 本文に記載なし（未確認） |
| c | 0.5。race・SLO で c を 0〜1 で変えた ablation（Fig. 4）で、c = 0 と 1 より良かった |
| OT | 本文に記載なし。公式実装は geomloss の `SamplesLoss("sinkhorn", p=2, blur=0.1)`（Sinkhorn divergence） |
| B^t_a の件数 | 本文に記載なし。公式 script の既定は 64（3D の script は 6） |
| 温度・`right_asymptote` | 本文に記載なし。公式 script の既定はどちらも 1（3D の script は温度 10） |

公式実装は B^t_a を群ごとの無限 loader（shuffle あり、`drop_last=False`）から引き、群ごとに別々に、train モード・
勾配なしで forward する。s^I には勾配が流れ、s^G には流れない。

公式実装の s^G は式 (5) と 2 点違う。OT の相手が B^t ではなく「全群の B^t_a の連結」で、softmax を群ではなく
バッチの標本について取る（`Fair_Loss_Scaler.forward`）。

### モデルの選び方と評価

本文に記載なし。公式実装は val の全体 AUC が最大の epoch を選び、その epoch の test を報告する。
評価は全体 AUC、ES-AUC（群間の AUC の差で割り引いた AUC）、群別 AUC、DPD、DEOdds、Mean / Max PSD
（群別 AUC のばらつき・最大差を全体 AUC で割ったもの）。5 回の平均と標準偏差。

## 3. この repo の実装

### ファイルと設定

| 役割 | 場所 |
| --- | --- |
| 損失 | [objectives/cohort.py](../../projects/hypernet_e2e/objectives/cohort.py) の `FISTaskLoss`。OT は同じ file の `sinkhorn_divergence`（中身は `_entropic_ot`） |
| strategy | [configs/training_strategy/fis.yaml](../../projects/hypernet_e2e/configs/training_strategy/fis.yaml)（`fusion` が c、`blur`、`group_sample_size` が B^t_a の件数） |
| B^t_a の抽出 | [data/group_datamodule.py](../../projects/hypernet_e2e/data/group_datamodule.py) の `GroupSampleBatchSampler` と `GroupImageDataModule.group_sample_batches` |
| B^t_a の forward | [objectives/cohort.py](../../projects/hypernet_e2e/objectives/cohort.py) の `GroupSampleSource`（`FISTaskLoss.callbacks()` が返す callback。LitModule は `configure_callbacks` で Trainer に渡すだけ） |
| クラス重み | [training.py](../../projects/hypernet_e2e/training.py) の `_resolve_inverse_class_weights` |
| 診断値の記録 | [callbacks/objective_diagnostics.py](../../projects/hypernet_e2e/callbacks/objective_diagnostics.py) の `ObjectiveDiagnosticsRecorder` |
| テスト | [tests/test_cohort_objectives.py](../../projects/hypernet_e2e/tests/test_cohort_objectives.py)（式 3〜5 との一致、勾配が s^I だけに流れること、B^t_a の件数） |
| 代表的な preset | `resnet_ham10000_fg_fis_sex_age`（性別 × 年齢の 4 群）、`resnet_ham10000_fg_fis_lhcf`（LHCF の 11 コホート、[worst-cohort.md](worst-cohort.md)） |

### 処理の流れ

**起動時（fit の前）。** HAM10000 の preset は `weighting: inverse` なので、`_resolve_inverse_class_weights` が
train のラベルから全群共通のクラス重み（平均 1 に正規化した逆頻度）を計算し、`class_weight` に書き込む。
B^t と B^t_a の両方の CE にこの重みが掛かる。

**学習の開始時（`GroupSampleSource.on_train_start`）。** `group_sample_batches(group_sample_size)` が
B^t_a の流れを作り、「次の B^t_a を推論して返す関数」を `FISTaskLoss.set_group_sample_source` で目的関数に渡す。`GroupSampleBatchSampler` が各群から `group_sample_size` 件ずつ引いて 1 つのバッチに連結する。
群の中は非復元で 1 周してから並べ直し、端数は次の周の先頭で埋めるので、常に同じ件数になる。transform は train と同じ。
この loader は `num_workers` 本の worker を学習の間ずっと常駐させる。

**学習の各ステップ。**

1. `FISTaskLoss.forward`（学習バッチ B^t の forward の後に呼ばれる）：B^t の標本ごとのクラス重み付き CE を出す（`_weighted_ce`）。
2. 渡された関数で B^t_a を 1 つ受け取る。B^t_a はモデルを train モードのまま勾配なしで forward する。全群の B^t_a を
   1 回でまとめて流す。公式実装と同じく B^t の forward の後に B^t_a を流すので、BatchNorm の running 統計は B^t、
   B^t_a の順に動く。
3. 勾配なしで、B^t の損失と群 a の B^t_a の損失の `sinkhorn_divergence` を群ごとに計算し、群について softmax して s^G にする（式 5）。
4. B^t の損失の softmax を s^I にする（式 4）。こちらは勾配を流す。
5. `((1 − c) · s^I_i + c · s^G_{a_i}) · ℓ_i` のバッチ平均を返す（式 3）。

`sinkhorn_divergence` は geomloss の定義（コスト |x − y|²/2、ε = blur²、自分自身との OT を差し引いて偏りを除く）を
1 次元で自前に書いたもの。ε を大きい値から目標値へ 50 段で下げて解く。

**設計上の選択。** 論文と公式実装が食い違う 2 点（OT の相手と softmax の向き）は論文の式に合わせた。
論文に書いていない所（Sinkhorn、blur 0.1、B^t_a 64 件、温度 1、勾配の流し方）は公式実装に合わせた。

**記録と checkpoint。**

- `ObjectiveDiagnosticsRecorder` が epoch ごとに `metrics/objective_diagnostics.jsonl` へ、s^G（`group_weight`）、
  OT 距離（`group_distance`）、重みの総和のうち群の項が占める割合（`group_term_share`）を書く。
  `group_term_share` で、1 節の「群の項の割合」を run ごとに確かめられる。
- checkpoint は既定の best val AUROC（`best_val_auroc_*.ckpt`）と `last.ckpt` に加え、`_fairness_ham10000.yaml` により
  val の性別 × 年齢の最悪群 AUROC で選んだ `best_val_worst_sexage_*.ckpt` を残す。公式実装に揃えるなら best val AUROC を使う。
  LHCF のコホートで学習した run を性別 × 年齢で選ぶと、属性ラベルを選択に使うことになる。
  予測は `analysis/common/predictions.py` に `--checkpoint <名前>` で渡す。

### 原論文との違い

| 項目 | 原論文・公式実装 | この repo |
| --- | --- | --- |
| s^G の正規化 | 論文は群について。公式実装はバッチの標本について | 群について（論文どおり） |
| OT の相手 | 論文は B^t と B^t_a。公式実装は B^t_a と全群の B^t_a の連結 | B^t と B^t_a（論文どおり） |
| B^t_a の forward | 群ごとに別々に | 全群の 64 × \|A\| 件を 1 回で。BatchNorm のバッチ統計と running stats の更新が公式と違う |
| B^t_a の抽出 seed | DataLoader の shuffle（全体の seed に従う） | 42 に固定（`GroupSampleSource` が seed を渡さず既定値になる）。run の seed を変えても同じ順に引く |
| B^t_a の端数 | 群の 1 周の終わりで 64 件未満になりうる | 常に 64 件 |
| Sinkhorn | geomloss | 自前の実装（定義は同じ） |
| 損失 | クラス重みなし（BCE か CE） | 逆頻度のクラス重み付き CE（2 出力） |
| データ・群 | 眼科画像、race 等の 2〜3 群 | HAM10000（良性 / 悪性の 2 クラス）、性別 × 年齢の 4 群か LHCF の 11 コホート |
| 最適化 | AdamW。CNN は lr 1e-4・batch 6・10 epoch | AdamW lr 1e-4・weight decay 0.01・batch 128・30 epoch |
| データ拡張 | 未確認 | あり（`train_transforms_medfair`） |
| モデルの選び方 | val の全体 AUC が最大の epoch | best val AUROC・最悪群 AUROC・last を残し、分析側で選ぶ |

c の効き方はバッチの大きさに強く依存する。batch 128 では c = 0.5 でも群の項が 9 割を超える（1 節の式で 4 群 0.97、11 群 0.92）。
原論文の CNN（batch 6、2〜3 群）では 0.67〜0.75 なので、個別の項と群の項の比を原論文に近づけたいなら
c を下げる（例：群の項を半分にするなら c = |A| / (|B^t| + |A|)、4 群で 0.030、11 群で 0.079）。
公式実装の正規化に戻すにはコードの変更が要る。

### 起動例

```bash
uv run python -m projects.hypernet_e2e.run experiment=resnet_ham10000_fg_fis_sex_age study=<slug> seed=42
# 個別の項と群の項を半々に効かせる場合（4 群）
uv run python -m projects.hypernet_e2e.run experiment=resnet_ham10000_fg_fis_sex_age study=<slug> seed=42 model.loss_fn.fusion=0.030
```

B^t_a の loader が worker を常駐させるので、並列で回すときは `data.num_workers` を下げる。
