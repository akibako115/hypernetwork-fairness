# FairDRO

正解クラスごとに群の重みを持ち、誤り率の高い群を重く、低い群を軽くして学習する手法。Jung et al., ICLR 2023。
狙いは Equalized Odds（EO）で、この repo では strategy `fair_dro`。

## 1. 手法のロジック

### 解く問題

普通の学習（ERM）は全サンプルの平均損失を下げるので、群（性別・年齢など）によって正解率が違っても直さない。
FairDRO が揃えたいのは、**正解クラスを固定したときの群ごとの正解率**である。原論文はその差を DCA
（Difference of Conditional Accuracy）で測る。

```
Δ_y   = max_{a,a'} | P(Ŷ=y | A=a, Y=y) − P(Ŷ=y | A=a', Y=y) |     A：群（敏感属性）、Y：正解クラス
DCA   = (1/|Y|) Σ_y Δ_y
```

二値分類では、DCA が 0 であることは EO（群の間で TPR と TNR が等しい）と同じ。

### 目的関数

クラス y ごとに群の重み q^y = (q^y_1, …, q^y_G) を持つ（G は群数）。L(θ, D^y_a) は群 a・クラス y のセル
（群 × クラスの組）の平均損失。クラスごとに最悪の q^y を取り、そのクラス平均を最小化する（式 6）。

```
min_θ  (1/|Y|) Σ_y  max_{q^y ∈ Q_ρ}  Σ_a q^y_a · L(θ, D^y_a)
Q_ρ = { q : Σ_a q_a = 1,  D_φ(q ‖ 一様分布) ≤ ρ }，φ(t) = (t − 1)²
```

Q_ρ は一様分布 1/G を中心とする χ² ダイバージェンスの球で、半径 ρ が大きいほど q は一様から離れられる。
**q ≥ 0 の制約が無い**ので、負の重み（擬似確率）も取れる。負の重みは、成績の良い群の損失を逆に上げる向きに働く。

この内側の最大化は、群損失の平均に「群損失の分散の平方根 × √ρ」を足したものに等しい（式 5、Xie et al. の結果）。
損失を 0-1 損失（誤りなら 1）にすると、FairDRO は「群・クラス均等の誤り率 + 定数 × √ρ × DCA」の最小化と同じになる
（系 1、式 7）。つまり DCA そのものを正則化項に持つ。

内側の最大化には閉形式の解がある（命題 2、式 8）。e_a = L(θ, D^y_a)、ē はその群平均。

```
q^y*_a = 1/G + √(ρ/G) · (e_a − ē) / ‖e − ē‖₂
```

各成分は [1/G − √(ρ(G−1))/G, 1/G + √(ρ(G−1))/G] に入る。下端が負になる ρ では、成績の良い群に負の重みが付く。
G = 2 では誤り率の差の符号だけで q* が決まり、2 つの値の間を行き来する。

### 学習の手順（原論文 Algorithm 1）

q を毎回閉形式の解に置き換える方法を IBR（iterated best response）という。そのままでは q が振動するので、
原論文は η_t = 1 − t/T（T は総 epoch 数）で混ぜる「平滑化 IBR」を使う（式 10）。

1. q^y を一様 1/G で始める。
2. epoch の中では、群 × クラスの各セルから均等にミニバッチを引き、q で重み付けしたセル平均の CE
   `(1/|Y|) Σ_y Σ_a q^y_a · CE_{a,y}` で θ を更新する（式 9）。
3. epoch の終わりに、**train 全件**で群 × クラス別の 0-1 誤り率を測り、式 8 で q* を出して
   `q ← (1 − η_t) q + η_t q*` とする。

θ の更新は微分できる CE、q を決めるのは 0-1 損失、と役割を分けている（4.4 節）。

## 2. 原論文の設定

- Jung, Park, Chun, Moon. *Re-weighting Based Group Fairness Regularization via Classwise Robust Optimization.*
  ICLR 2023（[arXiv:2303.00442](https://arxiv.org/abs/2303.00442)）
- 公式実装：[sangwon-jung94/FairDRO](https://github.com/sangwon-jung94/FairDRO)（`trainer/fairdro.py`、`arguments.py`）

### データセットとタスク

| データセット | 入力 | 予測ラベル（クラス数） | 群・敏感属性（群数） |
| --- | --- | --- | --- |
| UCI Adult | 表形式 | 年収が 5 万ドルを超えるか（2 クラス） | 性別（2） |
| ProPublica COMPAS | 表形式 | 再犯するか（2 クラス） | race：Caucasian か否か（2） |
| UTKFace | 顔画像 | 年齢 0〜19 / 20〜40 / 40 超（3 クラス） | ethnicity：White・Black・Asian・Indian（4） |
| CelebA（付録 D.2） | 顔画像 | 金髪か（2 クラス） | 性別（2） |
| CivilComments-WILDS | 文 | toxic か（2 クラス） | ethnicity：Black・White・Asian・Latino・Others（5） |

UTKFace だけが多クラスで、そこでは DCA は EO と一致しない（付録 D.1 で EO の差も報告）。

### ハイパーパラメータ

| 項目 | 値 |
| --- | --- |
| モデル | 表形式はロジスティック回帰、画像は ResNet18、文は事前学習済み BERT |
| 画像モデルの初期値 | Algorithm 1 は「θ をランダムに初期化」。公式 README のコマンドに `--pretrained` は無い |
| optimizer（表形式・画像） | AdamW、lr 1e-3 を cosine で減衰、70 epoch、batch 128、weight decay 1e-3（付録 C.1）。公式 README のコマンドは weight decay 1e-4 |
| optimizer（文） | AdamW、lr 2e-5（warm-up のあと線形減衰）、3 epoch、batch 24、weight decay 1e-3。q は 100 iteration ごとに更新 |
| データ拡張 | 本文に記載なし。公式の UTKFace は Resize 256 → RandomCrop 224 → 左右反転 |
| 抽出 | Algorithm 1 はセル均等。公式 README のコマンドの抽出（`--labelwise`）は公開版の引数に無く、未確認 |
| ρ | 格子 [1e-2, 1e2] を振る（表 C.1）。公式 README の値は Adult 5、COMPAS 3、UTKFace 0.3、CivilComments 0.3、CelebA 1.5 |
| q を決める損失 | 論文は 0-1 損失。公式の既定は CE で、`--use-01loss` で 0-1 になる |
| 誤りを測るときのモデル（公式） | train 全件を `model.train()` のまま、学習用 transform で推論する |

**ρ の単位に注意。** 公式の `_update_mw_margin` は `1/G + √(2ρ/G) · …` で、論文の式 8 の √(ρ/G) と違う。
同じ q を出すのは「論文の ρ = 2 × 公式の ρ」のとき。README の値は公式の単位。表 C.1 の格子がどちらの単位かは未確認。

公開版のコードについて：README のコマンドは `--method lgdro_chi`・`--optim-q ibr_ip`・`--trueloss`・`--labelwise`
など、公開版の `arguments.py` に無い引数を使っていて、そのままでは動かない。公開コードで論文の手順に当たるのは
`--method fairdro --optim-q smt_ibr --use-01loss`（`--q-decay` の既定 `linear` が η_t = 1 − t/T）。
既定の `--optim-q pd` は `adv_probs_dict` を初期化していないので、最初の q 更新で止まる。

### モデルの選び方

- 報告するのは**最終 epoch** のモデル（付録 C.1 末尾「All results are reported for the model at the last epoch」）。
- 評価は群 × クラスの balanced accuracy と DCA。ρ を振って accuracy と DCA のパレート曲線を描く。
- 単一の数値では、「Scratch（公平性なしの学習）の accuracy の 95% 以上を保つ設定のうち DCA が最良のもの」を報告する
  （付録 C.4）。この選択に val を使ったか test を使ったかは本文に記載が無い。4 seed の平均。

## 3. この repo の実装

### ファイルと設定

| 役割 | 場所 |
| --- | --- |
| 損失・閉形式・q の更新 | [objectives/fair_dro.py](../../projects/hypernet_e2e/objectives/fair_dro.py) の `FairDROTaskLoss`、`chi_square_best_response`、`nonnegative_radius_limit` |
| 共通の基底（セルの取り出し） | [objectives/conditional.py](../../projects/hypernet_e2e/objectives/conditional.py) の `ConditionalObjective` |
| strategy | [configs/training_strategy/fair_dro.yaml](../../projects/hypernet_e2e/configs/training_strategy/fair_dro.yaml)（`radius: 0.5`、`max_epochs: ${trainer.max_epochs}`） |
| epoch 末の誤り率と q の更新 | 同じ file の `FairDROEpochUpdate`（`callbacks()` が返す callback）・`_split_errors`。推論は [module.py](../../projects/hypernet_e2e/module.py) の `LitModule.eval_logits` |
| 起動時の検証と件数の注入 | [training.py](../../projects/hypernet_e2e/training.py) の `_resolve_conditional_population` |
| セル均等抽出・held-out | [data/group_class_sampler.py](../../projects/hypernet_e2e/data/group_class_sampler.py) の `GroupClassBatchSampler`・`GroupClassImageDataModule` |
| 群の定義 | data config（例 [chexpert_age_known_cells.yaml](../../projects/hypernet_e2e/configs/data/chexpert_age_known_cells.yaml)：65 歳未満/以上の 2 群、[ham10000_sex_age_cells.yaml](../../projects/hypernet_e2e/configs/data/ham10000_sex_age_cells.yaml)：性別 × 年齢の 4 群） |
| 記録・候補の選択 | [callbacks/objective_diagnostics.py](../../projects/hypernet_e2e/callbacks/objective_diagnostics.py) の `ObjectiveDiagnosticsRecorder`、[callbacks/conditional_fairness.py](../../projects/hypernet_e2e/callbacks/conditional_fairness.py) の `ConditionalFairnessRecorder`・`select_candidate` |
| 代表的な preset | `resnet_chexpert_age_fair_dro_race_known`（CheXpert 2 群、ImageNet から）、`resnet_chexpert_from_resnet_age_fair_dro_race_known`（同じ seed の ERM から継続）、`resnet_ham10000_fg_fairdro_sex_age`（HAM10000 4 群） |

### 処理の流れ

**起動時（fit の前）。**

- `_resolve_conditional_population`：single device・勾配蓄積なし・bf16 か fp32 でなければ止める。DataModule が
  `GroupClassImageDataModule` でなければ止める。`train.csv` から群 × クラスの件数 [G, 2] を数え、
  `population_cell_counts` に書く。空のセルがあれば止める。
- クラス重みは使わない。preset はどれも `weighting: none` で、`weighting=inverse` にすると `class_weight` が入り、
  `FairDROTaskLoss` が止まる。セル均等に抽出するので、クラスの偏りは抽出で消えているため。
- `FairDROTaskLoss.__init__`：q を buffer `q`（形 [2, G]、初期値 1/G）として持つ。ρ が
  `nonnegative_radius_limit` の 1/(2(G−1)) 以上なら、q* が負になり得ると警告する（止めはしない）。
  上限は G = 2 で 0.5、G = 4 で 0.167、G = 8 で 0.071 なので、既定の 0.5 ではどの preset でも警告が出る。
- `FairDROEpochUpdate.setup`・`RejectFitResume.on_load_checkpoint`：複数 GPU・勾配蓄積・途中再開を拒否する。

**学習の各ステップ。**

1. `GroupClassBatchSampler`：1 バッチに各セルから B/(2G) 件を復元抽出する（B = 128 なら G = 2 で 32 件、G = 4 で 16 件）。
   B は 2G の倍数でなければならない。
2. `FairDROTaskLoss.forward`：`_cells` で群 × クラスのマスクを作り、セルごとの平均 CE を [G, 2] で出す。
   これに q を掛けて足し、2（クラス数）で割る（式 9）。q は勾配を流さない。
3. モデルを更新する。q はこの時点では変えない（step ごとの双対の確定 `StepDualUpdate` は付かない）。

**epoch の終わり（`FairDROEpochUpdate.on_train_epoch_end`）。**

4. 誤り率を測る split を決める。DataModule が held-out を持てば held-out 全件、無ければ train 全件。
5. `_split_errors`：`LitModule.eval_logits`（eval モード）・評価 transform・sampler なしで 1 周し、`z1 − z0 > 0` を陽性としたときの
   群 × クラス別の誤り率を出す。行の欠けや空のセルがあれば止める。
6. `update_from_errors`：t を更新回数、T を `max_epochs` として η = 1 − t/T。クラスごとに
   `chi_square_best_response` で q* を出し、`q ← (1 − η) q + η q*`。最初の更新は η = 1 なので q = q* になる。
   `trainer.max_epochs` を変えると T も一緒に変わる。

**ρ の単位は公式実装に合わせた。** `chi_square_best_response` は `1/G + √(2ρ/G) · …` で、公式 README の値をそのまま
入れれば公式コードと同じ重みになる。論文の式や図の ρ を使うときは 1/2 にする。

**held-out を使える理由。** train 全件で測ると、モデルが少数セルを記憶して誤り率が 0 に近づき、q が一様に戻る。
これを避けるため、学習に使わない held-out で測る選択肢を持たせた（EO rate・Minimax AUC の held-out と同じ扱い）。
held-out を持つ data config は `*_dual_holdout`（train の 20% を held-out に分けた split）。FairDRO の preset には無いので、
使うときは data を差し替える（下の起動例）。

**記録と checkpoint。**

- `metrics.csv`：`train/fair_dro/q/y{c}_g{g}`（更新後の q）、`train/fair_dro/error/g{g}_y{c}`（誤り率）、`train/fair_dro/q_min`。
- `metrics/fair_dro_q.jsonl`：1 行 = 1 回の更新。`eta`・`q_star`・`q`・`error`・`count`・`source`（`train` か `holdout`）・`q_min`。
- `metrics/objective_diagnostics.jsonl`：バッチ内のセル別 CE（`risk`）と予測の陽性率（`rates`）。最後のバッチの値と `epoch_mean`。
- checkpoint は既定の best val AUROC と last。HAM10000 の preset は val の性別 × 年齢の最悪群 AUROC の best も残す
  （gmm8 preset でも最悪群は性別 × 年齢で測り、学習に使った群ではない）。
- CheXpert 年齢 2 群の preset は `ConditionalFairnessRecorder` が毎 epoch `checkpoints/candidate_XXX.ckpt` と
  `metrics/fairness_candidates.json`（val の群別 TPR・TNR・AUROC）を残す。継続学習 preset は fit の終わりに
  `select_candidate` で EO の規則で選ぶ：val の TPR 差・TNR 差がともに 0.1 以下、どの群の TPR・TNR も親 ERM から
  0.02 より下がらず、AUROC と最悪年齢 AUROC が親から 0.005 以内。満たす epoch のうち val の重み付き CE が最小のもの。
  無ければ `infeasible`。原論文に合わせた最終 epoch も `last_epoch` に書く。1 stage preset（`reference: deferred`）は
  選ばず、分析側で同じ seed の ERM の val を基準に選ぶ。
- held-out で q を更新した run は、epoch の選択に val を使う（held-out は q の更新に使ったため）。

### 原論文との違い

| 項目 | 原論文 | この repo |
| --- | --- | --- |
| タスク | 2 クラスと 3 クラス | 2 クラスだけ（q の形が [2, G]） |
| 誤りを測るデータ | train 全件 | train 全件。held-out を持つ data config なら held-out 全件 |
| 誤りを測るときのモデル | 公式は train モード・学習用 transform | eval モード・評価 transform |
| 抽出 | セル均等（Algorithm 1） | セル均等（セル内は復元抽出） |
| 初期値 | ランダム初期化 | ImageNet 重み。継続学習 preset は同じ seed の ERM |
| モデル | ResNet18 ほか | ResNet-50 か Spatial LoRA ResNet-50 |
| optimizer | AdamW lr 1e-3、cosine、70 epoch | AdamW lr 1e-4、weight decay 0.01、schedule なし、30 epoch |
| ρ | 格子を振ってパレート曲線 | 0.5 の 1 点（公式の単位） |
| モデルの選び方 | 最終 epoch | best val AUROC と last。CheXpert 2 群は EO の規則（上） |
| 評価 | balanced accuracy と DCA | 固定しきい値の群別 TPR・FPR（[research-premises.md](../research-premises.md)） |

原論文に揃えるなら、まず `model.loss_fn.radius` を格子で振る。負の重みを避けるなら上限 1/(2(G−1)) 未満にする。
最終 epoch で比べるなら全 epoch の重みを残し、optimizer の条件まで揃えるなら lr・epoch・scheduler を変える。

### 起動例

```bash
uv run python -m projects.hypernet_e2e.run experiment=resnet_ham10000_fg_fairdro_sex_age study=<slug> seed=42
```

```bash
# ρ を負の重みが出ない範囲に下げ、q を held-out 全件で測り、全 epoch の重みを残す（HAM10000 4 群）
uv run python -m projects.hypernet_e2e.run experiment=resnet_ham10000_fg_fairdro_sex_age study=<slug> seed=42 \
  model.loss_fn.radius=0.1 data=ham10000_sex_age_cells_dual_holdout \
  callbacks.model_checkpoint.save_top_k=-1 callbacks.model_checkpoint.save_weights_only=true
```

継続学習 preset（`*_from_resnet_*`）は `model.backbone_checkpoint_path=<同じ seed の ERM の checkpoint>` が必須。
