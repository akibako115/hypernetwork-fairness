# Minimax AUC と Population AUC

群ペアの AUC のうち最も低いものを上げるように学習する手法。Yang et al., AAAI 2023。
この repo では strategy `minimax_auc`。その対照として、ペアの重みを固定した `population_auc`（全体 AUC の最大化）も置く。

## 1. 手法のロジック

### 解く問題

AUC は「ランダムに選んだ陽性のスコアが、ランダムに選んだ陰性のスコアより高い確率」で、しきい値によらない
順位の良さを測る。群がある場合は、**陽性をどの群から、陰性をどの群から取るか**で AUC が分かれる。

```
AUC_{g,h} = P( f(X) > f(X') | Y = 1, Z = g, Y' = 0, Z' = h )      f：スコア、Z：群
```

g = h なら群内 AUC、g ≠ h なら群間 AUC と呼ぶ。G 群なら G² 個のペアがある（2 群で 4、4 群で 16、8 群で 64）。
群内 AUC だけを揃えても、群間 AUC が低ければ「ある群の陽性が別の群の陰性より低く並ぶ」不公平が残る。
原論文は、すべてのペアの AUC が等しい状態を公平と定める（式 3）。

全体の AUC は群ペアの AUC の重み付き和に分解できる（式 4）。重みは「陽性が群 g、陰性が群 h である確率」。

```
AUC = Σ_{g,h} P[Z = g | Y = 1] · P[Z' = h | Y' = 0] · AUC_{g,h}
```

普通に全体 AUC を上げる学習（原論文の AUCMax）は、重みの小さいペア（少数群が絡むペア）が低いままでも全体はあまり
下がらない。そこで、最も低いペアの AUC を上げる（式 5）。分解の形から、これは全体 AUC と大きくは衝突しないと主張する。

### 目的関数

AUC の指標関数は微分できないので、logistic 損失 ℓ(s) = log(1 + e^{−s}) に置き換える。R_{g,h}(θ) は、群 g の陽性と
群 h の陰性の全組について ℓ(f(x⁺) − f(x⁻)) を平均したもの（ペアの順位損失）。最大のペア損失を下げる問題を、
ペアの重み λ（G² 個の非負の値で合計 1）を使って書き換える（式 6・7）。

```
min_θ  max_{λ ∈ 単体}  Σ_{g,h} λ_{g,h} · R_{g,h}(θ)
```

### 学習の手順（原論文 Algorithm 1）

1. λ を `λ_{g,h} = n^{g+} · n^{h−} / (n⁺ · n⁻)` で始める（n^{g+} は群 g の陽性数、n⁺ は陽性の総数）。
   これは式 4 の重みで、この λ のままなら目的関数は全体 AUC の代理損失そのものになる。
2. 各 iteration で、群 × ラベルの層ごとに件数を全体の比率に比例させたミニバッチ B_t を引く（StratifiedSampler）。
3. θ を、1 つ前の λ で重み付けした損失の勾配で 1 歩動かす：`θ_t = θ_{t−1} − η_θ · λ_{t−1}ᵀ ∇R(θ_{t−1}; B_t)`。
4. λ を、同じバッチ・更新前の θ で測ったペア損失で指数的に更新して正規化する：
   `λ_t ∝ λ_{t−1} · exp(η_λ · R(θ_{t−1}; B_t))`。損失の大きいペアほど重みが増える。
5. 理論上の出力は、反復の中から一様に選んだ θ。

**Population AUC（この repo の対照）** は、手順 4 を省いて λ を初期値に固定したもの。式 4 により全体 AUC の代理損失を
最小化することになり、原論文の AUCMax に当たる。Minimax AUC との違いは λ を動かすかどうかだけになる。

## 2. 原論文の設定

- Yang, Ko, Varshney, Ying. *Minimax AUC Fairness: Efficient Algorithm with Provable Convergence.* AAAI 2023
  （[arXiv:2208.10451](https://arxiv.org/abs/2208.10451)）
- 公式実装：[zhenhuan-yang/MinimaxFairAUC](https://github.com/zhenhuan-yang/MinimaxFairAUC)（`src/rwm_models.py` の
  `RWMAUCLearner`、`src/config/*_RWMAllAUCLearner.json`）

### データセットとタスク

すべて表形式の 2 値分類で、群は 2 つ（ペアは 4）。ほかに 2 次元の合成データがある。

| データセット | 入力 | 予測ラベル（クラス数） | 群・敏感属性（群数） |
| --- | --- | --- | --- |
| Adult | 表形式（48,842 件、15 属性） | 年収が 5 万ドルを超えるか（2 クラス） | 性別（2） |
| Bank | 表形式（41,188 件、21 属性） | 定期預金に申し込むか（2 クラス） | 年齢：25 歳未満か 60 歳超か、それ以外か（2） |
| COMPAS | 表形式（11,757 件、53 属性） | 再犯するか（2 クラス） | race：Caucasian か否か（2） |
| Default | 表形式（30,000 件、24 属性） | 翌月に債務不履行になるか（2 クラス） | 性別（2） |

### ハイパーパラメータ

| 項目 | 値 |
| --- | --- |
| モデル | 2 隠れ層の MLP（幅は入力次元、ReLU）。出力のスコアに γ = 1・β = 0 固定の BatchNorm |
| 前処理 | カテゴリ属性は one-hot、数値属性は平均 0・分散 1 |
| 初期値 | 公平性手法はどれも AUCMax の出力から始める（付録 E） |
| optimizer | SGD |
| 探索（val で選ぶ） | batch ∈ {256, 512, …, 8192}（COMPAS は 8192 を除く）、η_θ ∈ {0.2, 0.1, 0.05, 0.01}、η_λ = κ · η_θ（κ ∈ {100, 10, 1, 0.1, 0.01}）、weight decay ∈ {0.1, 0.01, 0.001} |
| 分割・反復 | train / val / test = 60 / 20 / 20、25 回 |
| epoch | 本文に記載なし。公式の Adult の config は最大 1000 epoch |
| 公式の Adult の config | batch 8192、lr 0.1（50 iteration ごとに 0.1 倍）、momentum 0、weight decay 0.01。λ は 5 step ごとに、その間の損失の平均で更新し、ステップ幅は temperature 7 × その時点の lr |

公式の `RWMAUCLearner` もペアの損失は全組の −log σ(s⁺ − s⁻) の平均（`pAUCLoss`、`k=None`）で、本文と同じ。

### モデルの選び方

- 実験では、**val の最大ペア損失**で early stopping する（5 節）。ハイパーパラメータも val で選ぶ。
- 公式の `AbsClassifier.ckeck_n_save` は、val の最大ペア損失が保存時の値から tol（1e-5）を超えて**変わる**たびに
  モデルを保存し直し（悪化したときも保存する）、変化が tol 以下の epoch が 5 回（`num_tempers`）続くと止める。
  本文の「最大ペア損失が最小の epoch」とは一致しない。
- 評価は全体 AUC と、群ペア AUC の最小値 ÷ 最大値（min/max 比）。25 回の test の平均と標準偏差を報告する。

## 3. この repo の実装

### ファイルと設定

| 役割 | 場所 |
| --- | --- |
| 損失・q の更新 | [objectives/minimax_auc.py](../../projects/hypernet_e2e/objectives/minimax_auc.py) の `MinimaxAUCTaskLoss`（q を毎 step 更新）と、その子 class `PopulationAUCTaskLoss`（q を固定） |
| 共通の基底（セルの取り出し） | [objectives/conditional.py](../../projects/hypernet_e2e/objectives/conditional.py) の `ConditionalObjective` |
| strategy | [configs/training_strategy/minimax_auc.yaml](../../projects/hypernet_e2e/configs/training_strategy/minimax_auc.yaml)、[population_auc.yaml](../../projects/hypernet_e2e/configs/training_strategy/population_auc.yaml)（どちらも `dual_step_size: 0.01`、`pair_scope: all`） |
| q の更新の呼び出し・held-out | [objectives/conditional.py](../../projects/hypernet_e2e/objectives/conditional.py) の `StepDualUpdate`（`callbacks()` が返す callback）と、[module.py](../../projects/hypernet_e2e/module.py) の `LitModule.eval_logits`・`configure_callbacks` |
| 起動時の検証と件数の注入 | [training.py](../../projects/hypernet_e2e/training.py) の `_resolve_conditional_population` |
| セル均等抽出・held-out | [data/group_class_sampler.py](../../projects/hypernet_e2e/data/group_class_sampler.py) の `GroupClassBatchSampler`・`GroupClassImageDataModule.held_out_batches` |
| 群の定義 | data config（例 [ham10000_sex_age_cells_dual_holdout.yaml](../../projects/hypernet_e2e/configs/data/ham10000_sex_age_cells_dual_holdout.yaml)：性別 × 年齢の 4 群、[chexpert_age_race_known_cells_dual_holdout.yaml](../../projects/hypernet_e2e/configs/data/chexpert_age_race_known_cells_dual_holdout.yaml)：年齢 × race の 8 群） |
| 記録・候補の選択 | [callbacks/objective_diagnostics.py](../../projects/hypernet_e2e/callbacks/objective_diagnostics.py) の `ObjectiveDiagnosticsRecorder`、[callbacks/conditional_fairness.py](../../projects/hypernet_e2e/callbacks/conditional_fairness.py) の `ConditionalFairnessRecorder`・`select_candidate` |
| 代表的な preset | `resnet_ham10000_sex_age_minimax_auc_dual_holdout`（HAM10000 4 群、held-out で q）、`resnet_chexpert_from_resnet_age_minimax_auc_race_known` と `resnet_chexpert_from_resnet_age_population_auc_race_known`（CheXpert 2 群、同じ seed の ERM から継続）、`resnet_ham10000_fg_popauc_sex_age` |

### 処理の流れ

**起動時（fit の前）。**

- `_resolve_conditional_population`：single device・勾配蓄積なし・bf16 か fp32 でなければ止める。fp16 の GradScaler は
  optimizer step を飛ばすことがあり、q だけが進んでしまうため。DataModule が `GroupClassImageDataModule` でなければ止める。
  `train.csv` から群 × クラスの件数 [G, 2] を数えて `population_cell_counts` に書く。
- クラス重みは使わない（preset はどれも `weighting: none`。`class_weight` が入ると止まる）。
- `MinimaxAUCTaskLoss.__init__`：ペアの一覧 `pairs = [(g, h)]`（g が陽性の群、h が陰性の群）を作る。`pair_scope: all` で
  G² 個、`intra` で群内の G 個。q の初期値 n_{g+} · n_{h−} / (n⁺ · n⁻) を正規化して、対数で buffer `log_q` に持つ
  （公式の `groupgroup_ratios` と同じ値）。
- `StepDualUpdate.on_train_start`：DataModule が held-out を持てば、held-out からセル均等のバッチを流し続ける iterator を作る。
  `PopulationAUCTaskLoss` は q を更新しない（`uses_held_out = False`）ので、held-out を設定すると `LitModule.on_train_start` が止める。

**学習の各ステップ。**

1. `GroupClassBatchSampler`：1 バッチに各セルから B/(2G) 件を復元抽出する。B = 128 なら G = 4 で 16 件（ペアあたり 256 組）、
   G = 8 で 8 件（ペアあたり 64 組）。
2. `MinimaxAUCTaskLoss.forward`：スコアを s = z₁ − z₀ とし、各ペア (g, h) についてバッチ内の群 g の陽性と群 h の陰性を
   全部組にして `softplus(−(s⁺ − s⁻))` の平均を取る。これがペアの損失 R_{g,h}。更新前の q（勾配なし）との内積を返す。
   R は `_pending` に保留する。
3. `StepDualUpdate.on_train_batch_end`：モデルの更新の後に呼ばれる。held-out があれば held-out のバッチを 1 つ、
   `LitModule.eval_logits`（eval モード・更新後のモデル）で forward し、保留中の R を held-out の値に置き換える。
   LitModule はこの callback を Trainer の callback より先に並べるので、診断を記録する callback は確定の後の q を読む。
4. `commit_update`：`log_q ← log_softmax(log_q + dual_step_size · R)`。q · exp(η · R) を正規化するのと同じ計算で、
   対数で持つのは損失が大きいときに exp が溢れないようにするため。`PopulationAUCTaskLoss` は R を捨てるだけ。

held-out が無ければ、モデルの更新に古い q を使い、q の更新に同じバッチ・更新前のモデルの損失を使う。これは
Algorithm 1 の手順 3・4 と同じ順序になる。**held-out を使える理由**は、学習バッチで測るとモデルが少数セルを記憶して
ペアの損失の差が消え、q が初期値に戻るため。held-out の forward を eval モードにするのは、BatchNorm の統計を
held-out で動かさないため。held-out は train の 20% を分けた split（`*_dual_holdout` の data config）。

**記録と checkpoint。**

- `metrics/objective_diagnostics.jsonl`：`risk`・`q_before`・`q_after`・`q_entropy`・`q_max`・`cell_count`・`pair_count`。
  最後のバッチの値と `epoch_mean`、held-out 版では q の更新に実際に使った held-out の値の平均 `held_out_epoch_mean`。
  ペアの番号は `pair_scope: all` のとき g · G + h。
- dual-holdout preset は全 epoch の重みを残す。epoch は val で選ぶ（held-out は q の更新に使ったため）。
- HAM10000 の `fg` preset は、best val AUROC、val の性別 × 年齢の最悪群 AUROC の best、last だけを残す。
  gmm8 preset でも最悪群は性別 × 年齢で測るので、学習に使った群の最悪ペアでは選べない。
- CheXpert 年齢 2 群の preset は `ConditionalFairnessRecorder` が `metrics.csv` に `train/conditional/{名前}/{番号}`（epoch 平均）を、
  毎 epoch `checkpoints/candidate_XXX.ckpt` と `metrics/fairness_candidates.json` を残す。継続学習 preset は fit の終わりに
  `select_candidate` で選ぶ：AUROC と最悪年齢 AUROC が親 ERM から 0.005 以内の epoch のうち、val の群ペア AUROC の
  最小値 `min_pair_auroc` が最大のもの（同点なら最悪年齢 AUROC、AUROC の順）。`population_auc` も同じ規則で選ぶ。
  1 stage preset（`reference: deferred`）は選ばず、分析側で選ぶ。

### 原論文との違い

| 項目 | 原論文 | この repo |
| --- | --- | --- |
| データ・群 | 表形式、2 群（4 ペア） | 画像（CheXpert・HAM10000）、2・4・8 群（4・16・64 ペア） |
| q のステップ幅 | κ · η_θ を val で選ぶ。公式は temperature × 減衰する lr で、5 step の平均損失で更新 | 0.01 固定、毎 step、その step の損失で更新 |
| 抽出 | 層の件数を全体の比率に比例させる、非復元 | セル均等、復元。ペアの損失はセル内の平均なので期待値は同じで、分散が違う |
| q を測るデータ | 同じバッチ・更新前のモデル | held-out なし：同じ。held-out 版：held-out の別バッチ・更新後のモデル・eval モード |
| 初期値 | AUCMax の出力 | 1 stage は ImageNet 重み。継続学習 preset は CE で学習した同じ seed の ERM |
| モデル・optimizer | MLP、SGD | ResNet-50 か Spatial LoRA ResNet-50、AdamW lr 1e-4・weight decay 0.01、batch 128、30 epoch |
| モデルの選び方 | val の最大ペア損失で early stopping | 上の「記録と checkpoint」のとおり |
| 評価 | 全体 AUC と min/max 比 | 群別 AUROC と、固定しきい値の群別 TPR・FPR（[research-premises.md](../research-premises.md)） |

原論文に近づけるなら、`model.loss_fn.dual_step_size` を lr の κ 倍として振り、ERM から継続する preset を使い、
val の `min_pair_auroc` が最大の epoch を選ぶ。PopAUC は全体 AUC の代理なので、AUCMax に合わせるなら val の全体 AUROC で選ぶ。

### 起動例

```bash
uv run python -m projects.hypernet_e2e.run experiment=resnet_ham10000_sex_age_minimax_auc_dual_holdout study=<slug> seed=42
```

```bash
# q のステップ幅を変える／群内ペアだけにする
uv run python -m projects.hypernet_e2e.run experiment=resnet_ham10000_fg_minimaxauc_sex_age study=<slug> seed=42 \
  model.loss_fn.dual_step_size=0.1 model.loss_fn.pair_scope=intra
# PopAUC の対照（同じデータ・抽出・seed。pair_scope を変えたなら対照も揃える）
uv run python -m projects.hypernet_e2e.run experiment=resnet_ham10000_fg_popauc_sex_age study=<slug> seed=42 \
  model.loss_fn.pair_scope=intra
```

継続学習 preset（`*_from_resnet_*`）は `model.backbone_checkpoint_path=<同じ seed の ERM の checkpoint>` が必須。
