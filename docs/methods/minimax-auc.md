# Minimax AUC と Population AUC（Yang et al.）：原論文・実装・使い方

2026-10-16 作成。原論文の本文・付録と公式実装を読み直し、この repo の実装（commit `0f2af99` 時点）と突き合わせた結果をまとめる。
Minimax AUC と、その対照の Population AUC（PopAUC）の run を設計・起動・評価するセッションは、先にこれを読む。

## 先に押さえること

- **群ペアの AUC** を単位にする。AUC_{g,h} は「群 g の陽性が、群 h の陰性より高いスコアになる確率」。g = h なら群内 AUC、
  g ≠ h なら群間 AUC。G 群なら G² 個のペアがある（4 群で 16、8 群で 64）。
- **Minimax AUC は、その最小値を上げる**。各ペアの順位損失（陽性と陰性のスコア差の logistic 損失）に重み q を掛けて
  学習し、q は損失の大きいペアへ寄せる。q はペアについての確率分布（全成分 0 以上、合計 1）。原論文では λ と書く。
- **q の更新は指数勾配**：q ← q·exp(η × ペアの損失) を合計 1 に正規化する。損失の大きいペアほど重みが増える。
  毎 step、モデルを 1 step 更新した後に、その step の forward で測った損失で更新する（原論文 Algorithm 1 l.6–7 と同じ順序）。
- **PopAUC は q を初期値で固定した対照**。初期値は「全体 AUC を群ペアに分解したときの重み」n_{g+}·n_{h−}/(n⁺·n⁻)
  （n_{g+} は群 g の陽性数）なので、PopAUC は全体 AUC の代理損失そのもの（原論文の AUCMax、式 4）。
  損失の形とセル均等抽出は Minimax AUC と同じで、違いは q を動かすかどうかだけ。
- 実装は原論文と公式実装に合っている。違いは q のステップ幅（0.01 固定）、最終モデルの選び方、optimizer、初期値、
  held-out 版での q の測り方（下の表）。

## 原論文と公式実装

- Yang, Ko, Varshney, Ying. *Minimax AUC Fairness: Efficient Algorithm with Provable Convergence.* AAAI 2023
  （[arXiv:2208.10451](https://arxiv.org/abs/2208.10451)）
- 公式実装：[zhenhuan-yang/MinimaxFairAUC](https://github.com/zhenhuan-yang/MinimaxFairAUC)（`src/rwm_models.py` の
  `RWMAUCLearner`、`src/base_models.py` の `AUCLearner`、`src/config/*_const_RWMAllAUCLearner.json`）

### 原論文の条件（本文・付録と公式実装で確認済み）

| 項目 | 原論文・公式実装 | 出典 |
| --- | --- | --- |
| 目的 | 群ペア AUC の最小値の最大化（式 5）。logistic 代理損失 ℓ(s) = log(1+e^{−s}) で、min_θ max_{λ∈単体} Σ λ_{z,z'} R_{z,z'}(θ)（式 6・7） | 3 節 |
| 全体 AUC との関係 | 全体 AUC = Σ P[Z=z, Z'=z'] AUC_{z,z'}（式 4）。群ペアの最小値を上げても全体 AUC と矛盾しない、という主張の根拠 | 3 節 |
| アルゴリズム | λ_0 = n^{z+} n^{z'−}/(n⁺n⁻)。各 step で θ を λ_{t−1} で 1 step 更新し、λ ← λ_{t−1}·exp(η_λ R(θ_{t−1}; B_t)) を正規化 | Algorithm 1 |
| 抽出 | StratifiedSampler：群 × ラベルのセルごとに、**全体の比率に比例した件数**（⌈m·n_zy/n⌉）を非復元で引く | Algorithm 1 l.4、付録 B の Algorithm 2 |
| 出力するモデル | Algorithm 1 は反復の一様ランダム。実験は val の最大群損失による early stopping | Algorithm 1 l.9、5 節 |
| 初期値 | 公平性手法はすべて AUCMax の出力から始める | 付録 E |
| データと群 | Adult（性別）、Bank（年齢 <25 か >60）、COMPAS（race）、Default（性別）。どれも 2 群、4 ペア。ほかに 2 次元の合成データ | 5 節 表 1 |
| モデル | 2 隠れ層の MLP（幅 = 入力次元、ReLU、出力に固定の BatchNorm） | 付録 E |
| optimizer と探索 | SGD。batch ∈ {256,…,8192}、η_θ ∈ {0.2, 0.1, 0.05, 0.01}、η_λ = κ·η_θ（κ ∈ {100, 10, 1, 0.1, 0.01}）、weight decay ∈ {0.1, 0.01, 0.001} を val で選ぶ。60/20/20 分割、25 回 | 付録 E |
| 公式の設定（Adult） | batch 8192、lr 0.1（50 iteration ごとに 0.1 倍）、momentum 0、wd 0.01、最大 1000 epoch。λ は 5 step ごとに、その間の損失の平均で更新し、ステップ幅は temperature 7 × その時点の lr。val の最大ペア損失で patience 5 の early stopping | `config/adult_sex_mlp_const_RWMAllAUCLearner.json`、`rwm_models.py` |
| 評価 | 全体 AUC と、群ペア AUC の min/max 比（最小 ÷ 最大） | 表 2 |
| AUCMax（対照） | 群を区別せず全体の AUC 代理損失を最小化。公式は層別バッチの陽性・陰性を全部組にした損失で、val の全体損失で early stopping | 5 節、`base_models.py` |

## この repo の実装

`projects/hypernet_e2e/objectives/minimax_auc.py`。

| class | strategy（`configs/training_strategy/`） | q |
| --- | --- | --- |
| `MinimaxAUCTaskLoss`（`:12`） | `minimax_auc` | 毎 step 指数勾配で更新（`dual_update = "step"`） |
| `PopulationAUCTaskLoss`（`:94`） | `population_auc` | 初期値で固定（`dual_update = None`） |

- **ペア**：`pairs = [(g, h)]`（g が陽性の群、h が陰性の群）。`pair_scope: all` で G² 個、`intra` で群内の G 個。
  診断値の番号は `g·G + h`（`all` のとき）。
- **初期値**：`log_q` に n_{g+}·n_{h−}/(n⁺·n⁻) を入れる（`:49–52`）。件数は `training.py` の `_resolve_conditional_population` が
  train.csv から数えて注入する。公式の `groupgroup_ratios` と同じ。
- **損失**：`forward`（`:54`）。スコア s = z₁ − z₀ について、バッチ内の群 g の陽性と群 h の陰性を全部組にし、
  `softplus(−(s⁺ − s⁻))` を平均したものがペアの損失。これに q（更新前の値、勾配なし）を掛けて足す。
  公式の `pAUCLoss`（−log σ(s⁺ − s⁻)）と同じ。クラス重みは使わない（渡すと止める）。
- **q の更新**：`module.py` の `optimizer_step`（`:245`）が、モデルの step の後に `commit_update`（`minimax_auc.py:76`）を呼ぶ。
  `log_q ← log_softmax(log_q + 0.01 × 損失)`。損失は同じ step の forward（更新前の θ）で測ったもの。
- **held-out 版**：DataModule が `holdout_split` を持つと（`*_dual_holdout` preset）、モデルの step の後に held-out から
  セル均等のバッチを 1 つ取り、eval モードで forward した損失で q を更新する（`_measure_on_held_out`、`module.py:308`）。
  PopAUC に held-out を設定すると起動時に止める（q を更新しないので）。
- **抽出**：`GroupClassImageDataModule` 必須。1 バッチに各セル B/(2G) 件を復元抽出する（B=128、G=8 なら 8 件、ペアあたり 64 組）。
- **制約**：single device、勾配蓄積なし、bf16 か fp32（fp16 は optimizer step が飛び、q だけ進むため）。途中再開は未対応。

診断値（バッチごとに `risk`・`q_before`・`q_after`・`q_entropy`・`q_max`・`cell_count`・`pair_count`）：

- CheXpert 年齢 2 群 preset：`metrics.csv` の `train/conditional/{名前}/{番号}`（epoch 平均）。
- `metrics/objective_diagnostics.jsonl`（2026-10-17 から全 preset。それより前は HAM10000 と dual-holdout preset のみ）。最後のバッチの値、`epoch_mean`、
  held-out 版では q の更新に実際に使った held-out の損失の平均 `held_out_epoch_mean`。

### 原論文との違い

| 項目 | 原論文・公式実装 | この repo の preset |
| --- | --- | --- |
| q のステップ幅 | κ·η_θ を val で選ぶ。公式は temperature × 減衰する lr、5 step の損失平均で更新 | 0.01 固定、毎 step、その step の損失 |
| 抽出 | セルを全体の比率に比例して引く | セル均等。どちらもペアの損失の推定は偏らない（分散が違う） |
| q を測るデータ | 同じ train バッチ、更新前の θ | held-out なし：同じ。held-out 版：held-out の別バッチ、更新後の θ、eval モード |
| 初期値 | AUCMax の出力 | 1 stage は ImageNet 重み、継続学習 preset は CE で学習した同じ seed の ERM |
| モデル・optimizer | MLP、SGD | ResNet-50 か Spatial LoRA、AdamW lr 1e-4・wd 0.01、batch 128、30 epoch |
| 最終モデル | 反復の一様ランダム（理論）／val の最大ペア損失で early stopping（実験） | 下の「checkpoint の選び方」 |
| 評価 | 全体 AUC と min/max 比 | 群別 AUROC と、固定しきい値の群別 TPR・FPR |

## 起動

| preset | データと群 | 学習 |
| --- | --- | --- |
| `resnet_chexpert_age_minimax_auc_race_known`、`spatial_lora_chexpert_stage1_4_age_minimax_auc_race_known` | CheXpert、年齢 2 群（4 ペア） | ImageNet から 1 stage |
| `resnet_chexpert_from_resnet_age_{minimax,population}_auc_race_known` | CheXpert、年齢 2 群 | 同じ seed の ERM から継続（`model.backbone_checkpoint_path` 必須） |
| `spatial_lora_chexpert_age_race_minimax_auc_dual_holdout` | CheXpert、年齢 × race 8 群（64 ペア）、held-out で q | ImageNet から 1 stage、全 epoch 保存 |
| `{resnet,spatial_lora}_ham10000_sex_age_minimax_auc_dual_holdout` | HAM10000、性別 × 年齢 4 群（16 ペア）、held-out で q | ImageNet から 1 stage、全 epoch 保存 |
| `resnet_ham10000_fg_{minimaxauc,popauc}_{sex_age,gmm8}` | HAM10000、性別 × 年齢 4 群か GMM 8 クラスタ | ImageNet から 1 stage |

```bash
uv run python -m projects.hypernet_e2e.run experiment=resnet_ham10000_sex_age_minimax_auc_dual_holdout seed=<seed> study=<slug>
# q のステップ幅を変える／群内ペアだけにする
uv run python -m projects.hypernet_e2e.run experiment=<preset> seed=<seed> study=<slug> \
  model.loss_fn.dual_step_size=0.1 model.loss_fn.pair_scope=intra
```

PopAUC を Minimax AUC の対照にするときは、同じデータ・抽出・seed でそろえる（preset は組で用意してある）。

## checkpoint の選び方と予測の使い方

- **原論文に近い選び方**：val の群ペア AUROC の最小値（`min_pair_auroc`）が最大の epoch。原論文の目的そのもの。
  公式実装は val の最大ペア代理損失で early stopping しており、どちらも「最悪ペア」で選ぶ点は同じ。
- **CheXpert 年齢 2 群 preset**：`ConditionalFairnessRecorder` が毎 epoch `checkpoints/candidate_XXX.ckpt` と
  `metrics/fairness_candidates.json` を残す。継続学習 preset は fit 末に `select_candidate`
  （`callbacks/conditional_fairness.py:33`）で選ぶ：AUROC と最悪年齢 AUROC が親 ERM から 0.005 以内の epoch のうち、
  `min_pair_auroc` が最大のもの（同点なら最悪年齢 AUROC、AUROC の順）。1 stage preset は分析側で選ぶ（`reference: deferred`）。
- **dual-holdout preset**：全 epoch の重みが残る。val で選ぶ（held-out は q の更新に使ったので選択には使わない）。
- **HAM10000 の `fg` preset**：best val AUROC、best val 性別 × 年齢の最悪群 AUROC、last の 3 つだけ。gmm8 でも最悪群は
  性別 × 年齢で測るので、学習に使った群の最悪ペアでは選べない。
- PopAUC は全体 AUC の代理なので、原論文の AUCMax に合わせるなら val の全体 AUROC（公式は val の全体損失）で選ぶ。
- 予測は `analysis/common/predictions.py`。EO の規則など best val AUROC 以外で選んだ epoch は `--checkpoint` は名前で渡す（既定 `best` は run.json に記録された val AUROC の best、`xxx` は `checkpoints/xxx.ckpt`。例 `best_val_auroc_009`、`last`、`swad`）。
- q の推移は、`q_after` が val で最悪のペアへ寄ったかで確かめる。held-out 版は `held_out_epoch_mean` の `risk` と並べて読む。

## 2026-10-16 の照合結果

原論文（本文・付録）と公式実装を読み、この repo の実装と照合した。単体テスト（`tests/test_conditional_objectives.py`
の FairDRO・minimax 関連 23 件）は HEAD で通る。

**合っている点（確認済み）**

- ペアの順位損失（陽性群 g × 陰性群 h の全組の logistic 損失）、q の初期値 n_{g+}·n_{h−}/(n⁺n⁻)、
  指数勾配 + 正規化、更新の順序（モデルの step の後に、更新前の θ で測った同じ forward の損失で q を更新）。
- PopAUC は q を初期値に固定しており、式 4 の分解どおり全体 AUC の代理損失になっている。
- 抽出は原論文の StratifiedSampler（比例）ではなくセル均等だが、セル内平均で損失を測るので推定は偏らない。

**直した点**：なし（commit `0f2af99` は Minimax AUC・PopAUC に触れていない）。

**残している違い**：上の「原論文との違い」の表のとおり。

**既存の run（`run_inventory.csv`、2026-10-16 時点）**：q の更新の中身は、どの run の時点からも変わっていない。

- Minimax AUC 16 本：CheXpert 年齢 2 群の 1 stage 2 本（2026-10-07）、HAM10000 `fg` 6 本（2026-10-09、
  `archive/latent_bias_hypernet`）と起動時に失敗した 2 本、HAM10000 dual-holdout 6 本（2026-10-12、ResNet・Spatial LoRA 各 3 seed）。
- PopAUC 6 本：HAM10000 `fg`（性別 × 年齢・gmm8 各 3 seed、2026-10-10）。
- 2026-10-13 より前の held-out 版の run では、`objective_diagnostics.jsonl` が最後のバッチの値だけを持ち、それも held-out の値で
  上書きされている（`docs/reviews/20261013_fairness_objective_fixes.md`）。`q_after` は held-out の forward でも同じ q を
  指すので読めるはず（コードからの推測で、run では確かめていない）。`risk` は held-out の値として読む。
