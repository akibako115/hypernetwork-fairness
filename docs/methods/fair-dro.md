# FairDRO（Jung et al.）：原論文・実装・使い方

2026-10-16 作成。原論文の本文・付録と公式実装を読み直し、この repo の実装（commit `0f2af99` 時点）と突き合わせた結果をまとめる。
FairDRO の run を設計・起動・評価するセッションは、先にこれを読む。

## 先に押さえること

- **FairDRO は「正解クラスごとの GroupDRO」**。クラス y ごとに群の重み q^y（群数 G 個の重みで、合計 1）を持ち、
  群 × クラスの各セルの CE を q で重み付けして学習する。q は epoch 末に「誤り率の高い群を重く、低い群を軽く」する向きへ
  閉形式で決める。狙いは EO（二値分類では、群間で TPR と TNR をそろえること）。
- **q の動ける範囲が χ² 球**。一様分布 1/G からの χ² ダイバージェンスが半径 ρ 以下の重みの集合で、**負の重みも許す**。
  ρ が大きいほど q は一様から離れられる。
- **ρ は公式実装の単位で指定する**（2026-10-16 から）。q* = 1/G + √(2ρ/G)·(e − ē)/‖e − ē‖（e は群別の 0-1 誤り率、ē はその平均）。
  論文の式 8 は √(ρ/G) と書いており、論文の ρ は公式の 2 倍にあたる。preset の既定は 0.5（公式の既定値）。
- **G=2 では q* は誤り率の差の符号だけで決まる**：q* = 1/2 ± √(ρ/2)。ρ=0.5 で (1, 0)、それより大きいと片方が負になる。
  q* の全成分が 0 以上に収まる ρ の上限は 1/(2(G−1))（G=2 で 0.5、G=4 で 0.167、G=8 で 0.071）。
  手法の性質で、実装の誤りではない。公式 README の値（2 群の Adult 5、COMPAS 3、CelebA 1.5）もこの上限を大きく超える。
- **preset の 0.5 は、2026-10-16 より前の「ρ=1」と同じ重みを出す**（単位を変えただけ）。2026-10-16 の meeting 資料で
  「使わない」とした ρ=1 の FairDRO run と同じ設定なので、preset のまま回すと同じ振る舞いになる。回す前に ρ を決める。
- checkpoint は EO の規則（val の TPR・TNR の群間差）で選ぶ。best val AUROC ではない。原論文は最終 epoch を報告している。

## 原論文と公式実装

- Jung, Park, Chun, Moon. *Re-weighting Based Group Fairness Regularization via Classwise Robust Optimization.*
  ICLR 2023（[arXiv:2303.00442](https://arxiv.org/abs/2303.00442)）
- 公式実装：[sangwon-jung94/FairDRO](https://github.com/sangwon-jung94/FairDRO)（`trainer/fairdro.py`、`arguments.py`、
  `data_handler/dataloader_factory.py`）

### 原論文の条件（本文・付録と公式実装で確認済み）

| 項目 | 原論文・公式実装 | 出典 |
| --- | --- | --- |
| 目的 | クラスごとに χ² 球上の最悪の q^y を取り、その重み付き群損失のクラス平均を最小化（式 6）。0-1 損失なら「群・クラス均等の誤り + √ρ × DCA」の最小化と同じ（系 1、式 7） | 4.3 節 |
| 公平性の指標 | DCA：クラスごとの群間 accuracy 差の最大値を、クラスで平均（式 1）。二値分類では EO と同じ | 3 節 |
| q の閉形式 | 式 8（論文の単位）。公式 `_update_mw_margin` は `1/G + sqrt(2ρ/G)·dev/‖dev‖` | 命題 2、`trainer/fairdro.py` |
| q の更新（平滑化 IBR） | IBR（iterated best response）は、毎回 q を内側の最大化の厳密解 q* に置き換える方法。これを η_t = 1 − t/T で混ぜる：q ← (1−η_t) q + η_t q*（式 10）。**epoch 末に 1 回**、train 全件の群別損失で | 4.4 節、Algorithm 1 l.10 |
| q を決める損失 | 論文は 0-1、θ の更新は CE。公式の既定は CE（`--use-01loss` を付けると 0-1） | 4.4 節、`arguments.py` |
| 誤りの測り方（公式） | train 全件を `model.train()` のまま、学習用 transform（RandomCrop・Flip）で推論 | `trainer/fairdro.py`、`trainer_factory.py` |
| 抽出 | Alg. 1 はセル均等。公式 README のコマンドは `--balSampling` なし（自然抽出）。損失はバッチ内のセル平均 | Algorithm 1 l.6、README |
| データと群 | Adult（性別 2 群）、COMPAS（race 2 群）、UTKFace（ethnicity 4 群 × 年齢 3 クラス）、CelebA（性別 × Blond_Hair）、CivilComments（ethnicity 5 群） | 5 節、付録 D.2 |
| モデル | 表形式はロジスティック回帰、画像は ResNet18（README に `--pretrained` なし＝ランダム初期化）、言語は BERT | 5 節、README |
| optimizer | AdamW、lr 1e-3、cosine、70 epoch、batch 128。weight decay は付録 1e-3、README 1e-4。BERT は 3 epoch、q は 100 iteration ごと | 付録 C.1、5.3 節 |
| ρ | grid [1e-2, 1e2] を振ってパレート曲線を描く。README の値：Adult 5、COMPAS 3、UTKFace 0.3、CelebA 1.5、CivilComments 0.3（公式の単位） | 付録 C.3 表 C.1、README |
| checkpoint | **最終 epoch** | 付録 C.1 |
| 評価 | 群 × クラスの balanced accuracy と DCA。単一値は「Scratch の accuracy の 95% 以上で DCA 最良」、4 seed | 5 節、付録 C.4・表 D.1 |

公式 README のコマンドは `--method lgdro_chi`・`--optim-q ibr_ip`・`--trueloss` など、公開版の `arguments.py` に無い引数を使っており、
そのままでは動かない。公開コードで論文の手順に当たるのは `--method fairdro --optim-q smt_ibr --use-01loss`。
論文の数値をどの設定で出したかは、公開コードからは確定できない。

## この repo の実装

`projects/hypernet_e2e/objectives/fair_dro.py` の `FairDROTaskLoss`（`training_strategy=fair_dro`）。

- **閉形式**：`chi_square_best_response`（`fair_dro.py:14`）。公式 `_update_mw_margin` と同じ式。誤り率が全群で等しいときは一様。
- **上限の警告**：`nonnegative_radius_limit`（`:38`）が 1/(2(G−1)) を返し、ρ がそれ以上なら起動時に警告する（禁止はしない）。
  現在の preset はすべてこの警告が出る。
- **学習の損失**：`forward`（`:103`）。セルごとの CE 平均に q を掛け、クラスで平均する（`(q * risks).sum() / 2`、式 9）。
  クラス重みは使わない（`class_weight` を渡すと止める）。
- **q の更新**：`module.py` の `on_train_epoch_end`（`:323`）が、epoch 末に全件の群 × クラス別 0-1 誤り率を測り、
  `update_from_errors`（`fair_dro.py:136`）に渡す。η_t = 1 − t/T の T は `trainer.max_epochs`。最初の epoch は一様な q で学習する。
- **誤り率の測り方**：`_split_errors`（`module.py:358`）。eval モード・評価 transform・sampler なしで 1 周する。
  DataModule が held-out split を持てば held-out 全件、無ければ train 全件（`held_out_evaluation_dataloader`、
  `evaluation_dataloader("train")`）。どちらを使ったかは `fair_dro_q.jsonl` の `source` に残る。
- **抽出**：`GroupClassImageDataModule` 必須（`training.py` の `_resolve_conditional_population` が確認）。
  1 バッチに各セル B/(2G) 件を復元抽出する（B=128 なら G=2 で 32 件、G=8 で 8 件）。
- **制約**：single device、勾配蓄積なし、bf16 か fp32。途中再開は未対応（`module.py:216`）。

診断値：

| 場所 | 中身 |
| --- | --- |
| `metrics.csv` | `train/fair_dro/q/y{c}_g{g}`（更新後の q）、`train/fair_dro/error/g{g}_y{c}`（誤り率）、`train/fair_dro/q_min` |
| `metrics/fair_dro_q.jsonl` | 1 行 = 1 epoch。`eta`・`q_star`・`q`・`error`・`count`・`source`・`q_min` |
| `metrics/objective_diagnostics.jsonl` | HAM10000 preset のみ。バッチ内のセル別 CE（`risk`）と予測の陽性率（`rates`） |

### 原論文との違い

| 項目 | 原論文・公式実装 | この repo の preset |
| --- | --- | --- |
| 誤りを測るデータ | train 全件 | held-out があれば held-out 全件（preset には無い）、無ければ train 全件 |
| 測るときのモデル | 公式は train モード・学習 transform | eval モード・評価 transform |
| 抽出 | Alg. 1 はセル均等、公式 README は自然抽出 | セル均等 |
| 初期値 | ランダム初期化 | 1 stage は ImageNet 重み、継続学習 preset は同じ seed の ERM |
| optimizer | AdamW lr 1e-3、cosine、70 epoch | AdamW lr 1e-4、wd 0.01、schedule なし、30 epoch |
| ρ | grid でパレート曲線 | 0.5 の 1 点 |
| checkpoint | 最終 epoch | EO の規則（下） |
| 評価 | balanced accuracy と DCA | 固定しきい値の群別 TPR・FPR（research-premises） |

## 起動

FairDRO を使う preset（`configs/experiment/`）：

| preset | データと群 | 学習 |
| --- | --- | --- |
| `resnet_chexpert_age_fair_dro_race_known`、`spatial_lora_chexpert_stage1_4_age_fair_dro_race_known` | CheXpert、年齢 2 群 | ImageNet から 1 stage |
| `resnet_chexpert_from_resnet_age_fair_dro_race_known` | CheXpert、年齢 2 群 | 同じ seed の ERM から継続（`model.backbone_checkpoint_path` 必須） |
| `resnet_ham10000_fg_fairdro_sex_age` | HAM10000、性別 × 年齢 4 群 | ImageNet から 1 stage |
| `resnet_ham10000_fg_fairdro_gmm8` | HAM10000、DINOv2 特徴の GMM 8 クラスタ | ImageNet から 1 stage |

```bash
# ρ を負の重みが出ない範囲に下げ、q を held-out 全件で測り、全 epoch の重みを残す例（HAM10000 4 群）
uv run python -m projects.hypernet_e2e.run experiment=resnet_ham10000_fg_fairdro_sex_age seed=<seed> study=<slug> \
  model.loss_fn.radius=0.1 data=ham10000_sex_age_cells_dual_holdout \
  callbacks.model_checkpoint.save_top_k=-1 callbacks.model_checkpoint.save_weights_only=true
```

- `model.loss_fn.radius` は公式の単位。公式 README の値はそのまま使える。論文の図の ρ を使うときは 1/2 にする。
- `trainer.max_epochs` を変えると、η_t の T も一緒に変わる。

## checkpoint の選び方と予測の使い方

- **CheXpert の年齢 2 群 preset**：`ConditionalFairnessRecorder` が毎 epoch `checkpoints/candidate_XXX.ckpt` と
  `metrics/fairness_candidates.json`（val の群別 TPR・TNR・AUROC）を残す。継続学習 preset は fit 末に
  `select_candidate`（`callbacks/conditional_fairness.py:33`）で選び、`metrics/fairness_selection.json` に書く。
  規則は EO rate と共通：val の TPR 差・TNR 差がともに 0.1 以下、どの群の TPR・TNR も親 ERM から 0.02 以上下がらず、
  AUROC と最悪年齢 AUROC が親から 0.005 以内。満たす epoch のうち val の重み付き CE が最小のもの。
  満たす epoch が無ければ `infeasible`。原論文に合わせた最終 epoch も `last_epoch` に併記する。
- 1 stage preset は `reference: deferred` で、選択は分析側（同じ seed・同じ構造の ERM の val を親の代わりにする）。
- **HAM10000 preset**：best val AUROC、best val 性別 × 年齢の最悪群 AUROC、last の 3 つだけが残る。gmm8 preset でも
  最悪群は性別 × 年齢で測る（学習に使った GMM 群ではない）。EO の規則で選ぶなら、上の例のように全 epoch を残して回す。
- held-out で q を更新した run は、選択には val を使う（held-out は q の更新に使ったので、選択にも使うと二重に使う）。
- 予測は `analysis/common/predictions.py`。全 epoch を残した run では `--checkpoint` を必ず渡す。
- q の推移は `fair_dro_q.jsonl` で見る。q が誤り率の高い群へ向いたか、負の重みがどれだけ続いたか（`q_min`）を確かめる。

## 2026-10-16 の照合結果

原論文（本文・付録）と公式実装を読み、この repo の実装と照合した。単体テスト（`tests/test_conditional_objectives.py`
の FairDRO・minimax 関連 23 件）は HEAD で通る。

**合っている点（確認済み）**

- 閉形式（命題 2）、クラスごとの q、q を 0-1 誤りで決めて θ を CE で更新する分担（4.4 節）、
  epoch 末の平滑化 IBR（η_t = 1 − t/T）、q の初期値 1/G、損失のクラス平均。

**直した点（commit `0f2af99`）**

- ρ の単位を論文から公式実装へ変えた（`sqrt(ρ/G)` → `sqrt(2ρ/G)`）。公式 README の ρ をそのまま使えるようにするため。
- 非負の上限を 1/(G−1) から 1/(2(G−1)) に、preset の既定を 1.0 から 0.5 にした。どちらも単位の変更で、q の値は変わらない。

**残している違い**：上の「原論文との違い」の表のとおり。誤りを eval モードで測る点、held-out を使える点、
継続学習、AdamW の設定、EO の規則での選択。

**既存の run（`run_inventory.csv`、2026-10-16 時点で 12 本）**：どれも `0f2af99` より前で、ρ は旧単位で記録されている
（公式の単位では記録値の 1/2）。q の更新の中身は単位以外変わっていない。

- CheXpert 年齢 2 群（2026-10-07、`archive/conditional_fairness_one_stage`）：旧 ρ=1.0 の 2 本は完了記録が無い（stale_running）。
  `fair_dro.py` の docstring によれば、各クラスで片方の群の重みが 0 になり、一方の群だけを学ぶ解へ epoch ごとに振れた
  （上の G=2 の性質による。run の中身はこの照合では見ていない）。旧 ρ=0.1（公式 0.05）の 2 本は完走。
- HAM10000（2026-10-09、`archive/latent_bias_hypernet`）：旧 ρ=1.0（公式 0.5、現在の preset と同じ重み）で 6 本完走、
  2 本は起動時に失敗。完走した 6 本では、q* が 30 epoch 中 27〜30 epoch で負の成分を持ち、平滑化後の q の最小値は
  −0.07〜−0.16 だった（`fair_dro_q.jsonl` で確認）。4 群・8 群の上限（0.167・0.071）を ρ が超えているため。
