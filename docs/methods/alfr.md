# ALFR（Edwards & Storkey の交互更新）：原論文・実装・使い方

2026-10-16 作成。原論文の本文を読み直し、この repo の実装（修正 commit `0f2af99` を含む HEAD）と突き合わせた結果をまとめる。
ALFR の run を設計・起動・評価するセッションは、先にこれを読む。勾配反転（GRL）と adversary の作りは
[dann.md](dann.md) と共通なので、そちらも読む。

## 先に押さえること

- **何をする手法か**：表現から属性を当てる **adversary**（敵対者のネットワーク）と、表現を作る側（backbone と分類 head）を
  **minibatch ごとに交互に**更新する。偶数番目の minibatch では adversary だけ、奇数番目では表現側だけが動く。
  DANN は同じ minibatch で両方を同時に更新するので、違いは更新の順番だけ。
- **この repo の ALFR は原論文の式 9 の一部だけを使う**。再構成の項（autoencoder）は無く α = 0。
  属性の項の重み β = w·s、task の重み γ = 1 にあたる（w・s の意味は [dann.md](dann.md)）。
- **`0f2af99`（2026-10-16）で交互更新を原論文の Algorithm 1 に合わせた**。それより前は同じ minibatch で両方を更新し、
  adversary 側だけ BatchNorm を eval モードで通し、global_step が 2 倍に進んでいた。ただし **ALFR の run は 1 本も無い**ので、
  過去の結果への影響は無い。
- **表現側が更新されるのは minibatch の半分だけ**。同じ epoch 数の ERM と比べると、backbone の更新回数は半分になる。
- 原論文の model 選択は「val の accuracy − t × discrimination が最大」。この repo の既定は best val AUROC で、別物。

## 原論文と公式実装

- Edwards, Storkey. *Censoring Representations with an Adversary.* ICLR 2016（[arXiv:1511.05897](https://arxiv.org/abs/1511.05897)）
- 公式実装：論文に公開先の記載は無く、見つけていない（実装は theano と lasagne とだけ書かれている、5 節）。

### 原論文の条件（本文で確認済み）

| 項目 | 原論文 | 出典 |
| --- | --- | --- |
| 目的 | 敏感属性 S を予測できない表現 R を学ぶ。公平性は統計的パリティ P(η=1\|S=1) = P(η=1\|S=0) | 3 節、式 1 |
| 損失 | L = αC + βD + γE。C は再構成の二乗誤差、D は adversary の対数尤度（負の log-loss）、E は予測器の log-loss | 4 節、式 4–9 |
| minimax | min_θ max_φ L。adversary（φ）は D を上げ、表現側（θ：encoder・decoder・予測器）は L を下げる | 式 10 |
| 交互更新 | minibatch ごとに adversary か表現側の一方だけを 1 step 更新する（strict alternation）。最初は adversary | 4.4 節、Algorithm 1 |
| 停止の工夫（任意） | adversary の正解率が 90% を超えたら adversary を、60% を下回ったら表現側を更新しない。「改善することがあるが要検討」 | 4.4 節 |
| optimizer | Adam、既定の parameter | 5 節 |
| データと属性 | Adult（Gender）、Diabetes（Race を isCaucasian の 2 値に）。どちらも UCI の表データ | 5.1.1 節 |
| 分割 | Adult は train 35,000・val と test 約 5,000 ずつ。Diabetes は 80,000・約 10,000 ずつ。分割を変えて 5 回 | 5.1.1–5.1.2 節 |
| 構造 | autoencoder は 1〜3 層・幅 1〜100（ReLU）。adversary は 1〜3 層の ReLU。予測器は中央の層に載せたロジスティック回帰 | 5.1.2 節 |
| α・β・γ | α = 0.05 固定、β ~ U[0, 50]、γ ~ U[0, 10] を random search で 100 通り | 5.1.2 節 |
| model の選び方 | t ∈ [0, 3] ごとに、val で y_t = accuracy − t × discrimination が最大の model を選ぶ | 3 節 式 3、5.1.2 節 |
| 評価 | test の accuracy と discrimination（群間の陽性予測率の差の絶対値、式 2）。β/(β+γ) が大きいほど discrimination が下がる | 5.1.3 節、図 1–3 |

## この repo の実装

`projects/hypernet_e2e/objectives/attribute_invariance.py` の `AlternatingAttributeInvariantTaskLoss`（366 行）と、
`module.py` の手動最適化（manual optimization：Lightning の自動の backward・step を切り、自分で呼ぶ方式）の分岐。

- **手動最適化への切り替え**：loss が `requires_manual_optimization = True` を持つと、`module.py:87` が
  `automatic_optimization` を False にする。
- **optimizer は 2 つ**：`configure_optimizers`（405 行）が、表現側（net 全体と、adversary 以外の loss の parameter）用と、
  adversary 用を作る。adversary 用は model config の `adversary_optimizer`（無ければ表現側と同じ factory）。
  scheduler と `adversary_lr_scale ≠ 1` は受け付けない（止める）。
- **交互更新**：`_alternating_training_step`（472 行）。各 minibatch を train モードで 1 回だけ forward する。
  - `global_step` が偶数：adversary の step。表現を detach（勾配を切る）してから adversary 損失を計算し、adversary だけを更新する
    （`adversary_loss`、375 行）。表現側の損失は記録のためだけに勾配なしで計算する。
  - 奇数：表現側の step。adversary の parameter を `requires_grad=False` にし、`task + w · adv` を GRL 越しに backward する。
    backbone には ∂task − w·s·∂adv が届く（DANN と同じ式）。
  - 1 つの minibatch で step は 1 回なので、`global_step` は処理した minibatch 数と等しい（Lightning 2.6 は手動最適化の
    `global_step` を optimizer step の回数で数える。インストール済みのコードで確認）。
- **adversary と特徴量**：DANN と同じ（ResNet-50 の 2048 次元、`Linear(2048→256) → LeakyReLU(0.2)`、属性ごとの head、
  sex・race は cross-entropy、標準化した age は MSE、欠測は外す）。詳細は [dann.md](dann.md)。
- **記録**：`metrics.csv` に `train/task`、`train/attribute_adversary`、`train/attribute_adversary/<属性名>`、
  `train/alfr/adversary_loss`（adversary の step での損失の epoch 平均）、`train/adversary_scale`。val は素の CE だけ。

### 原論文との違い

| 項目 | 原論文 | この repo（`resnet50_attribute_invariant_alfr`） |
| --- | --- | --- |
| 再構成の項 | α = 0.05 | 無し（α = 0。原論文も「表現を他の task に再利用しないなら α = 0 でよい」と書く、4 節） |
| β・γ | β ~ U[0, 50]、γ ~ U[0, 10]（β/(β+γ) は平均で約 0.78） | β = w·s = 1、γ = 1（β/(β+γ) = 0.5）。探索していない |
| 敏感属性 | 2 値 1 つ | sex（2）・race（6）・age（連続）の損失の平均 |
| adversary | 1〜3 層の ReLU | 256 の 1 層、LeakyReLU(0.2) |
| 表現 | 小さい MLP を最初から学習 | ImageNet 事前学習の ResNet-50 を fine-tune |
| optimizer | Adam 既定（lr 1e-3、weight decay なし） | 両側とも AdamW lr 1e-4、weight decay 0.01 |
| adversary の損失の重み | adversary の step も β 倍（Algorithm 1 の L） | 重みなしの平均。Adam は勾配の定数倍で歩幅がほぼ変わらないので、実質の差は小さい（推測） |
| 停止の工夫 | 任意で使う | 無し |
| model の選び方 | val の accuracy − t × discrimination | best val AUROC（と last） |
| 評価 | accuracy と統計的パリティの差 | 固定しきい値 0.5 の群別 TPR・FPR（[research-premises.md](../research-premises.md)） |

違いについての補足（推測）：

- 原論文の公平性は統計的パリティ（予測の陽性率を群で揃える）で、この repo の主指標（群別の TPR・FPR の差）とは目標が違う。
  属性を表現から消すことは、TPR・FPR の差を縮めることを保証しない。
- 表現側の更新が半分になるので、ERM と epoch 数を揃えると task の学習が足りない可能性がある。

## 起動

| preset（`configs/experiment/`） | 中身 |
| --- | --- |
| `resnet_chexpert_attribute_invariant_alfr` | CheXpert、sex・race・age、w = 1、s = 1 一定（schedule なし） |

```bash
uv run python -m projects.hypernet_e2e.run experiment=resnet_chexpert_attribute_invariant_alfr seed=<seed> study=<slug>
```

| 変えたい点 | override |
| --- | --- |
| β を上げる（原論文の範囲へ） | `model.loss_fn.attribute_adversary_weight=<β>`（s = 1 なので β = w） |
| 原論文の adversary の形に寄せる | `'model.loss_fn.hidden_dim=[<幅>,<幅>]'` |
| adversary の lr | `model.adversary_optimizer.lr=<値>`（`adversary_lr_scale` は使えない） |
| β を 0 から上げる | `'model.loss_fn.gradient_schedule={name:dann,gamma:10.0,max_scale:1.0}'`（DANN の λ schedule。原論文には無い） |
| 表現側の更新回数を ERM に揃える | `trainer.max_epochs=<ERM の 2 倍>` |

HAM10000 の preset は無い。HAM10000 には race が無いので、作るときは `resnet_ham10000_fg_dann_sex_age` と同じく
`adversarial_attribute_names` を sex・age に指定し直す。

## checkpoint の選び方と予測の使い方

- 原論文どおりに選ぶなら、val の task 性能 − t × 公平性の差が最大の epoch。t を何通りか振り、t ごとに 1 つ選ぶ。
  この repo の主指標で何を「公平性の差」とするか（Eodds など）は未決。決めてから選ぶ。
- 既定の `model_checkpoint` は best val AUROC の 1 つと last だけを残す。原論文の選び方をするには、
  `callbacks.model_checkpoint.save_top_k=-1` で全 epoch を残す。
- 予測は `analysis/common/predictions.py`。net だけを復元し、adversary は読み込まない。best val AUROC 以外の epoch は
  `--checkpoint` は名前で渡す（既定 `best` は run.json に記録された val AUROC の best、`xxx` は `checkpoints/xxx.ckpt`。例 `best_val_auroc_009`、`last`、`swad`）。属性が消えたかは `--features` で表現を書き出し、probe で確かめる（[dann.md](dann.md) の手順）。
- 学習中の確認：`train/alfr/adversary_loss` が「事前分布だけを答える予測器」の水準に張り付いていないか見る。
  張り付いていれば、adversary が属性を追えていない（旧 DANN の失敗と同じ形）。

## 2026-10-16 の照合結果

- **一致を確認（コードを読んで確認）**：minibatch ごとの strict alternation と、最初が adversary であること（Algorithm 1）。
  adversary の step で表現を detach すること。表現側の step で adversary の parameter を止め、GRL で符号を反転すること。
  γ = 1・β = w·s・α = 0 にあたること。
- **`0f2af99` で直した**：(1) 同じ minibatch で両方を更新していた（adversary → 表現側）。(2) adversary 側の forward だけ
  BatchNorm を eval モードで通していたので、両側で表現の正規化が違っていた。(3) 1 minibatch で step が 2 回あり、
  global_step が 2 倍に進んでいた。λ schedule を使う設定では進捗 p が 2 倍速で進む計算になる（ALFR の preset は schedule なし）。
- **2026-10-13 に直した**（`7877f1c`・`dac24b1`）：trunk を LeakyReLU に、選んだ属性の一部がバッチ内で全件欠測のときの KeyError。
- **残る違い**：上の「原論文との違い」の表。特に β・γ を探索していない点と、model の選び方。
- **未確認**：公式実装は見つけていない。修正後の ALFR で adversary が属性を読めるか、表現から属性が消えるかは、run が無いので分からない。
- **既存の run**（run 台帳の集計、背景として）：ALFR の run は 0 本。
