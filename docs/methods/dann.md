# DANN（Ganin et al. の勾配反転）：原論文・実装・使い方

2026-10-16 作成。原論文の本文を読み直し、この repo の実装（修正 commit `0f2af99` を含む HEAD）と突き合わせた結果をまとめる。
属性不変化（GRL）の run を設計・起動・評価するセッションは、先にこれを読む。交互更新の版は [alfr.md](alfr.md)。

## 先に押さえること

- **何をする手法か**：backbone の表現から属性（sex・race・age）を当てる小さなネットワークを付ける。これを
  **adversary**（敵対者）と呼ぶ。adversary は属性を当てるように学習し、backbone は adversary が当てにくくなるように学習する。
  この逆向きの学習を 1 回の backprop で済ませるのが **GRL**（gradient reversal layer）。順伝播では値をそのまま通し、
  逆伝播では勾配に −s を掛けて backbone へ返す。
- **原論文はドメイン適応の手法**。source と target（2 値）を見分けられない表現を作る。この repo はドメインの代わりに
  患者属性を見分けられない表現を作る。公平性への転用は ALFR（Edwards & Storkey 2016）と同じ発想。
- **2 つの model config の違いは重み w だけ**（`0f2af99` 以降）。どちらも s を **λ schedule**（学習の進捗 p に応じて
  0 から 1 へ上げる式）で上げる。backbone へ届く逆向きの勾配の倍率は w·s で、既定 config は 0→0.1、`_dann` は 0→1（原論文と同じ）。
- **既存の DANN run 17 本は使わない**。全部 2026-10-13 の修正より前の ReLU 版で、adversary の head が属性を読めていなかった。
- **属性が消えたかは学習中の記録からは分からない**。val は素の cross-entropy だけを測り、adversary の val 損失は残らない。
  凍結した表現に別の probe を当てて確かめる（下の「checkpoint の選び方」）。

## 原論文と公式実装

- Ganin, Ustinova, Ajakan, Germain, Larochelle, Laviolette, Marchand, Lempitsky. *Domain-Adversarial Training of Neural
  Networks.* JMLR 17, 2016（[arXiv:1505.07818](https://arxiv.org/abs/1505.07818)）
- 公式実装：深層版（Caffe）は <http://sites.skoltech.ru/compvision/projects/grl/>（脚注 4）、浅い版は
  <http://graal.ift.ulaval.ca/dann/>（脚注 3）。**どちらも今回は読んでいない**。以下は本文だけで確認した。

### 原論文の条件（本文で確認済み）

| 項目 | 原論文 | 出典 |
| --- | --- | --- |
| 目的 | ラベルを予測でき、ドメインを見分けられない特徴を学ぶ | 3・4 節 |
| 不変にする変数 | ドメイン（source / target の 2 値）。ラベルは source だけにある | 4.2 節、式 10 |
| 更新式 | θf ← θf − μ(∂Ly/∂θf − λ∂Ld/∂θf)、θy ← θy − μ∂Ly/∂θy、θd ← θd − μλ∂Ld/∂θd | 式 13–15 |
| GRL | 順伝播は恒等（R(x) = x）、逆伝播で −1 倍（dR/dx = −I）。parameter なし | 4.2 節、式 16–17 |
| λ schedule | λp = 2/(1+exp(−γp)) − 1、γ = 10（調整していないと明記）。p は 0→1 の学習進捗 | 5.2.2 節 |
| λ の掛け先 | **λp は特徴抽出器の更新だけに使う**。ドメイン分類器は λ = 1 固定（「ラベル予測器と同じ速さで学ぶように」） | 5.2.2 節、脚注 6 |
| ドメイン分類器 | x→1024→1024→2 の全結合、ReLU（MNIST だけ x→100→2）。「この選び方は恣意的」と本文が認める | 5.2.2 節、図 4 |
| 表現力の条件 | ドメイン分類器の仮説クラスがラベル予測器のものを含む（Hy ⊆ Hd） | 4.2 節 |
| optimizer | SGD、momentum 0.9、lr μp = 0.01/(1+10p)^0.75。SVHN 系は lr を下げない | 5.2.2 節、5.2.4 節 |
| バッチ | 128 件、半分 source・半分 target | 5.2.2 節 |
| データ | MNIST→MNIST-M、Syn Numbers→SVHN、SVHN↔MNIST、Syn Signs→GTSRB、Office（事前学習 AlexNet） | 5.2.4 節、表 2・3 |
| model の選び方 | 深層実験は grid search せず固定の schedule で学習。浅い実験は reverse validation と source val の early stopping | 5.1.2 節、5.2.2 節 |
| 評価 | target ドメインの分類 accuracy | 表 2・3 |

## この repo の実装

`projects/hypernet_e2e/objectives/attribute_invariance.py` と `module.py`。

- **GRL**：`_GradientReverse`（15 行）。forward は恒等、backward は `-scale * gradient`。
- **adversary**：`_AttributeAdversary`（54 行）。入力は ResNet-50 の layer4 を global average pooling した 2048 次元
  （`models/resnet/network.py:205` の `forward_with_features`、fc の直前）。`Linear(2048→256) → LeakyReLU(0.2)` の trunk を
  全属性で共有し、属性ごとに head を持つ。カテゴリ属性は cross-entropy、age は MSE。age は train の平均と標準偏差で
  標準化してある（data config の `standardize_continuous: true`）。欠測の標本はその属性の損失から外す。
- **総損失**：`AttributeInvariantTaskLoss`（142 行）。`task + w · mean(属性ごとの adversary 損失)`。w は `attribute_adversary_weight`。
  task はクラス重み付き CE（`weighting=inverse` の重みは `training.py:269` で `task_loss` の下に入る）。
- **λ schedule**：`set_training_progress`（273 行）が s = max_scale · (2/(1+exp(−γp)) − 1) を計算する。p は
  `module.py:462` の `_update_adversary_progress` が毎 step の最初に `global_step / (総 step 数 − 1)` で渡す。
  s = tanh(5p) と同じ形で、学習の 1 割で 0.46、3 割で 0.91、5 割で 0.99 になる（30 epoch なら 3・9・15 epoch 分）。
- **optimizer**：AdamW 1 つで backbone・fc・adversary をまとめて更新する（自動最適化）。`adversary_lr_scale ≠ 1` のときだけ
  `_parameter_groups`（450 行）が adversary を別の param group にして lr を倍にする。
- **使える model**：`forward_with_features` を持つ ResNet だけ。他の net では `module.py:517` が TypeError で止める。
- **起動時の検査**：`training.py:200` の `_validate_attribute_invariance` が、ERM 以外の strategy との併用を拒否する。
  ただし experiment が `attribute_invariance.enabled: true` を持つときだけ働く（preset はすべて持つ）。
- **記録**：`metrics.csv` に `train/task`、`train/attribute_adversary`、`train/attribute_adversary/<属性名>`、
  `train/adversary_scale`（s の epoch 平均）。val は比較用の素の CE だけ。

各部分が受け取る勾配：

| 受け取る側 | この repo | 原論文（式 13–15、5.2.2 節） |
| --- | --- | --- |
| adversary | w · ∂adv | λ = 1 固定 |
| backbone | ∂task − w · s(p) · ∂adv | ∂Ly − λp · ∂Ld |
| 分類 head（fc） | ∂task | ∂Ly |

`_dann`（w = 1）はこの表で原論文と一致する。既定 config（w = 0.1）は adversary 側も 0.1 倍になるが、
AdamW は勾配の定数倍で歩幅がほとんど変わらないので、効くのは backbone 側の 0.1 倍だけと考えてよい（weight decay との比だけが変わる）。

### 原論文との違い

| 項目 | 原論文 | この repo |
| --- | --- | --- |
| 不変にする変数 | ドメイン（2 値） | CheXpert は sex（2）・race（6）・age（連続）の損失の平均。HAM10000 は sex・age |
| ラベル | source だけ | 全標本 |
| adversary | 1024→1024、ReLU | 256 の 1 層、LeakyReLU(0.2)、trunk を属性で共有 |
| backbone | 小さい CNN、AlexNet | ImageNet 事前学習の ResNet-50 を fine-tune |
| optimizer | SGD momentum 0.9、lr を 0.01 から下げる | AdamW lr 1e-4 一定、weight decay 0.01（adversary にも掛かる） |
| backbone 側の倍率 | λp：0→1 | w · s(p)：既定 0→0.1、`_dann` 0→1 |
| バッチ | source と target を半々 | 自然抽出 128 件 |
| model の選び方 | 深層実験は記載なし | best val AUROC（と last） |

違いについての補足：

- **LeakyReLU は原論文に無い**。2026-10-13 に足した。旧 ReLU 版では 256 unit 中 206 が val で一度も発火せず、
  head は事前分布だけを答えていた（`analysis/archive/adversary_strength/`、事実）。原因は「GRL の勾配を受けた backbone が
  trunk の入力を負に押し込み、head への情報と head の勾配を同時に断った」と推定している（推測、run での検証はまだ無い）。
- **adversary 損失が上がっても属性が消えたとは限らない**（推測）。backbone は「adversary に自信をもって外させる」ことでも
  損失を上げられる。age の MSE は特にそうで、情報を消さずに予測を系統的にずらすだけで大きくなる。

## 起動

| preset（`configs/experiment/`） | 中身 |
| --- | --- |
| `resnet_chexpert_attribute_invariant` | w = 0.1、λ schedule、sex・race・age |
| `resnet_chexpert_attribute_invariant_dann` | w = 1.0（原論文の λ に相当） |
| `resnet_ham10000_fg_dann_sex_age` | `_dann` を HAM10000 で。sex・age（race は無い）。val の最悪 sex × age 群 AUROC の checkpoint も残す |
| `spatial_lora_chexpert_from_attribute_invariant` | 2 段目。上の run の checkpoint を `model.backbone_checkpoint_path` で渡す |

```bash
uv run python -m projects.hypernet_e2e.run experiment=resnet_chexpert_attribute_invariant_dann seed=<seed> study=<slug>
```

| 変えたい点 | override |
| --- | --- |
| 原論文の adversary の形 | `'model.loss_fn.hidden_dim=[1024,1024]'` |
| s を一定にする（2026-10-16 より前の既定 config） | `model.loss_fn.gradient_schedule=null` |
| 旧 ReLU 版の再現 | `model.loss_fn.activation=relu` |
| adversary の lr だけ上げる | `model.loss_fn.adversary_lr_scale=<倍率>` |
| 属性ごとに別の trunk | `model.loss_fn.shared_trunk=false` |
| 対象の属性を絞る | `'model.loss_fn.adversarial_attribute_names.categorical=[sex]'` など |

HAM10000 には race が無いので、CheXpert 用の model config をそのまま使うと止まる。HAM10000 の preset のように属性を指定し直す。

## checkpoint の選び方と予測の使い方

- 原論文の深層実験は選び方を書いていない。固定の schedule で学習を終えた model を報告している、と読める（推測）。
- この repo は既定の `model_checkpoint`（best val AUROC の 1 つと last）。HAM10000 の preset は val の最悪 sex × age 群 AUROC の
  1 つも残す。全 epoch は残さない。
- **best val AUROC は逆向きの勾配が弱い epoch を選びうる**。s が 0.9 を超えるのは学習の 3 割（30 epoch なら 9 epoch 分）を
  過ぎてから。それより前の epoch が選ばれたら、その model は反転を十分に受けていない。選ばれた epoch と s の値を併記する。
- 評価の順番：先に属性が消えたかを probe で確かめ、それから公平性指標を読む。
  1. `analysis/common/predictions.py --features` で表現（fc の直前）を書き出す。adversary は読み込まない（net だけを復元する）。
  2. `analysis/archive/initial_resnet_vs_invariant/attribute_probe.py` と同じ手順（train で fit → val で選ぶ → test で測る）で、
     同じ構成の ERM と属性の読みやすさを比べる。
  3. 公平性指標は [research-premises.md](../research-premises.md) の固定しきい値 0.5 で測る。

## 2026-10-16 の照合結果

- **一致を確認（コードを読んで確認）**：GRL の forward・backward。総損失と勾配の流れ（`_dann` で式 13–15、λ_d = 1 と一致）。
  schedule の式と γ = 10。p が global_step に比例して 0→1 に進むこと。
- **`0f2af99` で直した**：既定 config は s = 1 の一定で、逆向きの勾配が step 0 から 0.1 倍で効いていた。λ schedule に変えた。
- **2026-10-13 に直した**（`7877f1c`・`dac24b1`）：trunk を LeakyReLU に、`adversary_lr_scale` を追加、選んだ属性の一部が
  バッチ内で全件欠測のときの KeyError。
- **残る違い**：上の「原論文との違い」の表。どこまで揃えるかは決めていない。
- **未確認**：公式実装（Caffe）は読んでいない。LeakyReLU で head が属性を読めるようになるかは、修正後の run がまだ無いので分からない。
- **既存の run**（run 台帳の集計、背景として）：17 本、2026-09-22〜10-10。CheXpert 14 本（既定 config 10 本：w = 0.1 の 3 seed と w = 0〜10 の振り。
  `_dann` 4 本）と HAM10000 3 本（seed 42〜44）。すべて旧 ReLU で、2026-10-13 の修正より前なので無効として扱う。
  既定 config の 10 本は schedule も無かった。
