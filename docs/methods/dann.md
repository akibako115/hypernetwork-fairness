# DANN

表現から属性を当てる小さなネットワーク（adversary）を付け、表現側は adversary が当てにくくなるように学習する手法。
Ganin et al., JMLR 2016。この repo では model config `resnet50_attribute_invariant`（重み w = 0.1）と
`resnet50_attribute_invariant_dann`（w = 1、原論文と同じ倍率）の 2 つ。どちらも strategy は `erm` のまま使う。
交互更新の版は [alfr.md](alfr.md)。両者で共通の考え方と実装はこの文書に書く。

## 1. 手法のロジック

### 解く問題

普通の学習（ERM）で得た表現（分類 head の直前の特徴量）には、予測に必要な情報のほかに、予測と関係の薄い変数の
情報も入る。原論文が扱うのは **ドメイン適応**（domain adaptation）で、ラベル付きの source ドメイン（例：合成の数字画像）
で学習し、ラベルの無い target ドメイン（例：実写の番地画像）で予測する。表現が「どちらのドメインの画像か」を
見分けられるなら、source で学んだ分類器は target では外れやすい。そこで、ラベルを予測でき、かつドメインを見分けられない
表現を学ぶ。

公平性への転用では、ドメインの代わりに **患者属性**（sex・race・age）を見分けられない表現を学ぶ。原論文の「ドメイン」は
画像の出どころ（source / target の 2 値）で、患者の属性ではない。この転用は ALFR（Edwards & Storkey 2016）と同じ発想で、
ALFR の論文も DANN の先行版から着想を得たと書いている。

### 目的関数

ネットワークを 3 つに分ける。

- 特徴抽出器 G_f（parameter θ_f）：画像から表現を作る。
- ラベル予測器 G_y（θ_y）：表現からラベルを予測する。損失 L_y。
- ドメイン分類器 G_d（θ_d）：表現からドメインを当てる。損失 L_d。公平性への転用ではこれが **adversary**（敵対者）。

```
E(θ_f, θ_y, θ_d) = L_y(θ_f, θ_y) − λ · L_d(θ_f, θ_d)

(θ_f, θ_y) は E を最小化し、θ_d は E を最大化する（＝ L_d を最小化する）。
```

λ は 2 つの目的の釣り合いを決める。G_d はドメインをよく当てるように学び、G_f は G_d が当てにくい表現を作るように学ぶ。
ラベル L_y は source の標本だけで、L_d は source と target の両方で計算する。

### 学習の手順と勾配反転層（GRL）

鞍点（一方が最小化、他方が最大化する点）を、次の勾配更新で探す（原論文の式 13–15。μ は学習率）。

```
θ_f ← θ_f − μ (∂L_y/∂θ_f − λ ∂L_d/∂θ_f)
θ_y ← θ_y − μ ∂L_y/∂θ_y
θ_d ← θ_d − μ λ ∂L_d/∂θ_d
```

θ_f の更新だけ L_d の勾配の符号が逆になる。これを普通の backprop 1 回で実現するのが **GRL**（gradient reversal layer）で、
G_f と G_d の間に挟む。順伝播では値をそのまま通し、逆伝播では勾配に −λ を掛けて G_f へ返す。parameter は持たない。
こうすると、L_y + L_d を普通に最小化するだけで、G_d は L_d を下げ、G_f は L_d を上げる向きに動く。1 つのミニバッチで
3 つを同時に更新する点が、交互に更新する ALFR との違い。

### λ schedule

λ を最初から 1 にすると、まだ意味のない G_d の勾配が G_f を乱す。原論文の深層実験では、学習の進捗 p（0 から 1 へ線形に進む）
に応じて λ を 0 から 1 へ上げる。

```
λ_p = 2 / (1 + exp(−γ p)) − 1      （γ = 10。tanh(5p) と同じ形）
```

p = 0.1 で 0.46、0.3 で 0.91、0.5 で 0.99。λ_p を掛けるのは G_f へ返す勾配だけで、G_d の更新は λ = 1 に固定する
（G_d がラベル予測器と同じ速さで学ぶように、と原論文は書く）。

### adversary の損失で分かること・分からないこと

adversary の損失が上がっても、表現から属性が消えたとは限らない。表現側は「adversary に自信をもって外させる」ことでも
損失を上げられる。連続値（age）を二乗誤差で当てる場合は特に、情報を残したまま予測を系統的にずらすだけで損失が大きくなる。
属性が消えたかは、学習を終えた表現に新しい分類器（probe）を学習させて確かめる。

## 2. 原論文の設定

- Ganin, Ustinova, Ajakan, Germain, Larochelle, Laviolette, Marchand, Lempitsky. *Domain-Adversarial Training of Neural
  Networks.* JMLR 17, 2016（[arXiv:1505.07818](https://arxiv.org/abs/1505.07818)）
- 公式実装：深層版（Caffe の GRL 拡張）は <http://sites.skoltech.ru/compvision/projects/grl/>（脚注 4）、浅い版は
  <http://graal.ift.ulaval.ca/dann/>（脚注 3）。どちらも読んでいない。以下は本文から取った。

### データセットとタスク

深層の画像実験（5.2 節）。どれも多クラス分類で、不変にする変数は **ドメイン（source / target の 2 値）**。
target のラベルは学習に使わない（教師なしドメイン適応）。

| データセット（source → target） | 入力 | 予測ラベル（クラス数） | 不変にする変数（値の数） |
| --- | --- | --- | --- |
| MNIST → MNIST-M（写真の断片に数字を合成） | 数字画像 | 数字（10 クラス） | ドメイン（2） |
| Syn Numbers（自作の合成数字）→ SVHN | 数字画像 | 数字（10 クラス） | ドメイン（2） |
| SVHN ↔ MNIST | 数字画像 | 数字（10 クラス） | ドメイン（2） |
| Syn Signs（合成標識）→ GTSRB | 交通標識の画像 | 標識の種類（43 クラス） | ドメイン（2） |
| Office（Amazon → Webcam、DSLR → Webcam、Webcam → DSLR） | 物体の写真 | 物体の種類（31 クラス） | ドメイン（2） |

ほかに、浅い 1 層版の実験（5.1 節：2 次元の toy 問題と Amazon レビューの感情 2 値分類）と、人物再同定の記述子学習
（5.3 節）がある。評価は target ドメインの分類 accuracy（表 2・3）。

### ハイパーパラメータ

深層の画像実験（5.2.2 節）：

| 項目 | 値 |
| --- | --- |
| 特徴抽出器 | 数字・標識は 2〜3 層の小さい CNN（図 4）。Office は ImageNet 事前学習の AlexNet を fine-tune |
| ドメイン分類器 | 全結合 x→1024→1024→2（MNIST だけ x→100→2）、ReLU。「この選び方は恣意的」と本文が認める |
| 損失 | L_y はロジスティック回帰の損失（多クラスの log-loss）、L_d は 2 値の cross-entropy |
| optimizer | SGD、momentum 0.9 |
| lr | μ_p = 0.01 / (1 + 10p)^0.75。SVHN ↔ MNIST は lr を下げない |
| batch | 128 件。半分が source、半分が target |
| epoch・反復回数 | 本文に記載なし（未確認） |
| 正則化 | SVHN の構成だけ dropout と ℓ2 ノルム制約。ほかは記載なし |
| 前処理・データ拡張 | 平均を引くだけ。データ拡張の記載なし |
| λ | 上の λ_p schedule（γ = 10、調整していないと明記）。G_d の更新は λ = 1 固定 |

表現力の条件として、G_d の仮説クラスが G_y のものを含む（H_y ⊆ H_d）ことを理論の前提に置く（4.2 節）。

### モデルの選び方

深層実験は grid search をせず、上の固定の schedule で学習する（5.2.2 節）。checkpoint をどう選んだかは書かれていない（未確認）。
浅い実験（5.1 節）は reverse validation（target に付けた疑似ラベルで逆向きに学習した分類器の source での誤差）で
ハイパーパラメータを選び、early stopping を使う。

## 3. この repo の実装

### ファイルと設定

| 役割 | 場所 |
| --- | --- |
| 損失・GRL・adversary | [objectives/attribute_invariance.py](../../projects/hypernet_e2e/objectives/attribute_invariance.py) の `AttributeInvariantTaskLoss`、`_AttributeAdversary`、`_GradientReverse` |
| model config | [configs/model/resnet50_attribute_invariant.yaml](../../projects/hypernet_e2e/configs/model/resnet50_attribute_invariant.yaml)（w = 0.1）、[resnet50_attribute_invariant_dann.yaml](../../projects/hypernet_e2e/configs/model/resnet50_attribute_invariant_dann.yaml)（w = 1） |
| 表現の取り出し | [models/resnet/network.py](../../projects/hypernet_e2e/models/resnet/network.py) の `ResNet.forward_with_features` |
| 進捗の受け渡し・optimizer | [module.py](../../projects/hypernet_e2e/module.py) の `LitModule._update_adversary_progress`・`_parameter_groups`・`_objective_components` |
| 起動時の検査・クラス重み | [training.py](../../projects/hypernet_e2e/training.py) の `_validate_attribute_invariance`・`class_weight_path` |
| 代表的な preset | `resnet_chexpert_attribute_invariant`（w = 0.1）、`resnet_chexpert_attribute_invariant_dann`（w = 1）、`resnet_ham10000_fg_dann_sex_age`（HAM10000、sex・age） |

### 処理の流れ

**起動時（fit の前）。**

- `_validate_attribute_invariance`：experiment が `attribute_invariance.enabled: true` を持つとき、strategy が `erm` でなければ止める。
  GroupDRO などの strategy は `model.loss_fn` を差し替えるので、そのままだと属性不変化の wrapper が黙って消えるため。
  preset はすべてこのフラグを持つ。
- `class_weight_path`：loss の `_target_` が `AttributeInvariantTaskLoss` で終わる（ALFR の class も含む）とき、`weighting=inverse` の
  クラス重みを `model.loss_fn.task_loss.class_weight` に書く。task の損失を wrapper が包むので 1 段深くなる。
  CheXpert・HAM10000 の base preset は `weighting: inverse` が既定。
- `AttributeInvariantTaskLoss.__init__`：data config の属性列（`input_attribute_names`）から、adversary に当てさせる列
  （`adversarial_attribute_names`、既定は sex・race・age）を選び、属性ごとの head を作る。

adversary（`_AttributeAdversary`）の作り：

- 入力は ResNet-50 の layer4 を global average pooling した 2048 次元（fc の直前、`forward_with_features` の 2 つ目の戻り値）。
- `Linear(2048→256) → LeakyReLU(0.2)` の trunk を全属性で共有し（`shared_trunk: true`）、属性ごとに出力層を持つ。
  カテゴリ属性は cross-entropy、age は二乗誤差（MSE）。age は data config の `standardize_continuous: true` で
  train の平均と標準偏差で標準化してある。
- 活性化を ReLU でなく LeakyReLU にしている理由：ReLU の trunk では、反転した勾配を受けた backbone が trunk の入力を全標本で
  負に押し込むだけで unit が発火しなくなり、head への情報と head の勾配が同時に断たれた（head が事前分布だけを答えた）。
  LeakyReLU は負側でも勾配が残るのでこの逃げ道を塞ぐ。`activation: relu` は旧 run の再現用に残してある。

**学習の各ステップ。**

1. `_update_adversary_progress`（`training_step` の最初）：p = `global_step / (総 step 数 − 1)` を `set_training_progress` に渡す。
2. `set_training_progress`：`gradient_schedule: {name: dann}` なら s = max_scale · (2/(1+exp(−γp)) − 1) を GRL の倍率にする
   （既定 γ = 10、max_scale = 1）。`gradient_schedule: null` なら s は `gradient_scale` の一定値。
3. `forward_with_features`：logits と表現を 1 回の forward で得る。
4. `loss_components`：task の損失（クラス重み付き CE）と、GRL を通した表現での属性ごとの adversary 損失を計算する。
   欠測の標本はその属性の損失から外す。属性ごとの損失を平均したものが adversary 損失。
5. `_objective_components`：`task + w · adversary` を総損失にする（w は `attribute_adversary_weight`）。Lightning の自動最適化で
   backward し、AdamW 1 つで backbone・fc・adversary をまとめて更新する。

各部分が受け取る勾配は次のとおり。

| 受け取る側 | この repo | 原論文（式 13–15） |
| --- | --- | --- |
| adversary | w · ∂adv | λ = 1 固定 |
| backbone | ∂task − w · s(p) · ∂adv | ∂L_y − λ_p · ∂L_d |
| 分類 head（fc） | ∂task | ∂L_y |

`_dann`（w = 1）はこの表で原論文と一致する。既定 config（w = 0.1）は adversary 側も 0.1 倍になるが、AdamW は勾配の定数倍で
歩幅がほとんど変わらないので、効くのは backbone 側の 0.1 倍だけと考えてよい（weight decay との比だけが変わる）。

adversary の lr だけを変えたいときは `adversary_lr_scale` を 1 以外にする。`_parameter_groups` が adversary を別の param group にして
lr をその倍率にする。事前学習済み backbone 向けの小さい lr では、ゼロから学ぶ adversary が backbone の変化に追いつかないことがあるため。

**記録と checkpoint。**

- `metrics.csv` に `train/task`、`train/attribute_adversary`、`train/attribute_adversary/<属性名>`、`train/adversary_scale`（s の epoch 平均）。
  val でも `val/attribute_adversary` と属性ごとの値を記録する（`validation_step` が task 以外の成分を記録するため）。ただしこれは
  学習中の adversary を val に当てた損失で、属性が表現から消えたかの判定には使えない（1 節の最後）。`val/loss` は素の CE。
- checkpoint は既定の `model_checkpoint`（best val AUROC の 1 つと last）。HAM10000 の preset は val の最悪 sex × age 群 AUROC で
  選んだ 1 つも残す（`_fairness_ham10000`）。全 epoch は残さない。
- best val AUROC は、反転した勾配がまだ弱い epoch を選びうる。s が 0.9 を超えるのは学習の 3 割（30 epoch なら 9 epoch 目）以降。
  選ばれた epoch と、そのときの `train/adversary_scale` を併記する。
- 予測は `analysis/common/predictions.py`。net（backbone と fc）だけを復元し、adversary は読み込まない。属性が消えたかは
  `--features` で表現を書き出し、同じ構成の ERM と並べて probe の成績を比べる。公平性指標は
  [research-premises.md](../research-premises.md) の固定しきい値 0.5 で測る。
- 対応する net は `forward_with_features` を持つ ResNet だけ。他の net では `_shared_step` が TypeError で止める。

### 原論文との違い

| 項目 | 原論文 | この repo |
| --- | --- | --- |
| 不変にする変数 | ドメイン（source / target の 2 値） | 患者属性。CheXpert は sex（2）・race（6）・age（連続）の損失の平均、HAM10000 は sex・age |
| ラベル | source の標本だけ | 全標本 |
| タスク | 10・31・43 クラス分類 | 2 値分類（CheXpert・HAM10000 とも `num_classes: 2`。HAM10000 は悪性かどうか） |
| adversary | 1024→1024、ReLU | 256 の 1 層、LeakyReLU(0.2)、trunk を属性で共有 |
| backbone | 小さい CNN、AlexNet | ImageNet 事前学習の ResNet-50 を fine-tune |
| optimizer | SGD momentum 0.9、lr を 0.01 から下げる | AdamW lr 1e-4 一定、weight decay 0.01（adversary にも掛かる） |
| backbone 側の倍率 | λ_p：0→1 | w · s(p)：既定 0→0.1、`_dann` 0→1 |
| バッチ | source と target を半々 | 自然抽出 128 件 |
| データ拡張 | 平均を引くだけ | あり（`train_transforms_medfair`） |
| クラス重み | なし | あり（`weighting=inverse`） |
| モデルの選び方 | 深層実験は記載なし | best val AUROC（と last） |
| 評価 | target の accuracy | しきい値 0.5 の群別 TPR・FPR |

原論文の adversary の形に寄せるなら `'model.loss_fn.hidden_dim=[1024,1024]'`、属性ごとに別の trunk にするなら
`model.loss_fn.shared_trunk=false`、s を一定にするなら `model.loss_fn.gradient_schedule=null` を渡す。
対象の属性は `adversarial_attribute_names` で絞れる。HAM10000 には race が無いので、CheXpert 用の model config をそのまま使うと止まる。

### 起動例

```bash
uv run python -m projects.hypernet_e2e.run experiment=resnet_chexpert_attribute_invariant_dann study=<slug> seed=42
uv run python -m projects.hypernet_e2e.run experiment=resnet_ham10000_fg_dann_sex_age study=<slug> seed=42
```

2 段学習の 2 段目（凍結した属性不変 backbone に spatial LoRA を載せる）は `spatial_lora_chexpert_from_attribute_invariant` で、
1 段目の checkpoint を `model.backbone_checkpoint_path=<run>/checkpoints/<名前>.ckpt` で渡す。
