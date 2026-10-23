# ALFR

表現から敏感属性を当てる adversary と、表現を作る側を、ミニバッチごとに交互に更新する手法。
Edwards & Storkey, ICLR 2016（ALFR は adversarial learned fair representations の略）。この repo では model config
`resnet50_attribute_invariant_alfr` で、strategy は `erm` のまま使う。adversary の作り・GRL・記録など DANN と共通の部分は
[dann.md](dann.md) に書き、ここには違う部分だけを書く。

## 1. 手法のロジック

### 解く問題

原論文の公平性は **統計的パリティ**（statistical parity）で、2 値の分類器 η の陽性予測率が敏感属性 S（2 値）で等しいこと。

```
P(η(X) = 1 | S = 1) = P(η(X) = 1 | S = 0)
```

敏感属性を入力から外すだけでは足りない。ほかの変数が S と相関していれば、S の情報が予測に漏れる。ALFR は、予測には役立ち、
S を当てられない **表現** R を学ぶ。表現に S の情報が無ければ、その上に載せたどんな分類器も S に依存しにくい。原論文は、
分類器の差別度（下の式 2）が表現の 2 群間の H-divergence（ある分類器の集合で 2 群をどれだけ見分けられるか）の半分で
上から抑えられることを示す。adversary で 2 群を見分けにくくすることは、この上限を下げることにあたる。

着想は DANN の先行版（Ganin & Lempitsky 2015）から来ていると原論文が書いている。DANN の「ドメイン」を「敏感属性」に置き換えた
形で、敵対的に変数を消すという考え方は [dann.md](dann.md) の 1 節と同じ。原論文も、S の値の違いを異なるドメインとみなせると書く。

### 目的関数

encoder（X から R を作る）、decoder（R から X を復元する）、predictor（R から Y を予測する）、adversary（R から S を当てる）の
4 つのネットワークを使う。θ は encoder・decoder・predictor の parameter、φ は adversary の parameter。

```
L(θ, φ) = α C + β D + γ E                                       （式 9）
  C = E‖X − Dec(R)‖²                  再構成の二乗誤差
  D = E[ S log Adv(R) + (1 − S) log(1 − Adv(R)) ]   adversary の対数尤度（log-loss の符号を反転したもの）
  E = log-loss(Y, Pred(R))            予測の誤差

min_θ max_φ L(θ, φ)                                              （式 10）
```

adversary は D を上げる（S をよく当てる）ように、表現側は L を下げる（予測がよく、S を当てにくく、復元できる）ように動く。
α・β・γ は 3 つの目的の釣り合い。表現を他のタスクに使い回さないなら α = 0 でよい、と原論文は書く。

### 学習の手順（原論文 Algorithm 1）

内側の max を毎回解ききるのは現実的でないので、GAN（Goodfellow et al. 2014）にならって交互に 1 歩ずつ動かす。

1. ミニバッチを 1 つ取り、L を計算する。
2. 今回が adversary の番なら φ だけを L が増える向きに 1 歩更新する。表現側の番なら θ だけを L が減る向きに 1 歩更新する。
3. 番を入れ替える。最初の番は adversary。

この「1 ミニバッチで片方だけを更新する」厳密な交互更新（strict alternation）が、同じミニバッチで両方を同時に更新する
DANN との違い。勾配の 1 歩は Adam などに置き換えてよい。

原論文は任意の工夫として、adversary の正解率が 90% を超えたら adversary を、60% を下回ったら表現側を更新しない、という
停止の規則も試している。adversary が強すぎると表現側への勾配が弱くなり、弱すぎると勾配が役に立たないため。
「改善することがあるが、さらに調べる必要がある」との書き方で、標準の手順には入っていない。

### 評価の指標

差別度（discrimination）は 2 群の陽性予測率の差の絶対値。

```
y_disc = | Σ_{i: s_i = 0} η(x_i) / N_0 − Σ_{i: s_i = 1} η(x_i) / N_1 |      （式 2）
y_t = y_acc − t · y_disc        （式 3。t ≥ 0 は accuracy と差別度の交換比）
```

## 2. 原論文の設定

- Edwards, Storkey. *Censoring Representations with an Adversary.* ICLR 2016（[arXiv:1511.05897](https://arxiv.org/abs/1511.05897)）
- 公式実装：論文に公開先の記載は無く、見つけていない。実装は theano と lasagne とだけ書かれている（5 節）。

### データセットとタスク

公平性の実験（5.1 節）はどちらも UCI の表形式データの 2 値分類で、敏感属性も 2 値。

| データセット | 入力 | 予測ラベル（クラス数） | 敏感属性（群数） |
| --- | --- | --- | --- |
| Adult（国勢調査） | 表形式、102 属性、45,222 件 | 年収が 5 万ドルを超えるか（2 クラス） | Gender（2） |
| Diabetes（米国の病院データ） | 表形式、235 属性、約 10 万件 | 再入院するか（2 クラス） | Race を isCaucasian の 2 値にしたもの（2） |

分割は Adult が train 35,000・val と test 約 5,000 ずつ、Diabetes が train 80,000・val と test 約 10,000 ずつ。分割を変えて 5 回繰り返す。
ほかに、顔写真に重ねた文字を消す画像の匿名化の実験（5.2 節。予測ラベルは無く、S は文字の有無）がある。

### ハイパーパラメータ

| 項目 | 値 |
| --- | --- |
| encoder・decoder | 1〜3 層（一様に選ぶ）、各層の幅 1〜100（全層同じ幅）、ReLU |
| adversary | 1〜3 層の ReLU。幅は記載なし（未確認） |
| predictor | encoder の中央の層に載せたロジスティック回帰 |
| optimizer | Adam、既定の parameter（lr の値は本文に無い。Adam 原論文の既定は 1e-3） |
| batch・epoch | 記載なし（未確認） |
| 正則化・データ拡張 | 記載なし（未確認） |
| α | 0.05 に固定 |
| β・γ | β ~ U[0, 50]、γ ~ U[0, 10] |
| 探索 | 上の分布から 100 通りを random search |

β/(β+γ) が大きいほど差別度が下がる（ある程度まで）と報告している（図 3）。

### モデルの選び方

t ∈ [0, 3] の値ごとに、val で y_t = accuracy − t × 差別度 が最大のモデルを 100 通りの中から選び、その test の y_t・accuracy・
差別度を報告する（5.1.2 節、図 1・2）。比較相手は LFR（Zemel et al. 2013）。

## 3. この repo の実装

### ファイルと設定

| 役割 | 場所 |
| --- | --- |
| 損失 | [objectives/attribute_invariance.py](../../projects/hypernet_e2e/objectives/attribute_invariance.py) の `AlternatingAttributeInvariantTaskLoss`（`AttributeInvariantTaskLoss` の子 class） |
| model config | [configs/model/resnet50_attribute_invariant_alfr.yaml](../../projects/hypernet_e2e/configs/model/resnet50_attribute_invariant_alfr.yaml) |
| 交互更新・optimizer | [module.py](../../projects/hypernet_e2e/module.py) の `LitModule.__init__`（`automatic_optimization` の切り替え）、`configure_optimizers`、`_alternating_training_step` |
| adversary・GRL・起動時の検査・クラス重み | DANN と共通（[dann.md](dann.md) の 3 節） |
| preset | `resnet_chexpert_attribute_invariant_alfr`（CheXpert、sex・race・age）。HAM10000 の preset は無い |

### 処理の流れ

**起動時（fit の前）。** DANN と同じ検査（`_validate_attribute_invariance`）とクラス重みの書き込み（`class_weight_path`）に加えて、
次が決まる。

- `LitModule.__init__`：loss が `requires_manual_optimization = True` を持つので、`automatic_optimization` を False にする。
  Lightning の自動の backward・step を切り、`_alternating_training_step` の中で自分で呼ぶ（manual optimization）。
- `configure_optimizers`：optimizer を 2 つ作る。表現側（net 全体と、adversary 以外の loss の parameter）用と、adversary 用。
  adversary 用は model config の `adversary_optimizer`（無ければ表現側と同じ factory）。scheduler と `adversary_lr_scale ≠ 1` は
  受け付けずに止める。adversary の lr は `adversary_optimizer.lr` で変える。

**学習の各ステップ（`_alternating_training_step`）。**

1. `_update_adversary_progress`：DANN と同じく GRL の倍率 s を決める。preset は `gradient_schedule: null` なので s = 1 の一定。
2. `_shared_step`：ミニバッチを train モードで 1 回だけ forward し、logits と表現を得る。
3. `global_step` が偶数なら adversary の番。`adversary_loss` が表現を detach（勾配を切る）してから属性ごとの adversary 損失の平均を
   計算し、adversary の optimizer だけを 1 歩進める。表現側の損失は記録のためだけに勾配なしで計算する。
4. 奇数なら表現側の番。adversary の parameter を一時的に `requires_grad=False` にし、`task + w · adversary` を GRL 越しに backward して、
   表現側の optimizer だけを 1 歩進める。backbone に届く勾配は ∂task − w·s·∂adv で、DANN と同じ式になる。

原論文の式 9 との対応は、再構成の項が無いので α = 0、β = w·s（preset では 1）、γ = 1。

**この形にした理由。**

- forward を 1 回にしているので、adversary と表現側が同じ BatchNorm の batch 統計で正規化した表現を見る。
- 1 ミニバッチで optimizer step は 1 回なので、`global_step`（Lightning は manual optimization ではすべての optimizer の step 回数を数える）は
  処理したミニバッチ数と等しい。交互の番も、λ schedule を使うときの進捗 p も、これで数える。
- 結果として、表現側が更新されるのはミニバッチの半分だけ。同じ epoch 数の ERM と比べると backbone の更新回数は半分になる。

**記録と checkpoint。**

- DANN と同じ項目（`train/task`、`train/attribute_adversary`、属性ごとの値、`train/adversary_scale`、val の adversary 損失）に加えて、
  `train/alfr/adversary_loss`（adversary の番での損失の epoch 平均）を `metrics.csv` に書く。これが「事前分布だけを答える予測器」の
  水準に張り付いていれば、adversary が属性を追えていない。
- checkpoint は既定の `model_checkpoint`（best val AUROC の 1 つと last）。原論文のように「val の性能 − t × 公平性の差」で epoch を
  選ぶには、`callbacks.model_checkpoint.save_top_k=-1` で全 epoch を残し、分析側で選ぶ。この repo の主指標で何を「公平性の差」と
  するかは未決。
- 予測と属性の probe は DANN と同じ（net だけを復元する。[dann.md](dann.md) の「記録と checkpoint」）。

### 原論文との違い

| 項目 | 原論文 | この repo（`resnet50_attribute_invariant_alfr`） |
| --- | --- | --- |
| 入力とタスク | 表形式データの 2 値分類 | 画像（胸部 X 線）の 2 値分類 |
| 再構成の項 | α = 0.05 | 無し（α = 0） |
| β・γ | β ~ U[0, 50]、γ ~ U[0, 10] を探索（β/(β+γ) は平均で約 0.78） | β = w·s = 1、γ = 1（β/(β+γ) = 0.5）。探索していない |
| 敏感属性 | 2 値 1 つ | sex（2）・race（6）・age（連続）の損失の平均 |
| adversary | 1〜3 層の ReLU、出力はシグモイド 1 つ | 256 の 1 層、LeakyReLU(0.2)、属性ごとの head |
| 表現 | 小さい MLP を最初から学習 | ImageNet 事前学習の ResNet-50 を fine-tune |
| optimizer | Adam 既定（weight decay なし） | 両側とも AdamW lr 1e-4、weight decay 0.01 |
| adversary の番での損失の重み | β 倍（Algorithm 1 の L） | 重みなしの平均。Adam は勾配の定数倍で歩幅がほぼ変わらないので、差は小さいと見込む（推測） |
| 停止の工夫（90% / 60%） | 任意で使う | 無し |
| モデルの選び方 | val の accuracy − t × 差別度 | best val AUROC（と last） |
| 評価 | accuracy と統計的パリティの差 | しきい値 0.5 の群別 TPR・FPR（[research-premises.md](../research-premises.md)） |

原論文の公平性は陽性予測率を群で揃えることで、この repo の主指標（群別の TPR・FPR の差）とは目標が違う。表現から属性を消しても、
TPR・FPR の差が縮むとは限らない。原論文の β の範囲に寄せるなら `model.loss_fn.attribute_adversary_weight=<β>`（s = 1 なので β = w）、
表現側の更新回数を ERM に揃えるなら `trainer.max_epochs` を ERM の 2 倍にする。HAM10000 で回すなら、
`resnet_ham10000_fg_dann_sex_age` と同じく `adversarial_attribute_names` を sex・age に指定し直した preset を作る。

### 起動例

```bash
uv run python -m projects.hypernet_e2e.run experiment=resnet_chexpert_attribute_invariant_alfr study=<slug> seed=42
```
