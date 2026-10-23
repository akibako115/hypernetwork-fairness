# SWAD

学習中の重みを、val 損失が谷にある間だけ毎 step 平均して 1 つのモデルにする手法。Cha et al., NeurIPS 2021。
公平性の手法ではなく、平坦な最小点を探して汎化を良くする手法である。この repo では experiment の `callbacks.swad` に
`SWADCallback` を足して使い、どの strategy とも組み合わせられる。

## 1. 手法のロジック

### 解く問題

原論文の課題は**ドメイン汎化**：いくつかのドメイン（例：写真・絵画・スケッチ）で学習し、学習に無いドメインで当てる。
損失の地形が鋭い最小点（少し重みを動かすと損失が大きく上がる点）は、ドメインが変わって地形が少しずれると成績が落ちやすい。
平坦な最小点ほど汎化しやすい、というのが出発点である。

重みの平均（SWA：学習の途中の重みを平均する）は平坦な最小点の中心に近づく方法として知られる。ただし普通の SWA は
数 epoch おきに重みを取るので平均する重みが少なく、小さいデータでは早く過学習に入るので、過学習した重みまで平均してしまう。

### SWAD の 2 つの工夫

- **密に取る（dense）**：毎 iteration の重みを平均に入れる。
- **過学習を避ける（overfit-aware）**：val 損失を見て、平均を始める点 t_s と終える点 t_e を決める。この間を**谷**と呼ぶ。

t_s と t_e は 3 つのハイパーパラメータで決める。Eval(i) を評価点 i の val 損失とする。

```
t_s：直近 N_s 点のうち最初の点が最小になった最初の点（損失がそれ以上下がらなくなった）
しきい値 l = r × (その N_s 点の val 損失の平均)
t_e：直近 N_e 点の最小が l を超えた点（損失が上がり続けている）
θ_SWAD = (1 / (t_e − t_s + 1)) Σ_{i = t_s..t_e} θ_i
```

N_s は収束の判定の窓、N_e は過学習の判定の窓、r はしきい値の余裕。谷の終わりを窓の最小で判定するのは、
val 損失の一時的な跳ね上がりで平均を打ち切らないためである。

### 学習の手順（原論文 Algorithm 1、付録 B.4）

一定の lr で学習しながら、評価点ごとに val 損失を測り、t_s が見つかるまでは収束の判定を、見つかった後は谷の終わりの判定を行う。
谷の終わりで学習を止め、t_s から t_e の重みの平均を返す。平均した 1 つのモデルを評価するので、checkpoint を選ぶ必要がない。

### この repo で使う理由

LHCF（Masroor et al., [worst-cohort.md](worst-cohort.md)）が、コホートを組み込む先の既存手法の 1 つに SWAD を挙げている。
SWAD 自体は群を使わないので、この repo では「ERM + SWAD」を群を使わない対照に、「最悪コホート損失 ℓ_wst + SWAD」を
LHCF の組み込み方の解釈として使う（後者は推測による解釈）。

## 2. 原論文の設定

- Cha, Chun, Lee, Cho, Park, Lee, Park. *SWAD: Domain Generalization by Seeking Flat Minima.* NeurIPS 2021
  （[arXiv:2102.08604](https://arxiv.org/abs/2102.08604)）
- 公式実装：[khanrc/swad](https://github.com/khanrc/swad)（`domainbed/swad.py` の `LossValley`、`domainbed/trainer.py`、
  `domainbed/hparams_registry.py`、`config.yaml`）

### データセットとタスク

DomainBed の 5 つのベンチマーク。どれも多クラスの画像分類で、群・敏感属性は無い。1 ドメインを test に回し、
残りのドメインで学習する。学習ドメインの 20% を val にする。

| データセット | 入力 | 予測ラベル（クラス数） | ドメイン数 |
| --- | --- | --- | --- |
| PACS（9,991 枚） | 画像（写真・絵画・漫画・スケッチ） | 物体（7 クラス） | 4 |
| VLCS（10,729 枚） | 画像 | 物体（5 クラス） | 4 |
| OfficeHome（15,588 枚） | 画像 | 物体（65 クラス） | 4 |
| TerraIncognita（24,788 枚） | カメラトラップ画像 | 動物（10 クラス） | 4 |
| DomainNet（586,575 枚） | 画像 | 物体（345 クラス） | 6 |

### ハイパーパラメータ

| 項目 | 値 |
| --- | --- |
| モデル | ImageNet 事前学習の ResNet-50。BatchNorm の統計は学習中固定（公式の `freeze_bn: True`） |
| optimizer | Adam、lr 5e-5（一定）、weight decay 0、dropout 0 |
| batch | ドメインあたり 32 枚 |
| 学習の長さ | 5,000 iteration（DomainNet は 15,000） |
| データ拡張 | 本文に記載なし。公式実装は DomainBed の既定（`data_augmentation: True`） |
| 評価の間隔 | VLCS 50、DomainNet 500、それ以外 100 iteration（付録 B.1。公式 README の実行例も `--checkpoint_freq` で同じ値） |
| N_s・N_e | 3・6。PACS で探索し、全データセットに流用 |
| r | VLCS 1.2、それ以外 1.3（{1.2, 1.3} から val で選ぶ）。公式の引数は `tolerance_ratio` = r − 1 |

公式実装は、評価の間隔ごとの区間の中で毎 step の重みを平均した「区間の平均」を作り、谷の判定はこの区間を単位に行う。
監視する値は学習ドメインの val の素の CE。収束しなかったときは最後の区間の平均を返す。
BatchNorm の取り直し（`update_bn`）は `freeze_bn: False` のときだけ走る。

### モデルの選び方と評価

SWAD は checkpoint を選ばない（谷の重みの平均を評価する）。lr などのハイパーパラメータは学習ドメインの val で選ぶ。
評価は未知ドメインの accuracy で、train / val の分け方を変えた 3 回の平均。

## 3. この repo の実装

### ファイルと設定

| 役割 | 場所 |
| --- | --- |
| 平均と谷の判定 | [callbacks/swad.py](../../projects/hypernet_e2e/callbacks/swad.py) の `SWADCallback`（`update`・`finalize`・`_update_bn`） |
| 設定 | experiment の `callbacks.swad`（`n_converge: 3`・`n_tolerance: 6`・`tolerance_ratio: 0.3`） |
| 監視する値 | [module.py](../../projects/hypernet_e2e/module.py) の `LitModule.validation_step` が出す `val/loss`（クラス重みなしの素の CE） |
| テスト | [tests/test_cohort_objectives.py](../../projects/hypernet_e2e/tests/test_cohort_objectives.py)（谷の始まりから終わりまでの平均、学習の終わりの残りの追加） |
| 代表的な preset | `resnet_ham10000_fg_swad`（ERM + SWAD）、`resnet_ham10000_fg_swadwst_lhcf`（ℓ_wst + SWAD、LHCF のコホート）、`spatial_lora_ham10000_fg_swadwst_lhcf`（Spatial LoRA） |

`resnet_ham10000_fg_swadwst_{sex_age,lhcfrand,label}` は ℓ_wst の群を性別 × 年齢・無作為な群・クラスに替えた対照
（[worst-cohort.md](worst-cohort.md)）。

### 処理の流れ

区間は 1 epoch。HAM10000（train 8,457 件、batch 128）では 1 区間 67 step、30 epoch で 30 区間になる。

**学習の各ステップ（`on_train_batch_end`）。** `net` のパラメータを float で足し込む。BatchNorm の統計（buffer）は足さない。

**epoch の終わり（`on_train_epoch_end`）。** Lightning はその epoch の val を回してから呼ぶので、その epoch の `val/loss` で判定する。

1. 足し込んだパラメータを step 数で割って区間の平均にし、`update` に渡す。
2. 収束前：直近 3 区間の最初が最小なら、その区間を収束点とし、しきい値を 3 区間の val 損失の平均 × 1.3 にする。
   収束点より前の区間も、しきい値以下が続いていた所までさかのぼって平均に足す。
3. 収束後：直近 6 区間の最も古い区間が収束点より後なら、6 区間の最小がしきい値を超えたかを見る。超えたら谷の終わりとし、
   以後は何もしない。超えなければ、その最も古い区間を平均に足す。各区間は 1 回だけ、step 数に比例した重みで足す。

**学習の終わり（`on_fit_end`）。**

1. `finalize`：谷が続いていれば、まだ足していない区間を、残りの最小がしきい値以下のあいだ古い順に足す（公式の `get_final_model` と同じ規則）。
2. 収束しなかったときは最後の区間の平均を使い、`swad.json` に `fallback_last_segment` を書く。
3. 平均した重みを `net` に書き込み、`_update_bn` で train loader を 1 周流して BatchNorm の統計を累積平均（`momentum=None`）で取り直す。
4. `checkpoints/swad.ckpt`（`{"state_dict": {"net.…": …}}`）と `metrics/swad.json`（`converge_epoch`・`averaged_epochs`・
   `dead_valley`・`threshold`）を書く。

**設計上の選択。** 学習中も BatchNorm の統計を更新しながら学習するので、平均した重みに合う統計を最後に取り直す。
谷の終わりでも学習は止めず、`max_epochs` まで回して他の checkpoint も残す。

**checkpoint と予測。** 評価するのは `swad.ckpt` だけで、val での選択には使わない。SWAD の run も best val AUROC・
最悪群 AUROC・last の checkpoint を残すが、これらは平均前の重みである。`run.json` の checkpoint の記録には `swad.ckpt` が載らないので、
予測を作るときは名前で渡す。`--checkpoint` を省くと既定の `best`（val AUROC の best）が選ばれ、失敗せずに平均前の重みを評価してしまう。

```bash
uv run python analysis/common/predictions.py --study <slug> --split test \
  --run-dir analysis/<slug>/runs/<run-id> --checkpoint swad
```

どの epoch を平均したかは `metrics/swad.json` の `averaged_epochs` で確かめる。

### 原論文との違い

| 項目 | 原論文・公式実装 | この repo |
| --- | --- | --- |
| 課題 | ドメイン汎化（未知ドメインの accuracy） | 群の公平性（HAM10000 の良性 / 悪性、2 クラス）。SWAD 自体は群を使わない |
| 区間の長さ | 50〜500 iteration（5,000 iteration で約 50 区間） | 1 epoch（67 step、30 区間） |
| 収束前へのさかのぼり | しきい値以下が続く区間のうち最も古い 1 つを飛ばして足す。収束した区間は谷の判定でもう 1 回足される | 飛ばさず、収束した区間も 1 回だけ足す（Algorithm 1 の形） |
| 早く収束したとき | 直近 6 区間がそろう前に収束すると、収束点より後の区間もさかのぼりに入り、同じ区間が何回か足される | そろっている区間の数に合わせて切り、各区間を 1 回だけ足す |
| 谷の終わり | 学習を止める | 学習は続け、平均だけをやめる |
| BatchNorm | 学習中は統計を固定し、取り直さない（既定） | 学習中も更新し、学習後に train を 1 周流して取り直す |
| optimizer・長さ | Adam lr 5e-5（一定）、weight decay 0、5,000 iteration | AdamW lr 1e-4、weight decay 0.01、30 epoch（約 2,000 step） |
| r | {1.2, 1.3} から val で選ぶ | 1.3 に固定 |

公式実装のさかのぼりの挙動は、`final_model` を収束した区間で初期化しながら、最初の `update_parameters` がそれを上書きする
ことから来ている。意図した挙動かは分からないので、この repo は「谷の区間を 1 回ずつ足す」Algorithm 1 の形に揃えた。
区間を原論文に近づけるには epoch の中で区切る実装が要る（未実装）。epoch 単位の val 損失は揺れが大きく、6 区間の最小では
上昇が隠れやすいので、谷の終わりが検出されず、ほぼ全 epoch の一様平均になることがある。

### 起動例

```bash
uv run python -m projects.hypernet_e2e.run experiment=resnet_ham10000_fg_swad study=<slug> seed=42
```

- val を毎 epoch 回す必要がある（`trainer.check_val_every_n_epoch=1`、既定）。`val/loss` が無いと `KeyError` で止まる。
- 別の preset に足すときは、experiment の `callbacks.swad` に `_target_: projects.hypernet_e2e.callbacks.swad.SWADCallback` と
  3 つの値を書く。
- `freeze_backbone: true` の run には使わない。`_update_bn` は `net` 全体を train モードにして統計を取り直すので、
  固定したはずの backbone の統計も変わる。
