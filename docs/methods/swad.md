# SWAD（Cha et al.）：原論文・実装・使い方

2026-10-16 作成。原論文の本文と公式実装（khanrc/swad）を読み、この repo の実装（HEAD `0f2af99`）と実 run に突き合わせた結果をまとめる。
SWAD の run を設計・起動・評価するセッションは、先にこれを読む。

## 先に押さえること

- **SWAD は「学習中の重みを平均したもの」を 1 つ作る手法で、checkpoint を選ばない。** 評価するのは
  `checkpoints/swad.ckpt` だけ。best val AUROC や last は SWAD の成果物ではない。
- **`analysis/common` の既定の選び方は `swad.ckpt` を選ばない。** `run_artifacts.selected_checkpoint` は
  `best_val_auroc_*.ckpt` を 1 つ拾うので、何も言わずに SWAD でない重みを評価してしまう。予測を作るときは
  `--checkpoint checkpoints/swad.ckpt` を必ず渡す（下の「checkpoint の選び方」）。
- **平均の単位が epoch で、原論文の 50〜500 step より粗い。** HAM10000 では 1 epoch ≈ 66 step。
- **これまでの 21 run は、どれも谷の終わり（過学習の始まり）を検出しなかった。** 平均の範囲はほぼ全 epoch（0〜3 から 28〜29）で、
  「過学習に入る前の重みだけを平均する」という SWAD の狙いは効いていない（下の「実 run での谷の判定」）。
- **原論文との違いは主に 3 点**：平均の単位、収束前の区間のさかのぼり方、谷の終わりで学習を止めないこと。
  谷の判定の規則そのものは公式実装と同じ。

用語：

- **区間（segment）**：重みを平均する単位。区間の中の step ごとの重みを平均したものを「区間の平均」と呼ぶ。
  原論文は評価の間隔（50〜500 step）、この repo は 1 epoch。
- **谷（loss valley）**：val 損失が下がりきってから上がり始めるまでの範囲。SWAD はこの範囲の重みだけを平均する。

## 原論文と公式実装

- Cha, Chun, Lee, Cho, Park, Lee, Park. *SWAD: Domain Generalization by Seeking Flat Minima.* NeurIPS 2021
  （[arXiv:2102.08604](https://arxiv.org/abs/2102.08604)）
- 公式実装：[khanrc/swad](https://github.com/khanrc/swad)（`domainbed/swad.py` の `LossValley`、`domainbed/trainer.py`、
  `domainbed/lib/swa_utils.py`、`domainbed/networks.py`、`domainbed/hparams_registry.py`、`config.yaml`、`README.md`）

### 原論文の条件（本文と公式実装で確認済み）

| 項目 | 原論文・公式実装 | 出典 |
| --- | --- | --- |
| 目的 | ドメイン汎化（学習に無いドメインでの accuracy）。平坦な最小点を探す | 1・3 節 |
| データ | DomainBed の PACS、VLCS、OfficeHome、TerraIncognita、DomainNet。1 ドメインを test に回し、残りの 20% を val にする | 4.1 節 |
| モデル | ImageNet 事前学習の ResNet-50 | 4.1 節、付録 B.1 |
| optimizer | Adam、lr 5e-5、weight decay 0、dropout 0 | 4.1 節、付録 B.1 |
| batch | ドメインあたり 32 枚（3 ドメインなら 96 枚） | 4.1 節、付録 B.1 |
| 学習の長さ | 5,000 iteration（DomainNet は 15,000） | 付録 B.1 |
| 評価の間隔（= 区間の長さ） | VLCS 50、DomainNet 500、それ以外 100 step。公式 README の実行例も同じ値を `--checkpoint_freq` で渡す（`datasets.py` の既定は 200・1000 で、論文の値とは違う） | 付録 B.1、`README.md` |
| 平均の取り方 | 毎 step の重みを平均する（dense）。区間ごとに平均を作り、谷の中の区間を等しい重みで足す | 3.2 節、`trainer.py`、`swad.py` |
| 監視する値 | val の素の CE（学習ドメインの val 部分の平均、`tr_outloss`） | `trainer.py`、`evaluator.py` |
| 谷の始まり（収束） | 直近 N_s 区間の最初が最小になったとき。N_s = 3 | 3.2 節、付録 B.4 |
| しきい値 | その N_s 区間の val 損失の平均 × r。r = 1.3（VLCS だけ 1.2）。公式の引数は `tolerance_ratio` = r − 1 = 0.3 | 4.1 節、付録 B.1、`config.yaml` |
| 谷の終わり | 直近 N_e 区間の最小がしきい値を超えたとき。N_e = 6。**ここで学習を止める** | 3.2 節、`trainer.py`（`dead_valley` で `break`） |
| 収束しなかったとき | 最後の区間の平均を返す | `swad.py` の `get_final_model` |
| BatchNorm | **学習中は統計を固定**（`freeze_bn: True` が既定で、BN 層を eval モードに保つ）。学習後の取り直し（`update_bn`、500 step）は `freeze_bn: False` のときだけ走る | 付録 B.1、`networks.py`、`hparams_registry.py`、`trainer.py` |
| ハイパーパラメータ探索 | N_s・N_e は PACS で決め全データセットに流用、r は {1.2, 1.3} から val で選ぶ | 付録 B.1 |
| checkpoint | 選ばない（平均した 1 つを評価する） | 4.1 節 |

## この repo の実装

`projects/hypernet_e2e/callbacks/swad.py` の `SWADCallback`。目的関数ではなく callback なので、`training_strategy` とは独立に
experiment の `callbacks.swad` で足す。

| 処理 | 場所 | 中身 |
| --- | --- | --- |
| 区間の平均 | `swad.py:64-71`、`:137-145` | 毎 step の `net` のパラメータを足し、epoch の終わりに step 数で割る。BN の統計（buffer）は平均しない（公式と同じ） |
| 収束の判定としきい値 | `swad.py:95-102` | 直近 3 区間の最初が最小なら収束。しきい値 = 3 区間の平均 × (1 + 0.3) |
| 収束前へのさかのぼり | `swad.py:103-111` | 収束した区間までの直近の区間を新しい順に見て、しきい値を超えた区間の次から収束した区間までを平均に足す |
| 谷の中の追加と終わり | `swad.py:113-118` | 直近 6 区間の最小がしきい値を超えたら終わり。そうでなければ 6 区間の最も古い 1 つを足す |
| 学習の終わり | `swad.py:120-135` | 谷が続いていれば、まだ足していない区間を、残りの最小がしきい値以下のあいだ古い順に足す（公式の `get_final_model` と同じ） |
| 収束しなかったとき | `swad.py:165-169` | 最後の区間の平均を使い、`swad.json` に `fallback_last_segment` を書く |
| BN の取り直し | `swad.py:147-160` | 平均した重みで train loader を 1 周流し、BN の統計を累積平均（`momentum=None`）で取り直す |
| 保存 | `swad.py:162-180` | `checkpoints/swad.ckpt`（`{"state_dict": {"net.…": …}}`）と `metrics/swad.json`（`converge_epoch`、`averaged_epochs`、`dead_valley`、`threshold`） |

- **監視する値**：`val/loss`。`module.py` の `validation_step` が計算する素の CE（クラス重みなし）で、train の目的関数が
  GroupDRO や ℓ_wst でも変わらない。公式の `tr_outloss` と同じ種類の値。
- **呼ばれる順序**：Lightning は epoch の終わりに val を回してから `on_train_epoch_end` を呼ぶので、その epoch の
  `val/loss` で判定している（照合で確認済み）。
- **区間の重み**：step 数に比例させる。HAM10000 では全 epoch が同じ step 数なので、等しい重みと同じ。
- **preset**（どれも HAM10000、`n_converge: 3`・`n_tolerance: 6`・`tolerance_ratio: 0.3`）：

| preset | 目的関数 | 群 |
| --- | --- | --- |
| `resnet_ham10000_fg_swad` | ERM | なし |
| `resnet_ham10000_fg_swadwst_lhcf` | ℓ_wst（[worst-cohort.md](worst-cohort.md)） | LHCF のコホート |
| `resnet_ham10000_fg_swadwst_sex_age` | ℓ_wst | 性別 × 年齢 |
| `resnet_ham10000_fg_swadwst_lhcfrand` | ℓ_wst | LHCF と同じ大きさの無作為な群 |
| `resnet_ham10000_fg_swadwst_label` | ℓ_wst | クラス（良性・悪性） |
| `spatial_lora_ham10000_fg_swadwst_lhcf` | ℓ_wst | LHCF のコホート（Spatial LoRA、条件は定数） |
| `spatial_lora_ham10000_fg_lhcfcond_swadwst_lhcf` | ℓ_wst | LHCF のコホート（Spatial LoRA、LHCF で条件付け） |

学習は AdamW lr 1e-4・weight decay 0.01・batch 128・30 epoch（`configs/model/resnet50.yaml`、`_base_ham10000.yaml`）。
SWAD の run も best val AUROC・最悪群 AUROC の checkpoint と last を残すが、これらは平均前の重み。

### 原論文との違い

| 項目 | 原論文・公式実装 | この repo |
| --- | --- | --- |
| 区間の長さ | 50〜500 step | 1 epoch（HAM10000 で約 66 step）。30 epoch で 30 区間（原論文は 5,000 / 100 = 50 区間） |
| 収束前へのさかのぼり | しきい値以下が続く区間のうち**最も古い 1 つを飛ばして**足す。収束した区間は後で**もう 1 回**足される（計 2 回） | 飛ばさず、収束した区間も 1 回だけ足す。docstring は「1 回だけ」にしか触れていない |
| 早く収束したとき | 直近 6 区間がそろう前に収束すると、収束した区間より後の区間もさかのぼりに入り、その後も同じ区間が何回か足される（コードを追って確認） | そろっている区間の数に合わせて切り、各区間を 1 回だけ足す |
| 谷の終わり | 学習を止める | 学習は続け、平均だけをやめる。last などはその後の重みになる |
| BatchNorm | 学習中は統計を固定し、学習後に取り直さない（既定） | 学習中も統計を更新する。学習後に train を 1 周流して取り直す（公式の `freeze_bn: False` の分岐に近い。公式は 500 step） |
| optimizer・長さ | Adam lr 5e-5、weight decay 0、5,000 step | AdamW lr 1e-4、weight decay 0.01、30 epoch（約 2,000 step） |
| 課題 | ドメイン汎化（未知ドメインの accuracy） | 群の公平性（最悪群 AUROC など）。群は使わない |
| r の選び方 | {1.2, 1.3} から val で選ぶ | 1.3 に固定 |

さかのぼりと 2 回足す点は、公式の `final_model` を収束区間のコピーで初期化しながら、最初の `update_parameters` が
それを上書きする（`n_averaged = 0` のとき代入になる）ことから来ている。意図した挙動かは分からない。この repo は
「谷の区間を 1 回ずつ足す」という原論文の Algorithm 1（付録 B.4）の形に揃えている。

## 起動

```bash
uv run python -m projects.hypernet_e2e.run experiment=resnet_ham10000_fg_swad seed=<seed> study=<slug>
```

- val を毎 epoch 回す必要がある（`trainer.check_val_every_n_epoch=1`、既定）。`val/loss` が無いと `KeyError` で止まる。
- 別の preset に足すときは、experiment の `callbacks.swad` に `_target_: projects.hypernet_e2e.callbacks.swad.SWADCallback` と
  3 つの値を書く。
- **`freeze_backbone: true` の run には使わない。** BN の取り直しは net の全 BN を train モードにして統計を書き換えるので、
  固定したはずの backbone の統計も変わる（コードから読める挙動。現行の SWAD preset はどれも `freeze_backbone: false`）。
- 谷の終わりを検出させたいなら、区間を細かくする（epoch 内で区切る）か、r を下げる。どちらも未実装で、試していない。

## checkpoint の選び方と予測の使い方

選ばない。`checkpoints/swad.ckpt` だけを評価する。val での選択に使わないので、val の予測は診断用にとどまる。

```bash
uv run python analysis/common/predictions.py --study <slug> --split test \
  --run-dir analysis/<slug>/runs/<run-id> --checkpoint checkpoints/swad.ckpt
```

- `swad.ckpt` は `state_dict` の key が `net.` 始まりなので、`predictions.py` の読み込み（`net.` を外して strict に読む）に
  そのまま通る。
- **`--checkpoint` を省くと `best_val_auroc_*.ckpt` が選ばれる。** `run_artifacts.selected_checkpoint`
  （`analysis/common/run_artifacts.py:135-160`）は `best_val_auroc_*.ckpt` が 1 つなら黙ってそれを返す。SWAD の run は
  best val AUROC の checkpoint を 1 つ持つので、失敗せずに平均前の重みを評価する。
- `run.json` の checkpoint の記録にも `swad.ckpt` は載らない（`docs/reviews/20261013_experiment_project_review.md` の R6）。
  ファイル名で指定するしかない。
- 旧分析（`analysis/archive/latent_bias_hypernet/fairness_eval.py:126-130`）は、run 名に `-swad` を含む run では
  どちらの選び方でも `swad.ckpt` の予測を使っていた。予測は `predict_ungrouped.py --swad` で作っていた。
- `metrics/swad.json` の `averaged_epochs` で、どの epoch を平均したかを必ず確かめる。

## 2026-10-16 の照合結果

公式実装（`swad.py`・`trainer.py`・`swa_utils.py`・`networks.py`・`hparams_registry.py`）と原論文を読み、HEAD `0f2af99` の
`callbacks/swad.py` と突き合わせた。`callbacks/swad.py` は `6a09df9`（10-13）で commit されてから変わっていない。
ただし下の 21 run（10-10〜10-11）は `b53c4f8` に未 commit の変更を載せた作業ツリーで学習しており、当時のコードが
今と同じかは記録からは確かめられない。s43 の run で、`val/loss` の推移から今の規則で計算し直したしきい値と平均範囲は
`swad.json` と一致した。

### 公式と同じ（確認済み）

- 収束の判定（直近 3 区間の最初が最小）、しきい値（その 3 区間の平均 × 1.3）、谷の終わり（直近 6 区間の最小が
  しきい値を超える）、学習の終わりの残りの足し方、収束しなかったときに最後の区間を使うこと。
- 監視する値が val の素の CE であること。
- 平均するのがパラメータだけで、BN の統計は平均しないこと。
- callback の順序：その epoch の val が終わってから `on_train_epoch_end` で判定している。

### 公式と違う（確認済み）

- 区間が 1 epoch（約 66 step）。原論文は 50〜500 step（PACS などは 100）。
- 収束前へのさかのぼりで、公式は最も古い区間を飛ばし、収束した区間を 2 回足す。この repo は飛ばさず 1 回だけ足す。
  docstring は後者（2 回足さない）にしか触れていない。
- 谷の終わりで学習を止めない。
- BN の扱い：公式の既定は学習中に統計を固定し、取り直さない。この repo は統計を更新しながら学習し、最後に train 1 周で
  取り直す。依頼時の想定（「公式は 500 step で取り直す」）は `freeze_bn: False` の分岐の話で、既定の経路ではない。
  BN を固定しない学習で平均した重みを使うなら取り直しは必要なので、この repo のやり方は筋が通っている。

### 実 run での谷の判定（確認済み）

`projects/hypernet_e2e/runs/` の SWAD 21 run（`fg-swad` 3、`fg-swadwst-*` ResNet 12、Spatial LoRA 6）の `metrics/swad.json` を見た。

- 21 run とも `dead_valley: false`。収束は epoch 0〜4、平均した範囲は epoch 0〜3 から 28〜29 まで。
- 例：`20261010T054745Z-resnet-ham10000-fg-swad-s43-3f4a`。val 損失は epoch 2 で 0.312、しきい値 0.447。その後 0.63 まで
  上がる epoch もあるが、どの 6 epoch の窓にもしきい値以下の epoch（後半でも 0.36〜0.40 台）が 1 つは入り、最小が
  しきい値を超えなかった。最後の epoch 29（0.534）だけが学習の終わりの処理で外れ、平均は epoch 0〜28。
- 推測（未検証）：epoch 単位の val 損失は揺れが大きく、6 区間の最小をとると上昇傾向が隠れる。区間を細かくすると
  原論文の 50 区間に近づくが、揺れはさらに大きくなるので、谷の終わりを検出できるようになるかは分からない。
  いまの SWAD の run は「ほぼ全 epoch の一様平均（SWA）＋ BN の取り直し」として読む。
