# GroupDRO（Sagawa et al.）：原論文・実装・使い方

2026-10-14 作成、2026-10-16 更新（HEAD `0f2af99` のコードと照合）。原論文の本文と公式実装を読み直し、この repo の実装と
突き合わせた結果をまとめる。GroupDRO の run を設計・起動・評価するセッションは、先にこれを読む。

## 先に押さえること

- **原論文の群は「属性 × ラベル」**（m = |A| × |Y|）。この repo の preset は属性だけの群（例：年齢 × race の 8 群、
  dw8）で、陽性と陰性を 1 つの群損失に混ぜている。原論文の形とは別の手法として読む。
- **原論文は強い正則化（強い ℓ2 か early stopping）が前提**で、それが論文の主張そのもの
  （副題 *On the Importance of Regularization for Worst-Case Generalization*）。この repo の preset は満たしていない。
- **checkpoint は val の最悪群の成績で選ぶ**（原論文の early stopping）。best val AUROC ではない。
  2026-10-14 から GroupDRO の strategy は全 epoch の重みを残す。それより前の run は best val AUROC と last しか無い。
- 属性だけの群では、**q は陽性率の高い群（CheXpert では若年群）に集まる**。陽性のクラス重みが陰性の約 8.7 倍なので、
  陽性率の高い群ほどクラス重み付きの群損失が大きくなるため（下の「q が向く群」）。最悪 AUROC・最小 TPR の群
  （高齢群）には向かない。公平性損失の代表として結論に使わない（ユーザー方針）。
- **q の更新は公式実装と同じ**（2026-10-16 に照合）。原論文と違うのは、上の群の切り方・クラス重み・抽出・
  checkpoint の選び方で、どれも意図して変えたもの（下の「照合結果」）。
- **GroupDRO の実装は `objectives/group.py` の 1 つだけ。** 以前あった 2 つ目（`SamplingCorrectedGroupTaskLoss` の
  `mode=dro`、strategy `sampling_corrected_group_dro`）は 1 step 前の q を使っていて数値が一致しなかったため、
  `0f2af99` で削除した。残っている `sampling_corrected_uniform_group` は群を等しく重み付けする別の損失
  （[uniform-group.md](uniform-group.md)）で、GroupDRO ではない。

## 原論文と公式実装

- Sagawa, Koh, Hashimoto, Liang. *Distributionally Robust Neural Networks for Group Shifts: On the Importance of
  Regularization for Worst-Case Generalization.* ICLR 2020（[arXiv:1911.08731](https://arxiv.org/abs/1911.08731)）
- 公式実装：[kohpangwei/group_DRO](https://github.com/kohpangwei/group_DRO)（`loss.py`、`run_expt.py`、`data/dro_dataset.py`）

### 原論文の条件（本文と公式実装で確認済み）

| 項目 | 原論文・公式実装 | 出典 |
| --- | --- | --- |
| 目的 | 群の損失の最大値を最小化。評価は最悪群 accuracy | 2 節 |
| 群 | **属性 × ラベル**（m = \|A\| × \|Y\|、主実験は 4 か 6 群）。Waterbirds は鳥の種類 × 背景、CelebA は髪色 × 性別 | 2 節 |
| アルゴリズム | オンライン。q を指数勾配で上げ（q_g ← q_g exp(η_q ℓ)）、全群で正規化し、q で重み付けした損失で θ を更新 | Algorithm 1 |
| η_q | 本文には記載なし。公式 `--robust_step_size` の既定は 0.01 | `run_expt.py` |
| 抽出 | Algorithm 1 は群を一様に引く。公式の例のコマンドは `--reweight_groups`（群の頻度の逆数で重み付けした `WeightedRandomSampler`、train だけ） | Algorithm 1、`README.md`、`data/dro_dataset.py` |
| 正則化 | **強い ℓ2**（Waterbirds λ = 1.0、CelebA λ = 0.1。lr 1e-5）**か early stopping**（ResNet50 は 1 epoch）。どちらも無いと、全群の train 損失が 0 に近づいて DRO が ERM と差を持たない | 3.2 節、付録 C.2 |
| group adjustment | 群の損失に C/√n_g を足す（C ∈ {0,…,5} を探索）。小さい群の汎化ギャップを見込む | 3.3 節 |
| モデルと optimizer | ImageNet 事前学習の ResNet50。SGD（momentum 0.9）、batch 128、固定 lr、データ拡張なし | 付録 C.2 |
| checkpoint | 「best early stopping epoch（robust validation accuracy で測る）」。val の最悪群 accuracy が最大の epoch | 付録 C.2、表 3 |
| ベンチマーク（表 3） | ℓ2 の強さ・epoch 数・C を grid search し、val 最悪群 accuracy が最大のモデルを報告 | 4 節 |

## この repo の実装

`projects/hypernet_e2e/objectives/group.py`（共通部分は `_ClassWeightedGroupDROTaskLoss`、`group.py:106-147`）。

| class | strategy（`configs/training_strategy/`） | クラス重み |
| --- | --- | --- |
| `GlobalClassWeightedGroupDROTaskLoss` | `group_dro_global` | 全群共通の `[num_classes]`（inverse）。preset の既定 |
| `SubgroupClassWeightedGroupDROTaskLoss` | `group_dro_subgroup` | 群ごとの `[num_groups, num_classes]` |

- **q の更新**（`group.py:75-82`、`:123-135`）：train の forward ごとに、公式と同じく先に
  `q *= exp(step_size * group_loss); q /= q.sum()` で q を更新し、更新後の q で群損失を重み付けして返す。
  バッチに現れない群の損失は 0 として扱い、全群で正規化する（公式と同じ。2026-10-13 にいったん入れた独自補正は撤回済み）。
  `step_size` の既定は 0.01（公式の `--robust_step_size` の既定と同じ）。
- **群損失**（`group.py:55-72`）：クラス重み付き CE の群内の単純平均（群内のクラス重みの和では割らない）。
  クラス重みはこの repo の追加で、原論文には無い。
  - `group_dro_global`：`[num_classes]` の逆頻度を平均 1 に正規化したもの（`training.py:284-296`）。
  - `group_dro_subgroup`：群ごとに `n_g / (C · n_{g,c})`（`training.py:249-266`）。群の中で陽性と陰性が半分ずつ効くので、
    群損失は「陽性の平均 CE と陰性の平均 CE の平均」になり、群の陽性率によらない。
  - どちらも `weighting=inverse` が必須（`training.py:220-224`）。
- **抽出**：preset（`*_group_dro_race_known`）は `train_sampling: uniform`（自然抽出）。
  B = 128 では、少数群がバッチに 1 件も入らないことがある（≥65 × Black は約 10% のバッチで欠ける）。
- **checkpoint**：2026-10-14 から `group_dro_global`・`group_dro_subgroup` の strategy が `save_top_k: -1`・
  `save_weights_only: true` を持ち、全 epoch を残す。それより前の run は
  `configs/callbacks/model_checkpoint.yaml` の既定どおり best val AUROC の 1 つと last だけ。
- **診断値**：`metrics.csv` の `train/group_dro/{metric}/g{id}`（epoch ごとの q と群損失）、
  群番号とラベルの対応は `metrics/group_dro_groups.json`。

### 原論文との違い

| 項目 | 原論文 | この repo の preset |
| --- | --- | --- |
| 群 | 属性 × ラベル | 属性だけ（陽性と陰性が同じ群損失に入る） |
| 抽出 | 群を均等に（`--reweight_groups`） | 自然抽出 |
| 正則化 | 強い ℓ2（0.1〜1.0）か 1 epoch の early stopping | AdamW の weight decay 0.01（decoupled）、30 epoch |
| optimizer | SGD momentum 0.9、固定 lr | AdamW lr 1e-4 |
| データ拡張 | なし | あり（`train_transforms_chexpert`） |
| group adjustment | C を探索 | なし（C = 0 相当） |
| checkpoint | val の最悪群 accuracy が最大の epoch | 2026-10-14 以降は全 epoch を残し、分析側で選ぶ（選び方は未決）。それより前は best val AUROC と last だけ |
| 群損失の大きさ | 群の平均 CE（クラス重みなし） | global のクラス重みでは陽性率の高い群ほど大きくなる（下の照合結果） |
| 複数 GPU | 単一 GPU | q を GPU 間で揃える処理は無い。preset はどれも `devices: 1` |
| 評価 | 最悪群 accuracy | 固定しきい値 0.5 の群別 TPR・FPR（research-premises） |

群を属性 × ラベルにすると、最悪群 accuracy は「各属性群の TPR と TNR の最小」になる。
CheXpert では TNR = 1 − FPR なので、research-premises の主指標（最悪 FPR・最小 TPR）とそのまま対応する。
### q が向く群（2026-10-14 に確認）

grid の年齢 × race 8 群の GroupDRO 15 run（5 構成 × seed 42〜44、`analysis/hypernet_conditioning_grid/runs/`）で、
最終 epoch の `train/group_dro/q_mean` を見た。

- q の合計は 15 run とも **<65 の 4 群でほぼ 1**（≥65 の 4 群は 15 run 平均で計 0.008）。最大は 12 run で <65 × White、3 run で <65 × Others。
- train の陽性率は <65 が 0.135〜0.159、≥65 が 0.053〜0.064。q と陽性率の相関は 0.36〜0.77。
- ERM（Spatial LoRA、seed 42）の val では、<65 は FPR が高く（0.34〜0.37）、TPR も高い（0.92〜0.93）。≥65 は FPR が低く
  （0.23〜0.26）、TPR と AUROC が低い（TPR 0.58〜0.76、AUROC 0.77〜0.83）。
- したがって q は「FPR の高い側」には向いているが、群ごとの最悪 FPR の群を選んでいるわけではない。
  最小 TPR・最悪 AUROC の側（高齢群）には向かない。陽性率で決まる損失の大小を追っているだけと読める。

## 起動

原論文に揃えるときに変える override の候補（どこまで揃えるかはユーザーと決める）：

| 揃える点 | override の例 |
| --- | --- |
| 学習の長さ（early stopping 型） | `trainer.max_epochs=<ERM の best val epoch 前後>` |
| 強い ℓ2 型 | `model.optimizer.weight_decay=<値>`（AdamW の decoupled weight decay は ℓ2 penalty と同じではない点に注意） |
| 属性 × ラベルの群・群を均等に引く | 構造が変わるので preset を足す（群 × クラスのセルを群にし、`GroupClassImageDataModule` を使う） |

## checkpoint の選び方と予測の使い方

- 原論文どおり：**学習に使った群の val 最悪群の成績が最大の epoch**。原論文は accuracy（属性 × ラベルの群）。
  属性だけの群で回した run を評価するときに、何を最悪群の成績とするか（最悪 FPR、最小 bacc など）は未決。決めてから選ぶ。
- 2026-10-14 より前の run は、原論文どおりには選べない（best val AUROC の 1 つと last しか無い）。
  `metrics.csv` の val 指標の epoch 推移で、選ばれた epoch と最悪群が最良の epoch がどれだけ離れているかは推測できる。
  `analysis/hypernet_conditioning_grid/README.md` の「GroupDRO は best（val AUROC）と last だけ」は、この grid の run
  （2026-10-12 学習）については正しい。今の strategy は全 epoch を残すので、新しく回す run には当てはまらない。
- 予測は EO rate と同じく `analysis/common/predictions.py`。全 epoch を残した run では `--checkpoint` を必ず渡す。
- q の推移は、`metrics.csv` の `train/group_dro/q/g*` で、最悪群（val で FPR が最大の群）に寄ったかを確かめる。

## 2026-10-16 の照合結果

原論文の Algorithm 1 と公式実装（`loss.py` の `compute_robust_loss`・`compute_group_avg`、`run_expt.py`）を読み、
HEAD `0f2af99` の `objectives/group.py` と突き合わせた。

### 公式と同じ（確認済み）

- **q の更新順序と式**：バッチの群損失で先に q を `q · exp(η · ℓ_g)` と更新し、全群の和で正規化し、更新後の q で
  重み付けした損失を返す。公式の `compute_robust_loss` と同じ順序。
- **バッチに現れない群**：群損失を 0 とする（公式の `compute_group_avg` も件数 0 の群は 0）。exp(0) = 1 で q は
  そのままだが、他の群が上がって正規化されるので、相対的に下がる。公式と同じ挙動。
- **η の既定**：0.01。

### 意図して変えたこと（確認済み）

- 群は属性だけ（原論文は属性 × ラベル）。
- クラス重みを足した。
- 自然抽出（原論文の例は群を均等に引く `--reweight_groups`）。
- group adjustment（C/√n_g）は無い（C = 0 相当）。
- checkpoint は学習側で選ばず、全 epoch を残して分析側に任せる。

### 注意すること

- **global のクラス重みでは、q は群の陽性率を追う。** 群損失を群内の件数で割るだけで、群内のクラス重みの和では
  割らない。陽性の重みが陰性より大きいので、予測の質が同じでも陽性率の高い群ほど群損失が大きくなる。照合時の計算例では、
  陽性・陰性とも 1 件あたりの CE が同じとき、陽性率 15% の群の群損失は 0.31、6% の群は 0.21（約 1.5 倍）。上の
  「q が向く群」で見た偏り（q が <65 の群に集まる）はこれで説明できる。subgroup のクラス重み
  （`group_dro_subgroup`）なら群損失が陽性率によらないので、この偏りは出ない。
- **複数 GPU への備えが無い。** q は GPU ごとに別々に更新され、揃える処理が無い。今の preset は単一 GPU なので
  問題にならないが、`trainer.devices` を増やすと GPU ごとに q がずれる。
- **2 つ目の実装は削除済み。** 2026-10-16 より前の文書や run に `sampling_corrected_group_dro`（`SamplingCorrectedGroupTaskLoss`
  の `mode=dro`）が出てきたら、それは今の GroupDRO とは別物。1 step 前の q で損失を重み付けしていた。
