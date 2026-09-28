# adversary head は属性を読めていたか

## 条件

- 対象: adversary loss の成分を記録して ws11 で回し直した 4 本（λ=0 / 1 / 3 / 10、seed 43）。[runs.md](../runs.md) の「adversary loss を記録した回し直し」
- 推移: `metrics/metrics.csv` の `val/attribute_adversary/{sex,race,age}`（30 epoch）
- 参照: 特徴を見ずに事前分布だけを答える予測器の loss。sex / race は train の観測行の頻度を答える cross-entropy、age は train の平均を答える MSE（標準化後）。いずれも属性が観測された val の行だけで計算する
- head の状態: λ=1 の選択 checkpoint（val AUROC 最大、epoch 10）の adversary の重みを、val の特徴量 cache に当てる
- 線形 probe: val を患者単位で半分に割り、片方で fit、もう片方で評価（logistic 回帰、class balanced）

notebook は [`adversary_loss_curves.ipynb`](../adversary_loss_curves.ipynb) と [`adversary_head_state.ipynb`](../adversary_head_state.ipynb)。
数値の正本は `results/adversary_loss_epochs.csv`、`results/adversary_loss_vs_prior.csv`、`results/adversary_head_sex_probe_val.csv`、
図は `figures/adversary_loss_epochs_val.png`。

## 観察事実

### adversary loss は事前分布の水準に張り付く

| λ | sex（事前分布 0.6737） | race（1.0786） | age（0.9813） |
|---:|---:|---:|---:|
| 0 | 0.6932 | 1.8416 | 0.9906 |
| 1 | 0.6745 | 1.0792 | 0.9579 |
| 3 | 0.6739 | 1.0800 | 0.9746 |
| 10 | 0.6739 | 1.0810 | 0.9755 |

最終 epoch の val loss。

- sex は λ=1 / 3 / 10 のそれぞれ epoch 4 / 1 / 3 から、事前分布との差が 0.005 未満に入り、最終 epoch まで出ない。
- race は epoch 21〜25 から同じ状態になる。
- age は epoch ごとに 0.93〜1.00 で揺れ、事前分布をわずかに下回る。R² に直すと 0.01〜0.05 程度。
- λ=1 から 10 まで、張り付く水準は変わらない。
- λ=0 は総損失が `task + 0·adv` なので head が学習されず、sex 0.693、race 1.84 に留まる。

### λ=1 の head は入力にある sex の情報を使っていない

| 見るもの | 値 |
|---|---:|
| trunk の unit 数 | 256 |
| val で一度も発火しない unit | 206 |
| 1 行あたりの平均発火 unit | 29.2 |
| sex head の AUROC | 0.4998 |
| sex head の P(sex=1) の範囲 | 0.385〜0.569 |
| sex head の cross-entropy | 0.6742 |

| 表現 | sex の線形 probe BAcc | AUROC |
|---|---:|---:|
| backbone の特徴量（adversary の入力、2048） | 0.856 | 0.935 |
| trunk の ReLU 前（256） | 0.783 | 0.862 |
| trunk の ReLU 後（256、head の入力） | 0.681 | 0.738 |

- trunk の 8 割は死んでいるが、残りの unit には sex が線形で読める情報が残っている。
- それでも sex head は並べ替えすらできていない（AUROC 0.50）。

## 解釈候補

- **adversary は λ によらず、学習の早い段階から機能していない可能性が高い。** head が事前分布だけを答えると、GRL が backbone に返す勾配は属性を消す方向の成分をほとんど持たない。λ=3 と λ=10 で probe の結果がほぼ同じだったこと（[attribute_probe.md](attribute_probe.md)）とも合う。
- 失敗の形は「trunk が全滅して head が bias しか使えない」ではない。「head の入力に sex の方向があるのに、head がそれを向いていない」である。backbone が GRL で head の今の重みの方向から sex を逃がし続け、head が追いつけない minimax の失敗が考えられる。死んだ unit は head が使える情報をさらに減らす（ReLU 前後で BAcc 0.783 → 0.681）。
- したがって、[subgroup_performance_and_gaps.md](subgroup_performance_and_gaps.md) で見えた λ による gap の変化を、adversary が属性を消した効果として読むことはできない。

## 追加確認

- head の学習率や更新回数を backbone とは別に上げる（head を k step 更新してから backbone を 1 step 更新する、など）。head が事前分布から離れられるかを見る。
- trunk の ReLU を LeakyReLU にするか、trunk の初期化を見直す。死んだ unit が 206 から減るか、head の AUROC が上がるかを見る。
- λ=3 / 10 と他の epoch の checkpoint で同じ確認をする。今回は λ=1 の epoch 10 だけであり、seed も 1 本である。
