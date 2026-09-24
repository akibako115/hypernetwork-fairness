# 属性 probe

## 条件

- 表現: selected checkpoint の分類 head 直前 feature
- probe: hidden 256 の MLP（学習時 adversary と同じ容量）
- fit / select / eval: train / val / test
- 対象 run: λ=0、0.1、1、3、10、全て seed 43
- 対象属性: sex、race、age_group_65、age
- race: White / Asian / Black の3群
- probe の設定: `MAX_EPOCHS=12`、`PATIENCE=3`、`BATCH_SIZE=8192`

ethnicity / frontal_lateral の control probe は、今回の主仮説に直接対応する adversarial target を先に確認するため、まだ実行していない。

linear probe も実行した。結果は [`results/attribute_probe_linear_test.csv`](../results/attribute_probe_linear_test.csv) と
[`results/attribute_probe_linear_by_method_test.csv`](../results/attribute_probe_linear_by_method_test.csv) に保存した。

数値の正本は [`results/attribute_probe_test.csv`](../results/attribute_probe_test.csv) と
[`results/attribute_probe_by_method_test.csv`](../results/attribute_probe_by_method_test.csv) である。

## test の観察事実

| λ | sex BAcc | race BAcc | age_group BAcc | age R² | age MAE (years) |
|---:|---:|---:|---:|---:|---:|
| 0 | 0.8475 | 0.5390 | 0.7312 | 0.3566 | 11.47 |
| 0.1 | 0.8333 | 0.5021 | 0.7182 | 0.3891 | 11.13 |
| 1 | 0.8151 | 0.5401 | 0.7228 | 0.4050 | 10.93 |
| 3 | 0.7794 | 0.5398 | 0.7624 | 0.5357 | 9.61 |
| 10 | 0.7788 | 0.5208 | 0.7649 | 0.5329 | 9.63 |

chance は sex / age_group の BAcc が 0.5、race の BAcc が 0.333、age の R² が 0、MAE は fit 年齢の中央値を答える基準で 14.38 歳である。

- sex は λ=0 の BAcc 0.8475 から λ=3〜10 の約 0.779 まで低下し、属性情報が減った方向に見える。
- race は λ=0.5390 から λ=0.5021〜0.5401 へ動くが、chance 0.333 からは離れている。race が消えたとは言えない。
- age_group_65 は λ=0.1 で少し低下するが、λ=3〜10 では BAcc 0.762 前後となり、baseline より高い。
- age は adversary の対象だが、λ=3 で R² が 0.5357、MAE が 9.61 歳となり、baseline より読める。age_group の結果とも整合し、age 情報が消えたとは言えない。

## 解釈候補

- adversary strength を上げると、sex の表現情報は抑えられる一方、age / age_group は抑えられていない可能性がある。
- λ=3〜10 の age probe 改善は、分類性能 collapse と同時に起きているため、表現が単純に「属性不変」になったとは解釈できない。
- λ=0.1〜1 の race BAcc 低下は見えるが、race の multi-class chance と比較すると残存情報は大きい。

### linear probe の確認

- sex の AUROC は λ=0 の 0.8867 から λ=10 の 0.8248 へ低下した。
- age_group_65 の AUROC は λ=0 の 0.7841 から λ=10 の 0.8447 へ上昇した。
- age の R² は λ=0 の 0.3481 から λ=10 の 0.5155 へ上昇し、MAE は 11.54 歳から 9.85 歳へ低下した。
- MLP だけでなく linear probe でも age / age_group の残存情報が確認されるため、単なる MLP の容量依存とは考えにくい。

## 追加確認

- ethnicity / frontal_lateral を control probe として追加し、表現全体が痩せた結果ではないかを確認する。
- probe の epoch 上限を元の 40 epoch に戻した確認を、候補 λ（0、1、3、10）で行う。今回の結果は水準探索であり、seed 1 本なので手法比較には使わない。
