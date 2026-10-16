# 全体性能の比較

## 条件

- split: test
- checkpoint: 各 run の validation AUROC 最大 checkpoint
- seed: 43 のみ（各 λ 1 run）
- 比較した λ: 0（baseline）、0.1、1、3、10
- 指標: AUROC、balanced accuracy、recall_0、precision_0、F1

数値の正本は [`results/overall_performance_test.csv`](../results/overall_performance_test.csv) とする。

## 観察事実

| λ | AUROC | BAcc | recall_0 | precision_0 | F1 |
|---:|---:|---:|---:|---:|---:|
| 0 | 0.8630 | 0.7962 | 0.7617 | 0.9758 | 0.4193 |
| 0.1 | 0.8597 | 0.7919 | 0.7928 | 0.9714 | 0.4340 |
| 1 | 0.8594 | 0.7941 | 0.7686 | 0.9744 | 0.4214 |
| 3 | 0.8592 | 0.7920 | 0.7743 | 0.9733 | 0.4230 |
| 10 | 0.8564 | 0.7848 | 0.7152 | 0.9777 | 0.3882 |

- λ=10 は baseline と比べて AUROC が -0.0066、BAcc が -0.0115、recall_0 が -0.0466、F1 が -0.0312。
- λ=10 の precision_0 は 0.9758 から 0.9777 に上がったが、recall_0 は最も低い。
- λ=0.1 は baseline より AUROC と BAcc が低い一方、recall_0 と F1 は高い。
- λ=1 と λ=3 は、今回の指標では λ=0.1 と大きく異なる挙動を示さない。

## 解釈候補

- λ=10 では adversary の勾配が分類性能を壊し始めている可能性がある。collapse の上限候補として扱う。
- λ=0.1〜3 の F1 や recall_0 の差は、seed 1 本では手法の効果とは確定できない。
- AUROC と hard prediction 指標の動きが一致しないため、閾値付近の予測分布も影響している可能性がある。

## 追加確認

- 公平性指標を同じ test cache から算出し、λ=10 の性能低下が gap 改善の代償になっているか確認する。
- 属性 probe で表現から属性が消えたか確認する。全体性能だけでは adversary が機能したか判断できない。
- 水準を絞った後、少なくとも複数 seed で再実行する。現状は n=1 の探索結果であり、手法比較には使わない。
