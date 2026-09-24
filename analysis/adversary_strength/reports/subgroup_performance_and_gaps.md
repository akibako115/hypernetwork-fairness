# subgroup 別性能と gap

## 条件と定義

- split: test
- subgroup: `age_group_65`、`sex`、`race`
- race: White / Asian / Black の3群に限定
- 指標: AUROC、BAcc、recall_0（TNR）、recall_1（TPR）
- `Eopp1`: recall_1 の max−min
- `Eopp0`: recall_0 の max−min（FPR の gap と同値）
- `Eodds`: `(Eopp0 + Eopp1) / 2`

全 subgroup の性能は [`results/subgroup_performance_test.csv`](../results/subgroup_performance_test.csv)、
gap は [`results/fairness_gaps_test.csv`](../results/fairness_gaps_test.csv) に保存した。

## 支持数

| 属性 | 最小 subgroup n | 最小 subgroup 陽性数 |
|---|---:|---:|
| age_group_65 | 9,998 | 547 |
| sex | 9,172 | 935 |
| race | 1,217 | 148 |

## gap の観察事実

| λ | 属性 | AUROC gap | BAcc gap | Eopp0 | Eopp1 | Eodds |
|---:|---|---:|---:|---:|---:|---:|
| 0 | age_group_65 | 0.0550 | 0.0506 | 0.1442 | 0.2453 | 0.1948 |
| 0.1 | age_group_65 | 0.0658 | 0.0729 | 0.1316 | 0.2774 | 0.2045 |
| 1 | age_group_65 | 0.0560 | 0.0569 | 0.0950 | 0.2088 | 0.1519 |
| 3 | age_group_65 | 0.0625 | 0.0518 | 0.0993 | 0.2030 | 0.1512 |
| 10 | age_group_65 | 0.0639 | 0.0385 | 0.0908 | 0.1678 | 0.1293 |
| 0 | race | 0.0330 | 0.0426 | 0.0552 | 0.0657 | 0.0604 |
| 0.1 | race | 0.0212 | 0.0199 | 0.0426 | 0.0330 | 0.0378 |
| 1 | race | 0.0189 | 0.0162 | 0.0438 | 0.0270 | 0.0354 |
| 3 | race | 0.0239 | 0.0414 | 0.0408 | 0.0420 | 0.0414 |
| 10 | race | 0.0252 | 0.0319 | 0.0470 | 0.0513 | 0.0491 |
| 0 | sex | 0.0154 | 0.0104 | 0.0020 | 0.0189 | 0.0104 |
| 0.1 | sex | 0.0151 | 0.0158 | 0.0033 | 0.0283 | 0.0158 |
| 1 | sex | 0.0194 | 0.0137 | 0.0169 | 0.0105 | 0.0137 |
| 3 | sex | 0.0187 | 0.0108 | 0.0039 | 0.0255 | 0.0108 |
| 10 | sex | 0.0104 | 0.0119 | 0.0004 | 0.0241 | 0.0122 |

## subgroup 性能の読み取り

- age group が一貫して最大の fairness gap を持つ。λ=1〜10 では baseline より Eodds が小さくなるが、λ=10 は全体 AUROC / BAcc / F1 の低下を伴う。
- λ=1 は age の Eopp0 を 0.1442 から 0.0950、Eopp1 を 0.2453 から 0.2088 に縮める。一方、age subgroup の AUROC gap は 0.0550 から 0.0560 とほぼ変わらない。
- race は λ=0.1〜1 で Eodds が 0.0604 から 0.0354〜0.0378 に縮むが、λ=10 では 0.0491 へ戻る。
- sex は baseline の Eodds が 0.0104 と小さく、λ による一方向の改善は見えない。
- これらは seed 43 の1本の水準探索であり、λ=1〜3 が最適だとはまだ確定できない。

## 次の確認

属性が実際に表現から消えたかを、同じ selected checkpoint に対する post-hoc attribute probe で確認する。特に age の gap 縮小が、属性情報の低下によるものか、単なる threshold / subgroup 性能の変化かを切り分ける。
