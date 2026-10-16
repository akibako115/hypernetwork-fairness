# cohort ごとの切片の補正

## 条件

- split: test。動作点の閾値だけ val で決める
- 対象: iterative 4 run（変調範囲 `fc` / `stage4+fc` × GroupDRO step size 1e-3 / 1e-2）の選択 checkpoint と baseline（ResNet ERM）
- seed: 42 のみ（各条件 1 run）
- cohort: 選択 checkpoint が属する stage02 の cohort（`cohort02`）。cohort は、モデルの入力と同じ metadata（sex・race・ethnicity・frontal/lateral・AP/PA・age と、それぞれの欠損フラグ）を metadata encoder に通し、その出力を k-means で 10 個に割ったもの
- 補正: margin（陽性と陰性の logit の差）から、cohort ごとの定数を引く。cohort 内の順位は変えない
  - `class-weight offset`: stage02 の class weight から `log(w_pos / w_neg)`
  - `val-fitted offset`: val で cohort ごとに切片だけを当てはめたもの
  - `shuffled offset`: 対照。`class-weight offset` を cohort 間で入れ替えたもの
- 属性群: [groups.py](../groups.py) の切り方（age group 65 歳・sex・race を White / Asian / Black に絞り、3 属性が非欠損の行）
- 動作点: 補正後の score で、val 全体の感度が 0.9 になる閾値。deploy で属性を見ずに 1 つの閾値を使う想定

notebook は [`cohort_logit_correction.ipynb`](../cohort_logit_correction.ipynb)。

- 数値の正本: `results/cohort_offset_ranking.csv`、`results/cohort_offset_baseline_control.csv`、`results/cohort_offset_attribute_metrics.csv`
- 図: `figures/cohort_offset_fitted_vs_class_weight.png`

## 背景

class weight 付きの cross-entropy を最小化する logit は、真の log odds に `log(w_pos / w_neg)` を足したものになる。
iterative の stage は cohort × class の class weight（`weighting=inverse`）を使うので、この定数が cohort ごとに違う。
cohort ごとに score の水準がずれると、cohort の中の順位は変わらないまま、cohort をまたいだ順位（pooled AUROC）が崩れる。
cohort はモデルの入力から決まるので、モデルはこのずれを容易に学習できる。

## 観察事実

### 学習された切片のずれ

| 条件 | class weight からの予測の幅 | val で当てはめた切片の幅 | 相関 | 傾き |
|---|---:|---:|---:|---:|
| fc / 1e-3 | 2.01 | 1.37 | 0.962 | 0.56 |
| fc / 1e-2 | 2.01 | 1.07 | 0.971 | 0.54 |
| stage4+fc / 1e-3 | 2.08 | 1.50 | 0.969 | 0.69 |
| stage4+fc / 1e-2 | 2.21 | 1.36 | 0.966 | 0.60 |

- モデルが学習した cohort ごとの切片は、class weight からの予測と強く相関する。大きさは予測の 5〜7 割である。

### pooled AUROC と cohort 内の AUROC（test）

baseline の pooled AUROC は 0.8597。

| 条件 | 補正なし | class weight どおり | val で当てはめ | シャッフル | cohort 内（iterative / baseline） |
|---|---:|---:|---:|---:|---:|
| fc / 1e-3 | 0.8459 | 0.8525 | 0.8540 | 0.8364 | 0.8153 / 0.8210 |
| fc / 1e-2 | 0.8468 | 0.8518 | 0.8536 | 0.8438 | 0.8155 / 0.8206 |
| stage4+fc / 1e-3 | 0.8479 | 0.8574 | 0.8583 | 0.8172 | 0.8230 / 0.8220 |
| stage4+fc / 1e-2 | 0.8468 | 0.8537 | 0.8551 | 0.8318 | 0.8183 / 0.8231 |

- baseline との差 0.012〜0.014 のうち、class weight どおりの補正で 39〜81%、val で当てはめた補正で 53〜89% が戻る。
- シャッフルした補正では、逆に 0.003〜0.031 下がる。
- cohort 内の AUROC は、同じ cohort で切った baseline とほぼ同じである（差は −0.006〜+0.001）。

### baseline に同じ補正をかけた対照

`val-fitted offset` は、class weight 由来に限らず cohort ごとの較正のずれを何でも吸収する。baseline の予測を各 iterative run の
cohort で切り、同じ手順で補正した（`results/cohort_offset_baseline_control.csv`）。

| cohort | baseline pooled（補正なし → 補正後） | 当てはめた切片の幅（baseline / iterative） |
|---|---|---|
| fc / 1e-3 | 0.8597 → 0.8594 | 0.39 / 1.37 |
| fc / 1e-2 | 0.8597 → 0.8597 | 0.39 / 1.07 |
| stage4+fc / 1e-3 | 0.8597 → 0.8600 | 0.46 / 1.50 |
| stage4+fc / 1e-2 | 0.8597 → 0.8601 | 0.51 / 1.36 |

- baseline は補正しても pooled AUROC が ±0.0004 しか動かない。cohort ごとの切片のずれは、iterative の stage に入ってから生じている。

### 属性群（test）

| 条件 | 補正 | worst AUROC（age / sex / race） | TPR gap（age / sex / race） | FPR gap（age / sex / race） |
|---|---|---|---|---|
| ResNet ERM | なし | 0.808 / 0.859 / 0.832 | 0.141 / 0.033 / 0.045 | 0.114 / 0.016 / 0.048 |
| fc / 1e-3 | なし | 0.801 / 0.850 / 0.823 | 0.104 / 0.035 / 0.049 | 0.058 / 0.012 / 0.062 |
| fc / 1e-3 | class weight どおり | 0.804 / 0.856 / 0.838 | 0.158 / 0.018 / 0.053 | 0.176 / 0.032 / 0.098 |
| fc / 1e-2 | なし | 0.798 / 0.847 / 0.820 | 0.104 / 0.022 / 0.030 | 0.071 / 0.003 / 0.052 |
| fc / 1e-2 | class weight どおり | 0.803 / 0.854 / 0.835 | 0.155 / 0.023 / 0.049 | 0.183 / 0.011 / 0.071 |
| stage4+fc / 1e-3 | なし | 0.802 / 0.849 / 0.825 | 0.094 / 0.007 / 0.032 | 0.038 / 0.016 / 0.035 |
| stage4+fc / 1e-3 | class weight どおり | 0.806 / 0.859 / 0.836 | 0.163 / 0.013 / 0.044 | 0.159 / 0.000 / 0.067 |
| stage4+fc / 1e-2 | なし | 0.798 / 0.850 / 0.820 | 0.116 / 0.019 / 0.026 | 0.045 / 0.022 / 0.078 |
| stage4+fc / 1e-2 | class weight どおり | 0.800 / 0.856 / 0.829 | 0.162 / 0.026 / 0.049 | 0.174 / 0.023 / 0.107 |

- 補正すると、sex と race の worst AUROC は baseline の水準に近づく。age はほぼ動かない（0.798〜0.802 → 0.800〜0.806、baseline 0.808）。
- 補正前の age の TPR gap・FPR gap は、baseline より小さい。補正すると baseline を超えて開く。
- race の FPR gap は、補正すると 0.035〜0.078 → 0.067〜0.107 に開く。

## 解釈候補

- **iterative で global AUROC が下がったのは、cohort ごとの class weight が cohort ごとの切片をずらしたためである。** cohort の中の順位は baseline とほぼ変わらず、崩れたのは cohort をまたいだ順位である。GroupDRO の `q` がほぼ一様な step size 1e-3 でも同じだけ下がることとも合う。
- **動作点での age の公平性の改善は、同じ切片のずれの裏側である。** 65 歳以上の cohort は陽性の重みが大きく（10〜12）、その cohort の score が持ち上がる。1 つの閾値で見ると、age ごとに閾値をずらしたのと同じ効果になる。補正して pooled AUROC を戻すと、age の群間差は baseline より開く。
- **この設定の GroupDRO は、順位（AUROC）を改善していない。** 改善したように見えた age の動作点の公平性は、class weight による暗黙の群別閾値である。
- **hidden cohort は、モデルの入力そのものを k-means で割ったものである。** cohort が sex × 撮影方向 × age（と race の欠損）で割れるのは、metadata の埋め込みを割っているためで、画像から未知の群を見つけているわけではない。

## 追加確認

- 群ごとの loss を、切片のずれに影響されない順位の loss（群内の陽性と陰性の組の loss）に置き換えた GroupDRO を回す。全体の較正は、全体共通の class weight の cross-entropy で保つ。
- checkpoint を val の worst-group AUROC で選ぶ。
- 動作点の公平性を目的に含めるなら、それは学習ではなく、閾値の決め方の問題として別に扱う。
- 補正後も fc で残る差（0.006〜0.008）が、cohort 内の順位の低下によるものかは、seed を足して確かめる。
