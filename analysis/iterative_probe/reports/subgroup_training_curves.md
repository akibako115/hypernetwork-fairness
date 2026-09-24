# 群ごとの epoch 推移（`q`・hidden cohort・属性群）

## 条件

- split: validation（学習中の記録が val しか無い）
- 対象: iterative 4 run（変調範囲 `fc` / `stage4+fc` × GroupDRO step size 1e-3 / 1e-2）。属性群の指標だけ baseline（ResNet ERM）の epoch 0〜11 を重ねる
- seed: 42 のみ（各条件 1 run）
- epoch: warmup 0–1、stage01 2–6、stage02 7–11 の通し番号。cohort は stage ごとに引き直すので、stage をまたいで cohort を対応させない。stage01 の cohort は、同じ変調範囲の 2 run で同じ集団になる
- 指標:
  - `q`: `train/group_dro/q_NN`（train epoch 終わり）
  - hidden cohort: `val/hidden_{auroc,bacc,loss}_NN` と、そこから後計算した min / max / gap（AUROC・bACC・loss）
  - 属性群: logger が記録した `val/<属性>/{worst_group_auroc,auroc_gap,worst_group_bacc,bacc_gap,Eopp0,Eopp1,Eodds}`。対象は sex・age group（65 歳）・race
- 注意:
  - race は学習側が 6 コードすべてを群にしており、README の評価軸（White / Asian / Black）と定義が違う。群ごとの値が記録されていないので、絞り直しはできない
  - 二値分類では `Eodds` = (`Eopp1` + `Eopp0`) / 2 で、独立な情報を持たない

notebook は [`subgroup_training_curves.ipynb`](../subgroup_training_curves.ipynb)。

- 数値の正本: `results/cohort_epoch_metrics.csv`、`results/cohort_q_picked.csv`、`results/attribute_fairness_epoch.csv`、`results/attribute_fairness_phase_mean.csv`
- 図: `figures/cohort_q_curves.png`、`hidden_cohort_curves.png`、`cohort_q_picked_curves.png`、`attribute_group_performance_curves.png`、`attribute_equalized_curves.png`

## 観察事実

### `q`

| 条件 | stage | max `q` | 上位 3 の合計 | entropy |
|---|---|---:|---:|---:|
| fc / 1e-3 | stage01 / 02 | 0.123 / 0.130 | 0.36 / 0.38 | 2.289 / 2.281 |
| fc / 1e-2 | stage01 / 02 | 0.324 / 0.354 | 0.73 / 0.83 | 1.732 / 1.596 |
| stage4+fc / 1e-3 | stage01 / 02 | 0.125 / 0.138 | 0.37 / 0.38 | 2.284 / 2.278 |
| stage4+fc / 1e-2 | stage01 / 02 | 0.259 / 0.354 | 0.71 / 0.78 | 1.675 / 1.735 |

一様なら max 0.1、上位 3 の合計 0.3、entropy 2.303。

- 1e-3 の `q` はほぼ一様のまま stage を終える。1e-2 は stage の終わりでも偏りが増え続けている。

### hidden cohort の worst と gap

- hidden 最小 AUROC は 0.76〜0.80 に留まる。stage の頭と終わりの差は −0.002〜+0.028 で、1e-3 の run でも 1e-2 と同じかそれ以上に動く。
- hidden bACC gap は stage4+fc で stage ごとに 0.027〜0.035 縮む。step size によらない。

### `q` の高い cohort と低い cohort の推移

stage ごとに、stage 内平均の `q` で上位 2 cohort と下位 2 cohort を選び、stage 内の推移を追った（図 `cohort_q_picked_curves.png`）。
stage 終わりの `q` で選んでも、4 cohort 中 3〜4 個は同じになる。

| 条件 | stage | AUROC の変化（上位 / 下位） | loss の変化（上位 / 下位） |
|---|---|---|---|
| fc / 1e-3 | stage01 | −0.012 / +0.010 | −0.100 / +0.016 |
| fc / 1e-3 | stage02 | +0.002 / −0.002 | −0.120 / +0.023 |
| fc / 1e-2 | stage01 | −0.013 / +0.003 | −0.111 / +0.011 |
| fc / 1e-2 | stage02 | −0.004 / +0.005 | −0.199 / +0.055 |
| stage4+fc / 1e-3 | stage01 | +0.007 / +0.003 | −0.001 / −0.030 |
| stage4+fc / 1e-3 | stage02 | +0.002 / +0.006 | −0.156 / +0.038 |
| stage4+fc / 1e-2 | stage01 | +0.006 / −0.005 | −0.016 / −0.026 |
| stage4+fc / 1e-2 | stage02 | +0.005 / −0.002 | −0.105 / −0.006 |

変化は stage 内の epoch 4 − epoch 0 から、10 cohort の平均の変化を引いた値（global の上下を除く）。各群 2 cohort の平均。

- **`q` の上位 cohort は、ほぼ AP を含まない（PA・Lateral の）cohort である。** 構成は [cohort_composition](cohort_composition.md) で見た。
- **`q` の上位 cohort は、弱いまま stage を終える。** 上位 2 cohort は 8 通りすべてで、val AUROC が 10 cohort の平均より低い位置を stage の最初から最後まで保つ。下位 2 cohort との差は縮まない。
- **下位 cohort には例外がある。** 下位 2 cohort の多くは平均より高いが、平均より低い cohort も 3 個ある（fc / 1e-3 の 01:07、stage4+fc / 1e-3 の 01:02、stage4+fc / 1e-2 の 02:03）。
- **AUROC では上位と下位の差が揃わない。** 平均を引いた変化で、上位が下位を上回る向きに揃わない。step size 1e-2 で `q` が 0.2〜0.35 まで上がった cohort でも同じである。
- **loss では上位が平均より下がる。** 8 通り中 6 通りで −0.10〜−0.20 になる。ただし step size 1e-3 でも同じ大きさで起きる。1e-3 では、上位の `q` でも stage 終わりで 0.12〜0.14 にしかならない。

### 属性群（stage01 / stage02 の区間平均。baseline は同じ epoch の区間）

| 属性 | 指標 | iterative 4 run | baseline |
|---|---|---|---|
| age group | Eodds | 0.084〜0.113 | 0.163 / 0.171 |
| age group | Eopp0 | 0.038〜0.074 | 0.127 / 0.117 |
| age group | Eopp1 | 0.130〜0.151 | 0.199 / 0.225 |
| age group | AUROC gap | 0.034〜0.051 | 0.040 / 0.046 |
| age group | worst-group AUROC | 0.799〜0.811 | 0.824 / 0.822 |
| sex | Eodds（stage02） | 1e-3: 0.009 / 0.018、1e-2: 0.039 / 0.028（fc / stage4+fc） | 0.009 |
| sex | worst-group AUROC | 0.835〜0.845 | 0.859 / 0.861 |

- age group の Eopp0 / Eopp1 / Eodds は、stage01 の最初の epoch（epoch 2）で下がり、stage02 まで低いまま保たれる。step size 1e-3 でも 1e-2 と同じだけ下がる。
- age group の AUROC gap と bACC gap は baseline と同じ範囲にある。
- sex の AUROC gap は 0.016 以下で、baseline と区別できない。step size 1e-2 の run は stage02 で sex の Eopp0 が epoch ごとに開く（fc で 0.02 → 0.05）。
- sex・age group の worst-group AUROC は、global AUROC が落ちる epoch（fc / 1e-2 の epoch 5、stage4+fc の epoch 8 と 11）に合わせて落ちる。
- race（6 コード）は epoch 間の振れ（worst-group AUROC で 0.72〜0.84）が条件差より大きい。val では code 5 が 30 件（陽性 3）、code 4 が 322 件しかない。

## 解釈候補

- **age group の Eopp が下がるのは、`q` の偏りによるものではない。** 1e-3 で `q` がほぼ一様でも同じだけ下がり、順位（AUROC gap）は変わらずに閾値で決まる TPR / TNR の群間差だけが縮む。stage に入った時点で変わるもう 1 つの要素は、cohort × class の class weight である。この重みが age group と連動していることは [cohort_composition](cohort_composition.md) で見た。
- **GroupDRO の `q` は、この 5 epoch の stage では弱い cohort の順位を上げていない。** `q` の上位 cohort で val loss が下がるのは、1e-3 でも同じ大きさで起きる。そのため重みの効果ではなく、`q` の順位を決める train の損失が高い cohort ほど val loss も戻りやすいことの表れと読むほうが自然である。1e-2 の stage02 で sex の Eopp が開くのも、閾値付近の較正が群ごとにずれた結果かもしれない。
- **worst-group AUROC が baseline より低いのは、群の間の差が広がったからではない。** global AUROC の低下が、群全体に一様に出ていると読める。

## 追加確認

- class weight の効果と `q` の効果を分ける。対照は次の 2 つ。
  - **`q` を固定する run:** `UniformGroupTaskLoss` で stage を回す（`q` は一様に固定し、cohort × class の class weight は今と同じ）。age group の Eopp が今と同じだけ下がれば、原因は `q` ではなく class weight である。
  - **cohort ごとの class weight を外す run:** warmup と同じ目的関数（全体共通の class weight の ERM）のまま、同じ Spatial LoRA で 12 epoch まで続ける。age group の Eopp が下がらなければ、cohort ごとの class weight が原因と確定する。
  - GroupDRO の目的関数は、全 group 共通の class weight を受け取らない（`projects/hypernet_iterative/loss.py`）。そのため、「共通の class weight で GroupDRO」という対照は作れない。
- 同じことを test split の予測 cache で確かめる。属性群・交差群の TPR / TNR と、race を White / Asian / Black に絞った worst / gap を見る。
- step size 1e-2 で `q` がまだ偏り続けているので、stage を長くしたとき（epoch を増やしたとき）に順位まで動くかを見る。
- race を 3 カテゴリに絞った epoch 推移が必要になったら、logger に群ごとの生の値（n・AUROC・bACC・TPR・TNR）を記録させる。
