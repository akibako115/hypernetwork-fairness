# hidden cohort の中身（属性・撮影方向の構成と class weight）

## 条件

- 対象: iterative 4 run（変調範囲 `fc` / `stage4+fc` × GroupDRO step size 1e-3 / 1e-2）の stage01 / stage02 の cohort（各 10 個）
- split: train（class weight も `q` も train で決まる）
- 入力:
  - `artifacts/cohorts/cohortNN/assignments.parquet` を train の split CSV と `image` でつないだもの
  - `stages/<stage>/config.yaml` の cohort × class の class weight（`weighting=inverse`）
  - `train/group_dro/q_NN` の stage 内平均。上位 2・下位 2 cohort の選び方は
    [subgroup_training_curves](subgroup_training_curves.md) と同じ（`_shared.rank_cohorts_by_q`）
- 属性:
  - 65 歳以上の割合、女性（`sex` = 1）の割合、race の各割合（0 = White、2 = Asian、3 = Black）。分母は各属性が欠損していない行
  - race の欠損の割合（全行が分母）。CheXpert は欠損の race に White と同じ 0 が入るので、`race_missing` で分ける
  - 撮影方向: `frontal_lateral` と `ap_pa` を AP / PA / Lateral にまとめたもの。AP / PA が欠損した Frontal 12 行は別の値
  - 「片方だけからなる」は、割合が 0.05 未満か 0.95 超

notebook は [`cohort_composition.ipynb`](../cohort_composition.ipynb)。

- 数値の正本: `results/cohort_composition.csv`、`results/cohort_q_picked_composition.csv`
- 図: `figures/cohort_q_picked_composition_<条件>.png`（条件ごとに 1 枚。行が cohort、列が race・sex・age・撮影方向）

## 観察事実

train 全体の割合は、65 歳以上 0.438、女性 0.406、White 0.641、race の欠損 0.112、AP 0.723、陽性率 0.101。

### cohort 全体

| 条件 | stage | sex で片方だけ | age で片方だけ | 撮影方向が 1 つだけ | AP を含まない | race 欠損が過半 | 陽性の重みの max / min |
|---|---|---:|---:|---:|---:|---:|---:|
| fc | stage01 | 10 | 4 | 4 | 2 | 2 | 4.7 |
| fc / 1e-3 | stage02 | 5 | 5 | 5 | 3 | 1 | 5.9 |
| fc / 1e-2 | stage02 | 7 | 5 | 5 | 3 | 1 | 5.9 |
| stage4+fc | stage01 | 9 | 4 | 4 | 2 | 2 | 4.8 |
| stage4+fc / 1e-3 | stage02 | 5 | 3 | 3 | 2 | 2 | 6.3 |
| stage4+fc / 1e-2 | stage02 | 7 | 5 | 5 | 3 | 1 | 6.9 |

stage01 は、同じ変調範囲の 2 run で割り当てが一致する（step size は stage01 の cohort に影響しない）。

- **cohort は sex・撮影方向・age で割れる。** stage01 はほぼ sex で割れ、stage02 では sex で割れる cohort が減る。撮影方向が 1 つだけの
  cohort は stage ごとに 3〜5 個ある。
- **race では割れないが、race の欠損が偏る。** White の割合（欠損を除く）は 0.18〜0.79 で、片方だけの cohort は無い。一方、race の欠損が
  半分を超える cohort が stage ごとに 1〜2 個ある。fc の stage01 の cohort 03 と 07 は欠損が 0.66 / 0.58 で、残りの行も大半が code 1 である。
- **class weight と age の連動:** 陽性の class weight は 65 歳以上の割合と強く連動する（順位相関 +0.68〜+0.86）。
  - 65 歳以上だけの cohort は陽性の重みが 10〜12、65 歳未満だけの cohort は 1.8〜4.4。
  - 女性の割合とはほぼ無相関（−0.13〜+0.15）。
- **train 全体の陽性率:** 65 歳未満 0.137、65 歳以上 0.054。男性 0.100、女性 0.102。AP 0.071、PA 0.188、Lateral 0.168。

### `q` の上位 2 cohort と下位 2 cohort

上位 − 下位（各 2 cohort の平均の差）。

| 条件 | stage | AP | PA | Lateral | 女性 | race の欠損 | 65 歳以上 | 陽性率 |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| fc / 1e-3 | stage01 | −0.75 | +0.42 | +0.33 | −1.00 | −0.35 | +0.23 | +0.01 |
| fc / 1e-3 | stage02 | −0.96 | +0.78 | +0.18 | −0.33 | −0.53 | +0.19 | +0.08 |
| fc / 1e-2 | stage01 | −0.67 | +0.36 | +0.31 | −1.00 | −0.06 | +0.37 | −0.01 |
| fc / 1e-2 | stage02 | −0.79 | +0.35 | +0.45 | −0.15 | −0.49 | +0.28 | +0.01 |
| stage4+fc / 1e-3 | stage01 | −0.36 | +0.48 | −0.12 | −0.47 | −0.36 | +0.46 | −0.03 |
| stage4+fc / 1e-3 | stage02 | −0.84 | +0.51 | +0.33 | −0.28 | −0.96 | −0.08 | +0.10 |
| stage4+fc / 1e-2 | stage01 | −0.36 | +0.48 | −0.12 | −0.47 | −0.36 | +0.46 | −0.03 |
| stage4+fc / 1e-2 | stage02 | −0.84 | +0.78 | +0.06 | −0.48 | −0.48 | +0.01 | +0.10 |

- **上位 2 cohort は、ほぼ AP を含まない cohort である。** 16 個（4 run × 2 stage × 2）のうち 14 個は AP が 0.05 未満で、PA だけ、
  Lateral だけ、または両者の混合からなる。例外の 2 個は stage4+fc の stage01 の cohort 00（65 歳以上の女性、AP 0.79）で、2 run で共通の集団である。
- **下位 2 cohort は、16 個すべてで AP が 0.62〜1.0。**
- **女性の割合と race の欠損の割合は、8 通りすべてで上位のほうが低い。** race の欠損が過半の cohort は延べ 13 個
  （run × stage で数えるので、stage01 の cohort は 2 回数える）あり、そのうち 8 個が下位 2 に入る。上位 2 には 1 個も入らない。
- **陽性率では上位と下位が分かれない**（差は −0.03〜+0.10）。age は向きが揃わない。
- 上の傾向は、`q` がほぼ一様な step size 1e-3 でも、1e-2 と同じ向きに出る。

## 解釈候補

- **cohort × class の class weight は、実質的に age group ごとに陽性の重みを変えている。**
  - これは age group ごとに判定の閾値を合わせ直すことに近い。
  - [subgroup_training_curves](subgroup_training_curves.md) で見た点と合う。stage に入ると、age group の Eopp0 / Eopp1 が `q` と無関係に下がり、AUROC gap は変わらない。
- **GroupDRO が重みを寄せているのは、撮影方向の少数派（PA・Lateral）の cohort である。**
  - train の 72% が AP なので、AP を含まない cohort は warmup の ERM で学習に占める割合が小さく、train の損失が高く残りやすい。
  - `q` の高い cohort が stage の間ずっと弱いまま、という [subgroup_training_curves](subgroup_training_curves.md) の観察とも合う。
  - 撮影方向は公平性の評価軸ではない（撮影プロトコル）。この設定の GroupDRO は、評価軸の属性群ではなく撮影条件の群の損失を下げにいっていることになる。
- **上位に男性が多いのは、AP を含まない cohort が男性側で切り出されやすいことの表れかもしれない。** sex そのものが `q` を決めているのかは、この表からは分けられない。
- **class weight は、sex ごとの閾値はほとんど動かさない。** cohort は sex で割れているが、sex では陽性率が変わらないためである。
- **race の欠損が偏った cohort があるのは、画像の側に欠損と結び付く特徴があることを示唆する。** 撮影時期や撮影元などの交絡の可能性がある。
- **この設定の hidden group は、既知の属性と撮影条件の近似になっている。** hidden cohort が sex × 撮影方向 × age をほぼそのまま拾っているためである。
  未知の群を見つけるという目的に照らしてこれでよいかは、別に判断が要る。

## 追加確認

- class weight の効果と `q` の効果を分ける。対照は次の 2 つ。
  - **`q` を固定する run:** `UniformGroupTaskLoss` で stage を回す（`q` は一様に固定し、cohort × class の class weight は今と同じ）。age group の Eopp が今と同じだけ下がれば、原因は `q` ではなく class weight である。
  - **cohort ごとの class weight を外す run:** warmup と同じ目的関数（全体共通の class weight の ERM）のまま、同じ Spatial LoRA で 12 epoch まで続ける。age group の Eopp が下がらなければ、cohort ごとの class weight が原因と確定する。
  - GroupDRO の目的関数は、全 group 共通の class weight を受け取らない（`projects/hypernet_iterative/loss.py`）。そのため、「共通の class weight で GroupDRO」という対照は作れない。
- 撮影方向ごとの val / test の性能を見る。PA・Lateral が AP より弱いなら、`q` が撮影方向の少数派に寄る理由がそれで説明できる。
- cohort が sex × 撮影方向 × age を拾う理由を、cohort を作る特徴量の側から見る（トピック 3 の特徴量分析）。
- race の欠損に偏った cohort が、何の画像特徴で集まっているかを見る。
