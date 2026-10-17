# latent_group_fairness

2026-10-13 開始。柱 B「hypernetwork で潜在サブグループを扱うのは妥当か、効果はあるか」の「改善」側。
[latent_subgroup_spaces](../latent_subgroup_spaces/README.md) で見た、生成 ΔW のクラスタ（属性の組み合わせを
陽性率の違いで束ねた群）に公平性損失をかけ、8 群（年齢 65 歳 × race）の動作点が改善するかを見る。

## 問い

1. 生成 ΔW の潜在群に公平性損失をかけると、評価の 8 群でも Eopp0・最悪 FPR が改善するか。
2. その改善は、同じ K で属性を直接 k-means した群にかけたときより大きいか（潜在空間が足したものがあるか）。
3. 条件付け（Spatial LoRA）の有無で、2 の答えは変わるか。

## 設計（2026-10-13、ユーザーと決定）

- 群：K = 8。学習前に固定し、学習中は動かさない（群を決めるモデルと学習するモデルを同時に回すと、群の定義が動く）。
  - `dw8`：1 段・ERM の Spatial LoRA（seed s）の生成 ΔW（stage3・stage4・fc の Gram の和）の kernel PCA 座標を、
    train の件数で重み付けした k-means
  - `attrk8`：モデル入力の属性（one-hot ＋ 年齢 / 100 ＋ 欠損フラグ）の k-means（対照）
  - seed s の run は seed s の ERM から作った群で学習する（群の揺れも手順の揺れに含める）
  - 作り方：`analysis/latent_subgroup_spaces/latent_groups.py`。群は train だけで決め、val・test は当てはめるだけ
- 格子（CheXpert race-known、1 段、seed 42・43・44）

  | 群 ＼ モデル × 目的 | ResNet-50 GroupDRO | ResNet-50 EO rate | Spatial LoRA GroupDRO | Spatial LoRA EO rate |
  |---|---|---|---|---|
  | 8 群（年齢 × race） | grid の既存 | grid の既存 | grid の既存 | grid の既存 |
  | `dw8` | 新規 | 新規 | 新規 | 新規 |
  | `attrk8` | 新規 | 新規 | 新規 | 新規 |

  新規は 8 条件 × 3 seed = 24 run。ERM と 8 群の行は [hypernet_conditioning_grid](../hypernet_conditioning_grid/README.md) の run を引用する。
- 目的関数・抽出・held-out・class weight は grid の 8 群の preset と同じで、群の定義だけを差し替える
  （`projects/hypernet_e2e/tests/test_latent_group_presets.py` が検算）。GroupDRO は代表にしない（q が最悪群に向かない）。
- 評価：research-premises の指標を、8 群と潜在群の両方で。参考に ERM ＋ 群別しきい値。

## 学習前の確認（新しい学習なし）

1 段・ERM の Spatial LoRA の予測で、群を train で作り、val で選んだ最悪群を test で測った（3 seed 平均）。

| 群 | test の最悪 FPR | 帰無分布の 95% 点 | 群の陽性率の幅 |
|---|---|---|---|
| 8 群 | 0.363 | 0.284 | 0.114 |
| `attrk8` | 0.413 | 0.272 | 0.115 |
| `dw8` | 0.499 | 0.278 | 0.268 |

- `dw8` は、同じ K の属性 k-means より悪い群を見つける。3 seed とも val で選んだ群が test でも最悪だった。
- 差は陽性率の分け方から来ている（陽性率の幅が 2 倍以上）。ΔW は属性の組を陽性率で束ねている。
- `dw8` と `attrk8` の一致（ARI）は 0.12〜0.25、`dw8` の seed 間の一致は 0.34〜0.37。
- 属性の k-means は尺度（年齢 / 100 と one-hot）の取り方に依存する。対照はこの 1 通り。
- 群の件数は、全 split（held-out を含む）で陽性 27 件以上・陰性 556 件以上（`latent_subgroup_spaces/results/latent_groups/counts.csv`）。

## preset

`projects/hypernet_e2e/configs/experiment/{resnet,spatial_lora}_chexpert_{dw8,attrk8}_{group_dro_race_known,eo_rate_dual_holdout}.yaml`。
split は `data/chexpert/splits_race_known_latent`・`splits_race_known_dual_holdout_latent`（列 `dw8_s<seed>`・`attrk8_s<seed>`）。

## 状態

ws11 が空いていないため、中心の 1 条件（Spatial LoRA × `dw8` × EO rate、seed 42）をローカルで試行中（[runs.md](runs.md)）。
