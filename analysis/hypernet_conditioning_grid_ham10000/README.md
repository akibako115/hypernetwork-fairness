# hypernet_conditioning_grid_ham10000

2026-10-13 開始。柱 A の格子（[hypernet_conditioning_grid](../hypernet_conditioning_grid/README.md)、CheXpert）を、
データだけ HAM10000 に替えて回す。条件付けが効くかどうかがデータに依存するかを見る。

## 問い

CheXpert の格子と同じ。同じデータ・同じ目的関数・同じ学習手順のもとで、属性で条件付けた hypernetwork（Spatial LoRA）は、
条件付けのない ResNet-50 より、全体の性能を落とさずに最悪群を上げるか。上がらないなら、どの条件で、なぜか。

評価の定義は [research-premises.md](../../docs/research-premises.md)（固定しきい値 0.5 の Eopp1・Eopp0・Eodds と
害の大きい側の最悪群の率が主、AUROC は副）。HAM10000 で害の大きい側は見落とし（最小の TPR）。

## 格子（HAM10000、sex × 年齢 65 歳の 4 群、seed 42・43・44）

HAM10000 には race・肌色の列が無いので、CheXpert の年齢 × race 8 群の代わりに、いまの評価軸と同じ sex × age_group_65 の
4 群を使う（2026-10-13、ユーザーと決めた）。群 ID は sex を上位桁にした順（female × <65、female × ≥65、male × <65、male × ≥65）。

| 手順 | モデル | ERM | GroupDRO | EO rate |
| --- | --- | --- | --- | --- |
| 1 段（ImageNet から全層） | ResNet-50 | 再利用 | 新規 | 新規 |
| | Spatial LoRA（stage3・stage4・fc） | 再利用 | 新規 | 新規 |
| | Spatial LoRA（fc のみ） | 再利用 | 新規 | 新規 |
| 2 段（親 ResNet ERM を凍結） | Spatial LoRA（stage3・stage4・fc） | 新規 | 新規 | 新規 |
| | Spatial LoRA（fc のみ） | 新規 | 新規 | 新規 |

15 条件 × 3 seed = 45 run。うち 1 段の ERM 3 条件（9 run）は既存の run を再利用し、新規は 12 条件 × 3 seed = 36 run。
CheXpert と違い、1 段の fc のみの ERM も既存の run がある。

### そろえたこと

CheXpert の格子と同じ方針で、データ以外は動かさない。

- **train**：ERM・GroupDRO は `data/ham10000/splits` の train 全体で学習する。EO rate は、双対変数を測る held-out
  （病変の 20%）を除いた `splits_dual_holdout` の train（80%）で学習する。val・test は共通。
- **ERM の再利用**：1 段の ResNet・Spatial LoRA は [ham10000_architecture_erm](../ham10000_architecture_erm/runs.md)、
  fc のみは 2026-10-09 の `spatial_lora_ham10000_fc`（[runs.md](runs.md)）。どれも `_base_ham10000` のレシピ
  （inverse class weight・30 epoch・AdamW lr 1e-4）で、今回の preset と同じ。10/13 の修正は ERM の学習に触れていない。
- **2 段の親**：同じ seed の上記 ResNet ERM の best val AUROC checkpoint。backbone と元の分類層を凍結し、
  Spatial LoRA の hypernetwork だけを学習する。
- **双対変数・群の重みの更新**：EO rate は held-out で測る。GroupDRO は原典どおり train のバッチで更新する。
- **抽出**：EO rate は sex × 年齢 × ラベルのセル均等、ERM・GroupDRO は自然抽出。
- **class weight**：全条件で全群共通の inverse。
- **checkpoint**：EO rate は全 epoch を残す。ERM・GroupDRO は best（val AUROC）と last だけ。

### 原典から変えたこと

- EO rate の初期 task 重みを 0.9 にした（CheXpert と同じ）。
- EO rate の双対変数を held-out で更新する。

### 読むときの注意

- CheXpert の格子の注意（抽出の違い、EO rate は 20% 少ない train、2 段の EO rate では held-out を親が学習済み）はそのまま当てはまる。
- test の陽性は群ごとに 28〜82 件しかない。群別の指標は CheXpert より雑音が大きい。
- test の 294 件は sex か age を欠く（うち 284 件は別施設の external）。属性条件付きモデルは学習中に見ていない欠損用の入力を受ける。
  **モデル間の主比較は属性の揃った 1,217 件で行う**（[ham10000_architecture_erm](../ham10000_architecture_erm/README.md)）。
- 10-09 の `resnet_ham10000_sex_age_group_dro` と既存の EO rate の run は修正前の実装なので使わず、回し直す。

## preset

`projects/hypernet_e2e/configs/experiment/` の次の名前。2 段は `_two_stage_frozen.yaml` を重ね、起動時に
`model.backbone_checkpoint_path` で親を渡す。GroupDRO は `_sex_age_groups_ham10000.yaml` で q に群のラベルを付ける。

| | ERM | GroupDRO | EO rate |
| --- | --- | --- | --- |
| 1 段 ResNet | `resnet_ham10000` | `resnet_ham10000_sex_age_group_dro` | `resnet_ham10000_sex_age_eo_rate_dual_holdout` |
| 1 段 Spatial LoRA | `spatial_lora_ham10000` | `spatial_lora_ham10000_sex_age_group_dro` | `spatial_lora_ham10000_sex_age_eo_rate_dual_holdout` |
| 1 段 fc のみ | `spatial_lora_ham10000_fc` | `spatial_lora_ham10000_fc_sex_age_group_dro` | `spatial_lora_ham10000_fc_sex_age_eo_rate_dual_holdout` |
| 2 段 Spatial LoRA | `spatial_lora_ham10000_from_resnet` | `spatial_lora_ham10000_from_resnet_sex_age_group_dro` | `spatial_lora_ham10000_from_resnet_sex_age_eo_rate_dual_holdout` |
| 2 段 fc のみ | `spatial_lora_ham10000_fc_from_resnet` | `spatial_lora_ham10000_fc_from_resnet_sex_age_group_dro` | `spatial_lora_ham10000_fc_from_resnet_sex_age_eo_rate_dual_holdout` |

## 状態

2026-10-13 に preset を用意した。学習は未起動。
