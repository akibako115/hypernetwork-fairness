# runs

この表を、この study が引用する run の正本とする。run 本体は Git 管理外で、完了済み run は
`analysis/two_stage_vs_e2e_age_race_group_dro/runs/` に取り込んである。

## 取り込み済み run

Spatial LoRA を ImageNet 初期化から E2E で age_group_65 × race の intersectional GroupDRO により最適化した対照条件。seed 42、30 epoch。

| run-id | project | run path | W&B | 条件 | 状態 |
|---|---|---|---|---|---|
| `20260928T025502Z-spatial-lora-chexpert-age-race-group-dro-s42-9502` | `hypernet_e2e` | `analysis/two_stage_vs_e2e_age_race_group_dro/runs/20260928T025502Z-spatial-lora-chexpert-age-race-group-dro-s42-9502` | [run](https://wandb.ai/kohki-akiba-kyushu-university/fairness_hypernet/runs/44c62t7k) | Spatial LoRA、age × race GroupDRO をE2E最適化 | succeeded |

第1段の ResNet checkpoint を初期値として、第2段の Spatial LoRA を
`age_group_65 × race` の intersectional GroupDRO で最適化した条件。seed 42、14群。

| run-id | project | run path | W&B | 条件 | 状態 |
|---|---|---|---|---|---|
| `20260924T122518Z-spatial-lora-chexpert-from-resnet-age-race-group-dro-s42-985c` | `hypernet_e2e` | `analysis/two_stage_vs_e2e_age_race_group_dro/runs/20260924T122518Z-spatial-lora-chexpert-from-resnet-age-race-group-dro-s42-985c` | [run](https://wandb.ai/kohki-akiba-kyushu-university/fairness_hypernet/runs/56wee8ew) | ResNet checkpoint → Spatial LoRA、age × race GroupDRO | succeeded |

親 checkpoint:

`projects/hypernet_e2e/runs/20260921T103036Z-resnet-chexpert-s42-5538/checkpoints/best_val_auroc_009.ckpt`

test split の予測 cache は同じ分析 package に作成し、5条件を同じ notebook で比較する。

## 比較対象: 同じアーキテクチャの E2E 条件

GroupDRO の効果を、アーキテクチャの差と分けて読むための対照。`E2E Spatial LoRA + GroupDRO` に
対しては同じ Spatial LoRA を CE で学習した run を、baseline ResNet に対しては同じ ResNet を
最初から同じ age × race GroupDRO で学習した run を置く。どちらも seed 42、30 epoch、inverse class
weight。

| run-id | project | run path | W&B | 条件 | 状態 |
|---|---|---|---|---|---|
| `20260929T082054Z-spatial-lora-chexpert-s42-b15c` | `hypernet_e2e` | `analysis/two_stage_vs_e2e_age_race_group_dro/runs/20260929T082054Z-spatial-lora-chexpert-s42-b15c` | [run](https://wandb.ai/kohki-akiba-kyushu-university/fairness_hypernet/runs/nrultp11) | Spatial LoRA を ImageNet 初期化から CE で E2E 最適化 | succeeded |
| `20260929T113619Z-resnet-chexpert-age-race-group-dro-s42-5474` | `hypernet_e2e` | `analysis/two_stage_vs_e2e_age_race_group_dro/runs/20260929T113619Z-resnet-chexpert-age-race-group-dro-s42-5474` | [run](https://wandb.ai/kohki-akiba-kyushu-university/fairness_hypernet/runs/cjaxqm9i) | ResNet-50 全体を age × race GroupDRO で E2E 最適化 | succeeded |

## 比較対象: baseline ResNet

2段階学習の第1段で用いた、ImageNet 初期化の ResNet-50 を全体 ERM で学習した baseline。
class weight は本実験と同じ inverse weighting、seed 42、30 epoch。第2段の親 checkpoint は
この run の `best_val_auroc_009.ckpt` である。

| run-id | project | run path | W&B | 条件 | 状態 |
|---|---|---|---|---|---|
| `20260921T103036Z-resnet-chexpert-s42-5538` | `hypernet_e2e` | `analysis/two_stage_vs_e2e_age_race_group_dro/runs/20260921T103036Z-resnet-chexpert-s42-5538` | [run](https://wandb.ai/kohki-akiba-kyushu-university/fairness_hypernet_e2e/runs/41gsahly) | ResNet-50 全体を ERM、inverse class weight | succeeded |
