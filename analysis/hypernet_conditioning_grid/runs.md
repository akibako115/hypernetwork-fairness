# runs

この package が読む run。設定・状態・数値の正本は各 run の artifact で、ここには run ID と状態だけを書く。

## CheXpert 1 段の ERM（再利用。study=two_stage_vs_e2e_age_race_group_dro）

| モデル | seed 42 | seed 43 | seed 44 |
| --- | --- | --- | --- |
| ResNet-50 | `20261002T124957Z-resnet-chexpert-race-known-s42-d6d3` | `20261002T142200Z-resnet-chexpert-race-known-s43-bbfc` | `20261002T155222Z-resnet-chexpert-race-known-s44-3d73` |
| Spatial LoRA | `20261002T125005Z-spatial-lora-chexpert-race-known-s42-a21c` | `20261002T145233Z-spatial-lora-chexpert-race-known-s43-424e` | `20261002T165551Z-spatial-lora-chexpert-race-known-s44-eb13` |

## 新規の 13 条件（study=hypernet_conditioning_grid）

未起動。
