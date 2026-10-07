# 対象run

2026-10-07 時点で、学習 run はまだ起動していない。

## 基準 ERM（引用）

選択の基準に使う。これらの run は別 study で生成したもので、`study` は書き換えない。

| arch | seed | project | run-id | 元run path（repo root基準） | W&B |
| --- | --- | --- | --- | --- | --- |
| ResNet-50 | 42 | hypernet_e2e | `20261002T124957Z-resnet-chexpert-race-known-s42-d6d3` | `projects/hypernet_e2e/runs/20261002T124957Z-resnet-chexpert-race-known-s42-d6d3` | [run](https://wandb.ai/kohki-akiba-kyushu-university/fairness_hypernet/runs/rswjo11h) |
| Spatial LoRA stage1–4 | 42 | hypernet_e2e | `20261003T142841Z-spatial-lora-chexpert-stage1-4-race-known-s42-a143` | `projects/hypernet_e2e/runs/20261003T142841Z-spatial-lora-chexpert-stage1-4-race-known-s42-a143` | [run](https://wandb.ai/kohki-akiba-kyushu-university/fairness_hypernet/runs/l4xfmu5d) |

## 1 stage の条件付き公平性 run

起動後に追記する。
