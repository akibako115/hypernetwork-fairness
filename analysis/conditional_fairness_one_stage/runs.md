# 対象run

## 基準 ERM（引用）

選択の基準に使う。これらの run は別 study で生成したもので、`study` は書き換えない。

| arch | seed | project | run-id | 元run path（repo root基準） | W&B |
| --- | --- | --- | --- | --- | --- |
| ResNet-50 | 42 | hypernet_e2e | `20261002T124957Z-resnet-chexpert-race-known-s42-d6d3` | `projects/hypernet_e2e/runs/20261002T124957Z-resnet-chexpert-race-known-s42-d6d3` | [run](https://wandb.ai/kohki-akiba-kyushu-university/fairness_hypernet/runs/rswjo11h) |
| Spatial LoRA stage1–4 | 42 | hypernet_e2e | `20261003T142841Z-spatial-lora-chexpert-stage1-4-race-known-s42-a143` | `projects/hypernet_e2e/runs/20261003T142841Z-spatial-lora-chexpert-stage1-4-race-known-s42-a143` | [run](https://wandb.ai/kohki-akiba-kyushu-university/fairness_hypernet/runs/l4xfmu5d) |

## 1 stage の条件付き公平性 run（ws11）

| 条件 | run-id | 状態 |
| --- | --- | --- |
| ResNet minimax AUC | `20261007T145210Z-resnet-chexpert-age-minimax-auc-race-known-s42-01ce` | 走行中 |
| Spatial LoRA minimax AUC | `20261007T145214Z-spatial-lora-chexpert-stage1-4-age-minimax-auc-race-known-s42-332d` | 走行中 |
| ResNet FairDRO ρ=1 | `20261007T145210Z-resnet-chexpert-age-fair-dro-race-known-s42-0be1` | 2026-10-08 に停止（q が端点で振動） |
| Spatial LoRA FairDRO ρ=1 | `20261007T145214Z-spatial-lora-chexpert-stage1-4-age-fair-dro-race-known-s42-ded9` | 2026-10-08 に停止（同上） |
| ResNet EO rate（λ 一様） | `20261007T145238Z-resnet-chexpert-age-equalized-odds-rate-race-known-s42-7130` | 2026-10-08 に停止（全予測が陰性） |
| Spatial LoRA EO rate（λ 一様） | `20261007T145216Z-spatial-lora-chexpert-stage1-4-age-equalized-odds-rate-race-known-s42-b744` | 2026-10-08 に停止（全予測が陽性） |

停止した 4 本の run.json は running のまま残っている。止めた理由は [README](README.md) にある。
