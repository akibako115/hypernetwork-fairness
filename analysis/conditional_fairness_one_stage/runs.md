# 対象run

## 基準 ERM（引用）

選択の基準に使う。これらの run は別 study で生成したもので、`study` は書き換えない。

| arch | seed | project | run-id | 元run path（repo root基準） | W&B |
| --- | --- | --- | --- | --- | --- |
| ResNet-50 | 42 | hypernet_e2e | `20261002T124957Z-resnet-chexpert-race-known-s42-d6d3` | `projects/hypernet_e2e/runs/20261002T124957Z-resnet-chexpert-race-known-s42-d6d3` | [run](https://wandb.ai/kohki-akiba-kyushu-university/fairness_hypernet/runs/rswjo11h) |
| Spatial LoRA stage1–4 | 42 | hypernet_e2e | `20261003T142841Z-spatial-lora-chexpert-stage1-4-race-known-s42-a143` | `projects/hypernet_e2e/runs/20261003T142841Z-spatial-lora-chexpert-stage1-4-race-known-s42-a143` | [run](https://wandb.ai/kohki-akiba-kyushu-university/fairness_hypernet/runs/l4xfmu5d) |

## 1 stage の条件付き公平性 run（ws11）

すべて seed 42、project は hypernet_e2e。元 run の path は `projects/hypernet_e2e/runs/<run-id>`。
metadata（`*.ckpt` を除く）は 2026-10-08 にローカルへ回収した。checkpoint は ws11 の
`/mnt/fast/kohkiakiba/hypernet-fairness/projects/hypernet_e2e/runs/<run-id>/checkpoints` に残っている。

| 条件 | run-id | 状態 | W&B |
| --- | --- | --- | --- |
| ResNet minimax AUC | `20261007T145210Z-resnet-chexpert-age-minimax-auc-race-known-s42-01ce` | succeeded | [run](https://wandb.ai/kohki-akiba-kyushu-university/fairness_hypernet/runs/it6ub7m5) |
| Spatial LoRA minimax AUC | `20261007T145214Z-spatial-lora-chexpert-stage1-4-age-minimax-auc-race-known-s42-332d` | succeeded | [run](https://wandb.ai/kohki-akiba-kyushu-university/fairness_hypernet/runs/9gfj1udg) |
| ResNet FairDRO ρ=0.1 | `20261007T153612Z-resnet-chexpert-age-fair-dro-race-known-s42-3ef9` | succeeded | [run](https://wandb.ai/kohki-akiba-kyushu-university/fairness_hypernet/runs/baxi2xlv) |
| Spatial LoRA FairDRO ρ=0.1 | `20261007T153605Z-spatial-lora-chexpert-stage1-4-age-fair-dro-race-known-s42-bbc7` | succeeded | [run](https://wandb.ai/kohki-akiba-kyushu-university/fairness_hypernet/runs/p6w57vng) |
| ResNet EO rate（初期 task 0.9） | `20261007T153614Z-resnet-chexpert-age-equalized-odds-rate-race-known-s42-b486` | succeeded | [run](https://wandb.ai/kohki-akiba-kyushu-university/fairness_hypernet/runs/a6dd0ys4) |
| Spatial LoRA EO rate（初期 task 0.9） | `20261007T153613Z-spatial-lora-chexpert-stage1-4-age-equalized-odds-rate-race-known-s42-2522` | succeeded | [run](https://wandb.ai/kohki-akiba-kyushu-university/fairness_hypernet/runs/kr2wocq5) |
| ResNet FairDRO ρ=1 | `20261007T145210Z-resnet-chexpert-age-fair-dro-race-known-s42-0be1` | 停止（q が端点で振動） | [run](https://wandb.ai/kohki-akiba-kyushu-university/fairness_hypernet/runs/4nu6d11y) |
| Spatial LoRA FairDRO ρ=1 | `20261007T145214Z-spatial-lora-chexpert-stage1-4-age-fair-dro-race-known-s42-ded9` | 停止（同上） | [run](https://wandb.ai/kohki-akiba-kyushu-university/fairness_hypernet/runs/iwhtw38g) |
| ResNet EO rate（λ 一様） | `20261007T145238Z-resnet-chexpert-age-equalized-odds-rate-race-known-s42-7130` | 停止（全予測が陰性） | [run](https://wandb.ai/kohki-akiba-kyushu-university/fairness_hypernet/runs/fgvnfpcb) |
| Spatial LoRA EO rate（λ 一様） | `20261007T145216Z-spatial-lora-chexpert-stage1-4-age-equalized-odds-rate-race-known-s42-b744` | 停止（全予測が陽性） | [run](https://wandb.ai/kohki-akiba-kyushu-university/fairness_hypernet/runs/j7vjfziw) |

停止した 4 本は、2026-10-08 に `docker stop` で止めた。強制終了なので `run.json` は running のまま残っている。
止めた理由と結果のまとめは [README](README.md) にある。package の `runs/` への取り込み（`common/runs.py import`）は、まだ行っていない。
