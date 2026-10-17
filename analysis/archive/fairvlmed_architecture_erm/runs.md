# 対象run

2026-10-07 01:27 JSTにws11で起動。GPU4〜6で各モデルのseed 42 → 43 → 44を逐次実行する。起動コマンドは`run_logs/fairvlmed_ws11_launch.sh`。
各seedのログはws11の`run_logs/fairvlmed_<preset>_s<seed>.log`。
ws11はGit checkoutではないため、`run.json`の`git_commit`は空になる。設定の正本は各runの`config.yaml`。

| モデル | seed | project | run-id | W&B | best epoch | best val AUROC |
| --- | --- | --- | --- | --- | --- | --- |
| ResNet-50 | 42 | hypernet_e2e | `20261006T162713Z-resnet-fairvlmed-s42-00c2` | [rnpla7tc](https://wandb.ai/kohki-akiba-kyushu-university/fairness_hypernet/runs/rnpla7tc) | 5 | 0.8132 |
| ResNet-50 | 43 | hypernet_e2e | `20261006T163427Z-resnet-fairvlmed-s43-f4ad` | [1qwfl5tk](https://wandb.ai/kohki-akiba-kyushu-university/fairness_hypernet/runs/1qwfl5tk) | 3 | 0.8307 |
| ResNet-50 | 44 | hypernet_e2e | `20261006T164136Z-resnet-fairvlmed-s44-269c` | [311t63vs](https://wandb.ai/kohki-akiba-kyushu-university/fairness_hypernet/runs/311t63vs) | 4 | 0.8229 |
| Spatial LoRA | 42 | hypernet_e2e | `20261006T162710Z-spatial-lora-fairvlmed-s42-804d` | [9d25vuzn](https://wandb.ai/kohki-akiba-kyushu-university/fairness_hypernet/runs/9d25vuzn) | 1 | 0.8132 |
| Spatial LoRA | 43 | hypernet_e2e | `20261006T163449Z-spatial-lora-fairvlmed-s43-bd2b` | [5xsidzw3](https://wandb.ai/kohki-akiba-kyushu-university/fairness_hypernet/runs/5xsidzw3) | 23 | 0.8271 |
| Spatial LoRA | 44 | hypernet_e2e | `20261006T164233Z-spatial-lora-fairvlmed-s44-f49c` | [cwkyk8rs](https://wandb.ai/kohki-akiba-kyushu-university/fairness_hypernet/runs/cwkyk8rs) | 18 | 0.8188 |
| HyperAdapt | 42 | hypernet_e2e | `20261006T162714Z-hyperadapt-fairvlmed-s42-92a4` | [knwm4fcv](https://wandb.ai/kohki-akiba-kyushu-university/fairness_hypernet/runs/knwm4fcv) | 4 | 0.8296 |
| HyperAdapt | 43 | hypernet_e2e | `20261006T164344Z-hyperadapt-fairvlmed-s43-f14d` | [v4d19z18](https://wandb.ai/kohki-akiba-kyushu-university/fairness_hypernet/runs/v4d19z18) | 8 | 0.8271 |
| HyperAdapt | 44 | hypernet_e2e | `20261006T170013Z-hyperadapt-fairvlmed-s44-1d27` | [blb412o4](https://wandb.ai/kohki-akiba-kyushu-university/fairness_hypernet/runs/blb412o4) | 1 | 0.8203 |

全9runが`succeeded`で30 epochを完走した（2026-10-07）。checkpointはbest / lastの両方を含めてws11から回収し、
HAM10000と合わせた全306 fileのSHA256がws11側と一致すること、best / lastが`run.json`記録のSHA256と一致することを確認した。
`runs/`へhardlinkで取り込み済み。best epochはcheckpoint名の0始まりのepoch番号。
