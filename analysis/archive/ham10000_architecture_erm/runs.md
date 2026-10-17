# 対象run

2026-10-07 00:58 JSTにws11で起動。GPU1〜3で各モデルのseed 42 → 43 → 44を逐次実行する。
起動コマンドは`run_logs/ham10000_ws11_launch.sh`。各seedのログはws11の`run_logs/ham10000_<preset>_s<seed>.log`。
ws11はGit checkoutではないため、`run.json`の`git_commit`は空になる。設定の正本は各runの`config.yaml`。

| モデル | seed | project | run-id | W&B | best epoch | best val AUROC |
| --- | --- | --- | --- | --- | --- | --- |
| ResNet-50 | 42 | hypernet_e2e | `20261006T155816Z-resnet-ham10000-s42-8a40` | [by85hbmp](https://wandb.ai/kohki-akiba-kyushu-university/fairness_hypernet/runs/by85hbmp) | 5 | 0.9202 |
| ResNet-50 | 43 | hypernet_e2e | `20261006T160730Z-resnet-ham10000-s43-1338` | [hfljvo2z](https://wandb.ai/kohki-akiba-kyushu-university/fairness_hypernet/runs/hfljvo2z) | 29 | 0.9250 |
| ResNet-50 | 44 | hypernet_e2e | `20261006T161647Z-resnet-ham10000-s44-d3bc` | [kpg0p6dx](https://wandb.ai/kohki-akiba-kyushu-university/fairness_hypernet/runs/kpg0p6dx) | 17 | 0.9331 |
| Spatial LoRA | 42 | hypernet_e2e | `20261006T155802Z-spatial-lora-ham10000-s42-d782` | [zbpsxe4z](https://wandb.ai/kohki-akiba-kyushu-university/fairness_hypernet/runs/zbpsxe4z) | 14 | 0.9360 |
| Spatial LoRA | 43 | hypernet_e2e | `20261006T160741Z-spatial-lora-ham10000-s43-bdd1` | [mmzk4vce](https://wandb.ai/kohki-akiba-kyushu-university/fairness_hypernet/runs/mmzk4vce) | 10 | 0.9276 |
| Spatial LoRA | 44 | hypernet_e2e | `20261006T161713Z-spatial-lora-ham10000-s44-513e` | [epptx1ds](https://wandb.ai/kohki-akiba-kyushu-university/fairness_hypernet/runs/epptx1ds) | 19 | 0.9307 |
| HyperAdapt | 42 | hypernet_e2e | `20261006T155759Z-hyperadapt-ham10000-s42-f78c` | [fz6xmqld](https://wandb.ai/kohki-akiba-kyushu-university/fairness_hypernet/runs/fz6xmqld) | 27 | 0.9336 |
| HyperAdapt | 43 | hypernet_e2e | `20261006T161735Z-hyperadapt-ham10000-s43-4db2` | [3yz15fjh](https://wandb.ai/kohki-akiba-kyushu-university/fairness_hypernet/runs/3yz15fjh) | 26 | 0.9271 |
| HyperAdapt | 44 | hypernet_e2e | `20261006T163711Z-hyperadapt-ham10000-s44-15e7` | [4hgo0qt1](https://wandb.ai/kohki-akiba-kyushu-university/fairness_hypernet/runs/4hgo0qt1) | 13 | 0.9375 |

全9runが`succeeded`で30 epochを完走した（2026-10-07）。checkpointはbest / lastの両方を含めてws11から回収し、
FairVLMedと合わせた全306 fileのSHA256がws11側と一致すること、best / lastが`run.json`記録のSHA256と一致することを確認した。
`runs/`へhardlinkで取り込み済み。best epochはcheckpoint名の0始まりのepoch番号。
