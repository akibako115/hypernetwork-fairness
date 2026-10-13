# runs

この package が読む run。設定・状態・数値の正本は各 run の artifact で、ここには run ID と状態だけを書く。

## HAM10000 1 段の ERM（再利用）

ResNet・Spatial LoRA は study=ham10000_architecture_erm（[runs.md](../ham10000_architecture_erm/runs.md)、ws11）。
fc のみは study=latent_bias_hypernet（ローカル）。解決済み config は Spatial LoRA と変調段だけが違う（実行場所の path と `persistent_workers` を除く）。
ResNet は 2 段の親を兼ねる（best val AUROC の checkpoint）。

| モデル | seed 42 | seed 43 | seed 44 |
| --- | --- | --- | --- |
| ResNet-50 | `20261006T155816Z-resnet-ham10000-s42-8a40` | `20261006T160730Z-resnet-ham10000-s43-1338` | `20261006T161647Z-resnet-ham10000-s44-d3bc` |
| 親の checkpoint | `best_val_auroc_005.ckpt` | `best_val_auroc_029.ckpt` | `best_val_auroc_017.ckpt` |
| Spatial LoRA | `20261006T155802Z-spatial-lora-ham10000-s42-d782` | `20261006T160741Z-spatial-lora-ham10000-s43-bdd1` | `20261006T161713Z-spatial-lora-ham10000-s44-513e` |
| Spatial LoRA（fc のみ） | `20261009T104332Z-spatial-lora-ham10000-fc-s42-b253` | `20261009T105230Z-spatial-lora-ham10000-fc-s43-6ded` | `20261009T110118Z-spatial-lora-ham10000-fc-s44-36f3` |

## 新規の 12 条件（study=hypernet_conditioning_grid_ham10000）

未起動。
