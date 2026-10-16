# runs

この package は学習しない。下の ERM の run（1 段・best val AUROC の checkpoint）を推論し直すだけで、
どれも 2026-10-13 の公平性損失の修正の影響を受けない。来歴（W&B・config・checkpoint の SHA256）は各 run の artifact が正本。
取り込み元は `projects/hypernet_e2e/runs/`（2026-10-15、`analysis/common/runs.py import`）。

| データ | モデル | seed | run-id | 元の study |
| --- | --- | --- | --- | --- |
| chexpert | HyperAdapt | 42 | `20261002T125004Z-hyperadapt-chexpert-race-known-s42-b901` | `two_stage_vs_e2e_age_race_group_dro` |
| chexpert | Spatial LoRA | 42 | `20261002T125005Z-spatial-lora-chexpert-race-known-s42-a21c` | `two_stage_vs_e2e_age_race_group_dro` |
| chexpert | Spatial LoRA | 43 | `20261002T145233Z-spatial-lora-chexpert-race-known-s43-424e` | `two_stage_vs_e2e_age_race_group_dro` |
| chexpert | Spatial LoRA | 44 | `20261002T165551Z-spatial-lora-chexpert-race-known-s44-eb13` | `two_stage_vs_e2e_age_race_group_dro` |
| chexpert | HyperAdapt | 43 | `20261002T181025Z-hyperadapt-chexpert-race-known-s43-4ab0` | `two_stage_vs_e2e_age_race_group_dro` |
| chexpert | HyperAdapt | 44 | `20261002T233012Z-hyperadapt-chexpert-race-known-s44-a7ba` | `two_stage_vs_e2e_age_race_group_dro` |
| ham10000 | HyperAdapt | 42 | `20261006T155759Z-hyperadapt-ham10000-s42-f78c` | `ham10000_architecture_erm` |
| ham10000 | Spatial LoRA | 42 | `20261006T155802Z-spatial-lora-ham10000-s42-d782` | `ham10000_architecture_erm` |
| ham10000 | Spatial LoRA | 43 | `20261006T160741Z-spatial-lora-ham10000-s43-bdd1` | `ham10000_architecture_erm` |
| ham10000 | Spatial LoRA | 44 | `20261006T161713Z-spatial-lora-ham10000-s44-513e` | `ham10000_architecture_erm` |
| ham10000 | HyperAdapt | 43 | `20261006T161735Z-hyperadapt-ham10000-s43-4db2` | `ham10000_architecture_erm` |
| ham10000 | HyperAdapt | 44 | `20261006T163711Z-hyperadapt-ham10000-s44-15e7` | `ham10000_architecture_erm` |
| ham10000 | Spatial LoRA（fc のみ） | 42 | `20261009T104332Z-spatial-lora-ham10000-fc-s42-b253` | `latent_bias_hypernet` |
| ham10000 | Spatial LoRA（fc のみ） | 43 | `20261009T105230Z-spatial-lora-ham10000-fc-s43-6ded` | `latent_bias_hypernet` |
| ham10000 | Spatial LoRA（fc のみ） | 44 | `20261009T110118Z-spatial-lora-ham10000-fc-s44-36f3` | `latent_bias_hypernet` |
