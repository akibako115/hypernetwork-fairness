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

ws06 で 2026-10-14 00:35 JST に起動（commit `89cf647`、`run_logs/hcgh_ham10000_ws06_launch_20261013.sh`）。
GPU 0〜3 に 3 container ずつ、各条件の 3 seed を同じ container で直列に回した。
36 run とも 2026-10-14 01:40 JST ごろまでに `succeeded` で終わった（failures ファイルは空）。checkpoint を除く metadata を 10-14 にローカルへ回収した。

| preset | seed 42 | seed 43 | seed 44 |
| --- | --- | --- | --- |
| `resnet_ham10000_sex_age_group_dro` | `20261013T153508Z-resnet-ham10000-sex-age-group-dro-s42-f3c4` | `20261013T154502Z-resnet-ham10000-sex-age-group-dro-s43-0cea` | `20261013T155507Z-resnet-ham10000-sex-age-group-dro-s44-5286` |
| `resnet_ham10000_sex_age_eo_rate_dual_holdout` | `20261013T153526Z-resnet-ham10000-sex-age-eo-rate-dual-holdout-s42-442d` | `20261013T155300Z-resnet-ham10000-sex-age-eo-rate-dual-holdout-s43-bee9` | `20261013T160533Z-resnet-ham10000-sex-age-eo-rate-dual-holdout-s44-778e` |
| `spatial_lora_ham10000_sex_age_group_dro` | `20261013T153531Z-spatial-lora-ham10000-sex-age-group-dro-s42-c226` | `20261013T155430Z-spatial-lora-ham10000-sex-age-group-dro-s43-6fdd` | `20261013T161032Z-spatial-lora-ham10000-sex-age-group-dro-s44-fbd7` |
| `spatial_lora_ham10000_sex_age_eo_rate_dual_holdout` | `20261013T153518Z-spatial-lora-ham10000-sex-age-eo-rate-dual-holdout-s42-5ba4` | `20261013T160450Z-spatial-lora-ham10000-sex-age-eo-rate-dual-holdout-s43-502f` | `20261013T162532Z-spatial-lora-ham10000-sex-age-eo-rate-dual-holdout-s44-f5ee` |
| `spatial_lora_ham10000_fc_sex_age_group_dro` | `20261013T153530Z-spatial-lora-ham10000-fc-sex-age-group-dro-s42-dc31` | `20261013T154506Z-spatial-lora-ham10000-fc-sex-age-group-dro-s43-032d` | `20261013T155421Z-spatial-lora-ham10000-fc-sex-age-group-dro-s44-a28d` |
| `spatial_lora_ham10000_fc_sex_age_eo_rate_dual_holdout` | `20261013T153531Z-spatial-lora-ham10000-fc-sex-age-eo-rate-dual-holdout-s42-99fc` | `20261013T155556Z-spatial-lora-ham10000-fc-sex-age-eo-rate-dual-holdout-s43-9920` | `20261013T161036Z-spatial-lora-ham10000-fc-sex-age-eo-rate-dual-holdout-s44-e590` |
| `spatial_lora_ham10000_from_resnet` | `20261013T153529Z-spatial-lora-ham10000-from-resnet-s42-855b` | `20261013T154735Z-spatial-lora-ham10000-from-resnet-s43-35aa` | `20261013T155932Z-spatial-lora-ham10000-from-resnet-s44-e782` |
| `spatial_lora_ham10000_from_resnet_sex_age_group_dro` | `20261013T153531Z-spatial-lora-ham10000-from-resnet-sex-age-group-dro-s42-2ea9` | `20261013T154522Z-spatial-lora-ham10000-from-resnet-sex-age-group-dro-s43-55dd` | `20261013T155507Z-spatial-lora-ham10000-from-resnet-sex-age-group-dro-s44-fbc0` |
| `spatial_lora_ham10000_from_resnet_sex_age_eo_rate_dual_holdout` | `20261013T153526Z-spatial-lora-ham10000-from-resnet-sex-age-eo-rate-dual-holdout-s42-13c6` | `20261013T160404Z-spatial-lora-ham10000-from-resnet-sex-age-eo-rate-dual-holdout-s43-b649` | `20261013T162629Z-spatial-lora-ham10000-from-resnet-sex-age-eo-rate-dual-holdout-s44-acae` |
| `spatial_lora_ham10000_fc_from_resnet` | `20261013T153530Z-spatial-lora-ham10000-fc-from-resnet-s42-f89c` | `20261013T154207Z-spatial-lora-ham10000-fc-from-resnet-s43-8cdb` | `20261013T154832Z-spatial-lora-ham10000-fc-from-resnet-s44-5b08` |
| `spatial_lora_ham10000_fc_from_resnet_sex_age_group_dro` | `20261013T153530Z-spatial-lora-ham10000-fc-from-resnet-sex-age-group-dro-s42-7152` | `20261013T154433Z-spatial-lora-ham10000-fc-from-resnet-sex-age-group-dro-s43-3802` | `20261013T155333Z-spatial-lora-ham10000-fc-from-resnet-sex-age-group-dro-s44-0d6d` |
| `spatial_lora_ham10000_fc_from_resnet_sex_age_eo_rate_dual_holdout` | `20261013T153526Z-spatial-lora-ham10000-fc-from-resnet-sex-age-eo-rate-dual-holdout-s42-fd15` | `20261013T155856Z-spatial-lora-ham10000-fc-from-resnet-sex-age-eo-rate-dual-holdout-s43-0591` | `20261013T161730Z-spatial-lora-ham10000-fc-from-resnet-sex-age-eo-rate-dual-holdout-s44-5c07` |

## 取り込み済み run

上の 45 run を `analysis/common/runs.py import` で取り込んだ（2026-10-14）。新規 36 run は best val AUROC の checkpoint だけを回収し（SHA256 は run.json と一致）、取り込み後に同じ inode の hardlink で `runs/` に足した。last と EO rate の各 epoch の checkpoint は ws06 にだけある。

| run-id | project | run path |
|---|---|---|
| `20261006T155816Z-resnet-ham10000-s42-8a40` | `hypernet_e2e` | `analysis/hypernet_conditioning_grid_ham10000/runs/20261006T155816Z-resnet-ham10000-s42-8a40` |
| `20261006T160730Z-resnet-ham10000-s43-1338` | `hypernet_e2e` | `analysis/hypernet_conditioning_grid_ham10000/runs/20261006T160730Z-resnet-ham10000-s43-1338` |
| `20261006T161647Z-resnet-ham10000-s44-d3bc` | `hypernet_e2e` | `analysis/hypernet_conditioning_grid_ham10000/runs/20261006T161647Z-resnet-ham10000-s44-d3bc` |
| `20261006T155802Z-spatial-lora-ham10000-s42-d782` | `hypernet_e2e` | `analysis/hypernet_conditioning_grid_ham10000/runs/20261006T155802Z-spatial-lora-ham10000-s42-d782` |
| `20261006T160741Z-spatial-lora-ham10000-s43-bdd1` | `hypernet_e2e` | `analysis/hypernet_conditioning_grid_ham10000/runs/20261006T160741Z-spatial-lora-ham10000-s43-bdd1` |
| `20261006T161713Z-spatial-lora-ham10000-s44-513e` | `hypernet_e2e` | `analysis/hypernet_conditioning_grid_ham10000/runs/20261006T161713Z-spatial-lora-ham10000-s44-513e` |
| `20261009T104332Z-spatial-lora-ham10000-fc-s42-b253` | `hypernet_e2e` | `analysis/hypernet_conditioning_grid_ham10000/runs/20261009T104332Z-spatial-lora-ham10000-fc-s42-b253` |
| `20261009T105230Z-spatial-lora-ham10000-fc-s43-6ded` | `hypernet_e2e` | `analysis/hypernet_conditioning_grid_ham10000/runs/20261009T105230Z-spatial-lora-ham10000-fc-s43-6ded` |
| `20261009T110118Z-spatial-lora-ham10000-fc-s44-36f3` | `hypernet_e2e` | `analysis/hypernet_conditioning_grid_ham10000/runs/20261009T110118Z-spatial-lora-ham10000-fc-s44-36f3` |
| `20261013T153508Z-resnet-ham10000-sex-age-group-dro-s42-f3c4` | `hypernet_e2e` | `analysis/hypernet_conditioning_grid_ham10000/runs/20261013T153508Z-resnet-ham10000-sex-age-group-dro-s42-f3c4` |
| `20261013T153518Z-spatial-lora-ham10000-sex-age-eo-rate-dual-holdout-s42-5ba4` | `hypernet_e2e` | `analysis/hypernet_conditioning_grid_ham10000/runs/20261013T153518Z-spatial-lora-ham10000-sex-age-eo-rate-dual-holdout-s42-5ba4` |
| `20261013T153526Z-resnet-ham10000-sex-age-eo-rate-dual-holdout-s42-442d` | `hypernet_e2e` | `analysis/hypernet_conditioning_grid_ham10000/runs/20261013T153526Z-resnet-ham10000-sex-age-eo-rate-dual-holdout-s42-442d` |
| `20261013T153526Z-spatial-lora-ham10000-fc-from-resnet-sex-age-eo-rate-dual-holdout-s42-fd15` | `hypernet_e2e` | `analysis/hypernet_conditioning_grid_ham10000/runs/20261013T153526Z-spatial-lora-ham10000-fc-from-resnet-sex-age-eo-rate-dual-holdout-s42-fd15` |
| `20261013T153526Z-spatial-lora-ham10000-from-resnet-sex-age-eo-rate-dual-holdout-s42-13c6` | `hypernet_e2e` | `analysis/hypernet_conditioning_grid_ham10000/runs/20261013T153526Z-spatial-lora-ham10000-from-resnet-sex-age-eo-rate-dual-holdout-s42-13c6` |
| `20261013T153529Z-spatial-lora-ham10000-from-resnet-s42-855b` | `hypernet_e2e` | `analysis/hypernet_conditioning_grid_ham10000/runs/20261013T153529Z-spatial-lora-ham10000-from-resnet-s42-855b` |
| `20261013T153530Z-spatial-lora-ham10000-fc-from-resnet-s42-f89c` | `hypernet_e2e` | `analysis/hypernet_conditioning_grid_ham10000/runs/20261013T153530Z-spatial-lora-ham10000-fc-from-resnet-s42-f89c` |
| `20261013T153530Z-spatial-lora-ham10000-fc-from-resnet-sex-age-group-dro-s42-7152` | `hypernet_e2e` | `analysis/hypernet_conditioning_grid_ham10000/runs/20261013T153530Z-spatial-lora-ham10000-fc-from-resnet-sex-age-group-dro-s42-7152` |
| `20261013T153530Z-spatial-lora-ham10000-fc-sex-age-group-dro-s42-dc31` | `hypernet_e2e` | `analysis/hypernet_conditioning_grid_ham10000/runs/20261013T153530Z-spatial-lora-ham10000-fc-sex-age-group-dro-s42-dc31` |
| `20261013T153531Z-spatial-lora-ham10000-fc-sex-age-eo-rate-dual-holdout-s42-99fc` | `hypernet_e2e` | `analysis/hypernet_conditioning_grid_ham10000/runs/20261013T153531Z-spatial-lora-ham10000-fc-sex-age-eo-rate-dual-holdout-s42-99fc` |
| `20261013T153531Z-spatial-lora-ham10000-from-resnet-sex-age-group-dro-s42-2ea9` | `hypernet_e2e` | `analysis/hypernet_conditioning_grid_ham10000/runs/20261013T153531Z-spatial-lora-ham10000-from-resnet-sex-age-group-dro-s42-2ea9` |
| `20261013T153531Z-spatial-lora-ham10000-sex-age-group-dro-s42-c226` | `hypernet_e2e` | `analysis/hypernet_conditioning_grid_ham10000/runs/20261013T153531Z-spatial-lora-ham10000-sex-age-group-dro-s42-c226` |
| `20261013T154207Z-spatial-lora-ham10000-fc-from-resnet-s43-8cdb` | `hypernet_e2e` | `analysis/hypernet_conditioning_grid_ham10000/runs/20261013T154207Z-spatial-lora-ham10000-fc-from-resnet-s43-8cdb` |
| `20261013T154433Z-spatial-lora-ham10000-fc-from-resnet-sex-age-group-dro-s43-3802` | `hypernet_e2e` | `analysis/hypernet_conditioning_grid_ham10000/runs/20261013T154433Z-spatial-lora-ham10000-fc-from-resnet-sex-age-group-dro-s43-3802` |
| `20261013T154502Z-resnet-ham10000-sex-age-group-dro-s43-0cea` | `hypernet_e2e` | `analysis/hypernet_conditioning_grid_ham10000/runs/20261013T154502Z-resnet-ham10000-sex-age-group-dro-s43-0cea` |
| `20261013T154506Z-spatial-lora-ham10000-fc-sex-age-group-dro-s43-032d` | `hypernet_e2e` | `analysis/hypernet_conditioning_grid_ham10000/runs/20261013T154506Z-spatial-lora-ham10000-fc-sex-age-group-dro-s43-032d` |
| `20261013T154522Z-spatial-lora-ham10000-from-resnet-sex-age-group-dro-s43-55dd` | `hypernet_e2e` | `analysis/hypernet_conditioning_grid_ham10000/runs/20261013T154522Z-spatial-lora-ham10000-from-resnet-sex-age-group-dro-s43-55dd` |
| `20261013T154735Z-spatial-lora-ham10000-from-resnet-s43-35aa` | `hypernet_e2e` | `analysis/hypernet_conditioning_grid_ham10000/runs/20261013T154735Z-spatial-lora-ham10000-from-resnet-s43-35aa` |
| `20261013T154832Z-spatial-lora-ham10000-fc-from-resnet-s44-5b08` | `hypernet_e2e` | `analysis/hypernet_conditioning_grid_ham10000/runs/20261013T154832Z-spatial-lora-ham10000-fc-from-resnet-s44-5b08` |
| `20261013T155300Z-resnet-ham10000-sex-age-eo-rate-dual-holdout-s43-bee9` | `hypernet_e2e` | `analysis/hypernet_conditioning_grid_ham10000/runs/20261013T155300Z-resnet-ham10000-sex-age-eo-rate-dual-holdout-s43-bee9` |
| `20261013T155333Z-spatial-lora-ham10000-fc-from-resnet-sex-age-group-dro-s44-0d6d` | `hypernet_e2e` | `analysis/hypernet_conditioning_grid_ham10000/runs/20261013T155333Z-spatial-lora-ham10000-fc-from-resnet-sex-age-group-dro-s44-0d6d` |
| `20261013T155421Z-spatial-lora-ham10000-fc-sex-age-group-dro-s44-a28d` | `hypernet_e2e` | `analysis/hypernet_conditioning_grid_ham10000/runs/20261013T155421Z-spatial-lora-ham10000-fc-sex-age-group-dro-s44-a28d` |
| `20261013T155430Z-spatial-lora-ham10000-sex-age-group-dro-s43-6fdd` | `hypernet_e2e` | `analysis/hypernet_conditioning_grid_ham10000/runs/20261013T155430Z-spatial-lora-ham10000-sex-age-group-dro-s43-6fdd` |
| `20261013T155507Z-resnet-ham10000-sex-age-group-dro-s44-5286` | `hypernet_e2e` | `analysis/hypernet_conditioning_grid_ham10000/runs/20261013T155507Z-resnet-ham10000-sex-age-group-dro-s44-5286` |
| `20261013T155507Z-spatial-lora-ham10000-from-resnet-sex-age-group-dro-s44-fbc0` | `hypernet_e2e` | `analysis/hypernet_conditioning_grid_ham10000/runs/20261013T155507Z-spatial-lora-ham10000-from-resnet-sex-age-group-dro-s44-fbc0` |
| `20261013T155556Z-spatial-lora-ham10000-fc-sex-age-eo-rate-dual-holdout-s43-9920` | `hypernet_e2e` | `analysis/hypernet_conditioning_grid_ham10000/runs/20261013T155556Z-spatial-lora-ham10000-fc-sex-age-eo-rate-dual-holdout-s43-9920` |
| `20261013T155856Z-spatial-lora-ham10000-fc-from-resnet-sex-age-eo-rate-dual-holdout-s43-0591` | `hypernet_e2e` | `analysis/hypernet_conditioning_grid_ham10000/runs/20261013T155856Z-spatial-lora-ham10000-fc-from-resnet-sex-age-eo-rate-dual-holdout-s43-0591` |
| `20261013T155932Z-spatial-lora-ham10000-from-resnet-s44-e782` | `hypernet_e2e` | `analysis/hypernet_conditioning_grid_ham10000/runs/20261013T155932Z-spatial-lora-ham10000-from-resnet-s44-e782` |
| `20261013T160404Z-spatial-lora-ham10000-from-resnet-sex-age-eo-rate-dual-holdout-s43-b649` | `hypernet_e2e` | `analysis/hypernet_conditioning_grid_ham10000/runs/20261013T160404Z-spatial-lora-ham10000-from-resnet-sex-age-eo-rate-dual-holdout-s43-b649` |
| `20261013T160450Z-spatial-lora-ham10000-sex-age-eo-rate-dual-holdout-s43-502f` | `hypernet_e2e` | `analysis/hypernet_conditioning_grid_ham10000/runs/20261013T160450Z-spatial-lora-ham10000-sex-age-eo-rate-dual-holdout-s43-502f` |
| `20261013T160533Z-resnet-ham10000-sex-age-eo-rate-dual-holdout-s44-778e` | `hypernet_e2e` | `analysis/hypernet_conditioning_grid_ham10000/runs/20261013T160533Z-resnet-ham10000-sex-age-eo-rate-dual-holdout-s44-778e` |
| `20261013T161032Z-spatial-lora-ham10000-sex-age-group-dro-s44-fbd7` | `hypernet_e2e` | `analysis/hypernet_conditioning_grid_ham10000/runs/20261013T161032Z-spatial-lora-ham10000-sex-age-group-dro-s44-fbd7` |
| `20261013T161036Z-spatial-lora-ham10000-fc-sex-age-eo-rate-dual-holdout-s44-e590` | `hypernet_e2e` | `analysis/hypernet_conditioning_grid_ham10000/runs/20261013T161036Z-spatial-lora-ham10000-fc-sex-age-eo-rate-dual-holdout-s44-e590` |
| `20261013T161730Z-spatial-lora-ham10000-fc-from-resnet-sex-age-eo-rate-dual-holdout-s44-5c07` | `hypernet_e2e` | `analysis/hypernet_conditioning_grid_ham10000/runs/20261013T161730Z-spatial-lora-ham10000-fc-from-resnet-sex-age-eo-rate-dual-holdout-s44-5c07` |
| `20261013T162532Z-spatial-lora-ham10000-sex-age-eo-rate-dual-holdout-s44-f5ee` | `hypernet_e2e` | `analysis/hypernet_conditioning_grid_ham10000/runs/20261013T162532Z-spatial-lora-ham10000-sex-age-eo-rate-dual-holdout-s44-f5ee` |
| `20261013T162629Z-spatial-lora-ham10000-from-resnet-sex-age-eo-rate-dual-holdout-s44-acae` | `hypernet_e2e` | `analysis/hypernet_conditioning_grid_ham10000/runs/20261013T162629Z-spatial-lora-ham10000-from-resnet-sex-age-eo-rate-dual-holdout-s44-acae` |

## EO を Cotter et al. の形で回し直した 5 条件（2026-10-15）

全ペアの EO rate（上の表）は原著の制約と形が違うので、EO の列を Communities の形で回し直す。各群の FPR を
全体（train の陰性の群構成で重み付け）と比べる片側の制約 `FPR_g − FPR_全体 ≤ 0.05` を群の数だけかけ、TPR には
かけない。λ は一様から始める。その他の設定は全ペアの版と同じ（`test_eo_fpr_overall_presets_change_only_the_constraint`）。
出力は best val AUROC ではなく、holdout で制約を満たす epoch のうち目的関数が最もよいものを分析側で選ぶ予定。

ws06 で 2026-10-14 23:58 JST に起動（commit `d3455cf`、`run_logs/hcgo_eo_fpr_overall_ws06_launch_20261014.sh`）。1 条件 1 container で 3 seed を直列に回す。30 run とも 2026-10-15 に succeeded（30 epoch、checkpoint は全 epoch + last）。metadata は 2026-10-16 に回収した。

| preset | seed 42 | seed 43 | seed 44 |
| --- | --- | --- | --- |
| `resnet_ham10000_sex_age_eo_fpr_overall_dual_holdout` | `20261014T145832Z-resnet-ham10000-sex-age-eo-fpr-overall-dual-holdout-s42-c545` | `20261014T152841Z-resnet-ham10000-sex-age-eo-fpr-overall-dual-holdout-s43-8cf1` | `20261014T155834Z-resnet-ham10000-sex-age-eo-fpr-overall-dual-holdout-s44-44ac` |
| `spatial_lora_ham10000_sex_age_eo_fpr_overall_dual_holdout` | `20261014T145835Z-spatial-lora-ham10000-sex-age-eo-fpr-overall-dual-holdout-s42-d946` | `20261014T153901Z-spatial-lora-ham10000-sex-age-eo-fpr-overall-dual-holdout-s43-bdb1` | `20261014T161900Z-spatial-lora-ham10000-sex-age-eo-fpr-overall-dual-holdout-s44-8ccf` |
| `spatial_lora_ham10000_fc_sex_age_eo_fpr_overall_dual_holdout` | `20261014T145836Z-spatial-lora-ham10000-fc-sex-age-eo-fpr-overall-dual-holdout-s42-e1b6` | `20261014T153240Z-spatial-lora-ham10000-fc-sex-age-eo-fpr-overall-dual-holdout-s43-b283` | `20261014T160548Z-spatial-lora-ham10000-fc-sex-age-eo-fpr-overall-dual-holdout-s44-683e` |
| `spatial_lora_ham10000_from_resnet_sex_age_eo_fpr_overall_dual_holdout` | `20261014T145821Z-spatial-lora-ham10000-from-resnet-sex-age-eo-fpr-overall-dual-holdout-s42-dc36` | `20261014T152142Z-spatial-lora-ham10000-from-resnet-sex-age-eo-fpr-overall-dual-holdout-s43-3771` | `20261014T154428Z-spatial-lora-ham10000-from-resnet-sex-age-eo-fpr-overall-dual-holdout-s44-b9d6` |
| `spatial_lora_ham10000_fc_from_resnet_sex_age_eo_fpr_overall_dual_holdout` | `20261014T145835Z-spatial-lora-ham10000-fc-from-resnet-sex-age-eo-fpr-overall-dual-holdout-s42-d47e` | `20261014T151639Z-spatial-lora-ham10000-fc-from-resnet-sex-age-eo-fpr-overall-dual-holdout-s43-2f40` | `20261014T153411Z-spatial-lora-ham10000-fc-from-resnet-sex-age-eo-fpr-overall-dual-holdout-s44-ac94` |
