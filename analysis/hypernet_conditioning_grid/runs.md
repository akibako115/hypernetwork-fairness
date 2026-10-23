# runs

この package が読む run（CheXpert と HAM10000）。設定・状態・数値の正本は各 run の artifact で、ここには run ID と状態だけを書く。

## CheXpert：1 段の ERM（再利用。study=two_stage_vs_e2e_age_race_group_dro）

| モデル | seed 42 | seed 43 | seed 44 |
| --- | --- | --- | --- |
| ResNet-50 | `20261002T124957Z-resnet-chexpert-race-known-s42-d6d3` | `20261002T142200Z-resnet-chexpert-race-known-s43-bbfc` | `20261002T155222Z-resnet-chexpert-race-known-s44-3d73` |
| Spatial LoRA | `20261002T125005Z-spatial-lora-chexpert-race-known-s42-a21c` | `20261002T145233Z-spatial-lora-chexpert-race-known-s43-424e` | `20261002T165551Z-spatial-lora-chexpert-race-known-s44-eb13` |

## CheXpert：新規の 13 条件（study=hypernet_conditioning_grid）

全ペアの EO rate 5 条件（15 run）は 2026-10-15 に表と `runs/` から外した（下の「外した run」）。

ws11 で 2026-10-13 02:30 JST に起動し、同日 18 時ごろまでに 39 本すべてが `succeeded`（commit `337e381`、
`run_logs/hcg_chexpert_ws11_launch_20261013.sh`）。各条件の 3 seed は同じ container で直列に回した。
checkpoint は ws11 にあり、ローカルへは未回収。

| preset | seed 42 | seed 43 | seed 44 |
| --- | --- | --- | --- |
| `resnet_chexpert_age_race_group_dro_race_known` | `20261012T173024Z-resnet-chexpert-age-race-group-dro-race-known-s42-0443` | `20261012T195356Z-resnet-chexpert-age-race-group-dro-race-known-s43-0eeb` | `20261012T221712Z-resnet-chexpert-age-race-group-dro-race-known-s44-958f` |
| `spatial_lora_chexpert_age_race_group_dro_race_known` | `20261012T173024Z-spatial-lora-chexpert-age-race-group-dro-race-known-s42-3f49` | `20261012T202127Z-spatial-lora-chexpert-age-race-group-dro-race-known-s43-976b` | `20261012T230150Z-spatial-lora-chexpert-age-race-group-dro-race-known-s44-7aaa` |
| `spatial_lora_chexpert_fc_age_race_group_dro_race_known` | `20261012T173017Z-spatial-lora-chexpert-fc-age-race-group-dro-race-known-s42-3c3d` | `20261012T194131Z-spatial-lora-chexpert-fc-age-race-group-dro-race-known-s43-fdbb` | `20261012T220517Z-spatial-lora-chexpert-fc-age-race-group-dro-race-known-s44-39f5` |
| `spatial_lora_chexpert_fc_from_resnet_age_race_group_dro_race_known` | `20261012T173037Z-spatial-lora-chexpert-fc-from-resnet-age-race-group-dro-race-known-s42-5660` | `20261012T191323Z-spatial-lora-chexpert-fc-from-resnet-age-race-group-dro-race-known-s43-72cb` | `20261012T204811Z-spatial-lora-chexpert-fc-from-resnet-age-race-group-dro-race-known-s44-e586` |
| `spatial_lora_chexpert_fc_from_resnet_race_known` | `20261012T173026Z-spatial-lora-chexpert-fc-from-resnet-race-known-s42-01b3` | `20261012T192345Z-spatial-lora-chexpert-fc-from-resnet-race-known-s43-2ece` | `20261012T212104Z-spatial-lora-chexpert-fc-from-resnet-race-known-s44-0fa5` |
| `spatial_lora_chexpert_fc_race_known` | `20261012T173007Z-spatial-lora-chexpert-fc-race-known-s42-281b` | `20261012T201302Z-spatial-lora-chexpert-fc-race-known-s43-91f9` | `20261012T230129Z-spatial-lora-chexpert-fc-race-known-s44-14bd` |
| `spatial_lora_chexpert_from_resnet_age_race_group_dro_race_known` | `20261012T173024Z-spatial-lora-chexpert-from-resnet-age-race-group-dro-race-known-s42-8732` | `20261012T200858Z-spatial-lora-chexpert-from-resnet-age-race-group-dro-race-known-s43-b570` | `20261012T223924Z-spatial-lora-chexpert-from-resnet-age-race-group-dro-race-known-s44-04d5` |
| `spatial_lora_chexpert_from_resnet_race_known` | `20261012T173024Z-spatial-lora-chexpert-from-resnet-race-known-s42-08b5` | `20261012T201053Z-spatial-lora-chexpert-from-resnet-race-known-s43-8a37` | `20261012T224546Z-spatial-lora-chexpert-from-resnet-race-known-s44-3d18` |

## CheXpert：図1〜6が読む取り込み済みrun

以下はrun artifactから生成した入力一覧。新規39本のcheckpointはws11に置き、そこで生成した予測cacheをローカルへ回収する。
比較では各runのbest val AUROCを使用し、cacheのcheckpoint SHA256をrun.jsonと照合する。

| run-id | project | run path | W&B |
| --- | --- | --- | --- |
| `20261002T124957Z-resnet-chexpert-race-known-s42-d6d3` | `hypernet_e2e` | `analysis/hypernet_conditioning_grid/runs/20261002T124957Z-resnet-chexpert-race-known-s42-d6d3` | [rswjo11h](https://wandb.ai/kohki-akiba-kyushu-university/fairness_hypernet/runs/rswjo11h) |
| `20261002T142200Z-resnet-chexpert-race-known-s43-bbfc` | `hypernet_e2e` | `analysis/hypernet_conditioning_grid/runs/20261002T142200Z-resnet-chexpert-race-known-s43-bbfc` | [kb7vff02](https://wandb.ai/kohki-akiba-kyushu-university/fairness_hypernet/runs/kb7vff02) |
| `20261002T155222Z-resnet-chexpert-race-known-s44-3d73` | `hypernet_e2e` | `analysis/hypernet_conditioning_grid/runs/20261002T155222Z-resnet-chexpert-race-known-s44-3d73` | [btr0ab3s](https://wandb.ai/kohki-akiba-kyushu-university/fairness_hypernet/runs/btr0ab3s) |
| `20261002T125005Z-spatial-lora-chexpert-race-known-s42-a21c` | `hypernet_e2e` | `analysis/hypernet_conditioning_grid/runs/20261002T125005Z-spatial-lora-chexpert-race-known-s42-a21c` | [cnsy8b78](https://wandb.ai/kohki-akiba-kyushu-university/fairness_hypernet/runs/cnsy8b78) |
| `20261002T145233Z-spatial-lora-chexpert-race-known-s43-424e` | `hypernet_e2e` | `analysis/hypernet_conditioning_grid/runs/20261002T145233Z-spatial-lora-chexpert-race-known-s43-424e` | [5769xfxy](https://wandb.ai/kohki-akiba-kyushu-university/fairness_hypernet/runs/5769xfxy) |
| `20261002T165551Z-spatial-lora-chexpert-race-known-s44-eb13` | `hypernet_e2e` | `analysis/hypernet_conditioning_grid/runs/20261002T165551Z-spatial-lora-chexpert-race-known-s44-eb13` | [gxwzjd75](https://wandb.ai/kohki-akiba-kyushu-university/fairness_hypernet/runs/gxwzjd75) |
| `20261012T173024Z-resnet-chexpert-age-race-group-dro-race-known-s42-0443` | `hypernet_e2e` | `analysis/hypernet_conditioning_grid/runs/20261012T173024Z-resnet-chexpert-age-race-group-dro-race-known-s42-0443` | [tsko2nj9](https://wandb.ai/kohki-akiba-kyushu-university/fairness_hypernet/runs/tsko2nj9) |
| `20261012T195356Z-resnet-chexpert-age-race-group-dro-race-known-s43-0eeb` | `hypernet_e2e` | `analysis/hypernet_conditioning_grid/runs/20261012T195356Z-resnet-chexpert-age-race-group-dro-race-known-s43-0eeb` | [smc6l83r](https://wandb.ai/kohki-akiba-kyushu-university/fairness_hypernet/runs/smc6l83r) |
| `20261012T221712Z-resnet-chexpert-age-race-group-dro-race-known-s44-958f` | `hypernet_e2e` | `analysis/hypernet_conditioning_grid/runs/20261012T221712Z-resnet-chexpert-age-race-group-dro-race-known-s44-958f` | [hifpy6up](https://wandb.ai/kohki-akiba-kyushu-university/fairness_hypernet/runs/hifpy6up) |
| `20261012T173024Z-spatial-lora-chexpert-age-race-group-dro-race-known-s42-3f49` | `hypernet_e2e` | `analysis/hypernet_conditioning_grid/runs/20261012T173024Z-spatial-lora-chexpert-age-race-group-dro-race-known-s42-3f49` | [z98zozx6](https://wandb.ai/kohki-akiba-kyushu-university/fairness_hypernet/runs/z98zozx6) |
| `20261012T202127Z-spatial-lora-chexpert-age-race-group-dro-race-known-s43-976b` | `hypernet_e2e` | `analysis/hypernet_conditioning_grid/runs/20261012T202127Z-spatial-lora-chexpert-age-race-group-dro-race-known-s43-976b` | [kvl5pmae](https://wandb.ai/kohki-akiba-kyushu-university/fairness_hypernet/runs/kvl5pmae) |
| `20261012T230150Z-spatial-lora-chexpert-age-race-group-dro-race-known-s44-7aaa` | `hypernet_e2e` | `analysis/hypernet_conditioning_grid/runs/20261012T230150Z-spatial-lora-chexpert-age-race-group-dro-race-known-s44-7aaa` | [coa23szj](https://wandb.ai/kohki-akiba-kyushu-university/fairness_hypernet/runs/coa23szj) |
| `20261012T173017Z-spatial-lora-chexpert-fc-age-race-group-dro-race-known-s42-3c3d` | `hypernet_e2e` | `analysis/hypernet_conditioning_grid/runs/20261012T173017Z-spatial-lora-chexpert-fc-age-race-group-dro-race-known-s42-3c3d` | [9g98hkqc](https://wandb.ai/kohki-akiba-kyushu-university/fairness_hypernet/runs/9g98hkqc) |
| `20261012T194131Z-spatial-lora-chexpert-fc-age-race-group-dro-race-known-s43-fdbb` | `hypernet_e2e` | `analysis/hypernet_conditioning_grid/runs/20261012T194131Z-spatial-lora-chexpert-fc-age-race-group-dro-race-known-s43-fdbb` | [k1tyoprv](https://wandb.ai/kohki-akiba-kyushu-university/fairness_hypernet/runs/k1tyoprv) |
| `20261012T220517Z-spatial-lora-chexpert-fc-age-race-group-dro-race-known-s44-39f5` | `hypernet_e2e` | `analysis/hypernet_conditioning_grid/runs/20261012T220517Z-spatial-lora-chexpert-fc-age-race-group-dro-race-known-s44-39f5` | [vm5lphg8](https://wandb.ai/kohki-akiba-kyushu-university/fairness_hypernet/runs/vm5lphg8) |
| `20261012T173037Z-spatial-lora-chexpert-fc-from-resnet-age-race-group-dro-race-known-s42-5660` | `hypernet_e2e` | `analysis/hypernet_conditioning_grid/runs/20261012T173037Z-spatial-lora-chexpert-fc-from-resnet-age-race-group-dro-race-known-s42-5660` | [7od36lbi](https://wandb.ai/kohki-akiba-kyushu-university/fairness_hypernet/runs/7od36lbi) |
| `20261012T191323Z-spatial-lora-chexpert-fc-from-resnet-age-race-group-dro-race-known-s43-72cb` | `hypernet_e2e` | `analysis/hypernet_conditioning_grid/runs/20261012T191323Z-spatial-lora-chexpert-fc-from-resnet-age-race-group-dro-race-known-s43-72cb` | [y95dsedk](https://wandb.ai/kohki-akiba-kyushu-university/fairness_hypernet/runs/y95dsedk) |
| `20261012T204811Z-spatial-lora-chexpert-fc-from-resnet-age-race-group-dro-race-known-s44-e586` | `hypernet_e2e` | `analysis/hypernet_conditioning_grid/runs/20261012T204811Z-spatial-lora-chexpert-fc-from-resnet-age-race-group-dro-race-known-s44-e586` | [ifiddt43](https://wandb.ai/kohki-akiba-kyushu-university/fairness_hypernet/runs/ifiddt43) |
| `20261012T173026Z-spatial-lora-chexpert-fc-from-resnet-race-known-s42-01b3` | `hypernet_e2e` | `analysis/hypernet_conditioning_grid/runs/20261012T173026Z-spatial-lora-chexpert-fc-from-resnet-race-known-s42-01b3` | [69a7on5e](https://wandb.ai/kohki-akiba-kyushu-university/fairness_hypernet/runs/69a7on5e) |
| `20261012T192345Z-spatial-lora-chexpert-fc-from-resnet-race-known-s43-2ece` | `hypernet_e2e` | `analysis/hypernet_conditioning_grid/runs/20261012T192345Z-spatial-lora-chexpert-fc-from-resnet-race-known-s43-2ece` | [p91w1zec](https://wandb.ai/kohki-akiba-kyushu-university/fairness_hypernet/runs/p91w1zec) |
| `20261012T212104Z-spatial-lora-chexpert-fc-from-resnet-race-known-s44-0fa5` | `hypernet_e2e` | `analysis/hypernet_conditioning_grid/runs/20261012T212104Z-spatial-lora-chexpert-fc-from-resnet-race-known-s44-0fa5` | [i47p6kke](https://wandb.ai/kohki-akiba-kyushu-university/fairness_hypernet/runs/i47p6kke) |
| `20261012T173007Z-spatial-lora-chexpert-fc-race-known-s42-281b` | `hypernet_e2e` | `analysis/hypernet_conditioning_grid/runs/20261012T173007Z-spatial-lora-chexpert-fc-race-known-s42-281b` | [jgqe92g4](https://wandb.ai/kohki-akiba-kyushu-university/fairness_hypernet/runs/jgqe92g4) |
| `20261012T201302Z-spatial-lora-chexpert-fc-race-known-s43-91f9` | `hypernet_e2e` | `analysis/hypernet_conditioning_grid/runs/20261012T201302Z-spatial-lora-chexpert-fc-race-known-s43-91f9` | [jwxkunb3](https://wandb.ai/kohki-akiba-kyushu-university/fairness_hypernet/runs/jwxkunb3) |
| `20261012T230129Z-spatial-lora-chexpert-fc-race-known-s44-14bd` | `hypernet_e2e` | `analysis/hypernet_conditioning_grid/runs/20261012T230129Z-spatial-lora-chexpert-fc-race-known-s44-14bd` | [p4aq2ty5](https://wandb.ai/kohki-akiba-kyushu-university/fairness_hypernet/runs/p4aq2ty5) |
| `20261012T173024Z-spatial-lora-chexpert-from-resnet-age-race-group-dro-race-known-s42-8732` | `hypernet_e2e` | `analysis/hypernet_conditioning_grid/runs/20261012T173024Z-spatial-lora-chexpert-from-resnet-age-race-group-dro-race-known-s42-8732` | [i7ur3kex](https://wandb.ai/kohki-akiba-kyushu-university/fairness_hypernet/runs/i7ur3kex) |
| `20261012T200858Z-spatial-lora-chexpert-from-resnet-age-race-group-dro-race-known-s43-b570` | `hypernet_e2e` | `analysis/hypernet_conditioning_grid/runs/20261012T200858Z-spatial-lora-chexpert-from-resnet-age-race-group-dro-race-known-s43-b570` | [z2bkl5vb](https://wandb.ai/kohki-akiba-kyushu-university/fairness_hypernet/runs/z2bkl5vb) |
| `20261012T223924Z-spatial-lora-chexpert-from-resnet-age-race-group-dro-race-known-s44-04d5` | `hypernet_e2e` | `analysis/hypernet_conditioning_grid/runs/20261012T223924Z-spatial-lora-chexpert-from-resnet-age-race-group-dro-race-known-s44-04d5` | [g58sbw4l](https://wandb.ai/kohki-akiba-kyushu-university/fairness_hypernet/runs/g58sbw4l) |
| `20261012T173024Z-spatial-lora-chexpert-from-resnet-race-known-s42-08b5` | `hypernet_e2e` | `analysis/hypernet_conditioning_grid/runs/20261012T173024Z-spatial-lora-chexpert-from-resnet-race-known-s42-08b5` | [rddfpkpd](https://wandb.ai/kohki-akiba-kyushu-university/fairness_hypernet/runs/rddfpkpd) |
| `20261012T201053Z-spatial-lora-chexpert-from-resnet-race-known-s43-8a37` | `hypernet_e2e` | `analysis/hypernet_conditioning_grid/runs/20261012T201053Z-spatial-lora-chexpert-from-resnet-race-known-s43-8a37` | [vik7ccfl](https://wandb.ai/kohki-akiba-kyushu-university/fairness_hypernet/runs/vik7ccfl) |
| `20261012T224546Z-spatial-lora-chexpert-from-resnet-race-known-s44-3d18` | `hypernet_e2e` | `analysis/hypernet_conditioning_grid/runs/20261012T224546Z-spatial-lora-chexpert-from-resnet-race-known-s44-3d18` | [lqrezm9c](https://wandb.ai/kohki-akiba-kyushu-university/fairness_hypernet/runs/lqrezm9c) |

## CheXpert：外した run（全ペアの EO rate、2026-10-15）

全ペアの制約（8 群の有向ペア 112 本、TPR・FPR とも ε=0.1、λ の初期 task 重み 0.9）は原著（Cotter et al.）の制約と形が違う。
EO の列は下の「EO を Cotter et al. の形で回し直した 5 条件」に置き換えるので、この 15 run を `runs/` の hardlink ごと外した。
run 本体は `projects/hypernet_e2e/runs/` に残っている。`compute.runs()` が拾わないよう、run ID は backtick で囲まずに書く。
2026-10-14 の図（`figures/requested_figures/`・`results/requested_figures/`）と予測 cache（`cache/` の 60 本）は、外す前のこの 15 run を含む。

- resnet-chexpert-age-race-eo-rate-dual-holdout：20261012T173008Z-resnet-chexpert-age-race-eo-rate-dual-holdout-s42-945a、20261012T201745Z-resnet-chexpert-age-race-eo-rate-dual-holdout-s43-4bbf、20261012T232426Z-resnet-chexpert-age-race-eo-rate-dual-holdout-s44-b7da
- spatial-lora-chexpert-from-resnet-age-race-eo-rate-dual-holdout：20261012T173014Z-spatial-lora-chexpert-from-resnet-age-race-eo-rate-dual-holdout-s42-749a、20261013T013113Z-spatial-lora-chexpert-from-resnet-age-race-eo-rate-dual-holdout-s43-f1c8、20261013T064947Z-spatial-lora-chexpert-from-resnet-age-race-eo-rate-dual-holdout-s44-3f2e
- spatial-lora-chexpert-age-race-eo-rate-dual-holdout：20261012T173018Z-spatial-lora-chexpert-age-race-eo-rate-dual-holdout-s42-8b3d、20261012T210435Z-spatial-lora-chexpert-age-race-eo-rate-dual-holdout-s43-ed72、20261013T003740Z-spatial-lora-chexpert-age-race-eo-rate-dual-holdout-s44-01a4
- spatial-lora-chexpert-fc-age-race-eo-rate-dual-holdout：20261012T173021Z-spatial-lora-chexpert-fc-age-race-eo-rate-dual-holdout-s42-0c92、20261012T214436Z-spatial-lora-chexpert-fc-age-race-eo-rate-dual-holdout-s43-d173、20261013T003123Z-spatial-lora-chexpert-fc-age-race-eo-rate-dual-holdout-s44-f301
- spatial-lora-chexpert-fc-from-resnet-age-race-eo-rate-dual-holdout：20261012T173024Z-spatial-lora-chexpert-fc-from-resnet-age-race-eo-rate-dual-holdout-s42-ff21、20261012T221641Z-spatial-lora-chexpert-fc-from-resnet-age-race-eo-rate-dual-holdout-s43-ea9c、20261013T011825Z-spatial-lora-chexpert-fc-from-resnet-age-race-eo-rate-dual-holdout-s44-09c7

## CheXpert：EO を Cotter et al. の形で回し直した 5 条件（2026-10-15）

全ペアの EO rate（上の表）は原著の制約と形が違うので、EO の列を Communities の形で回し直す。各群の FPR を
全体（train の陰性の群構成で重み付け）と比べる片側の制約 `FPR_g − FPR_全体 ≤ 0.05` を群の数だけかけ、TPR には
かけない。λ は一様から始める。その他の設定は全ペアの版と同じ（`test_eo_fpr_overall_presets_change_only_the_constraint`）。
出力は best val AUROC ではなく、holdout で制約を満たす epoch のうち目的関数が最もよいものを分析側で選ぶ予定。

ws06 で 2026-10-14 23:58 JST に起動（commit `d3455cf`、`run_logs/hcgo_eo_fpr_overall_ws06_launch_20261014.sh`）。各 run を 1 container で並べた。

| preset | seed 42 | seed 43 | seed 44 |
| --- | --- | --- | --- |
| `resnet_chexpert_age_race_eo_fpr_overall_dual_holdout` | `20261014T145832Z-resnet-chexpert-age-race-eo-fpr-overall-dual-holdout-s42-7a51` | `20261014T145834Z-resnet-chexpert-age-race-eo-fpr-overall-dual-holdout-s43-2d8e` | `20261014T145831Z-resnet-chexpert-age-race-eo-fpr-overall-dual-holdout-s44-0c75` |
| `spatial_lora_chexpert_age_race_eo_fpr_overall_dual_holdout` | `20261014T145828Z-spatial-lora-chexpert-age-race-eo-fpr-overall-dual-holdout-s42-847c` | `20261014T145828Z-spatial-lora-chexpert-age-race-eo-fpr-overall-dual-holdout-s43-ec31` | `20261014T145833Z-spatial-lora-chexpert-age-race-eo-fpr-overall-dual-holdout-s44-cf6d` |
| `spatial_lora_chexpert_fc_age_race_eo_fpr_overall_dual_holdout` | `20261014T145834Z-spatial-lora-chexpert-fc-age-race-eo-fpr-overall-dual-holdout-s42-923a` | `20261014T145837Z-spatial-lora-chexpert-fc-age-race-eo-fpr-overall-dual-holdout-s43-0003` | `20261014T145833Z-spatial-lora-chexpert-fc-age-race-eo-fpr-overall-dual-holdout-s44-c70f` |
| `spatial_lora_chexpert_from_resnet_age_race_eo_fpr_overall_dual_holdout` | `20261014T145806Z-spatial-lora-chexpert-from-resnet-age-race-eo-fpr-overall-dual-holdout-s42-3bb8` | `20261014T145829Z-spatial-lora-chexpert-from-resnet-age-race-eo-fpr-overall-dual-holdout-s43-81c6` | `20261014T145835Z-spatial-lora-chexpert-from-resnet-age-race-eo-fpr-overall-dual-holdout-s44-acec` |
| `spatial_lora_chexpert_fc_from_resnet_age_race_eo_fpr_overall_dual_holdout` | `20261014T145833Z-spatial-lora-chexpert-fc-from-resnet-age-race-eo-fpr-overall-dual-holdout-s42-5178` | `20261014T145837Z-spatial-lora-chexpert-fc-from-resnet-age-race-eo-fpr-overall-dual-holdout-s43-188d` | `20261014T145832Z-spatial-lora-chexpert-fc-from-resnet-age-race-eo-fpr-overall-dual-holdout-s44-8697` |

## HAM10000：1 段の ERM（再利用）

ResNet・Spatial LoRA は study=ham10000_architecture_erm（ws11）。
fc のみは study=latent_bias_hypernet（ローカル）。解決済み config は Spatial LoRA と変調段だけが違う（実行場所の path と `persistent_workers` を除く）。
ResNet は 2 段の親を兼ねる（best val AUROC の checkpoint）。

| モデル | seed 42 | seed 43 | seed 44 |
| --- | --- | --- | --- |
| ResNet-50 | `20261006T155816Z-resnet-ham10000-s42-8a40` | `20261006T160730Z-resnet-ham10000-s43-1338` | `20261006T161647Z-resnet-ham10000-s44-d3bc` |
| 親の checkpoint | `best_val_auroc_005.ckpt` | `best_val_auroc_029.ckpt` | `best_val_auroc_017.ckpt` |
| Spatial LoRA | `20261006T155802Z-spatial-lora-ham10000-s42-d782` | `20261006T160741Z-spatial-lora-ham10000-s43-bdd1` | `20261006T161713Z-spatial-lora-ham10000-s44-513e` |
| Spatial LoRA（fc のみ） | `20261009T104332Z-spatial-lora-ham10000-fc-s42-b253` | `20261009T105230Z-spatial-lora-ham10000-fc-s43-6ded` | `20261009T110118Z-spatial-lora-ham10000-fc-s44-36f3` |

## HAM10000：新規の 12 条件（study=hypernet_conditioning_grid_ham10000）

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

## HAM10000：取り込み済み run

上の 45 run を `analysis/common/runs.py import` で取り込んだ（2026-10-14）。新規 36 run は best val AUROC の checkpoint だけを回収し（SHA256 は run.json と一致）、取り込み後に同じ inode の hardlink で `runs/` に足した。last と EO rate の各 epoch の checkpoint は ws06 にだけある。

| run-id | project | run path |
|---|---|---|
| `20261006T155816Z-resnet-ham10000-s42-8a40` | `hypernet_e2e` | `analysis/hypernet_conditioning_grid/runs/20261006T155816Z-resnet-ham10000-s42-8a40` |
| `20261006T160730Z-resnet-ham10000-s43-1338` | `hypernet_e2e` | `analysis/hypernet_conditioning_grid/runs/20261006T160730Z-resnet-ham10000-s43-1338` |
| `20261006T161647Z-resnet-ham10000-s44-d3bc` | `hypernet_e2e` | `analysis/hypernet_conditioning_grid/runs/20261006T161647Z-resnet-ham10000-s44-d3bc` |
| `20261006T155802Z-spatial-lora-ham10000-s42-d782` | `hypernet_e2e` | `analysis/hypernet_conditioning_grid/runs/20261006T155802Z-spatial-lora-ham10000-s42-d782` |
| `20261006T160741Z-spatial-lora-ham10000-s43-bdd1` | `hypernet_e2e` | `analysis/hypernet_conditioning_grid/runs/20261006T160741Z-spatial-lora-ham10000-s43-bdd1` |
| `20261006T161713Z-spatial-lora-ham10000-s44-513e` | `hypernet_e2e` | `analysis/hypernet_conditioning_grid/runs/20261006T161713Z-spatial-lora-ham10000-s44-513e` |
| `20261009T104332Z-spatial-lora-ham10000-fc-s42-b253` | `hypernet_e2e` | `analysis/hypernet_conditioning_grid/runs/20261009T104332Z-spatial-lora-ham10000-fc-s42-b253` |
| `20261009T105230Z-spatial-lora-ham10000-fc-s43-6ded` | `hypernet_e2e` | `analysis/hypernet_conditioning_grid/runs/20261009T105230Z-spatial-lora-ham10000-fc-s43-6ded` |
| `20261009T110118Z-spatial-lora-ham10000-fc-s44-36f3` | `hypernet_e2e` | `analysis/hypernet_conditioning_grid/runs/20261009T110118Z-spatial-lora-ham10000-fc-s44-36f3` |
| `20261013T153508Z-resnet-ham10000-sex-age-group-dro-s42-f3c4` | `hypernet_e2e` | `analysis/hypernet_conditioning_grid/runs/20261013T153508Z-resnet-ham10000-sex-age-group-dro-s42-f3c4` |
| `20261013T153518Z-spatial-lora-ham10000-sex-age-eo-rate-dual-holdout-s42-5ba4` | `hypernet_e2e` | `analysis/hypernet_conditioning_grid/runs/20261013T153518Z-spatial-lora-ham10000-sex-age-eo-rate-dual-holdout-s42-5ba4` |
| `20261013T153526Z-resnet-ham10000-sex-age-eo-rate-dual-holdout-s42-442d` | `hypernet_e2e` | `analysis/hypernet_conditioning_grid/runs/20261013T153526Z-resnet-ham10000-sex-age-eo-rate-dual-holdout-s42-442d` |
| `20261013T153526Z-spatial-lora-ham10000-fc-from-resnet-sex-age-eo-rate-dual-holdout-s42-fd15` | `hypernet_e2e` | `analysis/hypernet_conditioning_grid/runs/20261013T153526Z-spatial-lora-ham10000-fc-from-resnet-sex-age-eo-rate-dual-holdout-s42-fd15` |
| `20261013T153526Z-spatial-lora-ham10000-from-resnet-sex-age-eo-rate-dual-holdout-s42-13c6` | `hypernet_e2e` | `analysis/hypernet_conditioning_grid/runs/20261013T153526Z-spatial-lora-ham10000-from-resnet-sex-age-eo-rate-dual-holdout-s42-13c6` |
| `20261013T153529Z-spatial-lora-ham10000-from-resnet-s42-855b` | `hypernet_e2e` | `analysis/hypernet_conditioning_grid/runs/20261013T153529Z-spatial-lora-ham10000-from-resnet-s42-855b` |
| `20261013T153530Z-spatial-lora-ham10000-fc-from-resnet-s42-f89c` | `hypernet_e2e` | `analysis/hypernet_conditioning_grid/runs/20261013T153530Z-spatial-lora-ham10000-fc-from-resnet-s42-f89c` |
| `20261013T153530Z-spatial-lora-ham10000-fc-from-resnet-sex-age-group-dro-s42-7152` | `hypernet_e2e` | `analysis/hypernet_conditioning_grid/runs/20261013T153530Z-spatial-lora-ham10000-fc-from-resnet-sex-age-group-dro-s42-7152` |
| `20261013T153530Z-spatial-lora-ham10000-fc-sex-age-group-dro-s42-dc31` | `hypernet_e2e` | `analysis/hypernet_conditioning_grid/runs/20261013T153530Z-spatial-lora-ham10000-fc-sex-age-group-dro-s42-dc31` |
| `20261013T153531Z-spatial-lora-ham10000-fc-sex-age-eo-rate-dual-holdout-s42-99fc` | `hypernet_e2e` | `analysis/hypernet_conditioning_grid/runs/20261013T153531Z-spatial-lora-ham10000-fc-sex-age-eo-rate-dual-holdout-s42-99fc` |
| `20261013T153531Z-spatial-lora-ham10000-from-resnet-sex-age-group-dro-s42-2ea9` | `hypernet_e2e` | `analysis/hypernet_conditioning_grid/runs/20261013T153531Z-spatial-lora-ham10000-from-resnet-sex-age-group-dro-s42-2ea9` |
| `20261013T153531Z-spatial-lora-ham10000-sex-age-group-dro-s42-c226` | `hypernet_e2e` | `analysis/hypernet_conditioning_grid/runs/20261013T153531Z-spatial-lora-ham10000-sex-age-group-dro-s42-c226` |
| `20261013T154207Z-spatial-lora-ham10000-fc-from-resnet-s43-8cdb` | `hypernet_e2e` | `analysis/hypernet_conditioning_grid/runs/20261013T154207Z-spatial-lora-ham10000-fc-from-resnet-s43-8cdb` |
| `20261013T154433Z-spatial-lora-ham10000-fc-from-resnet-sex-age-group-dro-s43-3802` | `hypernet_e2e` | `analysis/hypernet_conditioning_grid/runs/20261013T154433Z-spatial-lora-ham10000-fc-from-resnet-sex-age-group-dro-s43-3802` |
| `20261013T154502Z-resnet-ham10000-sex-age-group-dro-s43-0cea` | `hypernet_e2e` | `analysis/hypernet_conditioning_grid/runs/20261013T154502Z-resnet-ham10000-sex-age-group-dro-s43-0cea` |
| `20261013T154506Z-spatial-lora-ham10000-fc-sex-age-group-dro-s43-032d` | `hypernet_e2e` | `analysis/hypernet_conditioning_grid/runs/20261013T154506Z-spatial-lora-ham10000-fc-sex-age-group-dro-s43-032d` |
| `20261013T154522Z-spatial-lora-ham10000-from-resnet-sex-age-group-dro-s43-55dd` | `hypernet_e2e` | `analysis/hypernet_conditioning_grid/runs/20261013T154522Z-spatial-lora-ham10000-from-resnet-sex-age-group-dro-s43-55dd` |
| `20261013T154735Z-spatial-lora-ham10000-from-resnet-s43-35aa` | `hypernet_e2e` | `analysis/hypernet_conditioning_grid/runs/20261013T154735Z-spatial-lora-ham10000-from-resnet-s43-35aa` |
| `20261013T154832Z-spatial-lora-ham10000-fc-from-resnet-s44-5b08` | `hypernet_e2e` | `analysis/hypernet_conditioning_grid/runs/20261013T154832Z-spatial-lora-ham10000-fc-from-resnet-s44-5b08` |
| `20261013T155300Z-resnet-ham10000-sex-age-eo-rate-dual-holdout-s43-bee9` | `hypernet_e2e` | `analysis/hypernet_conditioning_grid/runs/20261013T155300Z-resnet-ham10000-sex-age-eo-rate-dual-holdout-s43-bee9` |
| `20261013T155333Z-spatial-lora-ham10000-fc-from-resnet-sex-age-group-dro-s44-0d6d` | `hypernet_e2e` | `analysis/hypernet_conditioning_grid/runs/20261013T155333Z-spatial-lora-ham10000-fc-from-resnet-sex-age-group-dro-s44-0d6d` |
| `20261013T155421Z-spatial-lora-ham10000-fc-sex-age-group-dro-s44-a28d` | `hypernet_e2e` | `analysis/hypernet_conditioning_grid/runs/20261013T155421Z-spatial-lora-ham10000-fc-sex-age-group-dro-s44-a28d` |
| `20261013T155430Z-spatial-lora-ham10000-sex-age-group-dro-s43-6fdd` | `hypernet_e2e` | `analysis/hypernet_conditioning_grid/runs/20261013T155430Z-spatial-lora-ham10000-sex-age-group-dro-s43-6fdd` |
| `20261013T155507Z-resnet-ham10000-sex-age-group-dro-s44-5286` | `hypernet_e2e` | `analysis/hypernet_conditioning_grid/runs/20261013T155507Z-resnet-ham10000-sex-age-group-dro-s44-5286` |
| `20261013T155507Z-spatial-lora-ham10000-from-resnet-sex-age-group-dro-s44-fbc0` | `hypernet_e2e` | `analysis/hypernet_conditioning_grid/runs/20261013T155507Z-spatial-lora-ham10000-from-resnet-sex-age-group-dro-s44-fbc0` |
| `20261013T155556Z-spatial-lora-ham10000-fc-sex-age-eo-rate-dual-holdout-s43-9920` | `hypernet_e2e` | `analysis/hypernet_conditioning_grid/runs/20261013T155556Z-spatial-lora-ham10000-fc-sex-age-eo-rate-dual-holdout-s43-9920` |
| `20261013T155856Z-spatial-lora-ham10000-fc-from-resnet-sex-age-eo-rate-dual-holdout-s43-0591` | `hypernet_e2e` | `analysis/hypernet_conditioning_grid/runs/20261013T155856Z-spatial-lora-ham10000-fc-from-resnet-sex-age-eo-rate-dual-holdout-s43-0591` |
| `20261013T155932Z-spatial-lora-ham10000-from-resnet-s44-e782` | `hypernet_e2e` | `analysis/hypernet_conditioning_grid/runs/20261013T155932Z-spatial-lora-ham10000-from-resnet-s44-e782` |
| `20261013T160404Z-spatial-lora-ham10000-from-resnet-sex-age-eo-rate-dual-holdout-s43-b649` | `hypernet_e2e` | `analysis/hypernet_conditioning_grid/runs/20261013T160404Z-spatial-lora-ham10000-from-resnet-sex-age-eo-rate-dual-holdout-s43-b649` |
| `20261013T160450Z-spatial-lora-ham10000-sex-age-eo-rate-dual-holdout-s43-502f` | `hypernet_e2e` | `analysis/hypernet_conditioning_grid/runs/20261013T160450Z-spatial-lora-ham10000-sex-age-eo-rate-dual-holdout-s43-502f` |
| `20261013T160533Z-resnet-ham10000-sex-age-eo-rate-dual-holdout-s44-778e` | `hypernet_e2e` | `analysis/hypernet_conditioning_grid/runs/20261013T160533Z-resnet-ham10000-sex-age-eo-rate-dual-holdout-s44-778e` |
| `20261013T161032Z-spatial-lora-ham10000-sex-age-group-dro-s44-fbd7` | `hypernet_e2e` | `analysis/hypernet_conditioning_grid/runs/20261013T161032Z-spatial-lora-ham10000-sex-age-group-dro-s44-fbd7` |
| `20261013T161036Z-spatial-lora-ham10000-fc-sex-age-eo-rate-dual-holdout-s44-e590` | `hypernet_e2e` | `analysis/hypernet_conditioning_grid/runs/20261013T161036Z-spatial-lora-ham10000-fc-sex-age-eo-rate-dual-holdout-s44-e590` |
| `20261013T161730Z-spatial-lora-ham10000-fc-from-resnet-sex-age-eo-rate-dual-holdout-s44-5c07` | `hypernet_e2e` | `analysis/hypernet_conditioning_grid/runs/20261013T161730Z-spatial-lora-ham10000-fc-from-resnet-sex-age-eo-rate-dual-holdout-s44-5c07` |
| `20261013T162532Z-spatial-lora-ham10000-sex-age-eo-rate-dual-holdout-s44-f5ee` | `hypernet_e2e` | `analysis/hypernet_conditioning_grid/runs/20261013T162532Z-spatial-lora-ham10000-sex-age-eo-rate-dual-holdout-s44-f5ee` |
| `20261013T162629Z-spatial-lora-ham10000-from-resnet-sex-age-eo-rate-dual-holdout-s44-acae` | `hypernet_e2e` | `analysis/hypernet_conditioning_grid/runs/20261013T162629Z-spatial-lora-ham10000-from-resnet-sex-age-eo-rate-dual-holdout-s44-acae` |

## HAM10000：EO を Cotter et al. の形で回し直した 5 条件（2026-10-15）

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
