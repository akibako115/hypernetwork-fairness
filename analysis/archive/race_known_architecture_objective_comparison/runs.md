# runs

今回の18条件 × seed 42・43・44 = 54 runをこのpackageで引用する。
条件IDと比較の目的は[実験計画](../../../docs/plans/20261002_experiment.md)を参照。

## 同期・取り込み完了（2026-10-05）

全54 runが正常終了し、各30epochを記録。DRO42 runはCSVに30epoch分・8群のqを記録している。
best / last checkpoint計108本のSHA256をrun.jsonと照合済み。
ws11の48runは同期済み、localの6runと合わせてhardlinkで取り込んだ。
取り込んだrun artifactは書き換えない。W&B URL・設定・split・親checkpoint来歴は各run.json等が正本。
初回ERM等のstudyは旧名のまま保持し、ここから引用する。

## 採用対象

| run-id | project | run path |
| --- | --- | --- |
| `20261002T124957Z-resnet-chexpert-race-known-s42-d6d3` | `hypernet_e2e` | `analysis/archive/race_known_architecture_objective_comparison/runs/20261002T124957Z-resnet-chexpert-race-known-s42-d6d3` |
| `20261002T125004Z-hyperadapt-chexpert-race-known-s42-b901` | `hypernet_e2e` | `analysis/archive/race_known_architecture_objective_comparison/runs/20261002T125004Z-hyperadapt-chexpert-race-known-s42-b901` |
| `20261002T125005Z-spatial-lora-chexpert-race-known-s42-a21c` | `hypernet_e2e` | `analysis/archive/race_known_architecture_objective_comparison/runs/20261002T125005Z-spatial-lora-chexpert-race-known-s42-a21c` |
| `20261002T142200Z-resnet-chexpert-race-known-s43-bbfc` | `hypernet_e2e` | `analysis/archive/race_known_architecture_objective_comparison/runs/20261002T142200Z-resnet-chexpert-race-known-s43-bbfc` |
| `20261002T145233Z-spatial-lora-chexpert-race-known-s43-424e` | `hypernet_e2e` | `analysis/archive/race_known_architecture_objective_comparison/runs/20261002T145233Z-spatial-lora-chexpert-race-known-s43-424e` |
| `20261002T155222Z-resnet-chexpert-race-known-s44-3d73` | `hypernet_e2e` | `analysis/archive/race_known_architecture_objective_comparison/runs/20261002T155222Z-resnet-chexpert-race-known-s44-3d73` |
| `20261002T160239Z-resnet-chexpert-age-race-group-dro-race-known-s42-eae5` | `hypernet_e2e` | `analysis/archive/race_known_architecture_objective_comparison/runs/20261002T160239Z-resnet-chexpert-age-race-group-dro-race-known-s42-eae5` |
| `20261002T160259Z-spatial-lora-chexpert-age-race-group-dro-subgroup-race-known-s42-8c3a` | `hypernet_e2e` | `analysis/archive/race_known_architecture_objective_comparison/runs/20261002T160259Z-spatial-lora-chexpert-age-race-group-dro-subgroup-race-known-s42-8c3a` |
| `20261002T160300Z-spatial-lora-chexpert-age-race-group-dro-race-known-s42-17f2` | `hypernet_e2e` | `analysis/archive/race_known_architecture_objective_comparison/runs/20261002T160300Z-spatial-lora-chexpert-age-race-group-dro-race-known-s42-17f2` |
| `20261002T160304Z-hyperadapt-chexpert-age-race-group-dro-subgroup-race-known-s42-ec14` | `hypernet_e2e` | `analysis/archive/race_known_architecture_objective_comparison/runs/20261002T160304Z-hyperadapt-chexpert-age-race-group-dro-subgroup-race-known-s42-ec14` |
| `20261002T160306Z-hyperadapt-chexpert-age-race-group-dro-race-known-s42-bf4b` | `hypernet_e2e` | `analysis/archive/race_known_architecture_objective_comparison/runs/20261002T160306Z-hyperadapt-chexpert-age-race-group-dro-race-known-s42-bf4b` |
| `20261002T165551Z-spatial-lora-chexpert-race-known-s44-eb13` | `hypernet_e2e` | `analysis/archive/race_known_architecture_objective_comparison/runs/20261002T165551Z-spatial-lora-chexpert-race-known-s44-eb13` |
| `20261002T180449Z-resnet-chexpert-age-race-group-dro-race-known-s43-ffae` | `hypernet_e2e` | `analysis/archive/race_known_architecture_objective_comparison/runs/20261002T180449Z-resnet-chexpert-age-race-group-dro-race-known-s43-ffae` |
| `20261002T181025Z-hyperadapt-chexpert-race-known-s43-4ab0` | `hypernet_e2e` | `analysis/archive/race_known_architecture_objective_comparison/runs/20261002T181025Z-hyperadapt-chexpert-race-known-s43-4ab0` |
| `20261002T181742Z-spatial-lora-chexpert-age-race-group-dro-race-known-s43-0a8f` | `hypernet_e2e` | `analysis/archive/race_known_architecture_objective_comparison/runs/20261002T181742Z-spatial-lora-chexpert-age-race-group-dro-race-known-s43-0a8f` |
| `20261002T181927Z-spatial-lora-chexpert-age-race-group-dro-subgroup-race-known-s43-ee6a` | `hypernet_e2e` | `analysis/archive/race_known_architecture_objective_comparison/runs/20261002T181927Z-spatial-lora-chexpert-age-race-group-dro-subgroup-race-known-s43-ee6a` |
| `20261002T200704Z-resnet-chexpert-age-race-group-dro-race-known-s44-4995` | `hypernet_e2e` | `analysis/archive/race_known_architecture_objective_comparison/runs/20261002T200704Z-resnet-chexpert-age-race-group-dro-race-known-s44-4995` |
| `20261002T203250Z-spatial-lora-chexpert-age-race-group-dro-race-known-s44-9ced` | `hypernet_e2e` | `analysis/archive/race_known_architecture_objective_comparison/runs/20261002T203250Z-spatial-lora-chexpert-age-race-group-dro-race-known-s44-9ced` |
| `20261002T203647Z-spatial-lora-chexpert-age-race-group-dro-subgroup-race-known-s44-1e6e` | `hypernet_e2e` | `analysis/archive/race_known_architecture_objective_comparison/runs/20261002T203647Z-spatial-lora-chexpert-age-race-group-dro-subgroup-race-known-s44-1e6e` |
| `20261002T212535Z-hyperadapt-chexpert-age-race-group-dro-subgroup-race-known-s43-f637` | `hypernet_e2e` | `analysis/archive/race_known_architecture_objective_comparison/runs/20261002T212535Z-hyperadapt-chexpert-age-race-group-dro-subgroup-race-known-s43-f637` |
| `20261002T213016Z-hyperadapt-chexpert-age-race-group-dro-race-known-s43-84d0` | `hypernet_e2e` | `analysis/archive/race_known_architecture_objective_comparison/runs/20261002T213016Z-hyperadapt-chexpert-age-race-group-dro-race-known-s43-84d0` |
| `20261002T220911Z-resnet-chexpert-age-race-group-dro-subgroup-race-known-s42-1823` | `hypernet_e2e` | `analysis/archive/race_known_architecture_objective_comparison/runs/20261002T220911Z-resnet-chexpert-age-race-group-dro-subgroup-race-known-s42-1823` |
| `20261002T225056Z-spatial-lora-chexpert-from-resnet-age-race-group-dro-race-known-s42-6388` | `hypernet_e2e` | `analysis/archive/race_known_architecture_objective_comparison/runs/20261002T225056Z-spatial-lora-chexpert-from-resnet-age-race-group-dro-race-known-s42-6388` |
| `20261002T225059Z-hyperadapt-chexpert-from-resnet-age-race-group-dro-race-known-s42-4915` | `hypernet_e2e` | `analysis/archive/race_known_architecture_objective_comparison/runs/20261002T225059Z-hyperadapt-chexpert-from-resnet-age-race-group-dro-race-known-s42-4915` |
| `20261002T233012Z-hyperadapt-chexpert-race-known-s44-a7ba` | `hypernet_e2e` | `analysis/archive/race_known_architecture_objective_comparison/runs/20261002T233012Z-hyperadapt-chexpert-race-known-s44-a7ba` |
| `20261003T001130Z-resnet-chexpert-age-race-group-dro-subgroup-race-known-s43-997e` | `hypernet_e2e` | `analysis/archive/race_known_architecture_objective_comparison/runs/20261003T001130Z-resnet-chexpert-age-race-group-dro-subgroup-race-known-s43-997e` |
| `20261003T002214Z-spatial-lora-chexpert-from-resnet-age-race-group-dro-race-known-s43-15e6` | `hypernet_e2e` | `analysis/archive/race_known_architecture_objective_comparison/runs/20261003T002214Z-spatial-lora-chexpert-from-resnet-age-race-group-dro-race-known-s43-15e6` |
| `20261003T015316Z-spatial-lora-chexpert-from-resnet-age-race-group-dro-race-known-s44-e6b1` | `hypernet_e2e` | `analysis/archive/race_known_architecture_objective_comparison/runs/20261003T015316Z-spatial-lora-chexpert-from-resnet-age-race-group-dro-race-known-s44-e6b1` |
| `20261003T021351Z-resnet-chexpert-age-race-group-dro-subgroup-race-known-s44-7f03` | `hypernet_e2e` | `analysis/archive/race_known_architecture_objective_comparison/runs/20261003T021351Z-resnet-chexpert-age-race-group-dro-subgroup-race-known-s44-7f03` |
| `20261003T024609Z-hyperadapt-chexpert-age-race-group-dro-subgroup-race-known-s44-257f` | `hypernet_e2e` | `analysis/archive/race_known_architecture_objective_comparison/runs/20261003T024609Z-hyperadapt-chexpert-age-race-group-dro-subgroup-race-known-s44-257f` |
| `20261003T024711Z-hyperadapt-chexpert-age-race-group-dro-race-known-s44-059e` | `hypernet_e2e` | `analysis/archive/race_known_architecture_objective_comparison/runs/20261003T024711Z-hyperadapt-chexpert-age-race-group-dro-race-known-s44-059e` |
| `20261003T031330Z-hyperadapt-chexpert-from-resnet-age-race-group-dro-race-known-s43-c8b6` | `hypernet_e2e` | `analysis/archive/race_known_architecture_objective_comparison/runs/20261003T031330Z-hyperadapt-chexpert-from-resnet-age-race-group-dro-race-known-s43-c8b6` |
| `20261003T074718Z-hyperadapt-chexpert-from-resnet-age-race-group-dro-race-known-s44-2240` | `hypernet_e2e` | `analysis/archive/race_known_architecture_objective_comparison/runs/20261003T074718Z-hyperadapt-chexpert-from-resnet-age-race-group-dro-race-known-s44-2240` |
| `20261003T135037Z-hyperadapt-chexpert-from-resnet-age-race-group-dro-subgroup-race-known-s42-201c` | `hypernet_e2e` | `analysis/archive/race_known_architecture_objective_comparison/runs/20261003T135037Z-hyperadapt-chexpert-from-resnet-age-race-group-dro-subgroup-race-known-s42-201c` |
| `20261003T135038Z-spatial-lora-chexpert-from-resnet-age-race-group-dro-subgroup-race-known-s42-3ccd` | `hypernet_e2e` | `analysis/archive/race_known_architecture_objective_comparison/runs/20261003T135038Z-spatial-lora-chexpert-from-resnet-age-race-group-dro-subgroup-race-known-s42-3ccd` |
| `20261003T142841Z-spatial-lora-chexpert-stage1-4-from-resnet-age-race-group-dro-race-known-s42-bfff` | `hypernet_e2e` | `analysis/archive/race_known_architecture_objective_comparison/runs/20261003T142841Z-spatial-lora-chexpert-stage1-4-from-resnet-age-race-group-dro-race-known-s42-bfff` |
| `20261003T142841Z-spatial-lora-chexpert-stage1-4-race-known-s42-a143` | `hypernet_e2e` | `analysis/archive/race_known_architecture_objective_comparison/runs/20261003T142841Z-spatial-lora-chexpert-stage1-4-race-known-s42-a143` |
| `20261003T142850Z-spatial-lora-chexpert-stage1-4-from-resnet-age-race-group-dro-subgroup-race-known-s42-380a` | `hypernet_e2e` | `analysis/archive/race_known_architecture_objective_comparison/runs/20261003T142850Z-spatial-lora-chexpert-stage1-4-from-resnet-age-race-group-dro-subgroup-race-known-s42-380a` |
| `20261003T142851Z-spatial-lora-chexpert-stage1-4-age-race-group-dro-race-known-s42-ccd6` | `hypernet_e2e` | `analysis/archive/race_known_architecture_objective_comparison/runs/20261003T142851Z-spatial-lora-chexpert-stage1-4-age-race-group-dro-race-known-s42-ccd6` |
| `20261003T142852Z-spatial-lora-chexpert-stage1-4-age-race-group-dro-subgroup-race-known-s42-eb6a` | `hypernet_e2e` | `analysis/archive/race_known_architecture_objective_comparison/runs/20261003T142852Z-spatial-lora-chexpert-stage1-4-age-race-group-dro-subgroup-race-known-s42-eb6a` |
| `20261003T152220Z-spatial-lora-chexpert-from-resnet-age-race-group-dro-subgroup-race-known-s43-8baa` | `hypernet_e2e` | `analysis/archive/race_known_architecture_objective_comparison/runs/20261003T152220Z-spatial-lora-chexpert-from-resnet-age-race-group-dro-subgroup-race-known-s43-8baa` |
| `20261003T164728Z-spatial-lora-chexpert-stage1-4-from-resnet-age-race-group-dro-subgroup-race-known-s43-6318` | `hypernet_e2e` | `analysis/archive/race_known_architecture_objective_comparison/runs/20261003T164728Z-spatial-lora-chexpert-stage1-4-from-resnet-age-race-group-dro-subgroup-race-known-s43-6318` |
| `20261003T164928Z-spatial-lora-chexpert-stage1-4-from-resnet-age-race-group-dro-race-known-s43-123c` | `hypernet_e2e` | `analysis/archive/race_known_architecture_objective_comparison/runs/20261003T164928Z-spatial-lora-chexpert-stage1-4-from-resnet-age-race-group-dro-race-known-s43-123c` |
| `20261003T165544Z-spatial-lora-chexpert-from-resnet-age-race-group-dro-subgroup-race-known-s44-37a8` | `hypernet_e2e` | `analysis/archive/race_known_architecture_objective_comparison/runs/20261003T165544Z-spatial-lora-chexpert-from-resnet-age-race-group-dro-subgroup-race-known-s44-37a8` |
| `20261003T170443Z-spatial-lora-chexpert-stage1-4-race-known-s43-9f83` | `hypernet_e2e` | `analysis/archive/race_known_architecture_objective_comparison/runs/20261003T170443Z-spatial-lora-chexpert-stage1-4-race-known-s43-9f83` |
| `20261003T171650Z-spatial-lora-chexpert-stage1-4-age-race-group-dro-subgroup-race-known-s43-b8a7` | `hypernet_e2e` | `analysis/archive/race_known_architecture_objective_comparison/runs/20261003T171650Z-spatial-lora-chexpert-stage1-4-age-race-group-dro-subgroup-race-known-s43-b8a7` |
| `20261003T172604Z-spatial-lora-chexpert-stage1-4-age-race-group-dro-race-known-s43-2eac` | `hypernet_e2e` | `analysis/archive/race_known_architecture_objective_comparison/runs/20261003T172604Z-spatial-lora-chexpert-stage1-4-age-race-group-dro-race-known-s43-2eac` |
| `20261003T181829Z-hyperadapt-chexpert-from-resnet-age-race-group-dro-subgroup-race-known-s43-35af` | `hypernet_e2e` | `analysis/archive/race_known_architecture_objective_comparison/runs/20261003T181829Z-hyperadapt-chexpert-from-resnet-age-race-group-dro-subgroup-race-known-s43-35af` |
| `20261003T190702Z-spatial-lora-chexpert-stage1-4-from-resnet-age-race-group-dro-race-known-s44-b2dc` | `hypernet_e2e` | `analysis/archive/race_known_architecture_objective_comparison/runs/20261003T190702Z-spatial-lora-chexpert-stage1-4-from-resnet-age-race-group-dro-race-known-s44-b2dc` |
| `20261003T190808Z-spatial-lora-chexpert-stage1-4-from-resnet-age-race-group-dro-subgroup-race-known-s44-a50b` | `hypernet_e2e` | `analysis/archive/race_known_architecture_objective_comparison/runs/20261003T190808Z-spatial-lora-chexpert-stage1-4-from-resnet-age-race-group-dro-subgroup-race-known-s44-a50b` |
| `20261003T194301Z-spatial-lora-chexpert-stage1-4-race-known-s44-b96f` | `hypernet_e2e` | `analysis/archive/race_known_architecture_objective_comparison/runs/20261003T194301Z-spatial-lora-chexpert-stage1-4-race-known-s44-b96f` |
| `20261003T200523Z-spatial-lora-chexpert-stage1-4-age-race-group-dro-subgroup-race-known-s44-4648` | `hypernet_e2e` | `analysis/archive/race_known_architecture_objective_comparison/runs/20261003T200523Z-spatial-lora-chexpert-stage1-4-age-race-group-dro-subgroup-race-known-s44-4648` |
| `20261003T201442Z-spatial-lora-chexpert-stage1-4-age-race-group-dro-race-known-s44-b58e` | `hypernet_e2e` | `analysis/archive/race_known_architecture_objective_comparison/runs/20261003T201442Z-spatial-lora-chexpert-stage1-4-age-race-group-dro-race-known-s44-b58e` |
| `20261003T224556Z-hyperadapt-chexpert-from-resnet-age-race-group-dro-subgroup-race-known-s44-ff87` | `hypernet_e2e` | `analysis/archive/race_known_architecture_objective_comparison/runs/20261003T224556Z-hyperadapt-chexpert-from-resnet-age-race-group-dro-subgroup-race-known-s44-ff87` |

## 除外した旧run

q時系列が欠損する旧M02（末尾`58a1`）および中断runは採用しない。
旧M02の取り込み・DRO予測cache・比較生成物は除去済み。学習側の元artifactは保持する。
取り込み履歴整理前の参照一覧は`run_logs`の退避文書を参照。
