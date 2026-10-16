# 対象run

2026-10-06現在、新しい学習runは未起動。
親ERMは以下の3runを選ぶ。2026-10-06にbest checkpointの実体SHA-256をrun.jsonと照合し、
train/val CSVのSHA-256も現在の `splits_race_known` と一致することを確認した。
このpackageへの取り込みは未実施。分析開始前に取り込み、入力はpackage内の `runs/` にそろえる。

計画は[実装計画](implementation_plan.md)と[実験計画](experiment_plan.md)を参照。
同seedのM01を親に、全層を追加30 epoch更新する。
初回の公平性学習は年齢のみの2群（65歳未満／65歳以上）。親ERMとsplitは変更しない。
年齢×raceのintersectionは、年齢2群の結果を確認した後の別条件として記録する。

| seed | project | run-id | 元run path（repo root基準） | W&B |
| --- | --- | --- | --- | --- |
| 42 | hypernet_e2e | `20261002T124957Z-resnet-chexpert-race-known-s42-d6d3` | `projects/hypernet_e2e/runs/20261002T124957Z-resnet-chexpert-race-known-s42-d6d3` | [run](https://wandb.ai/kohki-akiba-kyushu-university/fairness_hypernet/runs/rswjo11h) |
| 43 | hypernet_e2e | `20261002T142200Z-resnet-chexpert-race-known-s43-bbfc` | `projects/hypernet_e2e/runs/20261002T142200Z-resnet-chexpert-race-known-s43-bbfc` | [run](https://wandb.ai/kohki-akiba-kyushu-university/fairness_hypernet/runs/kb7vff02) |
| 44 | hypernet_e2e | `20261002T155222Z-resnet-chexpert-race-known-s44-3d73` | `projects/hypernet_e2e/runs/20261002T155222Z-resnet-chexpert-race-known-s44-3d73` | [run](https://wandb.ai/kohki-akiba-kyushu-university/fairness_hypernet/runs/btr0ab3s) |

引用理由：全条件で同じ学習済み表現から目的関数だけを変更し、追加学習予算をそろえる。
参照した取り込み済みartifactは `analysis/archive/race_known_architecture_objective_comparison/runs/<run-id>/`。
本studyの学習runが完了したら、条件ID・設定水準・run path・W&B URLをここへ追記する。
