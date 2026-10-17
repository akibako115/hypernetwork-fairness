# runs

この package が読む run。設定・状態・数値の正本は各 run の artifact で、ここには run ID と、なぜこの run かだけを書く。

## 試行（ローカル、2026-10-13）

ws11 が空いていないので、格子のうち中心の 1 条件（Spatial LoRA × `dw8` × EO rate、seed 42）をローカルの A6000 で先に回す。
学習が崩れないか、潜在群と評価の 8 群で差が縮むかを見て、残り 23 本へ進むかを決める。
未 commit の preset・群のスクリプトを含む状態で起動した（`run.json` の `git_commit` は 337e381）。

| run-id | project | 状態 |
|---|---|---|
| `20261013T144425Z-spatial-lora-chexpert-dw8-eo-rate-dual-holdout-s42-9bb9` | `hypernet_e2e` | 実行中 |

比較相手は grid の 8 群の同じ条件 `spatial_lora_chexpert_age_race_eo_rate_dual_holdout` s42（`20261012T173018Z-...-8b3d`）。

## 原論文型の EO rate（ローカル、2026-10-14）

9bb9 が退化解に寄ったので、Cotter et al. の実験の形に揃えて回し直した（`docs/methods/eo-rate.md`）。
制約は各 dw8 群の FPR を全体と比べる片側の 8 本（FPR_g − FPR_全体 ≤ 0.05）、M は一様に初期化、10 epoch。
`constraint_form`・`rates` は未 commit の変更を含む状態で起動した（`run.json` の `git_commit` は 89cf647）。
checkpoint は全 epoch を残し、val の dw8 制約で規則 A により選ぶ（epoch 9。式 18 も同じ）。

| run-id | project | run path |
|---|---|---|
| `20261014T142504Z-spatial-lora-chexpert-dw8-eo-rate-dual-holdout-s42-2f68` | `hypernet_e2e` | `analysis/latent_group_fairness/runs/20261014T142504Z-spatial-lora-chexpert-dw8-eo-rate-dual-holdout-s42-2f68` |

## GroupDRO（ローカル、2026-10-14）

dw8 群の GroupDRO を、全 epoch の checkpoint を残す設定で回し直した（前の fc60 は保存設定が古いので止めて削除）。
群は dw8 だけ（属性 × ラベルではない）、自然抽出、30 epoch で、原論文の条件とは違う（`docs/methods/group-dro.md`）。
checkpoint の選び方（val の最悪群の成績に何を使うか）は未決。

| run-id | project | run path |
|---|---|---|
| `20261014T144403Z-spatial-lora-chexpert-dw8-group-dro-race-known-s42-6870` | `hypernet_e2e` | `analysis/latent_group_fairness/runs/20261014T144403Z-spatial-lora-chexpert-dw8-group-dro-race-known-s42-6870` |
