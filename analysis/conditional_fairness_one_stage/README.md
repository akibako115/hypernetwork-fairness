# conditional_fairness_one_stage

## 問い

年齢 2 群（<65 / ≥65）に対する 3 つの条件付き公平性 objective を、ERM からの継続学習ではなく
ImageNet 初期化からの **1 stage（e2e）** で学習したとき、同じアーキテクチャ・同じ seed の ERM より
群間の差を縮められるか。また、その効果は ResNet と Spatial LoRA（stage1–4）で違うか。

継続学習版の比較は [minimax_auc_vs_eo](../minimax_auc_vs_eo/README.md) が持つ。この study は
目的関数の実装（`projects/hypernet_e2e/objectives/`）を共有し、学習の始め方だけを変える。

## 条件（seed 42）

| アーキテクチャ | 目的関数 | preset | class weight |
| --- | --- | --- | --- |
| ResNet-50 | minimax AUC（Yang et al. 2023） | `resnet_chexpert_age_minimax_auc_race_known` | なし |
| ResNet-50 | FairDRO（Jung et al. 2023、ρ=0.1） | `resnet_chexpert_age_fair_dro_race_known` | なし |
| ResNet-50 | EO rate（Cotter et al. 2019、初期 task 重み 0.9） | `resnet_chexpert_age_equalized_odds_rate_race_known` | inverse |
| Spatial LoRA stage1–4 | minimax AUC | `spatial_lora_chexpert_stage1_4_age_minimax_auc_race_known` | なし |
| Spatial LoRA stage1–4 | FairDRO（ρ=0.1） | `spatial_lora_chexpert_stage1_4_age_fair_dro_race_known` | なし |
| Spatial LoRA stage1–4 | EO rate | `spatial_lora_chexpert_stage1_4_age_equalized_odds_rate_race_known` | inverse |

- 共通部分（`_age_fairness_one_stage.yaml`）：
  - race_known split、AdamW（lr 1e-4）、30 epoch、bf16-mixed。いずれも ERM と同じ。
  - バッチは 4 セル（年齢群 × ラベル）各 32 件の均等抽出。
- 双対の設定（2026-10-08 の再起動後）。
  - minimax AUC：η=0.01、全ペア。継続学習版の既定値のまま。
  - EO rate：η=0.01、ε=0.1、τ=1。初期 λ の task 重みを 0.9 にする（`initial_task_weight`。preset に記述）。
  - FairDRO：ρ=0.1（起動時の override）。q は 0.5±0.16 に収まる。

## 基準と選択

sanity 時点のモデルは初期値なので、run 内では選ばない。
- recorder は `reference: deferred` で動く。全 epoch の候補 checkpoint と validation 指標、初期値の validation（`fairness_initial.json`）だけを残す。
- 選択は分析側で行う。基準は同じアーキテクチャ・同じ seed の ERM run の validation。
  - ResNet：`20261002T124957Z-resnet-chexpert-race-known-s42-d6d3`
  - Spatial LoRA：`20261003T142841Z-spatial-lora-chexpert-stage1-4-race-known-s42-a143`
- 規則は継続学習版と同じ。親を ERM 基準に読み替える。
  - EO rate と FairDRO：TPR・TNR の群間差が ≤ 0.10、ERM からの率の低下が ≤ 0.02、AUROC の低下が ≤ 0.005。
  - minimax AUC：AUROC と worst-age AUROC の低下が ≤ 0.005 の候補のうち、min pair AUROC が最大のもの。
  - 最終 epoch も併記する。

## 初回起動で分かったこと（2026-10-08）

seed 42 の 6 本を ws11 で同時に起動した。最初の 2〜5 epoch の時点で、FairDRO と EO rate の計 4 本を止めた。
run directory は残してある。強制終了なので `run.json` は running のまま。

- **FairDRO（ρ=1）**：G=2 のとき、閉形式の q* は誤り率の大小だけで決まる。
  - 偏差の向きは ±1/√2 しかないので、`q* = 1/2 ± √ρ/2` になる。ρ=1 では、各クラスで片方の群の重みが 0 になる。
  - 学習の結果、クラスごとに一方の群だけを学ぶ解へ毎 epoch 振れた。
  - val AUROC は ResNet で 0.85→0.62→0.81→0.75、Spatial LoRA で 0.85→0.39→0.77。
  - 実装の誤りではなく、ρ の選び方の問題。G=2 では ρ<1 にする。
- **EO rate（初期 λ 一様）**：epoch 0 から、train・val ともに全予測が片側に寄った。
  - ResNet は TPR=0・TNR=1、Spatial LoRA は TPR=1・TNR=0。
  - 未学習の初期値では、λ の 4/5 が proxy に載る。proxy は全スコアを閾値の片側へ押し出すと下がる。
  - 定数の予測は EO 制約を自明に満たすので、hard 違反は −ε のままで、λ を戻す圧力も生じない。
- **minimax AUC**：AUROC 0.84〜0.85、min pair AUROC 0.73〜0.80 で推移しており、継続中。

## 状態

2026-10-08：minimax AUC の 2 本が走行中。FairDRO（ρ=0.1）と EO rate（初期 task 重み 0.9）の 4 本は再起動の承認待ち。
