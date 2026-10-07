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
| ResNet-50 | FairDRO（Jung et al. 2023、ρ=1） | `resnet_chexpert_age_fair_dro_race_known` | なし |
| ResNet-50 | EO rate（Cotter et al. 2019） | `resnet_chexpert_age_equalized_odds_rate_race_known` | inverse |
| Spatial LoRA stage1–4 | minimax AUC | `spatial_lora_chexpert_stage1_4_age_minimax_auc_race_known` | なし |
| Spatial LoRA stage1–4 | FairDRO（ρ=1） | `spatial_lora_chexpert_stage1_4_age_fair_dro_race_known` | なし |
| Spatial LoRA stage1–4 | EO rate | `spatial_lora_chexpert_stage1_4_age_equalized_odds_rate_race_known` | inverse |

- 共通部分（`_age_fairness_one_stage.yaml`）：
  - race_known split、AdamW（lr 1e-4）、30 epoch、bf16-mixed。いずれも ERM と同じ。
  - バッチは 4 セル（年齢群 × ラベル）各 32 件の均等抽出。
- 双対の設定は継続学習版の既定値を使う。
  - minimax AUC：η=0.01、全ペア。
  - EO rate：η=0.01、ε=0.1、τ=1。
  - FairDRO：ρ=1。G=2 で q∈[0,1] になる境界。

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

## 状態

2026-10-07：preset を追加。seed 42 の 6 本を ws11 で起動する前。
