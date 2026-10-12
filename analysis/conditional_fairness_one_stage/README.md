# conditional_fairness_one_stage

> **2026-10-13 注記（公平性損失の実装修正）**：[修正記録](../../docs/reviews/20261013_fairness_objective_fixes.md)。EO rate の 4 run は旧 smoothstep 代理制約で回した。代理制約が予測を閾値の反対側へ動かせないので、
> EO rate の結果と「読めること」の EO の項は、EO 制約の効果として読まない。
> FairDRO ρ=0.1（G=2）と ERM の結果は影響を受けない。

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

## 結果（seed 42、2026-10-08）

再起動後の 4 本と minimax AUC の 2 本は、いずれも 30 epoch を完走した。
metadata（`*.ckpt` を除く）はローカルの `projects/hypernet_e2e/runs/` に回収済み。checkpoint（10 run で計 24 GB）は ws11 に残している。
以下の値はすべて validation で、`metrics.csv` の fairness_metrics の列を使う。ERM と同じ callback・同じ定義で算出している。
- epoch は 0 始まり。
- Eopp1 は年齢 2 群の TPR 差、Eodds は (TPR 差 + FPR 差)/2。
- 閾値は argmax。二値なので z1−z0>0 と同じになる。

| 条件 | AUROC 最大 epoch：AUROC / worst-age AUROC / Eopp1 / Eodds | 最終 epoch（29）：AUROC / Eopp1 / Eodds | Eodds 最小 epoch：AUROC / Eodds |
| --- | --- | --- | --- |
| ERM ResNet（基準） | ep8：0.868 / 0.824 / 0.241 / 0.187 | 0.855 / 0.331 / 0.217 | ep3：0.860 / 0.144 |
| ERM Spatial LoRA（基準） | ep11：0.869 / 0.828 / 0.294 / 0.215 | 0.856 / 0.388 / 0.251 | ep8：0.868 / 0.127 |
| ResNet minimax AUC | ep2：0.851 / 0.805 / 0.138 / 0.088 | 0.829 / 0.310 / 0.185 | ep1：0.831 / 0.046 |
| Spatial LoRA minimax AUC | ep6：0.854 / 0.802 / 0.180 / 0.110 | 0.832 / 0.272 / 0.161 | ep1：0.848 / 0.045 |
| ResNet FairDRO ρ=0.1 | ep2：0.856 / 0.804 / 0.231 / 0.176 | 0.824 / 0.360 / 0.210 | ep1：0.824 / 0.056 |
| Spatial LoRA FairDRO ρ=0.1 | ep4：0.850 / 0.800 / 0.324 / 0.208 | 0.815 / 0.257 / 0.148 | ep0：0.850 / 0.060 |
| ResNet EO rate（初期 task 0.9） | ep1：0.861 / 0.809 / 0.156 / 0.107 | 0.825 / 0.364 / 0.219 | ep1：同左 |
| Spatial LoRA EO rate（初期 task 0.9） | ep1：0.862 / 0.816 / 0.123 / 0.074 | 0.822 / 0.379 / 0.226 | ep1：同左 |

### 読めること

- **どの手法も ERM の AUROC に届かない**：AUROC の最大は 0.850〜0.862 で、ERM（0.868〜0.869）より 0.007〜0.019 低い。
  - 選択規則「AUROC の低下 ≤ 0.005」を ERM 基準で当てると、6 本とも候補が残らない（infeasible）。
  - 4 セル均等抽出に変えたこと自体の費用と、目的関数の費用はまだ分けられていない。
- **差が小さいのは学習の序盤だけ**：Eodds は ep0〜2 では 0.05〜0.11 で、ERM の AUROC 最大時（0.19〜0.22）より小さい。
  - しかし学習が進むと、ERM と同じく TPR 差が開き、最終 epoch では ERM 並み（Eodds 0.15〜0.23）に戻る。
  - ERM も ep3 から ep29 で Eopp1 が 0.16 から 0.33 に開く。差が開いていく傾向は手法によらない。
- **原因は ≥65 陽性セルの記憶と見ている**：
  - このセルは train に 3,869 枚しかない。4 セル均等抽出では、1 epoch に 1 枚あたり約 10 回使われる。
  - ResNet EO rate の train バッチ上の陽性予測率は、ep0 の 0.88 / 0.67（<65 / ≥65）から ep29 には 0.98 / 0.97 になる。同じ時点の val の ≥65 TPR は 0.25。
  - ResNet FairDRO が epoch 末に train 全件（eval mode）で測った ≥65 陽性の誤り率は ep29 で 0.18。val の約 0.79 よりずっと小さい。
  - 双対変数を動かす信号を train から取ると、記憶によって消える。
- **EO rate**：train 上では ep5 頃に 4 制約とも満たされた（hard 違反 ≤ 0）。
  - 以降、λ はタスク側へ戻り、ep29 で λ[0] = 1.0 になった。後半は実質「セル均等抽出の ERM」だった。
  - 初期 λ の修正で、初回のような全予測が片側に寄る崩壊は起きなかった。
- **FairDRO**：陰性クラスの q は、30 epoch を通して ρ=0.1 の上限 0.658 / 0.342（<65 / ≥65）に張り付いた。
  - 陽性クラスの q は、最初の 0.342 / 0.658 から ep29 には 0.46 / 0.54 程度まで戻った。
  - 2 群では、q* は毎回上限か下限のどちらかの端点になる。平滑化で、その平均に寄っていく動きと読める。
- **minimax AUC**：min pair AUROC は 0.80 → 0.74 と緩やかに下がった。
  - AUC 系の目的関数は閾値を制約しないので、TPR 差の拡大も止めない。
  - 序盤の Eodds 最小（ep1 で 0.045）は、AUROC も低い時点にあたる。

### 次の一手（提案）

1. **双対更新の信号を held-out から取る**：train の一部を学習に使わず、EO rate の制約違反と FairDRO の誤り率をそこで測る。
   - Cotter らは、同じ年（2019）の別論文で、制約の汎化のためにこの 2 データセット方式を提案している。
   - 今回の失敗（train の記憶で制約が満たされたことになる）に直接効く。
2. **抽出の影響を切り分ける**：
   - セル均等抽出の ERM（`sampling_corrected_erm` を 1 stage にしたもの）を同じ条件で回し、AUROC の低下が抽出によるのか目的関数によるのかを分ける。
   - ≥65 陽性の使い回しを抑える設定（抽出比の上限、augmentation の強化）も候補にする。
3. **ERM 基準での選択**：今回は ERM の val の群別 TPR / TNR が `metrics.csv` に無く、gap しか比べていない。
   - 選択規則の「率の低下 ≤ 0.02」を判定するには、ERM の best checkpoint で val を推論し直す必要がある。

## 状態

2026-10-08：seed 42 の 6 本が完走（初回に止めた 4 本は別 run として記録）。metadata は回収済み。上の次の一手は未着手。
