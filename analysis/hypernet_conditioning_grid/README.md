# hypernet_conditioning_grid

2026-10-13 開始。柱 A「hypernetwork による条件付けは、いつ効いて、いつ効かないか」
（[research_overview.md](../../docs/meetings/20261016/research_overview.md)）に答えるための対照実験の格子。
CheXpert と HAM10000 で同じ格子を回し、条件付けが効くかどうかがデータに依存するかも見る
（2026-10-17 に HAM10000 版の package `hypernet_conditioning_grid_ham10000` をここへまとめた）。

## 現状（2026-10-17）

- ここにある run はすべて旧前処理で学習している。2026-10-17 に CheXpert・HAM10000 の前処理を MEDFAIR に合わせたので
  （[docs/README.md](../../docs/README.md)）、全条件を新しい前処理で回し直す。
- EO rate の run は、2026-10-16 の修正（代理制約を hinge だけにした）より前の実装で回している。
- 下の「結果」（2026-10-14・10-15）は旧前処理・旧実装の run から作ったもので、結論には使わない。回し直した run で作り直す。

| file | 中身 |
| --- | --- |
| `grid_metrics.py`・`compute.py`・`comparisons.py`・[grid.ipynb](grid.ipynb) | CheXpert の指標・bootstrap・条件の比較 |
| `requested_analysis.py`・[requested_figures.ipynb](requested_figures.ipynb) | CheXpert の 6 種類の図 |
| `requested_analysis_ham10000.py`・[requested_figures_ham10000.ipynb](requested_figures_ham10000.ipynb) | HAM10000 の 6 種類の図（評価の向きと群だけを CheXpert から変える） |

## 問い

同じデータ・同じ目的関数・同じ学習手順のもとで、属性で条件付けた hypernetwork（Spatial LoRA）は、
条件付けのない ResNet-50 より、全体の性能を落とさずに最悪群を上げるか。上がらないなら、どの条件で、なぜか。

評価の定義は [research-premises.md](../../docs/research-premises.md)（固定しきい値 0.5 の Eopp1・Eopp0・Eodds と
害の大きい側の最悪群の率が主、AUROC は副）。

## 格子

どちらのデータでも、手順 × モデルの 5 通り × 目的関数 3 つ（ERM・GroupDRO・EO rate）× seed 42・43・44 の 45 run。

### 格子（CheXpert、年齢 65 歳 × race の 8 群、seed 42・43・44）

| 手順 | モデル | ERM | GroupDRO | EO rate |
| --- | --- | --- | --- | --- |
| 1 段（ImageNet から全層） | ResNet-50 | 再利用 | 新規 | 新規 |
| | Spatial LoRA（stage3・stage4・fc） | 再利用 | 新規 | 新規 |
| | Spatial LoRA（fc のみ） | 新規 | 新規 | 新規 |
| 2 段（親 ResNet ERM を凍結） | Spatial LoRA（stage3・stage4・fc） | 新規 | 新規 | 新規 |
| | Spatial LoRA（fc のみ） | 新規 | 新規 | 新規 |

新規は 13 条件 × 3 seed = 39 run。2 段の対照は、親の ResNet ERM そのもの（1 段の ResNet ERM と同じ run）。

決めた経緯（2026-10-13、ユーザーと）：

- class weight は全手法で inverse にそろえる。class weight を掛けられない minimax AUC（順位だけの損失で、
  0.5 の動作点が決まらない）と、損失の形がすでにクラスをそろえる FairDRO は外した。
- HyperAdapt は外し、fc だけを変調する Spatial LoRA を足した。
- 2 段の ResNet の継続学習（条件付けなしの対照）は回さない。
- 1 段と 2 段は同時に起動する。2 段の親は既存の race_known ResNet ERM なので、1 段の完了を待たない。

#### そろえたこと

- **train**：ERM・GroupDRO は `splits_race_known` の train 全体で学習する。EO rate は、双対変数を測る held-out
  （患者の 20%）を除いた `splits_race_known_dual_holdout` の train（80%）で学習する。val・test は共通。
- **ERM の再利用**：`race_known` の主比較で回した ResNet・Spatial LoRA × 3 seed（[runs.md](runs.md)）。条件（split・
  inverse class weight・30 epoch・学習率・Spatial LoRA の変調段）は今回の preset と同じ。commit は記録されていないが、
  10/13 の修正は ERM の学習に触れていない。
- **2 段の親**：同じ seed の上記 ResNet ERM の best val AUROC checkpoint。backbone と元の分類層を凍結し、
  Spatial LoRA の hypernetwork だけを学習する。
- **双対変数・群の重みの更新**：EO rate は held-out で測る（Cotter et al. の 2 データセット法。train で測ると
  少数セルの記憶で制約が満たされたことになる。`conditional_fairness_one_stage`）。GroupDRO は原典どおり train のバッチで更新する。
- **抽出**：EO rate は年齢 × race × ラベルのセル均等、ERM・GroupDRO は自然抽出（各 preset の既定）。
- **class weight**：全条件で全群共通の inverse。
- **checkpoint**：EO rate は全 epoch を残す。ERM・GroupDRO は best（val AUROC）と last だけ。

#### 原典から変えたこと

- EO rate の初期 task 重みを 0.9 にした（一様だと全予測が片側に寄って崩れた。2026-10-08）。
- EO rate の双対変数を held-out で更新する（上記）。

#### 読むときの注意

- ERM・GroupDRO（自然抽出）と EO rate（セル均等抽出）とでは抽出が違う。EO rate と ERM の差には抽出の効果も含まれる。
- EO rate は ERM より 20% 少ないデータで学習する。「EO rate vs ERM」の差は EO rate に不利な向きに偏る。
  同じ目的関数の中でのモデル間の比較は、train がそろっている。
- 2 段の EO rate では、held-out の患者を親の ResNet が学習済み（親は race_known の train 全体で学習）。
  held-out で測る制約違反には、親が記憶した分が混ざる。hypernetwork 自身は held-out を学習しない。

### 格子（HAM10000、sex × 年齢 65 歳の 4 群、seed 42・43・44）

HAM10000 には race・肌色の列が無いので、CheXpert の年齢 × race 8 群の代わりに、いまの評価軸と同じ sex × age_group_65 の
4 群を使う（2026-10-13、ユーザーと決めた）。群 ID は sex を上位桁にした順（female × <65、female × ≥65、male × <65、male × ≥65）。

| 手順 | モデル | ERM | GroupDRO | EO rate |
| --- | --- | --- | --- | --- |
| 1 段（ImageNet から全層） | ResNet-50 | 再利用 | 新規 | 新規 |
| | Spatial LoRA（stage3・stage4・fc） | 再利用 | 新規 | 新規 |
| | Spatial LoRA（fc のみ） | 再利用 | 新規 | 新規 |
| 2 段（親 ResNet ERM を凍結） | Spatial LoRA（stage3・stage4・fc） | 新規 | 新規 | 新規 |
| | Spatial LoRA（fc のみ） | 新規 | 新規 | 新規 |

15 条件 × 3 seed = 45 run。うち 1 段の ERM 3 条件（9 run）は既存の run を再利用し、新規は 12 条件 × 3 seed = 36 run。
CheXpert と違い、1 段の fc のみの ERM も既存の run がある。

#### そろえたこと

CheXpert の格子と同じ方針で、データ以外は動かさない。

- **train**：ERM・GroupDRO は `data/ham10000/splits` の train 全体で学習する。EO rate は、双対変数を測る held-out
  （病変の 20%）を除いた `splits_dual_holdout` の train（80%）で学習する。val・test は共通。
- **ERM の再利用**：1 段の ResNet・Spatial LoRA は study=ham10000_architecture_erm、
  fc のみは 2026-10-09 の `spatial_lora_ham10000_fc`（[runs.md](runs.md)）。どれも `_base_ham10000` のレシピ
  （inverse class weight・30 epoch・AdamW lr 1e-4）で、今回の preset と同じ。10/13 の修正は ERM の学習に触れていない。
- **2 段の親**：同じ seed の上記 ResNet ERM の best val AUROC checkpoint。backbone と元の分類層を凍結し、
  Spatial LoRA の hypernetwork だけを学習する。
- **双対変数・群の重みの更新**：EO rate は held-out で測る。GroupDRO は原典どおり train のバッチで更新する。
- **抽出**：EO rate は sex × 年齢 × ラベルのセル均等、ERM・GroupDRO は自然抽出。
- **class weight**：全条件で全群共通の inverse。
- **checkpoint**：EO rate は全 epoch を残す。ERM・GroupDRO は best（val AUROC）と last だけ。

#### 原典から変えたこと

- EO rate の初期 task 重みを 0.9 にした（CheXpert と同じ）。
- EO rate の双対変数を held-out で更新する。

#### 読むときの注意

- CheXpert の格子の注意（抽出の違い、EO rate は 20% 少ない train、2 段の EO rate では held-out を親が学習済み）はそのまま当てはまる。
- test の陽性は群ごとに 28〜82 件しかない。群別の指標は CheXpert より雑音が大きい。
- test の 294 件は sex か age を欠く（うち 284 件は別施設の external）。属性条件付きモデルは学習中に見ていない欠損用の入力を受ける。
  **モデル間の主比較は属性の揃った 1,217 件で行う**（[data_pipeline/README.md](../../data_pipeline/README.md) の HAM10000 の節）。
- 10-09 の `resnet_ham10000_sex_age_group_dro` と既存の EO rate の run は修正前の実装なので使わず、回し直す。

## preset

### CheXpert

`projects/hypernet_e2e/configs/experiment/` の次の名前。2 段は `_two_stage_frozen.yaml` を重ね、起動時に
`model.backbone_checkpoint_path` で親を渡す。

| | ERM | GroupDRO | EO rate |
| --- | --- | --- | --- |
| 1 段 ResNet | `resnet_chexpert_race_known` | `resnet_chexpert_age_race_group_dro_race_known` | `resnet_chexpert_age_race_eo_rate_dual_holdout` |
| 1 段 Spatial LoRA | `spatial_lora_chexpert_race_known` | `spatial_lora_chexpert_age_race_group_dro_race_known` | `spatial_lora_chexpert_age_race_eo_rate_dual_holdout` |
| 1 段 fc のみ | `spatial_lora_chexpert_fc_race_known` | `spatial_lora_chexpert_fc_age_race_group_dro_race_known` | `spatial_lora_chexpert_fc_age_race_eo_rate_dual_holdout` |
| 2 段 Spatial LoRA | `spatial_lora_chexpert_from_resnet_race_known` | `spatial_lora_chexpert_from_resnet_age_race_group_dro_race_known` | `spatial_lora_chexpert_from_resnet_age_race_eo_rate_dual_holdout` |
| 2 段 fc のみ | `spatial_lora_chexpert_fc_from_resnet_race_known` | `spatial_lora_chexpert_fc_from_resnet_age_race_group_dro_race_known` | `spatial_lora_chexpert_fc_from_resnet_age_race_eo_rate_dual_holdout` |

### HAM10000

`projects/hypernet_e2e/configs/experiment/` の次の名前。2 段は `_two_stage_frozen.yaml` を重ね、起動時に
`model.backbone_checkpoint_path` で親を渡す。GroupDRO は `_sex_age_groups_ham10000.yaml` で q に群のラベルを付ける。

| | ERM | GroupDRO | EO rate |
| --- | --- | --- | --- |
| 1 段 ResNet | `resnet_ham10000` | `resnet_ham10000_sex_age_group_dro` | `resnet_ham10000_sex_age_eo_rate_dual_holdout` |
| 1 段 Spatial LoRA | `spatial_lora_ham10000` | `spatial_lora_ham10000_sex_age_group_dro` | `spatial_lora_ham10000_sex_age_eo_rate_dual_holdout` |
| 1 段 fc のみ | `spatial_lora_ham10000_fc` | `spatial_lora_ham10000_fc_sex_age_group_dro` | `spatial_lora_ham10000_fc_sex_age_eo_rate_dual_holdout` |
| 2 段 Spatial LoRA | `spatial_lora_ham10000_from_resnet` | `spatial_lora_ham10000_from_resnet_sex_age_group_dro` | `spatial_lora_ham10000_from_resnet_sex_age_eo_rate_dual_holdout` |
| 2 段 fc のみ | `spatial_lora_ham10000_fc_from_resnet` | `spatial_lora_ham10000_fc_from_resnet_sex_age_group_dro` | `spatial_lora_ham10000_fc_from_resnet_sex_age_eo_rate_dual_holdout` |

## TODO：EO の列を Cotter et al. の形の run に差し替える（2026-10-15）

全ペアの EO rate（8 群の有向ペア 112 本、TPR・FPR とも ε=0.1、λ の初期 task 重み 0.9）は、原著の制約
（各群の率を全体の率と比べる）と形が違った。EO の列は `*_eo_fpr_overall_dual_holdout` の 15 run
（`FPR_g − FPR_全体 ≤ 0.05` を 8 本、TPR は制約しない、λ は一様から）に差し替える。ws06 で 2026-10-14 23:58 JST に起動した。

- [x] 全ペアの EO rate 15 run の hardlink を `runs/` から外し、`runs.md` の表からも外した（run ID は「外した run」に残す。
  run 本体は `projects/hypernet_e2e/runs/` にある）
- [x] `compute.condition()` が `eo-fpr-overall` を ERM と読まないようにした（目的関数名 `eo_fpr_overall`）
- [ ] 15 run が終わったら、metadata と全 epoch の checkpoint を ws06 から回収し、`runs/` に取り込む
- [ ] checkpoint の選び方を実装する：holdout で制約（`FPR_g − FPR_全体 ≤ 0.05`）を満たす epoch のうち、目的関数が最もよいもの。
  満たす epoch が無い run は、そのことを明示する。best val AUROC では選ばない（制約の効く前の epoch が選ばれる）
- [ ] 選んだ checkpoint で予測 cache を作り直す。`cache/` に残る全ペアの EO の cache（60 本）は使わない
- [ ] `grid.ipynb`・`requested_figures.ipynb` の目的関数の一覧（`eo_rate`）を `eo_fpr_overall` に差し替え、図と
  `results/` を作り直す。差し替えるまで、2026-10-14 の図と `reports/requested_figures.md` の EO の列は全ペアの版の結果である
- [ ] 主な指標は FPR 側（Eopp0・最悪群の FPR）。TPR は制約していないので、Eopp1 は狙った指標ではないと書く

## 結果（旧前処理・旧実装。結論には使わない）

### CheXpert：実行の記録

2026-10-13 に 39 run がすべて正常終了。起動は `run_logs/hcg_chexpert_ws11_launch_20261013.sh`（ws11、1 条件 1 container で 3 seed を直列、1 GPU に最大 2 container）。


### CheXpert：2026-10-14 の 6 種類の図

[requested_figures.ipynb](requested_figures.ipynb)で、45 runのbest validation AUROC checkpointをtest評価した。
ERMのモデル比較、公平性損失の比較、各8群の性能、指定GroupDRO runのq、validationのepoch推移を扱う。
[図一覧](figures/requested_figures/index.html)・[全15図PDF](figures/requested_figures/all_requested_figures.pdf)・
[分析メモ](reports/requested_figures.md)にまとめた。数値の正本は`results/requested_figures/`。

観察として、ERMの属性条件付けではAUROC系とFPR系の改善方向がそろわない。EO-rate（全ペアの版。上の TODO で差し替える）でEopp0が
縮む条件はあるが、全体性能の低下や群別FPRの悪化を伴う。一般的な有効性の結論は未確定とする。
worst-FPRの全epoch値は未記録なので、学習推移図の該当枠はbest checkpointの点のみ。
他の曲線は30 epochのログに基づき、global TPR/FPRはaccuracy・BACC・陽性率から復元した。

### HAM10000：実行の記録

2026-10-14 00:35 JST に ws06（GPU 0〜3）で 36 run を起動し、01:40 JST ごろまでに全 run が `succeeded` で終わった。run ID は [runs.md](runs.md)。checkpoint を除く metadata を回収し、45 run を `runs/` に取り込んだ。新規 36 run の checkpoint は best だけを回収した。

### HAM10000：2026-10-15 の ERM・GroupDRO の図

[requested_figures_ham10000.ipynb](requested_figures_ham10000.ipynb) で、CheXpert の [requested_figures](reports/requested_figures.md) と
同じ 6 種類の図を、ERM・GroupDRO の 30 run について作った。test は属性の揃った 1,217 件、最悪群は TPR 最小の群。
[分析メモ](reports/requested_figures_ham10000.md)、数値の正本は `results/requested_figures_ham10000/`、図は `figures/requested_figures_ham10000/`（再生成物）。

観察として、ERM の条件付けは全体 AUROC と worst-AUC を上げるが、Eopp1 は広がり worst-TPR は上がらない（2-stage では下がる）。
GroupDRO は Spatial LoRA で Eopp1 を下げ worst-TPR を上げる向きに動くが、同 seed 差の SD が平均と同程度以上で、FPR の上昇を伴う。
結論は未確定。EO rate がそろったら `OBJECTIVES` に足して再実行する。

予測 cache を作るために、`analysis/common/predictions.py` が GroupDRO の datamodule で属性欠損行を仮の群に寄せるようにした
（群 ID は network の入力にならない）。
