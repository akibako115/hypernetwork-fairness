# hypernet_conditioning_grid

2026-10-13 開始。柱 A「hypernetwork による条件付けは、いつ効いて、いつ効かないか」
（[research_overview.md](../../docs/meetings/20261016/research_overview.md)）に答えるための対照実験の格子。
過去の run のうち条件のそろう ERM は再利用し、修正の影響を受ける公平性損失は修正後の実装で回し直す。

## 問い

同じデータ・同じ目的関数・同じ学習手順のもとで、属性で条件付けた hypernetwork（Spatial LoRA）は、
条件付けのない ResNet-50 より、全体の性能を落とさずに最悪群を上げるか。上がらないなら、どの条件で、なぜか。

評価の定義は [research-premises.md](../../docs/research-premises.md)（固定しきい値 0.5 の Eopp1・Eopp0・Eodds と
害の大きい側の最悪群の率が主、AUROC は副）。

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

## 格子（CheXpert、年齢 65 歳 × race の 8 群、seed 42・43・44）

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

### そろえたこと

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

### 原典から変えたこと

- EO rate の初期 task 重みを 0.9 にした（一様だと全予測が片側に寄って崩れた。2026-10-08）。
- EO rate の双対変数を held-out で更新する（上記）。

### 読むときの注意

- ERM・GroupDRO（自然抽出）と EO rate（セル均等抽出）とでは抽出が違う。EO rate と ERM の差には抽出の効果も含まれる。
- EO rate は ERM より 20% 少ないデータで学習する。「EO rate vs ERM」の差は EO rate に不利な向きに偏る。
  同じ目的関数の中でのモデル間の比較は、train がそろっている。
- 2 段の EO rate では、held-out の患者を親の ResNet が学習済み（親は race_known の train 全体で学習）。
  held-out で測る制約違反には、親が記憶した分が混ざる。hypernetwork 自身は held-out を学習しない。

## preset

`projects/hypernet_e2e/configs/experiment/` の次の名前。2 段は `_two_stage_frozen.yaml` を重ね、起動時に
`model.backbone_checkpoint_path` で親を渡す。

| | ERM | GroupDRO | EO rate |
| --- | --- | --- | --- |
| 1 段 ResNet | `resnet_chexpert_race_known` | `resnet_chexpert_age_race_group_dro_race_known` | `resnet_chexpert_age_race_eo_rate_dual_holdout` |
| 1 段 Spatial LoRA | `spatial_lora_chexpert_race_known` | `spatial_lora_chexpert_age_race_group_dro_race_known` | `spatial_lora_chexpert_age_race_eo_rate_dual_holdout` |
| 1 段 fc のみ | `spatial_lora_chexpert_fc_race_known` | `spatial_lora_chexpert_fc_age_race_group_dro_race_known` | `spatial_lora_chexpert_fc_age_race_eo_rate_dual_holdout` |
| 2 段 Spatial LoRA | `spatial_lora_chexpert_from_resnet_race_known` | `spatial_lora_chexpert_from_resnet_age_race_group_dro_race_known` | `spatial_lora_chexpert_from_resnet_age_race_eo_rate_dual_holdout` |
| 2 段 fc のみ | `spatial_lora_chexpert_fc_from_resnet_race_known` | `spatial_lora_chexpert_fc_from_resnet_age_race_group_dro_race_known` | `spatial_lora_chexpert_fc_from_resnet_age_race_eo_rate_dual_holdout` |

## 状態

2026-10-13 に 39 run がすべて正常終了。起動は `run_logs/hcg_chexpert_ws11_launch_20261013.sh`（ws11、1 条件 1 container で 3 seed を直列、1 GPU に最大 2 container）。


## 2026-10-14：依頼された6種類の図

[requested_figures.ipynb](requested_figures.ipynb)で、45 runのbest validation AUROC checkpointをtest評価した。
ERMのモデル比較、公平性損失の比較、各8群の性能、指定GroupDRO runのq、validationのepoch推移を扱う。
[図一覧](figures/requested_figures/index.html)・[全15図PDF](figures/requested_figures/all_requested_figures.pdf)・
[分析メモ](reports/requested_figures.md)にまとめた。数値の正本は`results/requested_figures/`。

観察として、ERMの属性条件付けではAUROC系とFPR系の改善方向がそろわない。EO-rate（全ペアの版。上の TODO で差し替える）でEopp0が
縮む条件はあるが、全体性能の低下や群別FPRの悪化を伴う。一般的な有効性の結論は未確定とする。
worst-FPRの全epoch値は未記録なので、学習推移図の該当枠はbest checkpointの点のみ。
他の曲線は30 epochのログに基づき、global TPR/FPRはaccuracy・BACC・陽性率から復元した。
