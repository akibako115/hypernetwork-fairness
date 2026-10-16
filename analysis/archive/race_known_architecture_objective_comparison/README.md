# race_known_architecture_objective_comparison

> **2026-10-13 注記（公平性損失の実装修正）**：[修正記録](../../../docs/reviews/20261013_fairness_objective_fixes.md)。DRO の run はすべて自然抽出（`train_sampling: uniform`）で、バッチに出なかった群の q が系統的に縮んでいた。
> 少数群（≥65×Black など）ほど強く縮む。DRO の効果と q 推移についての観察は、旧実装の結果として読む。
> ERM 条件どうしの比較は影響を受けない。

2026-10-02に開始したrace-knownの主比較13条件 × seed 42・43・44を、このpackageにまとめる。
ResNet / Spatial LoRA / HyperAdapt、ERM / Group DRO、global / subgroup class weight、
1-stage / 2-stageを比較する。条件・GPU・実行順序は[実験計画](../../../docs/plans/20261002_experiment.md)を参照。
2-stageは同seedのResNet ERMから開始し、backboneと元classifierを凍結する。

追加で全stage Spatial LoRA（`[stage1,stage2,stage3,stage4,fc]`、conv2・rank 4）も扱う。
M16〜M20は1-stage ERM / DRO global / DRO subgroupと2-stage ERM→DRO global / subgroupの5条件。
既存M03・M04・M12・M06・M13に対し変調範囲を変更し、各3seedを比較する（adapter容量も増える）。

データは`splits_race_known`。DRO群は年齢2区分 × race 4区分の8群で、Othersも含める。
元split・14群の過去分析は、このpackageへ混ぜない。

## runの来歴と取り込み

初回の学習runには、起動時に流用した`study=two_stage_vs_e2e_age_race_group_dro`が残る。
この値は変更しない。q診断修正版のDRO runはこのpackageをstudyに指定する。
引用は[runs.md](runs.md)に明示し、
成功終了したrunを`runs/`へhardlinkで取り込む。学習中runや中断した旧runは比較に使わない。

## notebook

[仮説別の分析計画](analysis_plan.md)に従い、DRO追加→class weight→変調範囲→モデル構成→学習手順の順に分析する。
最初はM01対M02の3seed比較に限定し、全18条件を一括ランキングしない。

| 順序 | notebook | 主比較 |
| --- | --- | --- |
| 1 | [DRO追加](resnet_erm_vs_groupdro_global.ipynb) | M01 → M02 |
| 2 | [class weight](02_class_weight_scope.ipynb) | M02 → M11 |
| 3 | [Spatial LoRAの範囲](03_spatial_lora_stage_range.ipynb) | M04 → M17 |
| 4 | [アーキテクチャ](04_architecture_comparison.ipynb) | ERM：M01基準、M03・M16・M07 |
| 5 | [1-stage対凍結2-stage](05_one_stage_vs_frozen_two_stage.ipynb) | M04 → M06 |
| 6 | [ResNet基準のERM・DROモデル・学習手順](06_resnet_baselines_vs_groupdro_architectures.ipynb) | M01・M02基準、M03・M04・M06・M07・M08・M10 |
| 7 | [2-stageの感度・特異度](07_two_stage_operating_point.ipynb) | M04→M06、M08→M10。M01・M03・M07は補助対照 |
| 8 | [凍結2-stage Spatial LoRAの中身](08_two_stage_spatial_lora_inspection.ipynb) | M06・M13・M19・M20の生成ΔW・親M01からの予測変化・反実仮想metadata・encoder空間 |
| 9 | [ERMの群別陽性率と性能差](09_erm_prevalence_by_subgroup.ipynb) | M01の年齢2区分・race3区分（White/Asian/Black）別の陽性率・AUROC・BACC・TPR・TNR |
| 10 | [class weightと8群性能](10_class_weight_scope_by_group.ipynb) | ERM・DRO/G・DRO/SGのq推移と8群性能：ResNet（M01・M02・M11）、Spatial LoRA（M03・M04・M12、two-stage M06・M13）。別節でERMのモデル間比較（M01・M03・M07）の全体・8群性能 |

6冊目はglobal class weight・Spatial S34の8条件×3seed。ERM基準、DRO基準、同モデルの目的・手順差を分け、
AUROC・BACCの全体/worst-group、感度・特異度、8群、qを表示する。
AUROC gap・BACC gap・Eopp0・Eopp1は専用の表と図に示し、Eoddsも表に残す。
subgroup class weightとSpatial S14は混ぜない。BACCとCEは別の指標として扱う。

全冊が独立にseed 42・43・44のbest validation AUROC checkpointを同じtest CSVで評価する。
全体・8群性能、公平性、同seed差の平均と標準偏差、DROのq推移に限定し、追加比較は末尾のTODOに残す。
class weightの他アーキテクチャでの検証もTODOに含め、ResNetだけの結論を一般化しない。
指標・入力検証の共通処理はpackage内の[comparison.py](comparison.py)に置く。

予測cacheの生成コマンドはnotebook冒頭に掲載する。
cacheは`cache/`で共有し、表・図は`results/<topic>/`・`figures/<topic>/`へ分ける。
学習GPUを推論で自動使用しない。cache生成時は利用可能なdeviceを明示する。
各notebookは通常のRestart & Run Allで再生成できる。Git側のnotebook出力は空に保つ。
学習manifestにはtrain/valだけが記録されているため、testは現在のCSVのSHA256を来歴表へ保存し、
cacheとの画像行順・target一致を検算する。checkpointとtrain/valは学習記録のSHA256と照合する。

## 状態：全54 runの同期・取り込み完了

2026-10-05に全18条件×3seedをcheckpoint込みで取り込んだ。
best / last計108 checkpointのSHA256一致を確認済み。DRO42 runの全30epoch分のqもCSVで確認済み。
対象一覧は[runs.md](runs.md)。2026-10-05に5冊の主比較（9条件×3seed＝27run）を実行した。
各冊で全セル実行を確認済み。分析関連87テストとRuffも通過。
観察値と未確認事項は[主比較の実行記録](reports/primary_comparisons.md)を参照。
追加の6冊目も24runで全セル実行を確認。表・図・q推移をトピック別に保存した。
観察と未確認事項は[ResNet基準の比較記録](reports/resnet_baselines_vs_groupdro_architectures.md)。
分析関連89テストとRuffが通過。その他の条件での再現確認は未実施。

7冊目はglobal weightの7条件×3seed＝21runで全セル実行済み。
群別TPR/TNR、gap両端、全体BACCへの寄与、同症例の判定変更を保存・検算した。
両モデルの最小TPR群は≥65×Others（陽性68件）で、2-stageで平均TPRが低下する。
全体改善の内訳はSpatialでは特異度中心、HyperAdaptでは感度・特異度の両方。
qのbestまでの平均・best終端値と、群別の性能差・親ERMからの変化も照合した。
qは65歳未満へ集中する一方、≥65×Othersは2-stageでqが増えても1-stage比の平均感度が低下。
詳細は[動作点の分析記録](reports/two_stage_operating_point.md)。分析関連94テストとRuffが通過。

8冊目は2-stage Spatial LoRA 4条件×3seedの中身を、best（主）とlast（補助診断）で調べた。
画像forwardが要る予測probeは`spatial_lora_inspection.py`で`cache/*_spatial_probe.npz`へ書き、ΔW=0経路が同seed M01と一致することを検算した。
観察は[Spatial LoRAの中身の記録](reports/two_stage_spatial_lora_inspection.md)。

q時系列が欠損した旧M02を分析対象から除外し、旧runの取り込み・DRO予測cache・比較表・図・分析メモを外した。
ERM runと予測cacheは保持し、DROはq診断修正版のみを選択する。
総当たりや全条件ランキングは作らず、結果の採用・詳細な追加分析は主比較を読んでから判断する。
