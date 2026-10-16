# 仮説別の分析計画

## 進め方

18条件の総合ランキングは作らず、1トピックにつき1つの問いを扱う。
最初はResNetの対照でDROとclass weightを検証し、その後に変調範囲・モデル構成・学習手順を比較する。
以下の比較は結果を見る前に固定する。先にtestで良かった条件を選んで、後の比較の基準にしない。
この計画は分析の順序を定めるもので、全トピックの一括実行を意味しない。

略記：S34はSpatial LoRA `[stage3,stage4,fc]`、S14は`[stage1,stage2,stage3,stage4,fc]`。
両者ともconv2・rank 4。HはHyperAdapt。Gはglobal class weight、SGはsubgroup class weight。
条件IDとpresetの対応は[実験計画](../../../docs/plans/20261002_experiment.md)、入力は[runs.md](runs.md)。

## 1. DROを加えると性能・公平性はどう変わるか

主比較はM01 → M02（ResNet ERM → DRO/G）。最初のnotebookはこの2条件だけ、各3seedを扱う。
既存の`resnet_erm_vs_groupdro_global.ipynb`を3seed比較とq診断に拡張する。

| 役割 | baseline → 比較条件 | 固定するもの |
| --- | --- | --- |
| 主比較 | M01 → M02 | ResNet、1-stage、G |
| モデルを変えた再現確認 | M03 → M04 | S34、1-stage、G |
| モデルを変えた再現確認 | M07 → M08 | H、1-stage、G |
| 変調範囲を変えた再現確認 | M16 → M17 | S14、1-stage、G |

まず主比較を読み、残りは別パネル／別notebookで同じ現象があるかを確認する。
全体AUROCとworst-group AUROCの変化、改善・悪化した群、qが集中した群を対応づける。
qの集中だけから公平性改善・過学習・因果関係を断定しない。
M01はA100、M02はA6000で学習しており、ハードウェア差は完全には統制されていない。

## 2. class weightをglobalからsubgroupへ変えるとどうなるか

主比較はM02 → M11。DROとResNetを固定する。ERMとの比較をこの問いへ混ぜない。

| 役割 | G → SG | 固定するもの |
| --- | --- | --- |
| 主比較 | M02 → M11 | ResNet、1-stage、DRO |
| 1-stageの再現確認 | M04 → M12、M08 → M14、M17 → M18 | 各モデル、DRO |
| 2-stageの再現確認 | M06 → M13、M10 → M15、M19 → M20 | 各モデル、同seed M01親、凍結、DRO |

主比較でq・群loss・各群の陽性/陰性性能を検証し、再現確認は1-stageと2-stageを分ける。
class weightの変更は群lossの相対関係と数値スケールも変える。step sizeは固定のため、
差は「現在の重み付け方式全体」の効果として読む。異なる重み付けのraw学習lossを同じ尺度として比較しない。
重みなしの条件は追加しない。

## 3. Spatial LoRAをstage1・2にも広げる価値があるか

同じモデル系統・目的関数・学習手順でS34 → S14を比較する。
1-stage DRO/GのM04 → M17を主比較とし、残りは目的別の確認に分ける。

| 学習 / 目的 | S34 → S14 |
| --- | --- |
| 1-stage ERM | M03 → M16 |
| 1-stage DRO/G（主比較） | M04 → M17 |
| 1-stage DRO/SG | M12 → M18 |
| 2-stage DRO/G | M06 → M19 |
| 2-stage DRO/SG | M13 → M20 |

test性能・群性能に加え、trainable parameter数と学習時間を添える。
変調位置の拡張とadapter数・容量の増加は同時に起きるため、位置だけの因果効果とは断定しない。
1-stageと2-stageを同じ棒グラフで混ぜず、各ペアの差を示す。

## 4. 同じ学習条件でモデル構成を変えるとどうなるか

ここだけは、固定した目的関数ごとに最大4構成を同じパネルで比較する。
全18条件を並べず、各パネルで変える軸はモデル構成だけにする。

| パネル | 共通baselineと比較条件 | 問い |
| --- | --- | --- |
| 1-stage ERM（主比較） | M01基準でM03・M16・M07 | DROなしで各構成が何を改善するか |
| 1-stage DRO/G | M02基準でM04・M17・M08 | 同じDRO/Gでモデル差が残るか |
| 1-stage DRO/SG | M11基準でM12・M18・M14 | 重み付けを変えてもモデル差が残るか |
| 2-stage DRO/G | M06基準でM19・M10 | 同じM01親・凍結下でadapter方式がどう違うか |
| 2-stage DRO/SG | M13基準でM20・M15 | 同じM01親・凍結下でadapter方式がどう違うか |

S34対S14の詳細はトピック3に委ねる。ここではResNet対条件付きモデルと、Spatial LoRA対HyperAdaptを読む。
ResNetとの比較にはmetadata利用・adapter容量が含まれるため「metadataだけの効果」とは呼ばない。
S14とHyperAdaptはstage範囲を揃えてもconv種別・generator・parameter数が異なる。
HyperAdaptの優劣をadapter方式だけへ帰属させない。

## 5. 1-stageと凍結2-stageの学習手順はどう違うか

主比較はM04 → M06。残りは同じ比較を各構成・重み付けで確認する。

| モデル | DRO/G：1-stage → 2-stage | DRO/SG：1-stage → 2-stage |
| --- | --- | --- |
| S34 | M04 → M06 | M12 → M13 |
| S14 | M17 → M19 | M18 → M20 |
| H | M08 → M10 | M14 → M15 |

比較できるのは「現在の2つの学習手順の総合差」。1-stageは30epoch、2-stageはERM30＋DRO30epochで、
初期状態・DRO適用期間・凍結・計算量も違う。凍結だけ／2段構成だけの効果とは解釈しない。
M01 → M06・M19・M10は親からの変化を見る補助比較に限る。ERM→ERM対照がないため、
その変化をDRO単独の効果とは呼ばない。追加学習はこの分析計画に含めない。

## 全トピック共通の読み方

- 入力は取り込み済み54runのみ。checkpointは各runのbest validation AUROCに固定する。
  旧M02・中断run・元splitの過去実験・last checkpointは採用しない。
- 同じseed同士の差を3本示し、差の平均・標準偏差と各条件の平均・標準偏差を併記する。
  3seedだけで強い有意差・方式全体の優劣を主張しない。同seedは共通番号であって、異なるモデルの初期値が同一という意味ではない。
- 主評価はtest全体AUROCと、学習群と同じ8群のworst-group AUROC。どちらかだけで単一の総合順位を作らない。
- 補助評価はBACC、AUROC/BACC gap、Eopp0・Eopp1・Eodds。gapが縮んだ場合はworst改善かbest低下かを分ける。
  sex・race_group・age・ethnicityの単独属性は補足として扱う。
- 群別に件数・陽性/陰性数を示す。single-class群の未定義指標はNaNのまま扱う。
- DROでは全30epochのq_end・q_mean・群loss・countを各seed別に見る。best checkpoint epochを図へ示し、
  best選択後のepochの挙動と、評価対象checkpointの状態を混同しない。
- q最大群／最大学習loss群／最小test AUROC群の一致・不一致を記録する。
  weighted CEとAUROCは異なる目的なので、不一致をそのまま実装不良と判定しない。
- cacheはrun・checkpoint・split単位で共有し、表・図・レポートはトピック別に保存する。
  notebookは1つの問いごとに作り、観察事実・解釈候補・未確認事項を分離する。
- testで条件・閾値・比較ペアを選ばない。複数トピックは探索的検証として位置づける。

## 最初に実行する範囲

1. M01対M02の3seed比較：全体・8群性能・公平性とq推移。
2. 結果を確認後、M02対M11の3seed比較：class weightの方式とq／群lossの対応。
3. その後、トピック3→4→5へ進む。各段階で再現確認をどこまで実行するか確認する。

最初の範囲はM01・M02の6run。全54runの予測生成・総当たり比較は最初から行わない。

## notebook実装範囲（2026-10-05）

各トピックを独立した5冊に実装し、上記の主比較のみを1→5の順に検算する。
対象の和集合はM01・M02・M03・M04・M06・M07・M11・M16・M17の9条件×3seed＝27run。
他条件での再現確認・単独属性・詳細診断は各notebook末尾のTODOへ残す。
class weightの再現確認は全アーキテクチャを対象とする方針を維持する。
notebook一覧と実行状態は[README](README.md#notebook)に記録する。

## 追加比較：ResNet基準のERM・DROモデル・学習手順

対象はM01（ResNet ERM）・M02（ResNet DRO）、Spatial S34のM03・M04・M06、HyperAdaptのM07・M08・M10。
各3seed、DROはglobal class weight。`06_resnet_baselines_vs_groupdro_architectures.ipynb`にまとめる。

- M01基準でM02・M03・M04・M06・M07・M08・M10：ERM baselineとの差。
- M02基準でM04・M06・M08・M10：ResNet DROとの差。
- M03→M04、M07→M08：同モデルの1-stage ERM→DRO。
- M04→M06、M08→M10：同モデルの1-stage→凍結2-stage。

AUROCとBACCの全体/worst-groupに、感度・特異度、8群、公平性、q推移を添える。
AUROC gap・BACC gap・Eopp0（TNR gap）・Eopp1（TPR gap）は専用の表・図で比較し、Eoddsも表に含める。
ERM baselineとの差にはモデル・目的・学習手順の違いが含まれる。1-stage対2-stageも手順全体の差。
追加学習は行わず、subgroup weight・Spatial S14・詳細診断はTODOに残す。

## 追加診断：2-stageの感度・特異度

`07_two_stage_operating_point.ipynb`でM04→M06、M08→M10を主比較とする。
M01（同seedの親）、M03・M07（各モデルERM）を補助対照とし、global weightの21runを使う。
既存のbest validation AUROC・test・固定argmaxを維持し、閾値や条件をtestで選び直さない。

1. 8群のTPR/TNR/BACC差と各seedのgap最大・最小群。同率群を残し、陽性/陰性数を併記。
2. 全体BACC差を群別・陽性/陰性別に分解し、寄与の和を検算。群平均BACCも補助表示。
3. 同症例のFN↔TP・FP↔TN件数を集計し、正味変更件数と率の差を検算。

4. DRO12runのq推移、bestまでのq_meanのepoch平均・bestのq_endを群別性能差へ対応づける。
   同seedで1-stage対2-stageを比較し、親M01→2-stageの変化も別表にする。
   best後のepochは配分平均から除外し、期間差・weighted CEと動作点指標の違いを明記する。

ラベル別CE・群lossの詳細、スコア分布、閾値依存性、subgroup weightでの再現確認は次段階のTODO。
追加学習は行わず、凍結単独の因果効果は主張しない。
