# 2-stageの感度・特異度

M04→M06（Spatial S34）、M08→M10（HyperAdapt）を主比較とし、M01・M03・M07を補助対照とする。
全21run、seed 42–44、global class weight、race-known test 19,617件、固定argmax。
checkpointはbest validation AUROC。[来歴](../results/07_two_stage_operating_point/provenance.csv)にSHA256・親runを保存した。
予測cacheを再利用し、追加学習・閾値変更はしていない。

## 観察事実

以下は同seedの2-stage−1-stage差の3seed平均。

| 主比較 | 全体BACC差 | TPR側の寄与 | TNR側の寄与 | 群平均BACC差 |
| --- | --- | --- | --- | --- |
| Spatial | +0.01475 | −0.01199 | +0.02674 | +0.01202 |
| HyperAdapt | +0.01123 | +0.00717 | +0.00406 | +0.01555 |

寄与は各群の陽性/陰性数で重み付けした率の差の半分。
全seedでTPR側＋TNR側＝全体BACC差を検算した。
根拠：[寄与の分解](../results/07_two_stage_operating_point/bacc_decomposition_by_seed.csv)、
図：[群別寄与](../figures/07_two_stage_operating_point/bacc_contributions.png)。

### Eopp1：最小感度群が改善していない

主比較4条件・全seedで最小TPR群は≥65×Others（陽性68・陰性1,151件）。
SpatialのTPRは0.4706→0.4314（−0.0392）、TNRは0.8520→0.8746（+0.0226）。
HyperAdaptのTPRは0.4657→0.4461（−0.0196）、TNRは0.8598→0.8598（平均差は0）。
この群のBACCはそれぞれ−0.0083、−0.0098。

Spatialでは8群中7群で平均TPRが低下、≥65×Blackは平均差0。
≥65×Othersの低下幅は、各seedの最大TPR値の平均低下幅（−0.0325）より大きく、Eopp1が平均+0.0067。
HyperAdaptでは8群中6群で平均TPRが上昇する一方、≥65×Othersと<65×Blackは低下。
最大TPR値も平均+0.0134となり、Eopp1は平均+0.0330。
ただしEopp1の増加はSpatialでは2/3seed、HyperAdaptでも2/3seedで、全seed一様ではない。

親M01の≥65×Others TPRは0.4363。両2-stageはこの値に近いが、親の特徴を保持したことが原因とは断定できない。
根拠：[群別率](../results/07_two_stage_operating_point/subgroup_by_seed.csv)、
[gap両端](../results/07_two_stage_operating_point/gap_extrema_by_seed.csv)、
図：[TPR/TNR絶対値](../figures/07_two_stage_operating_point/group_tpr_tnr.png)。

### Eopp0：特異度の低い側の改善が大きい

Spatialでは全8群で平均TNRが上昇。
<65×Asianは0.6082→0.6871（+0.0789）、≥65×Othersは0.8520→0.8746（+0.0226）。
各seedの最小TNR値の平均は+0.0784、最大値は+0.0226で、Eopp0は−0.0558。
一部seedでは最小群が<65×Othersへ切り替わる。

HyperAdaptの<65×Asian TNRは0.6429→0.6612（+0.0184）。
最小TNR値の平均は+0.0184、最大値は+0.0032で、Eopp0は−0.0152。
ただし全群改善ではなく、<65×Othersと≥65×Asianは平均TNRがわずかに低下する。
Eopp0は両モデルとも全3seedで縮小している。

### 同症例の判定変更

≥65×Othersの陽性の正味改善件数（FN→TP minus TP→FN）は、
Spatialで平均−2.67件、HyperAdaptで−1.33件。陰性はそれぞれ+26件、0件。
陽性68件なので、数件の違いでもTPRは動く。これらは同じ症例集合を各seedで評価した平均であり、独立した204陽性症例ではない。
全群・全seedで正味変更件数/ラベル件数＝ΔTPRまたはΔTNRを検算した。
根拠：[変更件数](../results/07_two_stage_operating_point/prediction_transitions_by_seed.csv)、
図：[判定変更](../figures/07_two_stage_operating_point/prediction_transitions.png)。

## 学習中のqと性能変化

DRO12runの30epoch×8群、計2,880行を検証した。
評価checkpointまで（best epochを含む）のq_meanをepoch等重みで平均し、bestのq_endと併記する。
q_meanは各train forwardの更新後qのbatch平均。best後のepochは集計から除外する。
best epoch（0始まり）はM04が9/8/8、M06が3/7/9、M08が17/22/9、M10が11/10/6（seed42/43/44）。
集計期間は異なり、このq平均は勾配の累積寄与ではない。

### 重みは低感度群ではなく65歳未満へ集中

bestまでの平均qの合計は、65歳未満4群でM04 89.1%、M06 82.8%、M08 93.2%、M10 87.9%。
両2-stageで平均q最大群は<65×Others（Spatial 0.3044、HyperAdapt 0.3512）。
最小TPR群≥65×Othersの平均qはSpatial 0.0491、HyperAdapt 0.0348で、均等重み0.125より小さい。
best時の同群q_endはそれぞれ平均0.0109、0.0026。
qが重視する学習群と、testで感度が最も低い群は一致していない。

| ≥65×Others（陽性68件） | 1-stageの平均q | 2-stageの平均q | ΔTPR | ΔTNR | ΔBACC |
| --- | --- | --- | --- | --- | --- |
| Spatial | 0.0313 | 0.0491 | −0.0392 | +0.0226 | −0.0083 |
| HyperAdapt | 0.0190 | 0.0348 | −0.0196 | 0.0000 | −0.0098 |

同群の平均qは全seedで2-stageの方が大きい。それでもSpatialのTPR/BACCは全seedで低下、
HyperAdaptは2seedで低下・1seedで上昇する。悪化を「1-stageよりqが減ったから」とは説明できない。
一方、<65×Whiteは両モデルでqが減るが平均BACCが改善し、q増減と性能差は単調に対応しない。

根拠：[qと手順差の表](../results/07_two_stage_operating_point/q_vs_workflow_changes_by_seed.csv)、
[bestまでのq集計](../results/07_two_stage_operating_point/q_summary_to_best_by_seed.csv)。
図：[q時系列](../figures/07_two_stage_operating_point/q_history_by_seed.png)、
[配分](../figures/07_two_stage_operating_point/q_allocation_to_best.png)、
[Δqと性能差](../figures/07_two_stage_operating_point/delta_q_vs_group_changes.png)。
散布図は群・seedを識別する記述的な照合であり、24点を独立標本とした相関検定は行わない。

### 親ERMからの変化とは区別する

≥65×Othersの親M01→2-stageの平均TPR差はSpatial −0.0049、HyperAdapt +0.0098。
平均BACC差はそれぞれ−0.0014、−0.0015。
HyperAdaptは1-stage DRO比では感度が低下するが、親ERM比では感度が改善する。
したがって1-stage→2-stageの差は「2-stage学習そのものが感度を下げた」ことを意味しない。
根拠：[qと親からの変化](../results/07_two_stage_operating_point/q_vs_parent_changes_by_seed.csv)。

## 解釈候補と未確認

Spatialの全体BACC改善は感度の低下を特異度の改善が上回った結果。
HyperAdaptは全体の感度・特異度がともに改善する一方、最小感度群は取り残されている。
両者を同じ「全体性能向上・公平性悪化」という説明で一括りにはできない。
群平均BACCも上昇しており、全体改善を単に群サイズの重み付けだけへ帰属させることもできない。

3seedのばらつきと少数陽性群の不確実性は残る。凍結、初期値、総学習量が違うため、凍結単独の因果効果ではない。
qは学習中の群別global-weighted CEに応答し、testの感度を直接最適化するものではない。
低感度群のqが小さいことは観察できるが、q不足が性能低下の原因とは断定できない。
群内の陽性率・陽性/陰性別CEが群lossの順位に影響する可能性は次の検証対象。
bestの群lossはq集計表に保存したが、時間的なloss変化・ラベル別CE・スコア分布・閾値依存性、
subgroup weightでの再現確認は未実施。
