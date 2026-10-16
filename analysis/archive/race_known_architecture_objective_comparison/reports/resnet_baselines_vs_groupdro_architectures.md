# ResNet基準のERM・DROモデル・学習手順

8条件×seed 42・43・44の24run。同じrace-known test 19,617件をbest validation AUROC checkpointで評価。
DROはglobal class weight、Spatial LoRAはstage3–4・conv2・rank 4。
checkpoint・splitのSHA256と親runは[来歴表](../results/06_resnet_baselines_vs_groupdro_architectures/provenance.csv)。

## 観察事実

3seed平均。各条件の標準偏差と同seedの差は[比較表](../results/06_resnet_baselines_vs_groupdro_architectures/paired_delta_mean_std.csv)。

| 条件 | 全体AUROC | worst AUROC | 全体BACC | worst BACC |
| --- | --- | --- | --- | --- |
| ResNet ERM（M01） | 0.8612 | 0.7299 | 0.7946 | 0.6544 |
| ResNet DRO（M02） | 0.8543 | 0.7251 | 0.7883 | 0.6532 |
| Spatial ERM（M03） | 0.8621 | 0.7388 | 0.7920 | 0.6459 |
| Spatial DRO 1-stage（M04） | 0.8552 | 0.7251 | 0.7832 | 0.6613 |
| Spatial DRO 2-stage（M06） | 0.8643 | 0.7317 | 0.7980 | 0.6530 |
| HyperAdapt ERM（M07） | 0.8610 | 0.7371 | 0.7926 | 0.6554 |
| HyperAdapt DRO 1-stage（M08） | 0.8536 | 0.7188 | 0.7868 | 0.6628 |
| HyperAdapt DRO 2-stage（M10） | 0.8633 | 0.7260 | 0.7980 | 0.6530 |

公平性指標も3seed平均。8群のgapで、低いほど群間差が小さい。Eopp0はTNR gap、Eopp1はTPR gap。

| 条件 | AUROC gap | BACC gap | Eopp0 | Eopp1 |
| --- | --- | --- | --- | --- |
| ResNet ERM（M01） | 0.1498 | 0.1681 | 0.2096 | 0.4728 |
| ResNet DRO（M02） | 0.1498 | 0.1569 | 0.1951 | 0.4452 |
| Spatial ERM（M03） | 0.1411 | 0.1638 | 0.2237 | 0.4860 |
| Spatial DRO 1-stage（M04） | 0.1506 | 0.1442 | 0.2438 | 0.4639 |
| Spatial DRO 2-stage（M06） | 0.1506 | 0.1646 | 0.1881 | 0.4706 |
| HyperAdapt ERM（M07） | 0.1413 | 0.1733 | 0.2209 | 0.4781 |
| HyperAdapt DRO 1-stage（M08） | 0.1564 | 0.1501 | 0.2170 | 0.4408 |
| HyperAdapt DRO 2-stage（M10） | 0.1563 | 0.1815 | 0.2018 | 0.4738 |

標準偏差・Eoddsは[公平性比較表](../results/06_resnet_baselines_vs_groupdro_architectures/fairness_mean_std.csv)。

根拠：[seed別指標](../results/06_resnet_baselines_vs_groupdro_architectures/metrics_by_seed.csv)。
両モデルともDROの1-stage→2-stageで全体AUROC・BACCが3seedとも上昇。
worst-group BACCは平均で低下し、Spatialでは3seedとも低下、HyperAdaptでは2seed低下・1seed上昇。
8群の件数・陽性/陰性数、感度・特異度・公平性と、DRO15runの全30epochのqも保存した。

## 解釈と未確認

全体性能と最悪群の動作点性能は同じ結論ではない。3seedの差を有意差とは扱わない。
2-stageは同seed M01からbackbone・元classifierを凍結して学習し、総学習量は30対30+30epoch。
ERM基準との差はDRO単独の効果ではなく、モデル・目的関数・学習手順を含む比較。
subgroup weight、Spatial stage1–4、群CEのtrain/val診断、閾値選択はnotebook末尾のTODO。
