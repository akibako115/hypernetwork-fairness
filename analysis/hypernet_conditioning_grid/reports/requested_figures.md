# ERMの属性条件付けと公平性損失を、8群の性能で比較する

**ERMの属性条件付けで、公平性が一貫して改善したとはいえない。** 1-stageのSpatial LoRA（stage3–4＋fc）はResNetに比べ、平均AUC gapが0.1497から0.1411に縮み、worst-AUCが0.7299から0.7387に上がった。一方、Eopp0は0.2096から0.2237、worst-FPRは0.3371から0.3633に悪化した。

公平性損失を加えた結果も指標によって異なる。同じSpatial LoRAのEO-rateではEopp0とworst-FPRが下がるが、全体AUROCも0.8621から0.8461へ下がる。「公平性損失でも改善しない」と一括りにはできない。以下は3 seedの記述的な比較であり、効果の有無を有意差検定で確定したものではない。

- [図一覧（画像埋め込みHTML）](../figures/requested_figures/index.html)
- [全15図をまとめたPDF](../figures/requested_figures/all_requested_figures.pdf)
- [再実行用notebook](../requested_figures.ipynb)
- [条件ごとの平均・SD](../results/requested_figures/test_mean_sd.csv)／[同seed差の平均・SD](../results/requested_figures/paired_deltas_mean_sd.csv)

## 比較するのは15条件・45 runのbest checkpoint

CheXpert race-knownのtest 19,617件を、年齢（65歳未満／65歳以上）×race（White／Asian／Black／Others）の8群に分けた。モデルはResNet-50、Spatial LoRA（stage3–4＋fc）、Spatial LoRA（fcのみ）。Spatial LoRAは1-stageと、同seedのResNet ERMを凍結して補正する2-stageを含む。目的関数はERM、修正後GroupDRO、修正後EO-rate。seedは42・43・44。

図1〜4は各runのbest validation AUROC checkpointをtestで評価する。図6はvalidationの全30 epoch。2-stageのepochは補正学習開始から数え、親ResNetの30 epochを足さない。2-stageのERM図に置いたResNetは親の参考値であり、ResNetの継続学習ではない。

陽性はNo Finding。TPR/FPR/BACCの閾値は全群共通の0.5とした。AUC gapは8群のAUROC最大−最小、Eopp0はFPR最大−最小、worst-AUCはAUROC最小、worst-FPRはFPR最大。これらをseedごとに求めてから平均し、図のエラーバー・帯にseed間の標本SDを示す。SDは患者標本の不確実性を表さない。

## 図1・2：AUROCの改善とFPRの改善は一致しない

[図1：ERMの公平性](../figures/requested_figures/01_erm_fairness.png)では、同じ指標の軸範囲をそろえた。各seedの点と平均±SDを重ねている。

| ERM条件 | AUC gap ↓ | Eopp0 ↓ | worst-AUC ↑ | worst-FPR ↓ |
| --- | ---: | ---: | ---: | ---: |
| ResNet（親） | 0.1497 | 0.2096 | 0.7299 | 0.3371 |
| Spatial S3–4＋fc・1-stage | 0.1411 | 0.2237 | 0.7387 | 0.3633 |
| Spatial fc・1-stage | 0.1492 | 0.2269 | 0.7332 | 0.3398 |
| Spatial S3–4＋fc・2-stage | 0.1504 | 0.2058 | 0.7313 | 0.3310 |
| Spatial fc・2-stage | 0.1472 | 0.2095 | 0.7340 | 0.3419 |

観察として、1-stageの両Spatial構成はResNetより平均Eopp0・worst-FPRが高い。2-stageのS3–4＋fcでは両者が少し下がる一方、AUC gapは少し広がる。したがって、図の説明は「ERMではどの公平性指標も改善しない」より、「改善する指標と悪化する指標があり、属性条件付けだけで公平性が一貫して改善するわけではない」が適切である。

[図2・1-stage](../figures/requested_figures/02_erm_groups_one_stage.png)と[図2・2-stage](../figures/requested_figures/02_erm_groups_two_stage.png)は、AUROC・BACC・TPR・FPRを群別に示す。1-stageのS3–4＋fcでは、65歳未満×AsianのFPRがResNetの0.3371から0.3633へ上がる。65歳以上×OthersのAUROCは0.7299から0.7387へ上がるが、TPRは0.4363から0.4314で、BACCも0.6544から0.6459へ下がる。ランキング性能と固定閾値での性能を分けて読む必要がある。

群別の分母は図と[件数表](../results/requested_figures/test_group_counts.csv)に併記した。特に65歳以上×Blackは陽性22件・陰性303件、65歳以上×Othersは陽性68件・陰性1,151件。小さな群のTPRを、多数群と同じ精度の推定として扱わない。

## 図3・4：EO-rateの群間差縮小には、性能の変化が伴う

公平性指標は[図3・S3–4＋fc](../figures/requested_figures/03_objective_fairness_spatial_lora.png)と[図3・fcのみ](../figures/requested_figures/03_objective_fairness_spatial_lora_fc.png)。各モデル・各stageのERMを対照に置いた。

S3–4＋fcの結果は次のとおり。群別の値は[1-stage](../figures/requested_figures/04_objective_groups_spatial_lora_one_stage.png)と[2-stage](../figures/requested_figures/04_objective_groups_spatial_lora_two_stage.png)で確認できる。

| stage・目的 | Eopp0 ↓ | worst-FPR ↓ | worst-AUC ↑ | 全体AUROC ↑ | 全体BACC ↑ |
| --- | ---: | ---: | ---: | ---: | ---: |
| 1-stage ERM | 0.2237 | 0.3633 | 0.7387 | 0.8621 | 0.7921 |
| 1-stage GroupDRO | 0.1990 | 0.3531 | 0.7197 | 0.8540 | 0.7869 |
| 1-stage EO-rate | 0.1251 | 0.3128 | 0.7324 | 0.8461 | 0.7835 |
| 2-stage ERM | 0.2058 | 0.3310 | 0.7313 | 0.8642 | 0.7971 |
| 2-stage GroupDRO | 0.1919 | 0.3271 | 0.7309 | 0.8640 | 0.7981 |
| 2-stage EO-rate | 0.1275 | 0.3327 | 0.7214 | 0.8572 | 0.7927 |

観察として、1-stageのEO-rateではEopp0とworst-FPRが下がる。しかし65歳未満×AsianのFPRが0.3633から0.2949へ下がるのと同時に、65歳以上×OthersのFPRは0.1396から0.2369へ上がる。Eopp0の縮小には、FPRが高かった側の改善と、低かった側の悪化の両方が含まれる。

2-stageのEO-rateでもEopp0は下がるが、worst-FPRは0.3310から0.3327でほぼ変わらず、worst-AUCと全体AUROCは下がる。2-stage GroupDROは同stageのERMに近い結果で、全体AUROCの平均差は約−0.0002。これを効果の不存在や同等性の証明とはしない。

fcのみの群別性能も[1-stage](../figures/requested_figures/04_objective_groups_spatial_lora_fc_one_stage.png)・[2-stage](../figures/requested_figures/04_objective_groups_spatial_lora_fc_two_stage.png)に保存した。1-stageのEO-rateではEopp0が0.2269から0.1402へ下がる一方、worst-FPRは0.3398から0.3731へ上がる。変調範囲が違えば、同じEO-rateでも最悪群のFPRの動きは異なる。

解釈候補として、群間差を縮める制約が、全群の性能を同じ向きに改善するとは限らない。ただしEO-rateは訓練データ量と抽出方法もERM・GroupDROと異なるため、差を目的関数だけに帰属できない。

## 図5：指定runのqは65歳未満へ集中した

[図5](../figures/requested_figures/05_group_dro_weights.png)はSpatial S3–4＋fc、1-stage、GroupDRO、seed 42。runは結果の大小で選ばず、作図前に指定した。[元のq表](../results/requested_figures/selected_q_history.csv)にはrun-idを含む。

bestのepoch 14では、epoch末のqが65歳未満×Whiteに0.4891、同Othersに0.3574配分されていた。epoch 30ではWhiteが0.6906、Othersが0.1520となる。図はepoch内平均とepoch末の両方を示し、各時点の8群の和が1であることを検算した。

qが高いことは、その群のFPRやAUROCが最悪であることを直接意味しない。qは学習損失に対する重みであり、この図だけから「公平性上の最悪群へ重点を置いた」とは解釈しない。

## 図6：EO-rateの後半は、Eopp0とともにTPRも下がる

学習推移は5枚。各図の上段に公平性指標、下段に全体AUROC・BACC・TPR・FPRを置いた。

- [ResNet・1-stage](../figures/requested_figures/06_learning_resnet_one_stage.png)
- [Spatial S3–4＋fc・1-stage](../figures/requested_figures/06_learning_spatial_lora_one_stage.png)
- [Spatial fc・1-stage](../figures/requested_figures/06_learning_spatial_lora_fc_one_stage.png)
- [Spatial S3–4＋fc・2-stage](../figures/requested_figures/06_learning_spatial_lora_two_stage.png)
- [Spatial fc・2-stage](../figures/requested_figures/06_learning_spatial_lora_fc_two_stage.png)

EO-rateのbest validation AUROCは全15 runでepoch 1〜3にある。S3–4＋fc・1-stageのEO-rateはepoch 1→30で、平均Eopp0が0.1328→0.0438、全体FPRが0.2821→0.0517となる。一方、TPRは0.8396→0.3047、BACCは0.7787→0.6265、worst-AUCは0.7413→0.6372へ下がる。後半のEopp0低下だけを公平性改善として示すと、同時に起きた分類性能の低下を見落とす。

これは全EO-rate構成に同じ程度で起きるわけではない。fcのみ・2-stageのepoch 30では、平均TPRが0.8184、BACCが0.7971である。モデルと学習手順を分けて推移を見る必要がある。

**worst-FPRの全epoch推移は未取得。** ログに群別FPRがなく、ERM・GroupDROの途中checkpointも保存されていない。この枠には各seedのbest checkpointをvalidationで評価した点だけを置き、線で結んでいない。補間や他指標からの推定で埋めていない。

## 再現手順と、確認できた範囲

予測はws11で既に起動されていたGPUジョブから回収した。入力は[取り込み済みrun一覧](../runs.md)とpackageのcache。各cacheについてrun-id、split、checkpoint SHA256、split CSV SHA256、画像行順、target一致を確認した。bestとlastが同一バイト列のrunは、SHA256一致に基づき同じcacheをbestとして使う。[来歴表](../results/requested_figures/provenance.csv)に選択checkpoint名とcache内のcheckpoint名を両方残した。

```bash
uv run jupyter nbconvert --to notebook --execute --inplace \
  --ExecutePreprocessor.kernel_name=hypernet-fairness \
  analysis/hypernet_conditioning_grid/requested_figures.ipynb
```

図と比較条件はnotebook、入力検証と共通集計は[requested_analysis.py](../requested_analysis.py)。図15枚をPNG・PDFで、平均・SD・同seed差・群別値・epoch値をCSVで保存する。

global TPR/FPRの学習ログは直接残っていないため、micro accuracy、BACC、validation陽性率から復元した。復元したTP・FPは全1,350 epochで整数件数から0.001件以内にあり、固定混同行列のテストでも確認した。

学習時のvalidationはbf16-mixedでargmax判定、cacheはfloat32で陽性確率≥0.5の判定であり、完全一致はしていない。best epochでの差の最大絶対値はAUROC 0.000224、TPR 0.002525、FPR 0.002600、Eopp0 0.004775。[差分表](../results/requested_figures/validation_cache_log_errors.csv)に全runを保存した。精度・同点判定の違いが原因候補だが、その寄与を切り分ける再推論はしていない。図1〜4はcache、図6の線は学習ログに統一した。

EO-rateはtrainの20%を双対更新用に分け、残り80%をセル均等抽出する。ERM・GroupDROはtrain全体を自然抽出する。また2-stageの親ResNetは、EO-rateの双対更新用患者も学習済み。この違いと3 seed・少数群という限界があるため、次の判断は「どの指標と群を改善と呼ぶか」を図1〜4で絞るところまでとする。今回の値だけで、属性条件付けや公平性損失の一般的な有効性を結論づけない。
