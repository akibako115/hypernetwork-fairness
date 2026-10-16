# fairvlmed_architecture_erm

2026-10-07開始。[ham10000_architecture_erm](../ham10000_architecture_erm/README.md)と並行して、
CheXpert No Findingでhypernetworkの効果が見えにくいのはデータセットのせいかを、別のデータセットで確かめる。
CheXpertの主比較のうち、ERMのモデル構成比較（M01・M03・M07）と同じ3条件をHarvard-FairVLMedで回す。

- 問い：データセットだけを替えたとき、属性条件付きモデルとResNetの差は雑音より大きくなるか。
- HAM10000（皮膚・施設ショートカットあり・易しい）と性質の違うデータを並べる。
  FairVLMedはSLO眼底のglaucoma 2値で、先行研究のAUCは0.7〜0.8台と難しい側にある。

## 条件

データセット以外は、CheXpertの主比較のレシピに揃える。

| モデル | preset | CheXpert側の対応 |
| --- | --- | --- |
| ResNet-50 | `resnet_fairvlmed` | M01 |
| Spatial LoRA（`[stage3,stage4,fc]`・conv2・rank 4） | `spatial_lora_fairvlmed` | M03 |
| HyperAdapt（stage1〜4のconv1〜3とFC・rank 4） | `hyperadapt_fairvlmed` | M07 |

- 3条件×seed 42・43・44。ImageNetのResNet-50から1-stageでbackbone全体を学習する。
- batch 128、30 epoch、AdamW（lr `1e-4`、weight decay `1e-2`）、schedulerなし、uniform sampling。
  checkpointはbest `val/auroc`。
- class weightは掛けない（`weighting: none`）。trainの陽性率は50.5%で、global inverseにしても
  重みは約[1.01, 0.99]になり、CheXpertのレシピとの差は無視できる。
- データ：`data/fairvlmed/splits`。配布元の`use`列（train 7,000 / val 1,000 / test 2,000）をそのまま使う。
  1患者1枚なので患者単位の漏洩は無い。targetは`glaucoma`（yes=1）でFairCLIPと同じ。
- 前処理：評価はFairCLIPと同じbicubicで短辺224→CenterCrop 224。学習はその切り出し位置・左右・回転（±10°、黒埋め）を揺らす。
- 属性：入力はgender・race（Asian/Black/White）・ethnicity・language・maritalstatus・連続age。
  評価軸はFairCLIPの4軸（race・gender・ethnicity・language）と`age_group_65`。主軸はrace。
  trainにもethnicity・language・maritalstatusの欠損（`unknown`）があり、欠損用埋め込みは学習される。
  raceとgenderには欠損が無い。

## 評価上の注意

- raceで陽性率が違う（train：Black 65.1%、Asian 47.9%、White 48.0%）。属性条件付きモデルはこの事前確率差を
  入力から直接使える。群内AUROCは事前確率に依らないので、閾値依存の指標（Eopp・BACC）と分けて読む。
- testのAsianは158件（陽性79）、valのAsianは60件（陽性32）しかない。val上の群別指標は雑音が大きい。
- ほぼ黒い画像（`image_blank`）がtestで112件あり、Black（41/305）に偏る。主比較は全2,000件、
  感度分析として`image_blank`除外と、先行研究との突き合わせ用の`fairclip_eval_subset`（1,883件）を並べる。
- testのlanguageはother 11件・spanish 31件。学習中callbackのlanguage軸gapは参考値として扱う。

## 結論

2026-10-07、[01_architecture_comparison.ipynb](01_architecture_comparison.ipynb)で3モデル×3 seedを比較した（主比較はtest全2,000件）。

- 全体AUROC・群別AUROC・race軸のworst-group AUROCとAUROC gapは、3モデルで差が無い（区間が0をまたぐ）。
  blank画像を除いても、FairCLIPの評価対象でも同じ。
- HyperAdaptはrace軸のEopp0が+0.14［0.05, 0.19］悪化した。BlackのTNRが下がりWhiteのTNRが上がる形で、
  raceごとの陽性率の違いを出力に反映したと整合する。群内の順位付けは変わっていない。
- CheXpertと同じく、AUROCベースの公平性に属性条件付けの効果は見えない。データセット固有という説明はこの結果だけでは支持されない。

対象runは[runs.md](runs.md)。
