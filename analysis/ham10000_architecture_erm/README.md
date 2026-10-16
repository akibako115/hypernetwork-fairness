# ham10000_architecture_erm

2026-10-07開始。CheXpert No Findingでhypernetworkの効果が見えにくいのは、データセットのせいかを確かめる。
CheXpertの主比較のうち、ERMのモデル構成比較（M01・M03・M07）と同じ3条件を、HAM10000で回す。

- 背景：CheXpertでは、ResNet ERMに対するSpatial LoRA・HyperAdaptの差が、
  worst-group AUROCのtest標本雑音（paired bootstrap SE 0.010〜0.013）と区別できなかった
  （[機構診断](../archive/fairness_mechanism_probe/reports/mechanism_findings.md)）。
- 問い：データセットだけを替えたとき、属性条件付きモデルとResNetの差は雑音より大きくなるか。

## 条件

データセット以外は、CheXpertの主比較のレシピに揃える。

| モデル | preset | CheXpert側の対応 |
| --- | --- | --- |
| ResNet-50 | `resnet_ham10000` | M01 |
| Spatial LoRA（`[stage3,stage4,fc]`・conv2・rank 4） | `spatial_lora_ham10000` | M03 |
| HyperAdapt（stage1〜4のconv1〜3とFC・rank 4） | `hyperadapt_ham10000` | M07 |

- 3条件×seed 42・43・44。ImageNetのResNet-50から1-stageでbackbone全体を学習する。
- batch 128、30 epoch、AdamW（lr `1e-4`、weight decay `1e-2`）、schedulerなし、uniform sampling。
- class weightはtrain splitのglobal inverse。checkpointはbest `val/auroc`。
- データ：`data/ham10000/splits`。targetはMEDFAIRと同じ悪性2値（mel・akiec）。
  testはISIC2018 Task3の公式test 1,511件で、fairness指標はsexとageが揃う1,217件で計算する。
- 属性：入力はsex・localization（部位14区分）・連続age。評価軸はsexと`age_group_65`、その交差4群。
- MEDFAIR（ResNet-18・SGD・batch 1024・early stopping）とはbackboneと学習設定が違う。
  CheXpertとの対照を優先し、データ以外は動かさない。

## 評価上の注意

- trainとvalにはsexとageの欠損が無い。testの294件（うち284件は別施設の`dataset=external`）は欠損で、
  属性条件付きモデルには学習中に一度も見ていない欠損用の埋め込みと欠損フラグが入る。
  ResNetは属性を使わないので、全1,511件の全体AUROCはモデル構成と欠損入力の扱いを混ぜてしまう。
  **モデル間の主比較は、属性の揃った1,217件の全体・群別で行い**、全1,511件は補助に回す。
- testの陽性は群ごとに28〜82件（female×<65が最少）しかない。群別AUROCの雑音はCheXpertより大きい。
- 評価軸のage 2区分はMEDFAIRと同じ群になる（ageは5歳刻みで、MEDFAIRは60歳を若年側に入れる）。
  MEDFAIRと違い、class weightを掛け、checkpointはval全体AUROCで選ぶ（CheXpertの主比較と同じ）。

## 結論

2026-10-07、[01_architecture_comparison.ipynb](01_architecture_comparison.ipynb)で3モデル×3 seedを比較した（主比較は属性の揃う1,217件）。

- Spatial LoRA・HyperAdaptはResNet-50より全体AUROCが+0.024・+0.022高く、病変単位bootstrapの95%区間も0をまたがない。
  CheXpertと違い、モデル構成の差がtest標本の雑音より大きい。
- 上がり方は全群でほぼ一様で、AUROC gap・Eopp・Eoddsには差が無い。公平性ではなく全体性能の改善として現れている。
- ただしこの+0.024は、属性の欠損が集中する新施設（external）をほぼ含まない。全1,511件ではSpatial LoRA +0.015・HyperAdapt +0.005まで縮み、
  external単独ではResNet-50を下回る（0.846に対し0.822・0.799）。

### 属性の差し替え推論（2026-10-07、[counterfactual.py](counterfactual.py)）

学習済みの条件付きモデルへ、属性を差し替えてtestを推論し直した（結果は`results/02_metadata_counterfactual/`）。
差はexternalを除く既存施設の行で、ResNet-50に対する3 seed平均。

- **黒色腫 vs 母斑の改善（+0.02）は行ごとの属性による**。全行の属性を同じ値に固定すると消える（+0.004・-0.001）。
  メタデータだけのモデルでも0.82で分けられ、ResNet-50の出力へ後から足すだけでもほぼ再現する（0.927→0.940）。
- **黒色腫 vs bkl・bccの改善は属性に依存しない**。属性を固定してもSpatial LoRAでbkl +0.055・bcc +0.070が残る
  （固定前は+0.065・+0.071）。メタデータだけのモデルはこれらの組を分けられず（0.51・0.43）、後から足しても改善しない。
  改善はadapterを足した構成か、属性で事前確率を受け持たせて学習したことのどちらかにある。区別には属性を定数にして学習し直す必要がある。
- **externalでの劣化は年齢欠損の符号化による**。欠損年齢は配布CSVで0歳として入り、標準化後に約-3 SDの値になる。
  train/valには年齢欠損行が無いので、この入力は学習分布の外にある。年齢をtrain平均で埋めると、externalはSpatial LoRA 0.822→0.871・
  HyperAdapt 0.799→0.858とResNet-50（0.846）を上回り、HyperAdapt seed 43の崩れ（0.671）も0.880まで戻る。性別だけ埋めても戻らない。
- 属性を行間で入れ替えると、全行固定より悪くなる（黒色腫 vs 母斑でResNet-50を下回る）。モデルは属性を実際に使っている。

対象runは[runs.md](runs.md)。
