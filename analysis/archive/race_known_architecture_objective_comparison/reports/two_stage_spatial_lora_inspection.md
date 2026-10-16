# 凍結2-stage Spatial LoRAの中身（2026-10-05）

[08 notebook](../08_two_stage_spatial_lora_inspection.ipynb)の観察記録。対象はM06（S34・DRO/G、主対象）、
M13（S34・DRO/SG）、M19（S14・DRO/G）、M20（S14・DRO/SG）の各seed 42・43・44。
親は同seedのM01。数値は3seed平均で、表・図は`results/`・`figures/08_two_stage_spatial_lora_inspection/`。

best validation AUROCのepochは0〜10（M13 s44とM19 s42は0、M20は1〜3）。
lastは「生成器が育つとどうなるか」を見る補助診断であり、性能比較には使わない。

## 検算

- probeのΔW=0経路（凍結backbone＋元classifier）は、全12run・best/lastとも同seed M01の予測cacheと完全一致（最大差0）。
- M06 bestの実際の属性での出力は、既存の予測cacheと完全一致。
- ΔW分解は素朴な全展開計算と一致することをテストで固定（`analysis/tests/test_spatial_lora_inspection.py`）。

## 観察事実

### 生成ΔWの大きさ（元重みのFrobenius normとの比）

| 条件 | best：conv rms | best：shared割合 | last：conv rms | last：shared割合 |
| --- | --- | --- | --- | --- |
| M06 | stage3 0.17・stage4 0.23 | 0.92 | 0.49・0.68 | 0.74 |
| M13 | 0.12・0.14 | 0.82 | 0.42・0.60 | 0.61 |
| M19 | stage1〜4 0.06〜0.12 | 0.87 | 0.21〜0.58 | 0.70 |
| M20 | 0.03〜0.06 | 0.75 | 0.19〜0.49 | 0.57 |

shared割合は||μ||²/E||ΔW||²（conv層平均）。残りが属性で変わる部分で、8群間（between）と8群内に分かれる。
M06 bestではbetween 0.03、群内0.05にとどまる。lastでは各条件ともbetween 0.12〜0.20へ増える。
深い層ほどΔWが大きい。fcの相対ΔWはSGの条件（M13・M20）で大きい（last 0.22・0.26、Gでは0.04・0.06）。

### 親M01からの予測変化

| 条件/ckpt | mean abs Δmargin | 判定反転率 | 群平均で説明される割合 | AUROC（親0.8612） | offset-only AUROC |
| --- | --- | --- | --- | --- | --- |
| M06 best | 0.59 | 3.8% | 0.008 | 0.8643 | 0.8611 |
| M06 last | 1.21 | 7.8% | 0.114 | 0.8577 | 0.8478 |
| M13 best | 0.43 | 7.1% | 0.080 | 0.8604 | 0.8600 |
| M19 best | 0.47 | 5.2% | 0.009 | 0.8643 | 0.8612 |
| M20 best | 0.45 | 6.6% | 0.145 | 0.8604 | 0.8593 |

親のmarginの標準偏差は1.84。M06 bestでは全群でΔmarginが−0.4〜−0.55とほぼ一律に負（陽性側を下げる）。
群内の親との順位相関は0.96以上。lastでは≥65 × Othersなどで0.80まで下がる。
M06・M19 bestの全体AUROC +0.003は群平均のずれでは説明されない（offset-onlyは親と同値）。

### 反実仮想metadata（同じ画像でraceまたは年齢だけを差し替える）

| 条件/ckpt | 親からの|Δ| | race spread | age spread（50歳↔75歳） | race差し替えで判定反転 |
| --- | --- | --- | --- | --- |
| M06 best | 0.59 | 0.32 | 0.11 | 2.1% |
| M06 last | 1.21 | 1.97 | 0.40 | 7.5% |
| M19 best | 0.47 | 0.27 | 0.09 | 2.5% |
| M13 best | 0.43 | 0.41 | 0.24 | 4.6% |
| M20 best | 0.45 | 0.24 | 0.32 | 3.1% |

bestではrace spreadが親からの変化より小さい。lastでは逆転し、raceだけでmarginが2程度動く。
M06では「→White」がどの群でも最もmarginを上げる（best +0.1、last +0.7〜+1.3）。
全員に同じraceを入れても、bestの全体・8群AUROCは実際の属性とほぼ同じ（差は0.003以内）。

### metadata encoderの空間（train 3,467属性組）

- 16次元のうちparticipation ratioは3.5〜5.4。SGの条件（M13・M20）は第1主成分が0.43〜0.48と偏る。
- 分散割合η²：Gの条件ではrace 0.24〜0.38、AP/PA 0.19〜0.30、sex 0.14〜0.18。SGの条件では年齢（10歳刻み）が0.36〜0.45で最大。
- 1属性差し替えの移動量（cの広がりとの比）：sex・race・ethnicity・AP/PAは0.7〜1.0、年齢+10歳は0.2〜0.3。
- PCA図では、カテゴリ属性の組ごとに年齢で伸びる曲線群が並ぶ。lastで形は変わるが、曲線の構造は残る。
- 8群重心の距離行列のseed間Spearman相関は0.36〜0.89。

## 解釈の候補

- best checkpointのadapterは、主に属性に依らない共通の補正をかけている。ΔWのエネルギーの75〜92%がsharedで、
  予測変化もほぼ一律のmargin低下と群内の小さな並べ替え。ResNetとの差の大半は「凍結backboneへの共通fine-tuning」に近い。
- 属性ごとに異なるΔWは学習が進むと育つ（last）。ただし同時に全体AUROCは親を下回る。valid AUROCで選ぶと、
  属性別の補正が育つ前のepochが選ばれる。
- encoderはDRO群の属性（race・年齢）だけでなく、sex・viewも同程度に区別している。年齢の寄与はG条件で小さく、
  年齢は連続値の1次元入力で、+10歳の移動量はカテゴリの差し替えより小さい。
- SG（subgroup class weight）は、年齢方向のencoder分散とfcの属性別ΔWを大きくする傾向がある。

## 未確認

- 3seedで、best epochが0の run を含む。条件間の差を方式の優劣として読まない。
- 中間epochのΔWの推移（best/last以外のcheckpointは保存されていない）。
- 相対Frobenius normは凍結BNを挟むため、出力変化の大きさそのものではない。出力側は予測probeで見た。
- Othersはencoder code 1で代表させた。code 4・5（件数少）の差し替えは未確認。
- 1-stage Spatial LoRA・HyperAdapt 2-stageでの同じ分解は未実施。
