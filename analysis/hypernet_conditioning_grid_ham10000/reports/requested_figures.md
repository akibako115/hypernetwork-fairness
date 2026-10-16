# HAM10000 の ERM・GroupDRO を、4 群の見落としで比較する

**ERM で属性による条件付けを足しても、見落としの群間差は縮まない。** 1-stage の Spatial LoRA（S3–4＋fc）は ResNet より全体 AUROC が上がる（0.903→0.927）。一方で Eopp1（群間の TPR の差）は 0.196→0.259 と広がり、worst-TPR は 0.562→0.554 でほぼ変わらない。

**GroupDRO は、Spatial LoRA で Eopp1 を下げて worst-TPR を上げる向きに動く。ただし同じ seed どうしの差は seed 間で大きく揺れ、偽陽性の増加も伴う。** 3 seed の記述的な比較で、効果を確定したものではない。EO rate は回し直し中なので、この報告には含めていない。

- [再実行用 notebook](../requested_figures.ipynb)／[集計 module](../requested_analysis.py)
- 図：`figures/requested_figures/`（`index.html` に全図、`all_requested_figures.pdf` にまとめ）。図は再生成物なので Git で追跡しない
- 数値：[条件ごとの平均・SD](../results/requested_figures/test_mean_sd.csv)、[同 seed 差](../results/requested_figures/paired_deltas_mean_sd.csv)、[群別](../results/requested_figures/test_groups_mean_sd.csv)

## 何を比べたか

CheXpert の [requested_figures](../../hypernet_conditioning_grid/reports/requested_figures.md) と同じ図を、HAM10000 で作った。比べたのは 10 条件 × 3 seed（42・43・44）の 30 run で、モデルは ResNet-50、Spatial LoRA（S3–4＋fc）、Spatial LoRA（fc のみ）。それぞれに 1-stage と、同じ seed の ResNet ERM を凍結した 2-stage があり（ResNet は 1-stage のみ）、目的関数は ERM と GroupDRO の 2 つ。図 1〜4 は各 run の best val AUROC checkpoint で test を評価した値、図 6 は validation の全 30 epoch の推移。

CheXpert との違いは次の 3 点。

- **陽性は悪性**（mel / akiec）なので、害の大きい側は見落とし。最悪群は TPR が最小の群（worst-TPR）で、群間差は Eopp1 を主に見る。しきい値は全群共通の 0.5
- **群は sex × 年齢 65 歳の 4 群**。test の陽性は female × <65 が 35 件、female × ≥65 が 34 件、male × <65 が 59 件、male × ≥65 が 62 件。TPR が 1 件動くと 0.016〜0.029 変わる
- **test の 294 件は sex か age を欠く**ので群を決められない。全体指標も含めて、属性の揃った 1,217 件で計算した。全 1,511 件の AUROC は CSV の `auroc_all_rows` 列に残してある

gap と worst は seed ごとに 4 群から求めてから平均した。エラーバーは seed 間の標本 SD で、標本誤差は表さない。

## 図 1・2：ERM の条件付けでは、見落としの差は縮まない

| ERM 条件 | 全体 AUROC ↑ | AUC gap ↓ | Eopp1 ↓ | worst-AUC ↑ | worst-TPR ↑ | 全体 TPR | 全体 FPR |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| ResNet（親） | 0.903 | 0.060 | 0.196 | 0.859 | 0.562 ± 0.115 | 0.681 | 0.116 |
| Spatial S3–4＋fc・1-stage | 0.927 | 0.050 | 0.259 | 0.892 | 0.554 ± 0.034 | 0.677 | 0.066 |
| Spatial fc・1-stage | 0.925 | 0.044 | 0.228 | 0.890 | 0.629 ± 0.103 | 0.751 | 0.100 |
| Spatial S3–4＋fc・2-stage | 0.915 | 0.052 | 0.272 | 0.881 | 0.495 ± 0.016 | 0.660 | 0.073 |
| Spatial fc・2-stage | 0.910 | 0.060 | 0.209 | 0.867 | 0.508 ± 0.020 | 0.637 | 0.077 |

**観察事実**

- 条件付けたモデルはどれも、ResNet より全体 AUROC と worst-AUC が高く、全体 FPR が低い。
- Eopp1 はどの条件付けモデルも ResNet より大きい。worst-TPR が ResNet を平均で上回るのは fc のみ・1-stage だけで、その差（+0.067）は同 seed 差の SD（0.218）より小さい。
- TPR が最も低い群は、ERM の 5 条件すべてで female × <65（GroupDRO を含めると 10 条件中 9 条件。例外は fc のみ・1-stage の GroupDRO で、male × <65 の 0.684 が female × <65 の 0.705 を下回る）。
- 2-stage では全群で FPR が下がる（male × ≥65 で 0.239→0.169）のと同時に、female × <65 の TPR も 0.562→0.495 に下がる。

**解釈候補**：条件付けで動いたのは群間の差よりも全体の動作点で、偽陽性を減らす代わりに見落としを増やす側へ寄った。順位の性能（AUROC）は上がるが、固定しきい値 0.5 での見落としの差は縮まない。CheXpert で「AUROC 系と FPR 系の改善の向きがそろわない」と見たのと同じ形になっている。

## 図 3・4：GroupDRO は worst-TPR を上げる向きに動くが、揺れが大きく偽陽性も増える

同じモデル・同じ stage の ERM との、同じ seed どうしの差（平均 ± SD）。

| 条件 | Eopp1 | worst-TPR | 全体 AUROC | 全体 FPR |
| --- | ---: | ---: | ---: | ---: |
| ResNet・1-stage | +0.003 ± 0.043 | +0.024 ± 0.131 | +0.007 | −0.015 |
| Spatial S3–4＋fc・1-stage | −0.068 ± 0.096 | +0.043 ± 0.116 | −0.009 | +0.028 |
| Spatial fc・1-stage | −0.036 ± 0.061 | +0.033 ± 0.113 | −0.007 | +0.024 |
| Spatial S3–4＋fc・2-stage | −0.048 ± 0.060 | +0.048 ± 0.044 | −0.001 | +0.007 |
| Spatial fc・2-stage | −0.010 ± 0.016 | +0.010 ± 0.016 | 0.000 | −0.001 |

**観察事実**

- Spatial LoRA の 3 条件では、GroupDRO で Eopp1 が下がり worst-TPR が上がる向きにそろう。差の SD が平均より小さいのは、2-stage S3–4＋fc の worst-TPR だけ。
- 1-stage S3–4＋fc では、female × <65 の TPR が 0.562→0.600 に上がるのと同時に、female × ≥65 の FPR が 0.084→0.156 に上がる。Eopp1 の縮小は、最良群が下がったのではなく最悪群の TPR が上がったことによる。ただし偽陽性の増加を伴う。
- ResNet と 2-stage fc では、GroupDRO と ERM の差はほぼ無い。2-stage fc は seed 43・44 の test 指標が ERM とほぼ一致する（best epoch が ERM 3・4、GroupDRO 3・1 と早い）。

## 図 5：q は、見落としの多い群ではなく male × ≥65 へ寄る

事前に指定した 1 run（Spatial S3–4＋fc、1-stage、seed 42）。epoch 末の q は male × ≥65 が 0.263→0.342 と増え、test で TPR が最も低い female × <65 は 0.21 前後で、一様な 0.25 を下回ったまま。

q は train の損失に対する重みにあたる。**train の損失が大きい群と、固定しきい値で見落としが多い群とは一致していない。** GroupDRO が female × <65 の TPR を直接押し上げているとは、この図からは言えない。1 run だけの図なので、他の run に一般化しない。

## 図 6：validation の群間差は、epoch ごとに test の差より大きく揺れる

- 1-stage では、validation の Eopp1 が epoch ごとに 0.2〜0.35 の幅で揺れ、ERM と GroupDRO の帯は重なる。図 3 の 0.05 前後の差は、この揺れより小さい。
- 全体の TPR と FPR は、学習が進むにつれてどちらも下がる（動作点が特異度の側へ動く）。どの epoch を best に選ぶかで、見落としの率そのものが変わる。
- 2-stage の best epoch は 12 run 中 10 run が epoch 1〜4（残りは fc のみの seed 42 で、ERM 25・GroupDRO 27）。
- worst-TPR の全 epoch の推移は記録されていない（ログに群の最小 TPR が無く、途中の checkpoint も残していない）。図には best checkpoint の点だけを置き、線で結んでいない。

## 検算と、確認できていないこと

- 予測 cache は、checkpoint と split CSV の SHA256、画像の行順、target が一致することを確かめてから読んだ。run.json の best は val AUROC の最大値と一致し、30 epoch がそろい、q の和は 1。
- 学習ログから復元した global TPR/FPR は、整数件数から 0.01 件以内に収まった。
- **学習ログと cache の差は、CheXpert より大きい。** best epoch での最大の絶対差は、全体 AUROC 0.0106（2-stage で目立ち、ログのほうが低い）、群の AUROC 0.0045、Eopp1 0.056（val の陽性 33 件の群で 1〜2 件の判定が入れ替わった大きさ）。学習時は bf16-mixed、cache は float32 なので、精度の差が原因の候補。再推論での切り分けはしていない（[差分表](../results/requested_figures/validation_cache_log_errors.csv)）。図 1〜4 は cache、図 6 の線は学習ログに統一した。
- 属性で条件付けたモデルは、学習中に見ていない欠損用の入力を test の 294 件で受ける。全行の AUROC は条件付けモデルで下がる（例：Spatial S3–4＋fc・1-stage は 1,217 件で 0.927、全行で 0.910）。これは本文の比較から外した。

## 追加で確認するとよいこと

- EO rate の回し直しが終わったら、`OBJECTIVES` に足して同じ notebook を再実行する。
- GroupDRO の効果が seed の揺れを超えるかは、3 seed では判断できない。lesion 単位の bootstrap で同 seed 差の区間を見るか、seed を足す。
- female × <65 の見落としが多い理由（悪性の内訳 mel / akiec の比、病変部位）は、まだ見ていない。
