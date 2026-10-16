# latent_subgroup_spaces

2026-10-13 開始。柱 B「hypernetwork で潜在サブグループを扱うのは妥当か、効果はあるか」
（[research_overview.md](../../docs/meetings/20261016/research_overview.md)）の「捕捉」を確かめる。
1 段・ERM で学習した Spatial LoRA について、潜在サブグループを探す空間と群の作り方を並べ、何が見つかり、何が見つからないかを見る。

## 問い

1. 当初案（metadata encoder の出力 c・生成 ΔW の空間のクラスタリング）で見つかる群は何か。属性の組み合わせの束ね直しか（推論 1）。
   距離は「性能の悪さ」ではなく「予測の仕方の違い」を表すか（推論 2）。
2. 画像側の空間（モデル自身の特徴、基盤モデル RAD-DINO）と、誤差の信号を使う監査器（multiaccuracy）は、当初案が拾えないものを拾うか。
3. クラスタより小さい部分集団の偏りは、どこまで見つけられるか。

## 設計

- 対象 run：CheXpert race-known、1 段・ERM の Spatial LoRA（stage3・stage4・fc）× seed 42・43・44（[runs.md](runs.md)）。
  加工も新しい学習もしない。
- 空間：属性、c、ΔW、モデル自身の特徴、RAD-DINO
- 群の作り方
  - クラスタリング：k-means、GEORGE、LHCF
  - 監査器：ridge、回帰木
- 群は val で作り、test で測る。
- 物差しは 2 つ使う。
  - しきい値 0.5 の FPR（`docs/research-premises.md` の主指標。CheXpert では見落とし）
  - 得点をそろえたうえでの陽性数の O/E（較正のずれ）。画像を見る空間では FPR が得点の関数になり、群探しの目的には使えないため
- 群の中身は、属性と、病種（CheXbert の impression ラベル）・デバイス・術後の所見・レポート種別（報告書の正規表現）で読む。

| file | 中身 |
| --- | --- |
| `extract.py` | run ごとに val・test の logit 差・head 直前の特徴・c・セル番号と、セルごとの ΔW の Gram を `cache/` に書く（GPU） |
| `subgroups.py` | 読み込み、空間、群の作り方、物差し（FPR・帰無分布・O/E・走査統計） |
| [01_initial_spaces.ipynb](01_initial_spaces.ipynb) | 当初案の 2 空間 |
| [02_space_comparison.ipynb](02_space_comparison.ipynb) | 5 空間 × 作り方の横並び（FPR）と、細かい群・k 近傍の走査 |
| [03_calibration_bias.ipynb](03_calibration_bias.ipynb) | 同じ比較を、得点をそろえた O/E で |
| [04_three_lenses.ipynb](04_three_lenses.ipynb) | 01〜03 を群の悪さの 3 つの見方（動作点・較正・判別）で並べ直し、報告用の図を作る（`figures/04_three_lenses/`） |

notebook は repo root ではなく package の中で実行する。CPU のスレッドを絞らないと GMM が極端に遅くなる。

```bash
cd analysis/latent_subgroup_spaces && OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 MKL_NUM_THREADS=8 \
  uv run jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.kernel_name=hypernet-fairness <name>.ipynb
```

## 結論（暫定、ユーザー確認前）

[当初案の記録](reports/initial_spaces.md)、[空間の比較の記録](reports/space_comparison.md)。

- **当初案は、属性の組み合わせを陽性率の違いで束ねた群を見つける。** c・ΔW のクラスタの最悪 FPR（K = 4 で 0.42〜0.47）は 8 群（0.36）より悪い。
  ただし次の 3 点から、見ているのは主に陽性率の違いによる動作点の差だと考える。
  - クラスタの FPR は陽性率と相関 0.9 で並ぶ。
  - モデル入力の属性の構成で、FPR の違いの 8 割以上が説明できる。
  - 得点をそろえると、ずれは O/E 0.88〜0.94 にとどまる。
- **画像を見る空間で FPR を目的に群を探すと、難しい画像の塊が自明に見つかる**（モデル自身の特徴で FPR 1.0）。画像側の空間では、得点をそろえたずれを探す必要がある。
- **得点をそろえたずれ（較正）が最も大きく、test でも再現したのは、RAD-DINO の上の残差の監査器（multiaccuracy 型）だった。ただし主指標の性能では悪い群ではない。**
  - 大きさ：O/E 0.71、約 2,300 枚、3 seed とも負。平均残差（multiaccuracy の定義どおりの差）で選ぶと −0.18 に縮む
  - 中身：気胸・胸骨切開・縦隔拡大に寄り、属性ではほとんど決まらない。当初案が原理的に拾えない種類のずれである
  - 性能：FPR は補集合より低い（gap −0.07）。増えているのは正常を所見ありと出す誤り（TPR の gap −0.19）。AUROC は −0.05
- **較正のずれと性能の gap は別物で、群をまたぐと逆向きに並ぶことが多い。** 確率を高めに出す群は得点の低い群で、FPR は低い。性能の gap を目的にした、得点で自明にならない発見の方法はまだ無い。
- **小さな部分集団は、val の大きさでは見つけきれない。** 群を細かくすると、val で見えたずれは test で大きく戻る。k 近傍の走査で有意になるのは 500 件以上の球だけだった。
  次の手は次の 2 つ。
  - 発見に使うデータを train の cross-fit で大きくする
  - 候補を事前の要因で絞って、多重比較を減らす

## 3 つの見方での整理（2026-10-15、[04 notebook](04_three_lenses.ipynb)）

群の悪さを、動作点（0.5 の FPR）・較正（O/E）・判別（群の AUROC − 補集合）の 3 つに分けて、同じ方法の行で並べた。
各見方の物差しで val で最悪の群を選び、test で測る（3 seed 平均）。

- 動作点：属性の空間の群（c・ΔW）は FPR 0.50 前後で、陽性率で並ぶ。画像の空間では 0.58〜1.00 で、得点の高い画像を集めただけになる
- 較正：帰無分布を外れるのは RAD-DINO の監査器（−0.34）と、特徴の監査器（−0.17）、c の k-means（−0.13、わずかに外れる）
- 判別：RAD-DINO の GMM は、val の AUROC 最小でも 3 seed ともクラスタ 9（胸骨切開・中心静脈カテーテルに富む AP 画像）を選び、
  test の AUROC の差は −0.11。較正はほぼ合っている（log O/E −0.02）。AP 画像どうしで比べても AUROC 0.75 対 0.85
- 判別の差は c・ΔW の k-means（−0.09・−0.08）でも帰無分布を外れる。帰無分布は同じ大きさの無作為な群で作っており、
  得点の幅が狭い群ほど AUROC が低く出る分を含まない。特徴の GMM の −0.30 はこの効果が大きいと考える（未検証）
