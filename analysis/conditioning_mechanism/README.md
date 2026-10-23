# conditioning_mechanism

柱 A の機構：属性による条件付けは、どういうときに全体性能を上げ、どう効いているか。
2026-10-17 に `dataset_gain_sources`（2026-10-12 開始）と `attribute_interaction`（2026-10-15 開始）をまとめた。

## 現状（2026-10-17）

- 2 つの分析で 1 つの主張になる。条件付けが効くのは、属性が画像のスコアに無いラベルの情報を運ぶときだけで（1 節）、
  効き方はほぼスコアの切片をずらすこと（事前確率の注入）である（2 節）。
- 入力の run はすべて旧前処理で学習している。CheXpert・HAM10000 は前処理を MEDFAIR に、FairVLMed は学習の回転を
  ±10° から FairMedFM と同じ ±15° に変えた（2026-10-17）。CheXpert・HAM10000 は、格子
  （[hypernet_conditioning_grid](../hypernet_conditioning_grid/README.md)）の 1 段の ERM を回し直したら、その run を推論し直して
  作り直す。格子に無い run として、条件を定数にした Spatial LoRA（HAM10000 × 3 seed）と、FairVLMed の ResNet-50・
  Spatial LoRA・定数の Spatial LoRA（各 3 seed）を回す。
- 作り直しでは HyperAdapt と random8 を外す（格子でも HyperAdapt を外したため）。

| file | 中身 | 出力 |
| --- | --- | --- |
| `data.py`・`metrics.py` | 3 データの予測 cache と split の読み方、AUROC の分解と bootstrap | — |
| `gain_sources.py`・`gain_figure.py` | 1 節の物差し | `results/*.csv`、`figures/gain_vs_attribute_ceiling.png` |
| `condition_use.py` | 属性を行の間で入れ替えたときの低下（GPU） | `results/<dataset>_condition_use.csv` |
| `constant_condition.py` | HAM10000 で test 時だけ属性を固定・欠損を埋めた推論（GPU） | `cache/<run-id>_test_cf-<条件>.npz` |
| `interaction_counterfactual.py` | 評価セル内で属性をそろえ、入力の年齢を動かす推論（GPU） | `cache/<run-id>_test_interaction.npz` |
| `interaction_summary.py`・`interaction_figure.py` | 2 節の集計と図 | `results/{age_curve,contrasts,rank_change}.csv`、`figures/{age_curve,contrasts}.png` |

# 1. 条件付けで全体性能が上がるのはいつか（旧 dataset_gain_sources、2026-10-12）
2026-10-12 開始。属性条件付きモデル（Spatial LoRA・HyperAdapt）が ResNet-50 より上がるのは HAM10000 だけで、
FairVLMed・CheXpert では変わらない。その理由を、3 データセットで同じ物差しを当てて切り分ける。
hypernetwork が何をしているか（何があれば効き、何が無いと効かないか）を読むための分析。

- 学習・推論はしない。`cache/` の予測 cache を読む（対象 run は [runs.md](runs.md)）。
  例外は条件の入れ替え（[condition_use.py](condition_use.py)）と、HAM の属性の固定（[constant_condition.py](constant_condition.py)）で、これらは推論する。
- 主比較は test。HAM は施設が既知で sex・age が揃う 1,185 行（external は年齢欠損の符号化で崩れるので外す。
  [data_pipeline/README.md](../../data_pipeline/README.md) の HAM10000 の節）。区間は患者（病変）単位の対応のある bootstrap。
- 実行：`uv run python -m analysis.conditioning_mechanism.gain_sources` → `results/*.csv`、
  `uv run python -m analysis.conditioning_mechanism.gain_figure` → `figures/gain_vs_attribute_ceiling.png`

### 物差し

| 量 | 何を見るか | 出力 |
| --- | --- | --- |
| const を挟んだ分解 | 条件を定数にした Spatial LoRA を挟み、構成の効果（const − ResNet）と属性の効果（meta − const）を分ける | `gains.csv` |
| 後付けの上がり幅 | ResNet の出力に属性を線形に足したときの AUROC の上がり幅。**属性が画像のスコアに無い、ラベルの情報を運ぶ度合い**。条件付けで得られる幅の目安になる | `attribute_information.csv`・`per_attribute.csv` |
| 属性だけの予測力 | 属性だけのロジスティック回帰（train で学習）の AUROC | `attribute_information.csv` |
| 読み取りやすさ | ResNet の特徴から属性を当てる精度（val で学習・test で評価） | `readability.csv` |
| セル内・セル間 | AUROC を、属性の事前確率（属性だけのモデルの十分位）が同じ組と違う組に分ける。後付けの事前確率はセル間の組しか動かさない | `within_between.csv` |
| 条件の使用量 | test の属性を行の間で入れ替えたときの AUROC の低下 | `results/<dataset>_condition_use.csv` |
| HAM の機構 | dx の組ごとの AUROC、test 時だけ属性を固定した出力（`meta_fixed`）、良性内のスコアと年齢の相関 | `ham_pairs.csv`・`ham_age_coupling.csv` |

### 結果（2026-10-12）

![](figures/gain_vs_attribute_ceiling.png)

#### 3 データセットの比較（test、seed 平均）

| | HAM10000 | FairVLMed | CheXpert |
| --- | --- | --- | --- |
| 属性条件付き Spatial LoRA − ResNet | **+0.025**［+0.011, +0.040］ | −0.004［−0.012, +0.006］ | +0.001［−0.001, +0.003］ |
| HyperAdapt − ResNet | **+0.021**［+0.005, +0.037］ | +0.004［−0.004, +0.012］ | −0.000［−0.002, +0.002］ |
| 構成の効果：const − ResNet | +0.007［−0.005, +0.019］ | −0.008［−0.015, +0.000］ | （const の run なし） |
| 属性の効果：meta − const | **+0.018**［+0.003, +0.032］ | +0.004［−0.006, +0.014］ | — |
| 後付けの上がり幅（ResNet、val で学習） | +0.009 | +0.006 | +0.001 |
| 後付けの上がり幅（ResNet、test 上の cross-fit） | +0.013 | +0.008 | −0.000 |
| 後付けの上がり幅（属性条件付きモデル、cross-fit） | +0.001 | +0.005 | +0.000 |
| 属性だけの AUROC（val / test） | 0.749 / 0.800 | 0.689 / 0.676 | 0.699 / 0.704 |
| 主に効く属性（1 つずつ後付け） | 年齢 +0.011 | language +0.005・race +0.004 | すべて ±0.000 |
| 属性入れ替えでの AUROC の低下 | 0.04〜0.11 | 0.005〜0.02 | 0.004〜0.006 |
| test の標本誤差（SE、meta − ResNet） | 0.007 | 0.005 | 0.001 |
| val での meta − ResNet | +0.005 | −0.001 | +0.002 |

#### 読み

1. **条件付けの効果は「属性が画像のスコアに無いラベルの情報を運ぶか」で上限が決まる。** 属性だけの予測力は
   3 データセットとも 0.68〜0.80 あるが、ResNet の出力に後から足したときの上がり幅は HAM +0.009〜0.013、
   FairVLMed +0.006〜0.008、CheXpert 0.000 と分かれ、条件付きモデルの上がり方と同じ順に並ぶ。
   CheXpert では年齢・AP/PA などの属性が No Finding とよく相関するが（属性だけで 0.70）、その情報は画像の
   スコアにすでに入っている（AP/PA・正面側面は特徴から 0.97・1.00 で読める）。モデルは条件を使ってはいる
   （入れ替えで −0.004〜0.006）ものの、画像と同じ情報なので上がりようがない。標本誤差 0.001 で、これは検出力不足ではない。
2. **FairVLMed は、属性が運ぶ情報が小さく、Spatial LoRA の構成そのものの損と打ち消し合う。** 後付けの上限 +0.006 は
   test の SE 0.005 と同程度で、仮に全部取り込めても検出は難しい。meta − const は +0.004 で上限と整合する一方、
   const − ResNet が −0.008 なので、ResNet との差は 0 前後になる。条件付きモデルに後から属性を足すとまだ +0.005 上がり、
   HAM（+0.001）と違って属性の情報を取り込み切れてもいない。
3. **HAM の +0.025 は、2 つの機構に分かれる。**
   - **(a) test 時に渡す属性（主に年齢）の情報：黒色腫 vs 母斑。** 母斑は若年に多い（平均 45 歳 vs 黒色腫 61 歳）。
     この組の改善は、test 時に属性を固定すると消える（meta − meta_fixed +0.017［+0.008, +0.025］、meta_fixed − const −0.007）。
     const に後から属性を足すと 0.913 → 0.926 で、meta 0.931 の差 0.018 の約 7 割を再現する。
     fc だけの条件付けでも +0.015 が出る。この部分は「属性による事前確率の注入」で、深い層の条件付けは要らない。
   - **(b) 属性を入れて学習したことによる、画像側の判定の改善：黒色腫 vs bkl・bcc。** test 時に属性を固定しても
     const より良い（bkl +0.041［+0.010, +0.071］、bcc +0.036［+0.004, +0.072］）。
     事前確率が同じセルの中の組でも改善が出る（セル内の一致率は const 0.883 → meta 0.905、セル間は 0.916 → 0.933）。
     良性の中でスコアと年齢の順位相関は ResNet 0.15・const 0.13 に対し、meta_fixed 0.04・hyper_fixed 0.03 まで下がる。
     年齢を条件が受け持つことで、画像側が「年齢らしさ」を悪性の代わりの手がかりとして学ばなくなった、という
     **近道の肩代わり（explaining away）** と整合する。bkl・bcc は黒色腫と同じく高齢に多く、年齢の代わりの手がかりは
     この組を分けるのに役立たない。
4. **(b) は主に test に現れ、val ではほとんど見えない。** ResNet は val → test で bkl 0.829 → 0.791、bcc 0.880 → 0.798 と
   落ちるが、meta は 0.844 → 0.856、0.897 → 0.869 で保つ。施設の構成は val と test でほぼ同じで、施設内で比べても
   同じ形になる（vidir_modern の bcc：ResNet 0.775、meta_fixed 0.892）ので、施設の混ざり方の違いでは説明できない。
   属性だけの AUROC も val 0.749 → test 0.800 と上がり、ISIC 2018 の公式 test は train/val と分布がずれている。
   HAM の上がり幅は「全般的な性能向上」というより、**分布のずれへの頑健さ**として現れている。
5. **構成の効果（const − ResNet）はどのデータでも小さい**（HAM +0.007、FairVLMed −0.008、いずれも区間が 0 付近）。
   ランダム 8 区分の条件も const と変わらない（HAM +0.002、FairVLMed +0.004）。adapter を足したことや、条件の種類
   の数が増えたことで上がるのではない。

#### hypernetwork の作用としてのまとめ

- 条件付けが効くのは、**条件が画像に無いラベルの情報を運ぶとき**に限られ、その幅は「出力に属性を後から足す」こと
  でおおむね上限が分かる。CheXpert はこの上限が 0、FairVLMed は標本誤差と同程度だった。
- 効く場合の大半は、事前確率を入れる浅い作用（fc だけの条件付けでほぼ再現する）。
- それに加えて、学習時に条件が近道（ここでは年齢）を受け持つことで画像側の特徴が変わる作用が HAM で見えた。
  これは後付けでは得られない、条件付きで学習したことに固有の効果で、分布のずれのある test でだけ大きく現れた。

### 限界

- データセットは 3 つで、データの量（HAM 8.5 千枚、FairVLMed 7 千枚、CheXpert 15.8 万枚）・タスク・属性の種類が
  同時に違う。FairVLMed は HAM と同程度の量なのに上がらないので、量だけでは説明できないが、(b) の作用が小さい
  データでだけ出るかどうかは、CheXpert を間引いた学習でないと確かめられない。
- 後付けの上がり幅は点推定で、区間を付けていない。val で学習する版は val が checkpoint 選択に使われた split である点に注意。
- (b) の `meta_fixed` は、学習時に見た属性の組（典型値）を全行に入れた出力で、属性を「取り除いた」ものではない。
  典型値の選び方で値が少し動く（`constant_face` でも同じ向き。[constant_condition.py](constant_condition.py)）。
- 年齢の肩代わりは相関の証拠で、因果の確認には年齢だけで条件付ける・年齢だけ外して条件付ける学習が要る。
- 特徴からの読み取りやすさは、No Finding などのタスクで学習した ResNet の特徴を probe したもので、画像に含まれる
  属性の情報の下限にすぎない（CheXpert の年齢 R² 0.32 は、年齢が画像から読めないことを意味しない）。

### 次に確かめるなら

1. HAM を年齢だけで条件付ける学習と、年齢だけ外す学習（(a)・(b) が年齢によるかの因果の確認）。
2. CheXpert の train を 8 千枚に間引いて、ResNet・const・meta を学習する（(b) がデータの量に依存するか）。
3. 属性（臨床情報）が画像に無い情報を多く運ぶデータでの再現（例：PAD-UFES-20 のような臨床メタデータの豊富な
   皮膚画像）。上の 1. の読みが正しければ、後付けの上限が大きいほど条件付けの上がり幅も大きくなるはず。

# 2. 属性は切片をずらすだけか、画像の読み方も変えるか（旧 attribute_interaction、2026-10-15）
2026-10-15 開始。柱 A（hypernetwork による条件付けは、いつ効いて、いつ効かないか）の機構の確認。

### 問い

属性条件付きモデルは、属性を「スコアの切片をずらす」（陽性率の注入）だけに使っているのか、
「属性によって画像の読み方を変える」（属性と画像特徴の交互作用。例：80 歳ならこの程度の変化は正常と読む）
ことにも使っているのか。後者だけが、群別しきい値では代わりの利かない効果（群内 AUROC の改善）を生む。

既存の属性入れ替え推論（1 節の `condition_use.py`）は、モデルが属性を使っていることは
示したが、行の間で属性を混ぜるので、切片だけの使い方でも AUROC が下がり、両者を区別しない。

### 方法

学習済みの ERM の run（[runs.md](runs.md)、修正の影響を受けない）を、test で推論し直す（[interaction_counterfactual.py](interaction_counterfactual.py)）。

- 評価セル（HAM10000 は sex × 65 歳の 4 群、CheXpert は race × 65 歳の 8 群）の中では、全員に**同じ属性**を入れる。
  評価セルを決めない属性（HAM の localization、CheXpert の sex・ethnicity・正面側面・AP/PA）は train の最頻値に固定する。
- そのうえで、入力の年齢を 25〜85 歳で動かす。race（CheXpert）・sex（HAM）も差し替える。
- 属性が切片（と傾き）だけに効くなら、セル内のスコアの並びは変わらず、セル内 AUROC は入力によらず一定になる。

指標（[interaction_summary.py](interaction_summary.py)、図は [interaction_figure.py](interaction_figure.py)）:

- セル内スコアの順位相関（入力 45 歳と 75 歳）。切片だけなら 1。
- セル内 AUROC の差：セルの年齢群の代表値（45・75 歳）を入れたとき − 反対の年齢群の代表値。sex・race も同様に、自分の値 − 他の値の平均。
- 区間は患者（皮膚は病変）単位の cluster bootstrap 95%（1,000 回）。**test の標本の揺れだけ**を表す。seed は 3 本の平均で、学習の揺れは含まない。

### 結果（2026-10-15）

![](figures/age_curve.png)

1. **画像の並びは、入力属性でほとんど変わらない。** セル内の順位相関（45 歳 vs 75 歳）は、CheXpert 0.987〜0.999、
   HAM10000 0.982〜0.998。fc だけの条件付けは 0.998 で、ほぼ切片だけの使い方になっている。
2. **正しい年齢・属性を入れても、セル内 AUROC は上がらない。** 全セル平均の「自分の年齢群 − 反対の年齢群」は
   CheXpert +0.0005〜+0.0007、HAM10000 +0.0017〜+0.0028 で、区間はどれも 0 を含む。sex・race の差し替えも
   −0.003〜+0.0002 で 0 を含む。セル単位で 0 を含まないのは CheXpert の White <65（+0.001〜+0.002）と
   HyperAdapt の Others <65 の race（+0.0004）だけで、大きさは無視できる。
3. **HAM10000 で見える年齢曲線は「正しい年齢で読む」形ではない。** 入力年齢を 25〜35 歳にすると、
   セルの実年齢によらず全セルでセル内 AUROC が下がる（Spatial LoRA で最大 −0.03）。45 歳以上ではほぼ平ら。
   65 歳未満のセルでも若い年齢を入れたほうが悪いので、年齢に合わせた読み替えではなく、
   若い年齢の入力でスコアの並びが崩れる一般的な作用と見るのが自然。
4. **属性をそのまま入れたときのセル内 AUROC は、一定の属性より高い**（HAM10000 の 65 歳未満で +0.016〜+0.035）。
   これはセルの中での年齢（0〜64 歳）と部位のばらつきが、行ごとに違う事前確率として効いた分で、切片の作用である。

#### 読み

ERM で学習した属性条件付きモデル（Spatial LoRA・HyperAdapt・fc のみ）は、CheXpert と HAM10000 のどちらでも、
属性をほぼ切片（事前確率の注入）として使っている。「属性によって画像の読み方を変える」作用は、測れる大きさでは
起きていない。

### 限界

- ERM の run だけ。公平性損失で学習したモデルは見ていない。
- 評価セルを決めない属性は最頻値に固定した。HAM の localization・CheXpert の AP/PA との組み合わせで読み方が
  変わる効果は、この設計では見えない。
- 交互作用が「データに無い」のか「ERM が学ばない」のかは、この結果だけでは分けられない。分けるには、固定特徴量で
  特徴 × 年齢の交互作用項が効くかを見る（データ側）、または交互作用を植え込んだデータで回収できるかを見る。
