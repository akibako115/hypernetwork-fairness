# 損失ベースのGroup DROから評価指標に合わせた学習へ

2026-10-05。既存の[候補一覧](fairness-baseline-methods.md)、現在の実装、race-known実験、原著を確認した検討メモ。次の目的関数を選ぶための提案であり、新手法の実装・学習・予測cacheの追加分析はまだ行っていない。

2026-10-06、検証する主候補をMinimax AUCとEO rate制約に絞った。比較計画（`analysis/archive/minimax_auc_vs_eo/README.md`。2026-10-17 に消し、snapshot ブランチに残る）に、原著と簡易版の区別、陽性率の影響を切り分ける対照、今日追加された機構診断を踏まえた評価方針を記録した。
同日の計画更新で、初回の群定義を年齢のみの2群（65歳未満／65歳以上）に変更した。
Minimaxは4群ペア、EOは4制約で検証し、結果を確認してから年齢×raceのintersectionへ展開する。
以下の8群に関する記述は既存結果と当初案であり、初回実験の正本は上記の比較計画とする。

**現行の主指標がworst-group AUROCなので、追加する学習手法の第一候補はMinimax AUCとする。** 共通閾値での感度・特異度の差を改善するなら、rate制約によるEqualized Odds（EO）学習を別の候補にする。単純な群バランスと群別閾値を対照に置けば、学習による改善が必要な部分を切り分けられる。

## 現在の結果は「DROで何も改善しない」とまでは示していない

元の候補一覧は手法調査が中心で、失敗原因の実証までは含んでいない。今回の判断には、ResNet基準の比較レポート（`analysis/archive/race_known_architecture_objective_comparison/reports/resnet_baselines_vs_groupdro_architectures.md`。2026-10-17 に消し、snapshot ブランチに残る）も使った。以下は同じtest 19,617件を使った3seed平均で、global class weight、best validation AUROC checkpointでの結果。

| 条件 | 全体AUROC | worst AUROC | TPR gap（Eopp1） | TNR gap（Eopp0） |
| --- | ---: | ---: | ---: | ---: |
| ResNet ERM | 0.8612 | 0.7299 | 0.4728 | 0.2096 |
| ResNet DRO | 0.8543 | 0.7251 | 0.4452 | 0.1951 |
| Spatial LoRA ERM | 0.8621 | 0.7388 | 0.4860 | 0.2237 |
| Spatial LoRA DRO・1-stage | 0.8552 | 0.7251 | 0.4639 | 0.2438 |
| HyperAdapt ERM | 0.8610 | 0.7371 | 0.4781 | 0.2209 |
| HyperAdapt DRO・1-stage | 0.8536 | 0.7188 | 0.4408 | 0.2170 |

DROでTPR gapが縮む条件はあるが、全体・最悪群のAUROCは3モデルとも低下している。これは現行設定での観察であり、有意差や「損失重み付けが原因」という因果関係は確定していない。群別class weightによる切片のずれとは別に、global weightでも改善しない問題が残る。

現在の群は年齢2区分×race 4区分の8群。初回メモの14群とは異なる。評価指標・群定義・閾値は[実験記録](../plans/20261002_experiment.md)を正本とする。

## 群の損失の大小と、改善したい指標は一致しない

[現行実装](../../projects/hypernet_e2e/objectives/group.py)は、class weight付きCEを群内で平均し、次の更新を行う。

\[
L_g=\mathbb E[w_Y\,\mathrm{CE}(f(X),Y)\mid G=g],\qquad
q_g\leftarrow\frac{q_g\exp(\eta L_g)}{\sum_hq_h\exp(\eta L_h)}.
\]

高いCEに重みを寄せる仕組みは、低いAUROCや低いTPRに重みを寄せる仕組みとは異なる。AUROCは陽性と陰性の順位、TPR・FPRは閾値を超える割合を測る。CEは予測確率の大きさにも依存する。例えば同じ順位を保ったままlogitを温度で変えると、AUROCを保ったままCEが変わる。

また、完全に較正されたモデルでも、重みなしCEの群内最小期待値は条件付きエントロピーに対応する。画像から判別しにくい群やラベルの曖昧な群には、他群より高い損失が残りうる。この議論は目的関数の性質からの推論であり、今回のデータでラベルノイズが原因だと確認したわけではない。

したがって、**CEによる再重み付けの変更だけでなく、順位またはエラー率を学習対象にする比較**が必要と考える。離散的なAUROC・TPRをそのままautogradに渡すことはできないので、各手法が定める微分可能な代理目的を使う。

## 最悪群のAUROCにはMinimax AUCが合う

[Yang et al., AAAI 2023](https://ojs.aaai.org/index.php/AAAI/article/view/26405)は、群内と群間の陽性・陰性ペアのAUCをともに扱い、その最小値を最大化する。通常のGroup DROからの変更点は、単一サンプルのCEからペアの順位損失へ移ること。最悪ケースを重視する考え方は保てる。

このプロジェクト向けの簡易案としては、群内の陽性・陰性ペアに対し、例えば次の順位損失を定義する。

\[
R_g^{\mathrm{rank}}=
\mathbb E_{X^+\sim P(X\mid Y=1,G=g),\,X^-\sim P(X\mid Y=0,G=g)}
\left[\log(1+\exp(-(s(X^+)-s(X^-))))\right].
\]

\[
\min_\theta\;L_{\mathrm{CE}}+\lambda\max_gR_g^{\mathrm{rank}},
\qquad s=z_1-z_0.
\]

これは本メモの実装提案であり、原著の損失・最適化アルゴリズムの再現ではない。「Minimax AUC原著」と「CE＋群内minimax ranking」を別条件として記録する。代理損失の最小化から、実測AUROCの改善が必ず得られるとは限らない。

群内順位だけなら群別のlogit切片を変えても目的値は変わらない。そのため、全体AUROCの維持にはCEを残す案や、異なる群の陽性・陰性を比較する項も検討する。原著を採用する場合は群間ペアも省かない。8群では陽性群×陰性群で64通りになる。

課題は少数群の陽性ペア不足。群×クラスの抽出や複数バッチにまたがる推定が必要なら、CEの学習分布への影響も明示する。胸部X線での有効性は今回未確認なので、まず既存ERMの特徴を凍結し、最終層で動作とvalidationの改善を調べる。合格した目的関数をSpatial LoRAへ移す。

全体AUCを目的に、AUCの公平性を制約にする別案もある。[Yao et al., AISTATS 2023](https://proceedings.mlr.press/v206/yao23b.html)は、この種の制約付きAUC最適化を扱う。原著の制約と今回の8群のworst AUROCは同一ではないため、そのままの対応は前提にしない。

## 感度・特異度の差にはrate制約を使う

共通閾値を使うとき、EOは群間のTPRとFPRをそろえる条件。現在の二値分類では `argmax(logits)` がスコア `s=z₁−z₀` の閾値0に対応する。次の制約は、この評価に直接対応するプロジェクト向けの提案になる。

\[
\min_\theta L_{\mathrm{CE}},\qquad
\max_g\mathrm{TPR}_g-\min_g\mathrm{TPR}_g\le\epsilon_+,\quad
\max_g\mathrm{FPR}_g-\min_g\mathrm{FPR}_g\le\epsilon_-.
\]

実装時には、全群ペアの差に上下両方向の制約を置くなどして表現できる。[Cotter et al., JMLR 2019](https://jmlr.org/papers/v20/18-616.html)のproxy-Lagrangianは、モデルの更新には微分可能な代理制約、制約側の更新には元のrateを使う枠組み。違反している制約に学習の圧力を配分するので、群CEの大小で配分するDROとは判断基準が異なる。

前のメモではEO marginと同系統として予備にしたが、今回は検討対象へ戻す。指定した閾値でのrate制約と、閾値を含まないmargin正則化は、評価指標との対応が違うため。

簡易試作では `σ((s−t)/τ)` を指示関数の代わりに使い、陽性・陰性で別々に平均する。ただし、この平均の差が小さくても実際のTPR・FPR gapが小さい保証はない。`τ` と制約許容値はvalidationで選び、hard rateも毎epoch記録する。この単純なsoft penaltyをproxy-Lagrangian原著と呼ばない。

EOだけを強めると、全群の性能を下げて差を消す解もありうる。全体AUROCとworst AUROC、worst TPR・TNRの下限を採用基準に加える。TPRを重視するEopp条件なら、FPR側の悪化も併記する。

## 既存候補は目的との対応で絞る

| 候補 | 今回の位置付け | 理由・確認点 |
| --- | --- | --- |
| 群バランスのsampling | 必須の単純対照 | 群内の陽性率を保ち、少数群の学習機会を増やす。適応重みが必要かを測る |
| 群別閾値 | 学習前の参照線 | 順位を変えずに、動作点の調整で改善する部分を測る |
| Minimax AUC | AUROC目的の主候補 | CEの代わりに陽性・陰性の順位を最悪ケースの対象にする |
| rate制約・proxy-Lagrangian | 共通閾値でのEO目的の主候補 | 違反しているTPR・FPR制約を対象にする |
| FairDRO | EO目的の先行手法（2026-10-07追加） | クラスごとの群重みを0-1誤りから閉形式で決め、負の重みも許す。群CEの陽性率混合と、CEと誤り率のずれの両方を避ける |
| worst-group EO margin | CXRでの比較候補 | 実装は比較的小さいが、EOの代理目的であり原著と群定義が異なる |
| Y条件付きMMD | 分布合わせの比較候補 | 特定閾値より強い条件を課す。全群の順位性能を下げないか確認する |
| Y条件付きadversary | MMDと別機構の予備 | 追加ネットワークとminimax学習の調整が必要 |
| FIS | 優先度を下げる | 損失分布に基づく再重み付けなので、目的指標との不一致が残る |
| FaMI・条件付けなしの独立性 | 対照に限定 | 群ごとの陽性率の違いと診断に必要な情報まで消すおそれがある |

群バランスを必須とする根拠は、[Zhang et al., CHIL 2022](https://proceedings.mlr.press/v174/zhang22a.html)のCXR比較。そこで最悪群性能を狙う手法は単純なdata balancingを上回らず、群間差を消す手法は全群の性能を落とす場合があった。ただし、今回の8群で同じ順位になるかは未検証。

群別閾値の参照は[Hardt et al., NeurIPS 2016](https://arxiv.org/abs/1610.02413)に基づく。TPRだけをそろえるEoppと、TPR・FPRをともにそろえるEOは分ける。厳密なEOは一群一閾値では実現できない場合があり、原著の後処理にはランダム化も含まれる。元スコアの群内AUROCは閾値を変えても改善しない。

Y条件付きの分布合わせでは、各クラスで `P(score | Y, G)` を合わせる。MMDでこれを近似する先行研究は[Pfohl et al.](https://arxiv.org/abs/2007.10306)を参照。同研究では性能低下も報告されている。特定閾値でのgapを縮める目的には、分布全体の一致は必要以上に強い可能性がある。

FISの[原著](https://arxiv.org/html/2310.02492v3)は、損失に基づくサンプル・群のscalingを使う。今回の「CEと公平性指標が一致しない」という仮説を直接検証する手法としては、順位目的とrate制約を先に比較するほうが適切と考える。

## EO marginの原著と初回メモの差に注意する

Nikhil Cherian Kurian et al. *Worst-Group Equalized Odds Regularization for Multi-Attribute Fair Medical Image Classification.* 2026年5月19日公開のarXivプレプリント。[論文ページ](https://arxiv.org/abs/2605.19214)／[本文](https://arxiv.org/html/2605.19214v1)。

分類損失に、陽性側・陰性側のEOを狙う正則化項を加える。

\[
L=L_{\mathrm{BCE}}+\lambda_+L_{\mathrm{EO}+}+\lambda_-L_{\mathrm{EO}-}.
\]

陽性側では陽性の平均予測確率が最も低い群、陰性側では陰性の平均予測確率が最も高い群を選ぶ。選択した群のサンプルと、バッチ内の逆クラスのサンプルとのlogitの分離を、log-sum-expとhingeで促す。群CEの大小で重みを決めるGroup DROに対し、陽性・陰性それぞれの弱点を別に学習対象とする点が異なる。

著者らはMIMIC-CXRと眼科画像で、全体AUCへの影響を抑えながらEO・Eoppが改善したと報告している。胸部X線への近さと目的関数の導入しやすさから、最近のEO研究の中では先に検討する候補と考える。ただし、今回への適用では次を区別する。

- 原著はage、race、sexの属性値ごとの群から最悪群を選ぶ。交差8群から選ぶ版は適用上の変更であり、交差群での効果は別途評価する。
- 最悪群選択に使うクラス条件付き平均確率は、実際のTPR・FPRではない。marginもEOの成立を保証しない。
- 原著のEOddsは最悪・最良群のrate比から計算する。現行コードの `Eodds=(TPR gap+FPR gap)/2` と数値を直接比較しない。
- 原著はvalidationのYouden's Jで動作点を選ぶ。現行のargmax固定と区別する。EMAの追加も原著再現とは別条件にする。

実装上の推論：原著のlog-sum-expは標本数で正規化されていない。スコアが同じでもプールの件数で値が変わるため、batchサイズ・samplingを変えたときに `λ=0.5〜0.6` をそのまま流用しない。原著の再現と件数補正の変更を混ぜず、空セルによる項のskip率も記録する。

## カーネルEOの新しい研究は理論面の参照にする

Yijin Ni, Xiaoming Huo. *Kernel-based Equalized Odds: A Quantification of Accuracy-Fairness Trade-off in Fair Representation Learning.* NeurIPS 2025、査読済み。[論文ページ・PDF](https://papers.nips.cc/paper_files/paper/2025/hash/85a47d32fbf29bb8d42cdd980b719959-Abstract-Conference.html)。

カーネルを使ったEOの基準 `EO_k` を提案し、予測精度と公平性のトレードオフを分析する。EOに対応するクラス条件付き独立性に加え、条件付けなしの独立性や較正との関係も扱う。経験的な推定量と、その推定誤差に関する理論結果を示している。

著者らが明記しているとおり、中心は理論の構築である。今回のCXRモデルへ学習損失として導入した際の改善は未確認なので、実装候補のEO marginと区別し、条件付きの分布合わせを設計・評価する際の参照にする。

2026-10-05の追加検討では、Cotterらの2019年論文をrate制約の基礎、Kurianらの2026年論文を医用画像での実装候補、Ni・Huoの2025年論文をEOの理論面の参照として整理した。新しいEO研究を検討する場合も、EOのrate差を制約する方法と、margin・分布の代理目的を正則化する方法を区別する。

## 次の比較は目的関数を固定した小さな実験から始める

1. **既存ERMのvalidation予測で診断する。** 群別CE、陽性・陰性別CE、陽性率、AUROC、TPR・FPR、DROのqを照合する。共通閾値の曲線と群別閾値の参照も作り、順位の問題と動作点の問題を分ける。qと性能の相関だけで因果関係を判断しない。
2. **ResNetで目的関数を比較する。** ERM・現行DROに群バランス対照を加える。現行のAUROC主目的にはMinimax AUC、EOも研究目的にするならrate制約を加える。EO marginはCXR文献との比較用に続ける。全候補を最初から3アーキテクチャへ展開しない。
3. **有望な目的関数だけSpatial LoRAへ移す。** 同seedのERM checkpointを共有し、最終層のみの学習とadapterの学習を比較する。最終層で十分か、画像表現の変更が必要かを調べる。学習予算は対照間でそろえる。

比較では次を共通にする。

- split、交差8群、seed、初期化、class weightの規約をそろえる。CEの重みの正規化も確認し、CEと公平性項の相対尺度が手法間で意図せず変わらないようにする。
- 群バランスsamplingと群損失の等重み平均は、有限バッチでは同じ挙動にならない。群の出現頻度と空セル率を記録する。
- 条件付き公平性項にはCEのclass weightを掛けない。群×クラスで抽出するなら、条件付き統計の推定用なのか、分類損失の分布変更なのかを明示する。後者を避ける場合はimportance補正や別バッチを使う。
- EMAやcacheの統計値は勾配を持たない。最悪群の選択を安定させても、選択された群の現在のサンプルから勾配を得る必要がある。
- 全体AUROCの許容低下幅と、worst AUROC・worst TPR/TNRの採用基準を探索前に決める。validationで候補を選び、testは選定後の報告に使う。
- 現行best validation AUROC選択は、worst AUROCやEO目的とは一致しない。過去比較用の選択を残しつつ、目的に合わせたvalidation選択も別に報告する。
- 群別件数、陽性・陰性数、全群の指標を残す。3seed平均に加え、患者IDが利用できる場合は患者単位のbootstrapで小さい群の不確実性を見る。

現時点の提案は、**群バランス対照とMinimax AUCを最初のAUROC比較に加え、EOを主目的にする場合はrate制約も検証する**こと。どの方法もこのデータでの改善は未確認であり、gapの縮小だけで採用を決めない。
