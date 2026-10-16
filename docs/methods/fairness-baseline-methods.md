# 公平性手法の比較候補

demographic 群（age × race）の GroupDRO 実験に、比較対象として加える公平性手法の候補と、その選定理由をまとめる。
2026-09-29〜30 の文献調査とコード調査に基づく。どの手法を実際に回すかの決定と run の記録は、各分析 package を正本とする。

2026-10-05追記：現在のrace-known 8群の結果を踏まえた優先候補と、原著との対応で注意が必要な点は、[損失ベースのGroup DROから評価指標に合わせた学習へ](fairness-objective-alternatives.md)を参照。以下の候補一覧は初回調査の記録として残す。

同日の追加調査では、Kurian et al.（2026年5月公開のarXivプレプリント）のEO marginを医用画像での実装候補として詳述し、Ni & Huo（NeurIPS 2025、査読済み）のKernel-based Equalized Oddsを理論面の参照として追加した。目的関数・Group DROとの違い・今回への適用上の注意は上記メモに記録した。

同日、先行研究をクラスの扱い（A・B・C）で分類し直し、原典の論文と公式コードで確認した。[クラスの扱いによる先行研究の分類](#2026-10-05追記クラスの扱いによる先行研究の分類)を参照。

## 選定の基準

- 仕組みの系統が重ならないこと（損失の再重み付け、敵対的学習、分布合わせ、margin、独立性、後処理）
- 胸部 X 線での使用例があるか、広く引用された出典があること
- 多数の交差群（調査時点では 14 群）にそのまま適用できること
- 論文の記述から再実装できること。公開実装がないことは除外の理由にしない

## 比較手法セット（提案）

| 系統 | 手法 | 出典 | 狙う基準 | クラス条件付け |
| --- | --- | --- | --- | --- |
| 単純な基準 | 群バランスの resampling（**属性単位**） | Zhang et al. 2022、Idrissi et al. 2022 | 最悪群 | なし（群内の陽性率は保たれる） |
| 医用画像の標準ベースライン | FIS | Luo et al., FairVision | 群間の AUC の均等化 | なし |
| 最悪群の損失（実施済み） | GroupDRO（原著どおり） | Sagawa et al., ICLR 2020 | 最悪群 | なし |
| Eodds：敵対的学習 | Y で条件付けた adversary | Zhang, Lemoine & Mitchell, AIES 2018 | Eodds | あり |
| Eodds：分布合わせ | MMDMatch（Y で条件付け） | Pfohl et al.（Zhang et al. 2022 で CXR に使用） | Eodds（閾値によらない） | あり |
| Eodds：最悪群の margin | worst-group Eodds 正則化 | Kurian et al., arXiv 2026 | Eodds（TPR 側と FPR 側を別の項で） | あり |
| Eodds：classwise DRO（2026-10-07追加） | FairDRO | Jung et al., ICLR 2023 | Eodds（DCA） | あり（クラスごとの群重み） |
| 独立性（対照） | FaMI（原著どおり、条件付けなし） | Sadri et al., MICCAI 2025 | 独立性（Demographic Parity 型） | なし |
| 参照線（学習なし） | 群別閾値の後処理 | Hardt et al., NeurIPS 2016 | Eodds / Eopp | — |

### 各手法の要点

- **群バランスの resampling**
  - CXR のベンチマークで最も強い単純な基準として報告されている（Zhang et al. 2022、MEDFAIR、Yang et al. 2024）。
  - 抽出は群単位にする。「群 × クラス」のセル単位で均等に抽出すると、群ごとに logit の切片がずれて全体 AUROC が下がる。
- **FIS**
  - サンプル単位の重みと群単位の重みを `c = 0.5` で混ぜ、損失に掛ける。群単位の重みは、群の損失分布と全体の損失分布の Sinkhorn 距離から決める。
  - 群の重みの正規化が、論文では群単位、公開実装ではバッチ内のサンプル単位になっている。論文の式に従う。
- **GroupDRO**
  - 「実施済み」の run には、群ごとの inverse class weight（陽性の重みが群によって 3.1〜15.2）で学習したものがある。この設定は群ごとの切片のずれを起こすので、全体共通の class weight の run と区別して扱う。
- **Y で条件付けた adversary**
  - adversary に (logit, y) を入れて群を予測させ、勾配を反転する。出力を群の数にすれば多群でそのまま動く。
  - 大きい群が adversary の学習を支配しないよう、adversary の損失を群ごとに正規化する（LAFTR に倣う）。
- **MMDMatch**
  - y = 0 と y = 1 のそれぞれで、各群の予測スコアの分布を全体の分布に MMD で合わせる。
  - 群 × クラスのセルが小さくなるので、群単位の sampler と併用する。
- **worst-group Eodds 正則化（Kurian et al.）**
  - ミニバッチの陽性で平均予測確率が最も低い群を選び、その群の陽性と全群の陰性との logit の margin に hinge をかける（log-sum-exp で滑らかにする）。陰性側も同様。
  - 損失は `BCE + λ⁺·L_EO⁺ + λ⁻·L_EO⁻`、論文では λ = 0.5〜0.6。
  - MIMIC-CXR で GroupDRO・JTT・FairBatch・敵対的学習・FaMI より joint Eodds が小さかった（GroupDRO の 0.163 に対し 0.127）。
  - 群の数が多いと最悪群の選択がノイズに振り回されるので、群ごとの統計量を EMA で平滑化する。
- **FaMI**
  - `CE + μ·I(u; v)` の勾配を score function difference で推定する。μ = 0.5。推定の詳細は引用先（Babaie-Zadeh & Jutten, 2004/2005）に委ねられている。
  - 正解クラスで条件付けないので、群ごとの陽性率の差まで消す方向に働く。この弊害を示す対照として原著どおりに入れる。
- **群別閾値の後処理**
  - 学習で得た Eopp の縮小が「閾値を群ごとに変えれば済む程度か」を測る物差しにする。既存の予測 cache だけで計算できる。

## 外した手法と理由

| 手法 | 理由 |
| --- | --- |
| FairDi（Masroor et al. 2024） | 群の数だけ教師モデルが必要で、多数の交差群には向かない |
| LAFTR、CFair | 二値属性が前提 |
| Agarwal et al. の reductions | 分類器の再学習を繰り返すため、ResNet では現実的でない |
| LHCF（Masroor et al. 2026） | 人口統計の群を使わない枠組みで、今回の問いから外れる |
| Cotter et al. の proxy-Lagrangian | Kurian et al. と同じ rate 制約の系統なので予備に回す |
| FairCLIP-OT | クラスで条件付けない分布合わせで、MMDMatch がその条件付き版にあたる |

## 別の軸として検討する対照

- **DFR、または最終層だけを GroupDRO で学習し直す条件**
  - 「凍結した ERM の特徴の上で最終層だけを学習し直せば足りる」という先行研究（Kirichenko et al. 2023）がある。第 2 段で Spatial LoRA を学習する意味を示すために必要。
  - 関連研究として、FairTune（Dutt et al. 2024、CheXpert を含む）と Mao et al. 2023（公平性の制約を E2E で課すとその基準に過学習する）がある。

## 実装の約束事（提案）

- 原著どおりに実装し、手法ごとの工夫はしない。論文と公開実装が食い違う場合は論文の式に従い、その旨を記録する。
- class weight は全群共通の inverse weighting を、分類損失の項にだけ掛ける。群ごとの class weight とセル単位のバランスは使わない。
- 多数の群で動かすための実装上の選択（群ごとの統計量の EMA、群単位の sampler、adversary の損失の群ごとの正規化）は全手法でそろえる。

## 2026-10-05追記：クラスの扱いによる先行研究の分類

先行研究の公平性手法を「正解クラス Y を目的関数のどこに入れるか」で分類し、原典の論文と公式コードで確認した。コードの行番号は確認時のコミットに対応する（MEDFAIR `2fae424`、FairVision `f16fa27`、FairSeg `b043bda`、FairCLIP `91b05b6`、shortcut-ood-fairness `970ddb5`、CXR_Fairness `c2a0e88`、ICLR2020-CFair `4f03012`、group_DRO `cbbc1c5`、MMPF `9ed8eac`、BlindParetoFairness `67a9e16`、fairbatch `4f65276`）。

- **A：Y を入れない**。群（属性・クラスタ・client）の損失や誤り率を、陽性と陰性を混ぜたまま比べる。
- **B：Y を群の定義に入れる**。群を「属性 × ラベル」のセルにする。群別の class weight と同じ働きになり、各群の学習時の陽性率がそろう。
- **C：Y を公平性項の条件に入れる**。公平性の比較を y ごとに行う。陽性率に依存しなくなるのは公平性項だけで、分類損失側は別に扱う必要がある。

### A の帰結：陽性率が群のリスクに入る

群平均の損失は、陽性と陰性の損失を群の陽性率 π_g で混ぜたものになる。

R_g = π_g · R_g⁺ + (1 − π_g) · R_g⁻

- クラスごとの損失 R_g⁺ と R_g⁻ が全群で同じでも、π_g が違えば R_g は違う。陽性が少数派で R⁺ > R⁻ のとき、陽性率の高い群ほどリスクが高く見える。
- 切片を全群で共有するモデルでは、π_g が全体の陽性率から離れた群ほど較正がずれ、R_g⁺ と R_g⁻ 自体も変わる。陽性率の差は、混合の重みと較正の両方を通ってリスクに入る。
- 誤り率でも同じ形になる（Err_g = π_g · FNR_g + (1 − π_g) · FPR_g）。
- 損失分布を比べる FIS では、群の損失分布が陽性と陰性の分布の π_g による混合になる。全体との距離は |π_g − π| で大きくなる。向きは「陽性率が高いほど」ではなく「全体の陽性率から離れるほど」。
- 群のリスクが高く見えた群を上げ重みすると、主にその群の切片が動く。順位の性能（群内 AUROC）が改善するとは限らない。

### 確認済みの分類

**A：群は属性・クラスタ・client だけ**

| 手法 | 群 | 重み・制約に使う量 | class weight | 根拠と注記 |
| --- | --- | --- | --- | --- |
| MEDFAIR GroupDRO | 属性 | 群平均 BCE、step size 0.01 固定 | なし | `models/GroupDRO/utils.py:14,65,78`、`GroupDRO.py:25,41`。汎化補正は `generalization_adjustment="0"` で無効（`GroupDRO.py:27`、`utils.py:61-62`）。論文 App. B.2.2 は η を探索したと書くが、sweep の `groupdro_alpha`・`groupdro_gamma` は損失に効かない。`WeightedRandomSampler` を併用し、単位の既定が `parse_args.py:88`（group）と `sweep_batch.py:19`（class）で食い違う |
| MEDFAIR resampling（本表） | 属性 | 群均等の抽出（損失は使わない） | なし | 論文 Sec. 4 "upsampled the minority groups"。セル単位（`balanced`、`BaseDataset.py:118-136`）は App. E の分析のみ。損失を使わないので、下の「難しさ」の問題は起きない |
| Zhang et al. 2022 GroupDRO | 属性 | 群平均 CE | なし | `cxr_fairness/algorithms.py:72,83`。同じ論文の MMDMatch・Adversarial・FairALM は C |
| FIS（FairVision） | 属性 | 群の BCE 損失分布と比較相手との Sinkhorn 距離、c = 0.5 | なし | 論文 Eq. (3)–(5)、Sec. 5。比較相手は論文では学習バッチ、コードでは群ごとの抽出サンプルの連結（`src/modules.py:760-765`）。実装は論文の式に合わせた（[記録](../reviews/20261015_lhcf_paper_alignment.md)） |
| LHCF（Masroor et al. 2026） | 画像埋め込みの GMM クラスタ（属性ラベルなし） | クラスタ平均損失の max、または max − min（Eq. 3） | 未確認（コード非公開） | Table 1 の悪性率 0.00〜66.39% は隠れクラスタの値で、人口統計の群の値ではない。FIS・FEBS などの属性の入る場所に群の ID を入れる枠組み（各手法への組み込み方・λ・ℓ_wst と ℓ_gap のどちらを使ったかは未記載）。FaMI の文献 [34] は FairDomain を指すが、手法は Sadri et al. |
| Minimax Pareto Fairness（Martinez et al. 2020） | 属性（定式化） | 群リスク r_a(h) の minimax | なし | §3。実験では MIMIC-III で転帰を群に入れ（B 相当）、HAM10000 では群 = 診断クラス（§6.2、§6.3） |
| AFL（Mohri et al. 2019）/ q-FFL（Li et al. 2020） | client・domain | client 内の平均損失の minimax / q 乗重み | なし | AFL §3 Eq. (1)、q-FFL §3.2 Eq. (2)。Fashion-MNIST 実験は domain = クラス |
| FEBS（FairSeg） | 属性 | 背景を含むクラス別 Dice の和（Eq. 3） | なし | `SAMed/utils.py:58-78`。分類の陽性率ではなく、セグメンテーションでの類似構造として参考扱い |
| Zafar et al. 2017 OMR、accuracy parity（Zhao & Gordon） | 属性 | 群の誤分類率の一致 | なし | Zafar §2 Eq. (3)、Zhao & Gordon（JMLR 版）§2.2 Definition 3 |

**群を使わず、サンプルの損失で重みを決めるもの（A の個体版）**

陽性が少数派だと陽性サンプルの損失が高く、陽性が上げ重みされる。その結果、陽性率の高い群ほど重みを多く受ける。

| 手法 | 重みを決める量 | クラス不均衡への言及 |
| --- | --- | --- |
| χ² DRO（Hashimoto et al. 2018） | 損失が η を超えるサンプルを上げ重み（§4.2 Eq. 5） | なし |
| ARL（Lahoti et al. 2020） | adversary の出力。入力は (x, y)（§3.2、公式コードの `adversary_include_label=True`） | あり。§5 で「副次的に class imbalance を解決した」と利点として記述（Fig. 4、5d） |
| JTT（Liu et al. 2021） | ERM が誤分類したサンプル（Eq. 7, 8） | あり。§5.3・App. C.2 で、まれなラベルの群が上げ重みされると記述 |
| LfF（Nam et al. 2020） | 相対 CE | 論文にはなし。公式コードは比を取る前にクラス別の最大損失で正規化（`train.py:214-222`） |
| Blind Pareto Fairness（Martinez et al. 2021） | 未知の分割 p(A\|X,Y) 上の minimax | 分割が Y に依存してよい設計（§2、§3.2） |

**B：群を属性 × ラベルのセルにする**

| 手法 | 根拠と注記 |
| --- | --- |
| GroupDRO 原典（Sagawa et al. 2020） | §2.1 "m = \|A\| × \|Y\| groups"。コードは `group_array = y*(n_groups/n_classes) + confounder`（`data/celebA_dataset.py:53-54` など）。step size の既定 0.01（`run_expt.py:39`）、汎化補正の既定 C = 0 |
| Yang et al. 2024（GroupDRO、ReSample） | `learning/algorithms.py:97` の `y * num_attributes + a`。論文本文には群の定義がない。ReSample は各属性内の陽性率の**期待値**を 50% にする（復元抽出の重み付き sampler）。DFR はコードにあるが論文の評価対象外 |
| MEDFAIR balanced resampling | App. E の分析のみ |

**C：公平性の比較を Y で条件付ける**

| 手法 | 根拠と注記 |
| --- | --- |
| Zhang et al. 2022 MMDMatch・MeanMatch | §3、`algorithms.py:141-151`。y ∈ {0, 1} ごとに群のスコア分布を全体に合わせる |
| Zhang et al. 2022 Adversarial・FairALM | `algorithms.py:172,188`、`271-305` |
| CFair（Zhao et al. 2020） | 全体の陽性率から作る class weight（BER、§3.4、`main_adult.py:63`）と、Y ごとの adversary（§3.1、`models.py:123-142`）。属性は 2 値 |
| FairBatch eqopp・eqodds（Roh et al. 2021） | 抽出は (y, z) セル単位、比較は y の中で z 間（`FairBatchSampler.py:161-225`）。accuracy parity 版はない（§5 で future work） |
| Zafar et al. 2017 FPR・FNR | §2 Eq. (4)、(5) |

**DP 型（A とは別の問題）**

公平性基準そのものが陽性率の差と衝突する。

- FairCLIP：画像とテキストの類似度の分布を群間で揃える。ラベルは学習に使わない（`finetune_FairCLIP.py:187-204`）。
- FairBatch dp：比較の向きが本文・App. A.1・コード（`FairBatchSampler.py:227-266`）で食い違う。

### 上の比較手法セットとの関係

- 上の表の「GroupDRO（原著どおり）」は群を属性で定義している。原典の群は属性 × ラベルのセルなので、更新式だけが原典どおりで、群の定義は MEDFAIR・Zhang et al. 2022 と同じ A の形にあたる。
- 2026-10-07：年齢 2 群の study（`analysis/archive/minimax_auc_vs_eo`）に FairDRO（C9）を追加した。
  - 既存の GroupDRO は「MEDFAIR 流の変種」と位置付ける。
  - FairDRO の内側最大化は、クラス内の群間の誤り率の分散に対応する（Prop. 1、Lemma 1）。重みは 0-1 誤りで決まり、負の値もとる。
- 「class weight に言及する先行研究がほとんどない」とは書けない。CFair は全体の class weight を明示的に使い、ARL・JTT・MMPF は陽性やまれなラベルへの重みの偏りに触れている。書けるのは「クラスの扱いは副作用・実装の細部・実験ごとの選択として現れ、定式化の段階では分けられていない」まで。

### 未確認

- MEDFAIR の本表の実験で使われた `resample_which` の値
- LHCF の class weight（コード非公開）
- JTT の公式コードが class weight を足しているか
- CFair の adversary 側の重み（`main_adult.py:66-68`）が論文の BER の定義と一致するか

## 出典

**ベンチマーク**

- Zong, Yang & Hospedales. MEDFAIR: Benchmarking Fairness for Medical Imaging. ICLR 2023. <https://arxiv.org/abs/2210.01725>
- Zhang et al. Improving the Fairness of Chest X-ray Classifiers. CHIL 2022. <https://proceedings.mlr.press/v174/zhang22a.html>（コード：<https://github.com/MLforHealth/CXR_Fairness>）
- Jin et al. FairMedFM: Fairness Benchmarking for Medical Imaging Foundation Models. NeurIPS 2024 D&B. <https://arxiv.org/abs/2407.00983>
- Yang et al. The limits of fair medical imaging AI in real-world generalization. Nature Medicine 2024. <https://www.nature.com/articles/s41591-024-03113-4>

**比較手法セット**

- Idrissi et al. Simple data balancing achieves competitive worst-group-accuracy. CLeaR 2022. <https://proceedings.mlr.press/v177/idrissi22a.html>
- Luo et al. FairVision: Equitable Deep Learning for Eye Disease Screening via Fair Identity Scaling. <https://arxiv.org/abs/2310.02492>（コード：<https://github.com/Harvard-Ophthalmology-AI-Lab/FairVision>）
- Sagawa et al. Distributionally Robust Neural Networks for Group Shifts. ICLR 2020. <https://arxiv.org/abs/1911.08731>
- Jung, Park, Chun & Moon. Re-weighting Based Group Fairness Regularization via Classwise Robust Optimization (FairDRO). ICLR 2023. <https://arxiv.org/abs/2303.00442>
- Zhang, Lemoine & Mitchell. Mitigating Unwanted Biases with Adversarial Learning. AIES 2018. <https://dl.acm.org/doi/10.1145/3278721.3278779>
- Kurian et al. Worst-Group Equalized Odds Regularization for Multi-Attribute Fair Medical Image Classification. arXiv 2026. <https://arxiv.org/abs/2605.19214>
- Sadri, DeSilvio & Viswanath. Mutual Information Regularization for Fairness-aware Deep Imaging Representations. MICCAI 2025. <https://papers.miccai.org/miccai-2025/paper/1404_paper.pdf>
- Hardt, Price & Srebro. Equality of Opportunity in Supervised Learning. NeurIPS 2016. <https://proceedings.neurips.cc/paper/2016/hash/9d2682367c3935defcb1f9e247a97c0d-Abstract.html>

**2026-10-05追加：最近のEO研究**

- Kurian et al. Worst-Group Equalized Odds Regularization for Multi-Attribute Fair Medical Image Classification. arXiv、2026年5月19日公開（上の比較手法セットと同じ論文）。[論文ページ](https://arxiv.org/abs/2605.19214)／[本文](https://arxiv.org/html/2605.19214v1)。
- Ni & Huo. Kernel-based Equalized Odds: A Quantification of Accuracy-Fairness Trade-off in Fair Representation Learning. NeurIPS 2025。[論文ページ・PDF](https://papers.nips.cc/paper_files/paper/2025/hash/85a47d32fbf29bb8d42cdd980b719959-Abstract-Conference.html)。理論面の参照とし、比較手法セットへの採用は未決定。

**外した手法・別の軸**

- Masroor et al. Fair Distillation: Teaching Fairness from Biased Teachers in Medical Imaging. arXiv 2024. <https://arxiv.org/abs/2411.11939>
- Madras et al. Learning Adversarially Fair and Transferable Representations. ICML 2018. <https://proceedings.mlr.press/v80/madras18a.html>
- Zhao et al. Conditional Learning of Fair Representations. ICLR 2020. <https://arxiv.org/abs/1910.07162>
- Agarwal et al. A Reductions Approach to Fair Classification. ICML 2018. <https://proceedings.mlr.press/v80/agarwal18a.html>
- Masroor et al. Fairness Beyond Demographics: Optimizing Performance Across Appearance-Based Hidden Cohorts in Medical Imaging. arXiv 2026. <https://arxiv.org/abs/2605.29827>
- Cotter et al. Optimization with Non-Differentiable Constraints with Applications to Fairness, Recall, Churn, and Other Goals. JMLR 2019. <https://jmlr.org/papers/v20/18-616.html>
- Luo et al. FairCLIP: Harnessing Fairness in Vision-Language Learning. CVPR 2024. <https://arxiv.org/abs/2403.19949>
- Kirichenko, Izmailov & Wilson. Last Layer Re-Training is Sufficient for Robustness to Spurious Correlations. ICLR 2023. <https://arxiv.org/abs/2204.02937>
- Dutt et al. FairTune: Optimizing Parameter Efficient Fine Tuning for Fairness in Medical Image Analysis. ICLR 2024. <https://arxiv.org/abs/2310.05055>
- Mao et al. Last-Layer Fairness Fine-tuning is Simple and Effective for Neural Networks. ICML 2023 Workshop. <https://arxiv.org/abs/2304.03935>

**2026-10-05追加：クラスの扱いの分類**

- Lahoti et al. Fairness without Demographics through Adversarially Reweighted Learning. NeurIPS 2020. <https://arxiv.org/abs/2006.13114>（コード：<https://github.com/google-research/google-research/tree/master/group_agnostic_fairness>）
- Hashimoto et al. Fairness Without Demographics in Repeated Loss Minimization. ICML 2018. <https://arxiv.org/abs/1806.08010>
- Liu et al. Just Train Twice: Improving Group Robustness without Training Group Information. ICML 2021. <https://arxiv.org/abs/2107.09044>（コード：<https://github.com/anniesch/jtt>）
- Nam et al. Learning from Failure: Training Debiased Classifier from Biased Classifier. NeurIPS 2020. <https://arxiv.org/abs/2007.02561>（コード：<https://github.com/alinlab/LfF>）
- Martinez, Bertran & Sapiro. Minimax Pareto Fairness: A Multi Objective Perspective. ICML 2020. <https://proceedings.mlr.press/v119/martinez20a.html>（コード：<https://github.com/natalialmg/MMPF>）
- Martinez et al. Blind Pareto Fairness and Subgroup Robustness. ICML 2021. <https://proceedings.mlr.press/v139/martinez21a.html>（コード：<https://github.com/natalialmg/BlindParetoFairness>）
- Mohri, Sivek & Suresh. Agnostic Federated Learning. ICML 2019. <https://arxiv.org/abs/1902.00146>
- Li et al. Fair Resource Allocation in Federated Learning. ICLR 2020. <https://arxiv.org/abs/1905.10497>
- Roh et al. FairBatch: Batch Selection for Model Fairness. ICLR 2021. <https://arxiv.org/abs/2012.01696>（コード：<https://github.com/yuji-roh/fairbatch>）
- Zafar et al. Fairness Beyond Disparate Treatment & Disparate Impact: Learning Classification without Disparate Mistreatment. WWW 2017. <https://arxiv.org/abs/1610.08452>
- Zhao & Gordon. Inherent Tradeoffs in Learning Fair Representations. NeurIPS 2019 / JMLR 2022. <https://arxiv.org/abs/1906.08386>
- Tian et al. FairSeg: A Large-Scale Medical Image Segmentation Dataset for Fairness Learning Using Segment Anything Model with Fair Error-Bound Scaling. ICLR 2024. <https://arxiv.org/abs/2311.02189>（コード：<https://github.com/Harvard-Ophthalmology-AI-Lab/FairSeg>）
- 既出の論文の公式コード：MEDFAIR <https://github.com/ys-zong/MEDFAIR>、Yang et al. 2024 <https://github.com/YyzHarry/shortcut-ood-fairness>、Sagawa et al. <https://github.com/kohpangwei/group_DRO>、CFair <https://github.com/hanzhaoml/ICLR2020-CFair>、FairCLIP <https://github.com/Harvard-Ophthalmology-AI-Lab/FairCLIP>
