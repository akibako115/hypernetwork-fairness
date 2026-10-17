# Fair Identity Scaling（FIS、FairVision）：原論文・実装・使い方

2026-10-16 作成。FairVision の本文と公式実装を読み直し、この repo の実装（HEAD `0f2af99`）と突き合わせた結果をまとめる。
FIS の run を設計・起動・評価するセッションは、先にこれを読む。

## 先に押さえること

- **FIS は CE に標本ごとの重みを掛ける手法。** 重みは (1 − c)·s^I + c·s^G。s^I は損失の softmax で、損失の大きい標本を重くする。
  s^G は「その標本の群の損失の分布が、バッチ全体の損失の分布からどれだけ離れているか」の softmax で、離れた群を重くする。
  分布の離れ具合は **OT**（最適輸送：一方の分布を他方へ動かすのに要る最小のコスト）で測る。実装は **Sinkhorn divergence**
  （OT にエントロピーの正則化を足して速く解けるようにし、その偏りを差し引いたもの）。
- **s^G の正規化が論文と公式実装で違う。この repo は論文に合わせた**（2026-10-15、commit は 10-16 の `671750e`）。
  論文の形では s^I が 1/B 前後、s^G が 1/|A| 前後になる（B はバッチの件数、|A| は群の数）。そのため c = 0.5 でも、
  重みのうち群の項が占める割合は B/(B + |A|) になる。B = 128 なら 11 群で 0.92、4 群で 0.97。**個別の項はほとんど効かない。**
  原論文の CNN の設定（B = 6、|A| = 2〜3）では 0.67〜0.75 で、両方の項が効いていた（推測）。c を何の意味で使うかは見直しを勧める（下の「起動」）。
- B^t_a（群ごとに別に引くミニバッチ）の forward は、公式実装と 2 点違う。全群を 1 回でまとめて forward する（BatchNorm の統計が変わる）。
  抽出の seed が 42 に固定で、run の seed によらない。
- **既存の FIS run 6 本（2026-10-10）は使わない。** 10-13 と 10-15 の修正より前の版で、B^t_a を引いていない。

## 原論文と公式実装

- Luo, Khan, Tian, Shi, Dou, Elze, Fang, Wang. *FairVision: Equitable Deep Learning for Eye Disease Screening via Fair Identity Scaling.*
  [arXiv:2310.02492](https://arxiv.org/abs/2310.02492)（v3、2024-04）
- 公式実装：[Harvard-Ophthalmology-AI-Lab/FairVision](https://github.com/Harvard-Ophthalmology-AI-Lab/FairVision)
  （`src/modules.py` の `Fair_Loss_Scaler`、`scripts/train_glaucoma_fair_fis.py`）

### 原論文の条件（本文と公式実装で確認済み）

| 項目 | 原論文・公式実装 | 出典 |
| --- | --- | --- |
| 損失 | (1/\|B^t\|) Σ_i ((1 − c) s^I_i + c s^G_i) ℓ_i | 4.3 節、式 (3) |
| s^I | バッチ B^t の損失の softmax（Σ_i s^I_i = 1） | 式 (4) |
| s^G | exp(OT(B^t の損失, B^t_a の損失)) を群 a = 1..\|A\| について正規化（和が 1）。B^t_a は群 a の train から別に引いたミニバッチ | 式 (5) |
| c | 0.5（c = 1 で群の項だけ、0 で個別の項だけ） | 4.3 節、5 節 |
| 公式実装の s^G | 各群の B^t_a と全群の B^t_a の連結との Sinkhorn divergence を、**バッチの標本について** softmax。B^t は使わない | `modules.py:756–769` |
| OT の中身 | 本文に記載なし。公式は geomloss `SamplesLoss("sinkhorn", p=2, blur=0.1)` | `modules.py:751` |
| B^t_a | 本文に件数の記載なし。公式は群ごとの無限 loader から 64 件（`drop_last=False`）。各群を別々に、train モード・勾配なしで forward | `train_glaucoma_fair_fis.py:171–189、349–357` |
| 温度・`right_asymptote` | 公式の train script の既定はどちらも 1 | 同 95–96 行 |
| データ | Harvard-FairVision。AMD・DR・緑内障それぞれ 10,000 件（6,000 / 1,000 / 3,000）。SLO 眼底（2D）と OCT B-scan（3D） | 3 節、5 節 |
| 群 | race（3 群）、gender（2）、ethnicity（2）を 1 属性ずつ | 5 節、6.2 節 |
| モデル | ResNet50、DenseNet121、EfficientNet、ViT-B、Swin-B。3D は 3D ResNet | 5 節 |
| 最適化 | AdamW。CNN は lr 1e-4・10 epoch・batch 6。Swin-B は 1e-4・50 epoch・64。3D は 5e-5・50 epoch。公式は betas (0.0, 0.1) | 5 節、script 530–556 行 |
| checkpoint | 本文に記載なし。公式は val の全体 AUC が最大の epoch の test を報告 | script 626 行付近 |
| 指標 | 全体 AUC、ES-AUC、群別 AUC、DPD、DEOdds、Mean / Max PSD。5 回の平均と標準偏差 | 5 節 |

## この repo の実装

- **損失**：`projects/hypernet_e2e/objectives/cohort.py:89–156` の `FISTaskLoss`。
  - s^I：バッチの CE の softmax（L154）。勾配を流す（公式と同じく detach しない）。
  - s^G：B^t の損失と各群の B^t_a の損失の Sinkhorn divergence を、群について softmax（L148–152）。勾配は流さない。
  - 損失：((1 − c) s^I_i + c s^G_{a_i}) ℓ_i のバッチ平均（L155–156）。
  - Sinkhorn は `sinkhorn_divergence`（L58–86）の自前実装。1 次元・コスト |x − y|²/2・ε = blur²・ε を 50 段で下げる。
- **B^t_a**：`data/group_datamodule.py:104–142` の `GroupSampleBatchSampler` が各群から 64 件ずつ引いて連結する
  （群の中は非復元で 1 周してから並べ直す）。`group_sample_batches`（L261–282）が train と同じ transform で loader にする。
  `module.py:276–288` が学習開始時に流れを作り、`module.py:291–305` が毎 step、train モード・勾配なしで forward する。
  1 step の forward は学習バッチ 128 枚 + 群数 × 64 枚（LHCF の 11 群なら 704 枚）。B^t_a 用に別の worker（`num_workers` 本）が常駐する。
- **選び方**：`training_strategy=fis`。`fusion` が c、`blur`、`group_sample_size` が B^t_a の件数。
- **クラス重み**：HAM10000 の preset は `weighting: inverse` なので、B^t と B^t_a の CE の両方に逆頻度の重みが掛かる。
- **記録されるもの**：通常の `metrics.csv` と checkpoint に加え、`metrics/objective_diagnostics.jsonl`（1 行 = 1 epoch）に
  s^G（`group_weight`）、OT 距離（`group_distance`）、標本の重みのうち群の項が占める割合（`group_term_share`）を残す。
  `group_term_share` で、c が実際にどれだけ群の項を効かせたか（上の 0.92〜0.97）を run ごとに確かめられる。
  2026-10-17 から `ObjectiveDiagnosticsRecorder` が既定の callback に入り、全 preset で書き出す。

### 原論文との違い

| 項目 | 原論文 | この repo |
| --- | --- | --- |
| s^G の正規化 | 群について（式 (5)） | 群について（論文どおり。公式実装とは違う） |
| OT の相手 | B^t と B^t_a | 同じ（公式実装は B^t_a と全 B^t_a の連結） |
| B^t_a の forward | 公式は群ごとに 64 件ずつ別々に | 全群の 64 × \|A\| 件を 1 回で。BatchNorm のバッチ統計と running stats の更新が公式と違う |
| B^t_a の抽出 seed | 公式は DataLoader の shuffle（全体の seed に従う） | 42 に固定（`module.py:288` が seed を渡さず、既定の 42 になる）。seed 42・43・44 で同じ順に引く |
| B^t_a の端数 | 公式は群の 1 周の終わりで 64 件未満になりうる | 常に 64 件 |
| Sinkhorn | geomloss の既定（ε-scaling 0.5） | 自前。収束値との差は 1% 未満。geomloss 既定の値とは最大 1.7 倍違う（監査時の確認） |
| データ・群 | 眼科画像、race 等の 2〜3 群 | HAM10000、性別 × 年齢の 4 群か LHCF の 11 コホート |
| 最適化 | AdamW、CNN は batch 6・10 epoch | AdamW lr 1e-4、batch 128、30 epoch |
| checkpoint | 公式は val 全体 AUC 最大 | best val AUROC ほか（下） |

## 起動

| preset（`configs/experiment/`） | 群 |
| --- | --- |
| `resnet_ham10000_fg_fis_lhcf` | LHCF の 11 コホート（[worst-cohort.md](worst-cohort.md)） |
| `resnet_ham10000_fg_fis_sex_age` | 性別 × 年齢の 4 群 |

```bash
uv run python -m projects.hypernet_e2e.run experiment=resnet_ham10000_fg_fis_sex_age seed=42 study=<slug>
```

**c の扱い（提案）**：論文の式は残し、「個別の項と群の項を半々に効かせる」意図なら、c を |A|/(B + |A|) にする。
11 群で `model.loss_fn.fusion=0.079`、4 群で `0.030`。論文の数字（c = 0.5）をそのまま使う run と区別して記録する。
公式実装の正規化に戻すにはコードの変更が要る。

メモリ：B^t_a の loader が `num_workers` 本の worker を常駐させる。並列で回すときは `data.num_workers` を下げる。

## checkpoint の選び方と予測の使い方

- 公式実装どおりなら **val の全体 AUC が最大の epoch**（`best_val_auroc_*.ckpt`）。
- preset は `_fairness_ham10000.yaml` により、val の性別 × 年齢の最悪群 AUROC で選んだ `best_val_worst_sexage_*.ckpt` と
  `last.ckpt` も残す。LHCF のコホートで学習した run を性別 × 年齢で選ぶと、属性ラベルを選択に使うことになる。
- 予測は `analysis/common/predictions.py`。best val AUROC 以外は `--checkpoint` を渡す。

## 2026-10-16 の照合結果

- **一致を確かめたこと**：式 (3)–(5) の形（s^G の群についての正規化、B^t と B^t_a の比較）。blur 0.1、群あたり 64 件、温度 1、
  `right_asymptote` 1 は公式と同じ。s^I に勾配を流し、s^G に流さない点も公式と同じ。自前の Sinkhorn は収束値と 1% 未満の差。
- **文書に無かった違い**：B^t_a を全群まとめて forward すること（BatchNorm）、B^t_a の seed が 42 に固定であること。
  [reviews/20261015_lhcf_paper_alignment.md](../reviews/20261015_lhcf_paper_alignment.md) は forward を「公式実装と同じ」と書くが、
  train モード・勾配なしの点だけが同じで、群ごとに分けない点は違う。
- **c の意味**：上のとおり、論文の正規化では c = 0.5 が実効で 0.92〜0.97 の混合比になる。意図どおりか見直すことを勧める。
- **実装の履歴と既存 run**：
  - 〜10-12：B^t_a の代わりにバッチ内の群の部分集合、s^G は群について正規化、blur 0.05（`6a09df9` の docstring と run の記録による）。
  - 10-13〜10-15：公式実装の正規化（標本について softmax）。この版で回した run は無い。
  - 10-15〜：論文の式（今の版）。10-16 に `671750e` で commit。この版で回した run はまだ無い。
  - 既存の 6 本（`resnet-ham10000-fg-fis-{lhcf,sex-age}`、seed 42〜44、10-10）は最初の版で、使わない。
