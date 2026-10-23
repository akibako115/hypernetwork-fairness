# 公平性手法の文書

`projects/hypernet_e2e` に実装した学習手法ごとに 1 本ずつ置く。どの文書も次の 3 節からなる。

1. **手法のロジック**：解く問題、目的関数、学習の手順。repo の実装には触れない。
2. **原論文の設定**：対象としたデータセットとタスク（何クラス分類か）、ハイパーパラメータ、モデルの選び方。
3. **この repo の実装**：ファイルと設定、処理の流れ、原論文との違い、起動例。

2026-10-16 の原論文・公式実装との照合の記録や、それより前の run がどの版の実装で学習したかは、
3 節構成に書き直す前の版（commit `e693e02` の `docs/methods/`）に残っている。

## 手法の一覧

| 文書 | 手法 | 原論文 | 選び方（Hydra） |
| --- | --- | --- | --- |
| [erm.md](erm.md) | ERM と不均衡への対処（クラス重み、クラス逆頻度抽出、群×クラス均等抽出、抽出補正 ERM） | — | `training_strategy=erm`、`weighting`、`data.train_sampling`、`data=*_cells` |
| [uniform-group.md](uniform-group.md) | 群を等しく重み付けする損失（2 系統） | Sagawa et al. 2020 の `--reweight_groups` ほか | `uniform_group`、`sampling_corrected_uniform_group` |
| [group-dro.md](group-dro.md) | GroupDRO | Sagawa et al., ICLR 2020 | `group_dro_global`、`group_dro_subgroup` |
| [fair-dro.md](fair-dro.md) | FairDRO | Jung et al., ICLR 2023 | `fair_dro` |
| [minimax-auc.md](minimax-auc.md) | Minimax AUC と Population AUC | Yang et al., AAAI 2023 | `minimax_auc`、`population_auc` |
| [eo-rate.md](eo-rate.md) | EO rate 制約（proxy-Lagrangian） | Cotter et al., JMLR 2019・ICML 2019 | `equalized_odds_rate` |
| [worst-cohort.md](worst-cohort.md) | ℓ_wst（最悪コホート損失） | LHCF（arXiv 2605.29827） | `worst_cohort` |
| [fis.md](fis.md) | Fair Identity Scaling | Luo et al.（FairVision） | `fis` |
| [fami.md](fami.md) | FaMI | Sadri et al., MICCAI 2025 | `fami` |
| [swad.md](swad.md) | SWAD | Cha et al., NeurIPS 2021 | experiment の `callbacks.swad` |

属性を敵対的に消す DANN・ALFR は hypernet_e2e から外した。使うときは別の project に置く。実装と文書は
外す前の commit（`2d2f1f6`）に残っている。

## 手法の選定に使った調査

- [fairness-baseline-methods.md](fairness-baseline-methods.md)：比較候補の調査（2026-09-29〜10-06）
- [fairness-objective-alternatives.md](fairness-objective-alternatives.md)：評価指標に合わせた学習の検討

どちらもエージェントが書いた調査で、手法を選んだ当時の判断材料として残している。実装の正本は上の各文書とコード。
