# 公平性手法の文書

`projects/hypernet_e2e` に実装した学習手法ごとに、原論文の条件・この repo の実装・違い・使い方を 1 本ずつ置く。
どの文書も同じ節立てにしている（先に押さえること → 原論文と公式実装 → この repo の実装 → 原論文との違い → 起動 →
checkpoint の選び方 → 2026-10-16 の照合結果）。

2026-10-16 に、全手法を原論文と公式実装に照らして照合し、誤りを直した（commit `0f2af99`）。各文書の
「照合結果」の節がその記録で、それより前の run がどの版の実装で学習したかもそこに書く。

文書中の `analysis/archive/...` は 2026-10-17 に消した。README・コード・notebook は `snapshot/agent-era-20261016` ブランチと commit `c5f7deb` に残っている（[analysis/README.md](../../analysis/README.md) の「消した package」）。

## 手法の一覧

| 文書 | 手法 | 原論文 | 選び方（Hydra） |
| --- | --- | --- | --- |
| [erm.md](erm.md) | ERM と不均衡への対処（クラス重み、クラス逆頻度抽出、群×クラス均等抽出、抽出補正 ERM） | — | `training_strategy=erm`、`weighting`、`data.train_sampling`、`data=*_cells` |
| [uniform-group.md](uniform-group.md) | 群を等しく重み付けする損失（2 系統） | Sagawa et al. 2020 の `--reweight_groups` ほか | `uniform_group`、`sampling_corrected_uniform_group` |
| [group-dro.md](group-dro.md) | GroupDRO | Sagawa et al., ICLR 2020 | `group_dro_global`、`group_dro_subgroup` |
| [fair-dro.md](fair-dro.md) | FairDRO | Jung et al., ICLR 2023 | `fair_dro` |
| [minimax-auc.md](minimax-auc.md) | Minimax AUC と Population AUC | Yang et al., AAAI 2023 | `minimax_auc`、`population_auc` |
| [eo-rate.md](eo-rate.md) | EO rate 制約（proxy-Lagrangian） | Cotter et al., JMLR 2019・ICML 2019 | `equalized_odds_rate` |
| [dann.md](dann.md) | 勾配反転（GRL）による属性不変化、DANN | Ganin et al., JMLR 2016 | `/model: resnet50_attribute_invariant*` |
| [alfr.md](alfr.md) | ALFR（adversary との交互更新） | Edwards & Storkey, ICLR 2016 | `/model: resnet50_attribute_invariant_alfr` |
| [worst-cohort.md](worst-cohort.md) | ℓ_wst（最悪コホート損失） | LHCF（arXiv 2605.29827） | `worst_cohort` |
| [fis.md](fis.md) | Fair Identity Scaling | Luo et al.（FairVision） | `fis` |
| [fami.md](fami.md) | FaMI | Sadri et al., MICCAI 2025 | `fami` |
| [swad.md](swad.md) | SWAD | Cha et al., NeurIPS 2021 | experiment の `callbacks.swad` |

## 手法の選定に使った調査

- [fairness-baseline-methods.md](fairness-baseline-methods.md)：比較候補の調査（2026-09-29〜10-06）
- [fairness-objective-alternatives.md](fairness-objective-alternatives.md)：評価指標に合わせた学習の検討

どちらもエージェントが書いた調査で、手法を選んだ当時の判断材料として残している。実装の正本は上の各文書とコード。
