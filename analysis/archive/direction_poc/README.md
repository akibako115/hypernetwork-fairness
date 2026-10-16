# direction_poc

2026-10-12開始。`latent_bias_hypernet` の結論（潜在クラスタの条件は E2E で使われない、振り分けは規則が食い違う場でだけ効く）を
受けて、次の研究の方向 5 案それぞれを、凍結特徴と既存の予測だけで当たり付けする PoC。

前提：hypernet の振り分けが効くには、(i) 条件が画像にない情報を運ぶこと、(ii) 部分群ごとに本当に別の判定規則が
要る場であること、の 2 つが要る。

| # | 方向 | script | 結果 | 報告 |
| --- | --- | --- | --- | --- |
| 1 | 画像から計算できない条件（学習時だけの情報）で振り分ける | `d1_privileged.py` | `results/d1/` | `reports/d1_privileged.md` |
| 2 | 規則の食い違いを実データ（アーチファクト）で先に測る | `d2_artifact_rules.py` | `results/d2/` | `reports/d2_artifact_rules.md` |
| 3 | 区分を判定の境界に合わせる方法（区分と補正の同時学習） | `d3_learned_partition.py` | `results/d3/` | `reports/d3_learned_partition.md` |
| 4 | Multiaccuracy-Boost と [h; e] の振り分けの直接比較 | `d4_multiaccuracy.py` | `results/d4/` | `reports/d4_multiaccuracy.md` |
| 5 | CheXpert の年齢差をラベルの意味の違いとして分解する | `d5_label_semantics.py` | `results/d5/` | `reports/d5_label_semantics.md` |

全体のまとめは [reports/summary.md](reports/summary.md)。

学習 run は新しく回していない（凍結特徴の上の head と、既存の予測の再評価だけ）。実行ログは `run_logs/`。
入力は `latent_bias_hypernet/cache/`・`fairness_mechanism_probe/cache/`・`race_known_architecture_objective_comparison/cache/` を
読み取り専用で使う。
