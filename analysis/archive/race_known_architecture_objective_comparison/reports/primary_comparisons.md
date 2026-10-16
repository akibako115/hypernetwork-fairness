# 主比較の実行記録（2026-10-05）

独立した5冊を1→5の順に実行。対象は9条件×seed 42・43・44の27run。
同じrace-known test 19,617件を、各runのbest validation AUROC checkpointで評価した。
採用checkpoint・SHA256は各トピックの`results/<topic>/provenance.csv`、固定した比較は[分析計画](../analysis_plan.md)。

## 観察事実

比較条件−baseline、3seedの平均±標準偏差（AUROCの絶対値）。

| 問い | 比較 | Δ全体AUROC | Δworst-group AUROC |
| --- | --- | --- | --- |
| DRO追加 | M01 → M02 | −0.00697 ± 0.00134 | −0.00474 ± 0.00688 |
| class weight | M02 → M11 | −0.00091 ± 0.00420 | +0.01061 ± 0.02421 |
| Spatial範囲 | M04 → M17 | −0.00141 ± 0.00106 | −0.00275 ± 0.00624 |
| ERMモデル構成 | M01 → M03 | +0.00091 ± 0.00268 | +0.00889 ± 0.02236 |
| ERMモデル構成 | M01 → M16 | −0.00101 ± 0.00674 | +0.01492 ± 0.01684 |
| ERMモデル構成 | M01 → M07 | −0.00024 ± 0.00315 | +0.00719 ± 0.01217 |
| 凍結2-stage | M04 → M06 | +0.00912 ± 0.00057 | +0.00658 ± 0.01738 |

根拠は各トピックの`paired_deltas.csv`・`paired_delta_mean_std.csv`。
8群の件数・陽性/陰性数、BACC・公平性は同じフォルダへ保存。
DROの全30epochのq_end・q_mean・群loss・countも保存し、q_end図にbest epochを表示した。

## 解釈と未確認

上表は固定した各ペアの観察であり、全条件ランキングや有意差の判定ではない。
class weightの他モデルへの一般化、変調位置だけの効果、凍結だけの効果は未確認。
他条件での再現確認・単独属性・患者単位の不確実性は各notebook末尾のTODOに残す。
