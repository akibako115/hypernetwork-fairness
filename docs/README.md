# docs

研究全体にかかわる文書の置き場。個々の仮説の計画・結果は `analysis/<study>/` が正本で、ここには置かない。

## 現在地

**いまの問いの枠組みは [meetings/20261016/research_overview.md](meetings/20261016/research_overview.md)（2026-10-13）。**
柱 A「hypernetwork による条件付けは、いつ効いて、いつ効かないか」と、柱 B「hypernetwork で潜在サブグループを扱うのは
妥当か、効果はあるか」の 2 本。評価の定義は [research-premises.md](research-premises.md) に従う。

この節は、正本が変わるたびに書き換える。ここに挙がっていない計画・分析文書は、過去の記録として参考にとどめる。

- 2026-10-13 に公平性損失の実装を直した（[reviews/20261013_fairness_objective_fixes.md](reviews/20261013_fairness_objective_fixes.md)）。
  それ以前の EO rate・自然抽出の GroupDRO・FIS・DANN の結果は、手法の効果として読めない。
  過去の結果は、修正の影響を受けず格子に合うものだけを再利用し、影響を受けるものは使わずに回し直す
  （research_overview の「過去の実験結果の扱い」）。
- 2026-10-15 に FIS と FaMI を論文の式に合わせた（[reviews/20261015_lhcf_paper_alignment.md](reviews/20261015_lhcf_paper_alignment.md)）。
  それ以前の FIS・FaMI の結果は使わない。
- Q1〜Q3 の区分（[plans/20261011_chexpert_restructure_plan.md](plans/20261011_chexpert_restructure_plan.md)）は柱 A・B に置き換えた。
  結果の一部は上の不具合の影響を受ける。

## 置き場

| 場所 | 中身 |
| --- | --- |
| [research-premises.md](research-premises.md) | 研究の前提と評価の定義。計画・手法選定・結論の書き方はこれに反しない形で進める |
| [experiment-workflow.md](experiment-workflow.md) | 実験 1 本を仮説から結論まで進める手順 |
| [plans/](plans/) | 日付つきの実験計画と実行記録。どれが現役かは上の「現在地」が決める。いまある 2 本（10-02 の主比較の実行記録、10-11 の Q1〜Q3）はどちらも過去の記録 |
| [methods/](methods/) | 公平性手法の調査。[比較候補](methods/fairness-baseline-methods.md)と、[評価指標に合わせた学習への検討](methods/fairness-objective-alternatives.md)。手法ごとの原論文の条件・実装・checkpoint の選び方は [EO rate](methods/eo-rate.md)、[GroupDRO](methods/group-dro.md) |
| [reviews/](reviews/) | 実装・実験基盤のレビューと修正記録 |
| [meetings/](meetings/) | ミーティング用メモ。`<YYYYMMDD>/` ごとに置く |
| [slides/](slides/) | スライド。`<name>/` に図を作る `figures.py`、pptx を組む `build.js`、生成物を置く |

スライドのスクリプトは repo root から実行する（例 `uv run python docs/slides/chexpert-q1/figures.py`、
`node docs/slides/chexpert-q1/build.js`）。各 `build.js` の冒頭に手順がある。
