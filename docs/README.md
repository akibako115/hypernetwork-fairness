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
- 2026-10-16 に全手法を原論文・公式実装と照合し、誤りを直した（[methods/README.md](methods/README.md)、commit `0f2af99`）。
  照合の記録（それより前の run がどの版で学習したか）は、手法文書を 3 節構成に書き直す前の版（commit `e693e02` の
  `docs/methods/`）に残っている。
- 2026-10-17 に CheXpert と HAM10000 の画像の前処理を MEDFAIR に合わせた（縦横比を保たず 256×256 → train はランダム、評価は中央で 224 を切り出す）。
  256×256 への縮小も MEDFAIR の cv2 と同じくアンチエイリアスなしにし、MEDFAIR の経路との差は全画素で ±1 階調以内になった。
  それより前の CheXpert の run は train と評価で画像の形が揃っていない旧前処理、HAM10000 の run は短辺 256 の旧前処理で学習している。どれを回し直すかは、
  既存の分析・run の棚卸しをしながら決める。
- Q1〜Q3 の区分（[plans/20261011_chexpert_restructure_plan.md](plans/20261011_chexpert_restructure_plan.md)）は柱 A・B に置き換えた。
  結果の一部は上の不具合の影響を受ける。

## 置き場

| 場所 | 中身 |
| --- | --- |
| [research-premises.md](research-premises.md) | 研究の前提と評価の定義。計画・手法選定・結論の書き方はこれに反しない形で進める |
| [experiment-workflow.md](experiment-workflow.md) | 実験 1 本を仮説から結論まで進める手順 |
| [plans/](plans/) | 日付つきの実験計画と実行記録。どれが現役かは上の「現在地」が決める。いまある 2 本（10-02 の主比較の実行記録、10-11 の Q1〜Q3）はどちらも過去の記録 |
| [methods/](methods/README.md) | 学習手法ごとの文書（手法のロジック・原論文の設定・この repo の実装）。一覧は [methods/README.md](methods/README.md) |
| [reviews/](reviews/) | 実装・実験基盤のレビューと修正記録 |
| [meetings/](meetings/) | ミーティング用メモ。`<YYYYMMDD>/` ごとに置く |
| [slides/](slides/) | スライド。`<name>/` に図を作る `figures.py`、pptx を組む `build.js`、生成物を置く |

スライドのスクリプトは repo root から実行する（例 `uv run python docs/slides/chexpert-q1/figures.py`、
`node docs/slides/chexpert-q1/build.js`）。各 `build.js` の冒頭に手順がある。
