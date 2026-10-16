# analysis

1 つの仮説を検証する単位を `analysis/<slug>/` の package として持つ。`projects/` が学習の実装を
所有するのに対し、ここは **その run を読んで何が言えるかを所有する**。学習コードはここに置かない。

package は project ごとに分けず、`analysis/` 直下へ平らに並べる。仮説は project をまたぐことが
あり（e2e と iterative の比較など）、project で階層を切るとその比較の置き場が無くなる。

package の slug は学習側の `study` と同じ値になる。run は起動時に `study=<slug>` を受け取り、
`run.json` にその値を残す。解決済み設定ごと W&B にも載るので、**仮説 → run → dashboard が
1 つの名前でつながる**（dashboard 側は `config.study` で group by する）。新しい仮説を始める
ときは、先にこの package を作る。

ただし `study` が記録するのは**生成の理由**だけで、1 run に 1 つしか付かない。**どの package が
その run を引用したか**は多対多で、あとから増える。こちらの正本は各 package の `runs.md` が
持つ。baseline のように複数の仮説が読む run は、`study` が指すのと別の package からも参照される。
`runs.md` には少なくとも `project`、`run-id`、repo root からの相対 `run path`、W&B URL を記録し、
分析コードが読む入力を明示する。

`scratch/` は例外で、まだ仮説に紐づかない run の逃げ道として置いている。結論の根拠にはしない。

```text
common/        run 記録の読み方と予測 cache。全 package で共有する
styles/        図の style。全 package で共有する
tests/         共有層と、指標の定義が学習側と一致することのテスト
archive/<slug>/ 終わった仮説の package。中の構成は <slug>/ と同じ
<slug>/
  README.md     仮説、対象 run、結論
  runs.md       この仮説に紐づく run の一覧と状態
  runs/         取り込んだ run（projects/*/runs/ への hardlink。Git 管理外）
  <name>.ipynb  分析トピックごとの読み込み・集計・可視化・考察（分析の主単位）
  <module>.py   notebook から切り出した処理（共有する・重い・テストで固定したいものだけ）
  cache/        予測 cache（再生成できる中間物）
  results/      集計した表
  figures/      レポートが参照する図
  reports/      読み手向けの分析メモ
```

**生成物の依存関係は一方向に保つ。** ただし、分析の実装を最初からすべて script に分解することは
要求しない。探索段階では、notebook の中で入力の読み込みから分析・可視化・考察までを完結させてもよい。
同じ処理を複数の分析で使う、実行に時間がかかる、または notebook が読みにくくなった段階で、必要な
部分だけを module や script に切り出す。

```text
projects/*/runs/  →  runs/  →  notebook  →  figures/（必要な場合）
                    runs.py import       ↘ cache / results（必要な場合）

再利用・高コストな処理だけを外部化する場合:

projects/*/runs/  →  runs/  →  cache/  →  results/  →  notebook  →  figures/（必要な場合）
```

外部化した script / module は前段の生成物だけを入力に取る。notebook は必要に応じて split CSV、
checkpoint、run artifact を直接読む。分析固有の可視化は notebook で行い、複数 package で再利用する図だけを
`figures/` に保存する。script と notebook の使い分けは [AGENTS.md](AGENTS.md) を正本とする。

`common/` で共有するのは **run 契約の読み方**（`run.json`・`metrics.csv`・W&B transaction log・
checkpoint 選択・予測 cache）までとする。**群の切り方・指標の定義・図は package に閉じる。**
仮説ごとの判断を共有層へ上げると、別の仮説がその判断を暗黙に引き継ぐ。

| module | 責務 |
| --- | --- |
| `common/paths.py` | repo root と、run・split CSV・style の標準 path |
| `common/runs.py` | 学習が書いた run を package の `runs/` へ hardlink で取り込む（CLI 付き） |
| `common/run_artifacts.py` | run artifact の読み取りと checkpoint 選択、CSV 書き出し |
| `common/predictions.py` | 選択済み checkpoint → split 予測 `.npz`（CLI 付き） |
| `common/studies.py` | study に属する run の一覧と `runs.md` 用の表 |

## package の索引

2026-10-14 時点。いまの問い（柱 A・B）との対応は [docs/README.md](../docs/README.md) の「現在地」を正本とし、
ここでは各 package が何を扱ったかだけを書く。

### 現役（直下）

柱 A（条件付けのモデル構成と目的関数）

| package | 扱ったこと |
| --- | --- |
| `hypernet_conditioning_grid/` | 柱 A の主実験。条件付けの有無（ResNet-50 / Spatial LoRA）× 目的関数 × 学習手順の格子を CheXpert で回す。条件のそろう過去の ERM は再利用し、公平性損失は修正後の実装で回し直す |
| `hypernet_conditioning_grid_ham10000/` | 上の格子をデータだけ HAM10000（sex × 年齢 65 歳の 4 群）に替えて回す。条件付けの効きがデータに依存するかを見る |
| `ham10000_architecture_erm/` | 主比較の ERM 3 構成を HAM10000 で回す |
| `fairvlmed_architecture_erm/` | 主比較の ERM 3 構成を FairVLMed で回す |
| `dataset_gain_sources/` | 条件付けが HAM10000 でだけ全体性能を上げる理由を、3 データセットで切り分ける |

柱 B（潜在サブグループ）

| package | 扱ったこと |
| --- | --- |
| `latent_group_fairness/` | 柱 B の「改善」側。生成 ΔW のクラスタに公平性損失をかけ、評価の 8 群（年齢 × race）の動作点が良くなるかを、属性を直接 k-means した群と比べる |
| `latent_subgroup_spaces/` | 1 段・ERM の Spatial LoRA（CheXpert）で、属性・metadata encoder の c・生成 ΔW・モデル自身の特徴・RAD-DINO の 5 空間 × クラスタリング／見落としの監査器で潜在サブグループを探し、同じ物差し（0.5 の FPR、val で作り test で測る）で並べる。小さな部分集団（細かい群・k 近傍の走査）も扱う |
| `conditioned_subgroup_discovery/` | 属性条件付き Spatial LoRA の生成 ΔW を属性セルで読み、規則の反転を植え込んで回収できるかを見る（README なし。`attribute_rule_planting.py`・`delta_probe.py` の docstring が説明） |

その他

- `scratch/` — まだ仮説に紐づかない run の逃げ道。結論の根拠にしない

### 退避済み（`archive/`）

終わった仮説と、柱 A・B に置き換わる前の探り。slug は学習側の `study` と W&B に残っているので名前は変えず、
置き場だけを `archive/<slug>/` へ移している。`common/paths.py` の `study_dir()` は直下に無い slug を
`archive/` から探すので、`runs.py import <slug>` などはそのまま動く。import は `analysis.archive.<slug>` になる。

✱ は 2026-10-13 の公平性損失の修正（[修正記録](../docs/reviews/20261013_fairness_objective_fixes.md)）の
影響を受ける結果を含む package で、README の冒頭に注記がある。過去の結果を再利用するときは
research_overview の「過去の実験結果の扱い」に従う。

| package | 扱ったこと |
| --- | --- |
| `archive/race_known_architecture_objective_comparison/` ✱ | CheXpert race-known の主比較 13 条件 × 3 seed。ResNet / Spatial LoRA / HyperAdapt × ERM / GroupDRO × class weight × 1・2 stage |
| `archive/two_stage_vs_e2e_age_race_group_dro/` ✱ | 元 split・14 群での 2 stage と e2e の GroupDRO 比較（旧実験）。race-known 分は上へ分離 |
| `archive/hyperadapt_two_stage_vs_e2e/` | HyperAdapt の 2 stage と e2e（CheXpert、seed 43） |
| `archive/conditional_fairness_one_stage/` ✱ | 年齢 2 群の FairDRO・EO rate・minimax AUC を 1 stage で回す（CheXpert、seed 42） |
| `archive/minimax_auc_vs_eo/` ✱ | Minimax AUC と EO rate 制約の実装・実験計画 |
| `archive/fairness_mechanism_probe/` | 群間差の機構診断（条件付けの伸びしろ・ラベルの意味・動作点） |
| `archive/iterative_probe/` ✱ | `hypernet_iterative` の初期方針を決める探り（GroupDRO の step size など） |
| `archive/initial_resnet_vs_invariant/` ✱ | ResNet と attribute-invariant ResNet（DANN）の初期比較 |
| `archive/adversary_strength/` ✱ | DANN の adversary の強さと、属性が消えるまでの水準 |
| `archive/latent_bias_hypernet/` ✱ | 1 段目を凍結して hypernetwork の補正にバイアスを溜め、潜在サブグループを発見・改善する設計。部分群の定義の比較（Q1）、公平性手法 11 種の比較、Q2 の損失比較もここにある |
| `archive/direction_poc/` | `latent_bias_hypernet` を受けた次の方向 5 案を、凍結特徴と既存予測で当たり付けする PoC |

仮説が終わった package は、同じ要領で `archive/` へ移す。移すときは `analysis.<slug>` の import と
`analysis/<slug>` の path を repo 全体で書き換え、package 内の `../` で始まる相対リンクを 1 段深くする。

## 追跡境界

追跡するのは **再生成の手順と、そこから読み取った結論**に限る。

- 追跡する: `README.md` / `runs.md` / `reports/` / 集計 script / notebook / `common/` /
  `styles/` / `tests/`
- 追跡しない: `results/` の表（`*.csv` / `*.parquet`）、`figures/` の図（`*.png`）、
  `cache/` / `outputs/` / `checkpoints/` と `*.npz` / `*.pt` / `*.pth`

図も表も run artifact から再生成できる。repo に残すのは、再生成できない判断のほうとする。
run そのもの（`projects/*/runs/<run-id>/` と、取り込んだ `analysis/<slug>/runs/`）と実行ログ（`run_logs/`）も追跡しない。

## run の取り込み

学習は `projects/<project>/runs/<run-id>/` に書く。**分析は自分の package の `runs/` だけを読む。**
使う run は、分析を始める前に package へ取り込む。

```bash
uv run python analysis/common/runs.py import <slug> projects/hypernet_e2e/runs/<run-id> ...
```

取り込みは directory を作り直し、file を hardlink にする（`cp -al` と同じ）。

- **disk は増えない。** checkpoint が 270MB/本あっても、取り込みにかかるのは directory の分だけ
- **同じ run を複数の package が取り込める。** baseline は、引用する package ごとに取り込む。
  run の置き場を 1 つの package に決めない（引用は多対多で、あとから増える）
- **package が単体で完結する。** `projects/` 側の run を整理しても、取り込んだ file は消えない

守ること:

- **終わった run（`status` が `succeeded` / `failed`）だけ取り込む。** 取り込んだ後に学習が作る
  file（checkpoint など）は package 側に現れない。`runs.py` は実行中の run を拒む
- **取り込んだ file を書き換えない。** hardlink は元と同じ inode なので、書き換えると正本も変わる。
  run artifact は不変という約束がそのまま効く
- 同じ run を 2 回渡しても何もしない。出力される表の行を `runs.md` に貼る
- 別マシンへ rsync するときは `-H` を付ける。付けないと hardlink が実体の複製になる

controller log（`run_logs/`）は分析の入力ではないので取り込まない。分析が読む数値は run artifact
（e2e の `metrics/metrics.csv`、iterative の `stages/*/metrics/metrics.csv`、`run.json`）にある。

`run_logs/` のファイル名は起動時に人が付けるため run-id とは対応しないが、ログ本文に run dir が
出るので run-id から引ける。落ちた run の理由を確かめる時だけこれを使う。

```bash
grep -l <run-id> run_logs/*.log
```

結論の根拠になる数値は `results/` へ書き出し、そこから読み取ったことを `reports/` と各 package の
`README.md` に残す。ログの行そのものが根拠になる場合は、該当箇所だけを `reports/` に引用する。
