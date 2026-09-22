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

`scratch/` は例外で、まだ仮説に紐づかない run の逃げ道として置いている。結論の根拠にはしない。

```text
common/        run 記録の読み方と予測 cache。全 package で共有する
styles/        図の style。全 package で共有する
tests/         共有層と、指標の定義が学習側と一致することのテスト
<slug>/
  README.md     仮説、対象 run、結論
  runs.md       この仮説に紐づく run の一覧と状態
  collect.py    run artifact → results/*.csv
  groups.py     群の定義と群別指標 → results/*.csv
  plots.py      results/*.csv → figures/*.png
  <name>.ipynb  主張・表の表示・考察
  cache/        予測 cache（再生成できる中間物）
  results/      集計した表
  figures/      レポートが参照する図
  reports/      読み手向けの分析メモ
```

**生成物は一方向にしか流れない。**

```text
run artifact  →  cache/  →  results/  →  figures/  →  notebook
```

各 script は単独で走り、前段の生成物だけを入力に取る。notebook から split CSV も checkpoint も
直接読まない。この向きを守ると、図を 1 枚描き直すために notebook を全部実行する状態にならない。
script と notebook の書き方は [AGENTS.md](AGENTS.md) を正本とする。

`common/` で共有するのは **run 契約の読み方**（`run.json`・`metrics.csv`・W&B transaction log・
checkpoint 選択・予測 cache）までとする。**群の切り方・指標の定義・図は package に閉じる。**
仮説ごとの判断を共有層へ上げると、別の仮説がその判断を暗黙に引き継ぐ。

| module | 責務 |
| --- | --- |
| `common/paths.py` | repo root と、run・split CSV・style の標準 path |
| `common/run_artifacts.py` | run artifact の読み取りと checkpoint 選択、CSV 書き出し |
| `common/predictions.py` | 選択済み checkpoint → split 予測 `.npz`（CLI 付き） |
| `common/studies.py` | study に属する run の一覧と `runs.md` 用の表 |

現在の package:

- `initial_resnet_vs_invariant/` — 通常の ResNet と attribute-invariant ResNet の初期比較
- `iterative_probe/` — 反復学習の初期方針を決める探り分析
- `scratch/` — まだ仮説に紐づかない run の逃げ道

## 追跡境界

追跡するのは **再生成の手順と、そこから読み取った結論**に限る。

- 追跡する: `README.md` / `runs.md` / `reports/` / 集計 script / notebook / `common/` /
  `styles/` / `tests/`
- 追跡しない: `results/` の表（`*.csv` / `*.parquet`）、`figures/` の図（`*.png`）、
  `cache/` / `outputs/` / `checkpoints/` と `*.npz` / `*.pt` / `*.pth`

図も表も run artifact から再生成できる。repo に残すのは、再生成できない判断のほうとする。
run そのもの（`projects/*/runs/<run-id>/`）と実行ログ（`run_logs/`）も追跡しない。

## ログの扱い

分析パッケージへログを持ち込まない。hardlink も copy も作らない。分析が読む数値は run artifact
（`run.json`、`stages/*/metrics/metrics.csv`）にあり、実行ログには入っていない。

`run_logs/` のファイル名は起動時に人が付けるため run-id とは対応しないが、ログ本文に run dir が
出るので run-id から引ける。落ちた run の理由を確かめる時だけこれを使う。

```bash
grep -l <run-id> run_logs/*.log
```

結論の根拠になる数値は `results/` へ書き出し、そこから読み取ったことを `reports/` と各 package の
`README.md` に残す。ログの行そのものが根拠になる場合は、該当箇所だけを `reports/` に引用する。
