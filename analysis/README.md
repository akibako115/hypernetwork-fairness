# analysis

1 つの仮説を検証する単位を `analysis/<slug>/` の package として持つ。`projects/` が学習の実装を
所有するのに対し、ここは **その run を読んで何が言えるかを所有する**。学習コードはここに置かない。

```text
<slug>/
  README.md    仮説、対象 run、結論
  runs.md      この仮説に紐づく run の一覧と状態
  results/     集計した表（CSV / JSON）
  figures/     レポートが参照する図
  reports/     読み手向けの分析メモ
```

## 追跡境界

`README.md` / `runs.md` / `results/` / `figures/` / `reports/` は Git で追跡する。分析の結論と、
その根拠になった表と図は repo に残す。

`outputs/` / `cache/` / `checkpoints/` と `*.npz` / `*.pt` / `*.pth` は追跡しない。再生成できる
中間物を repo に入れない。図は repo 全体の `*.png` 除外を `figures/` だけ打ち消して追跡する。

run そのもの（`projects/*/runs/<run-id>/`）と実行ログ（`run_logs/`）も追跡しない。

## ログの扱い

分析パッケージへログを持ち込まない。hardlink も copy も作らない。分析が読む数値は run artifact
（`run.json`、`stages/*/metrics/metrics.csv`）にあり、実行ログには入っていない。

`run_logs/` のファイル名は起動時に人が付けるため run-id とは対応しないが、ログ本文に run dir が
出るので run-id から引ける。落ちた run の理由を確かめる時だけこれを使う。

```bash
grep -l <run-id> run_logs/*.log
```

結論の根拠になる数値は `results/` の CSV / JSON へ写す。追跡されるのはこちらで、ログではない。
ログの行そのものが根拠になる場合は、該当箇所だけを `reports/` に引用する。
