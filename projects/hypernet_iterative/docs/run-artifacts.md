# iterative run artifact 契約

`hypernet_iterative` の 1 run は、warmup と複数の cohort stage を所有する parent run である。
生成物は `projects/hypernet_iterative/runs/<run-id>/` にのみ保存し、入力データ配下には書き込まない。

```text
<run-id>/
  config.yaml
  run.json
  data_manifest.json
  preflight.json
  logs/train.log
  artifacts/cohorts/cohort01/
  stages/warmup/{config.yaml,result.json,checkpoints/,metrics/}
  stages/stage01/{config.yaml,result.json,checkpoints/,metrics/}
```

`run.json` は parent run の状態と stage ごとの result を保持する。`study` はこの run が属する仮説で、
`analysis/<study>/` が正本になる。状態は `running`、`succeeded`、`failed` のいずれかで、
`selected_checkpoint` には最後の stage の best val/auroc を記録する。run 全体の best では
ないので、stage をまたぐ選択は `stages.*.checkpoints` の score から分析側で決める。
既存 run の再開・上書きはしない。

各 stage の `result.json` は fit 完了時点の scalar metrics、`checkpoints`、そして epoch ごとの
metric を書いた `metrics_csv` の絶対 path を持つ。`metrics/metrics.csv` は `CSVLogger` の出力で、
epoch ごとに全 callback の記録を残す。`callback_metrics` のスナップショットは最終 epoch の 1 点しか
持たないので、推移はこの CSV が正本である。親はここを読んで W&B run へ集約する。

`data_manifest.json` は train / val / test split のファイル hash、行数、画像集合 hash、target 分布と画像 root を持つ。
`preflight.json` は起動前に実行する project-local golden preflight のコマンド、出力、golden hash、Git commit を持つ。

各 `artifacts/cohorts/cohortNN/` は直前 stage の `last` checkpoint に基づく cohort である。
次 stage の warm-start も同じ checkpoint を使う。best を引き継ぐと、GroupDRO が効いた更新ほど
stage 境界で巻き戻る。
`cohort.json` に参照 checkpoint の絶対 path と SHA-256、KMeans の設定、group 数を記録し、`assignments.parquet` を
後続 stage の固定 cohort DataModule へ渡す。optimizer、scheduler、GroupDRO の内部状態は stage をまたいで引き継がない。

`iteration.cohort_training_strategy` は `group_dro`、`uniform_group_iterative`、`uniform_group` を
選べる。`uniform_group` だけは warm-start を持たず、各 stage が backbone の初期化からやり直す。
反復の対照には `uniform_group_iterative` を使う。群内クラス均衡は目的関数ではなく `weighting` で選ぶ。
`weighting=inverse` の cohort stage は必ず group ごとの重みを使い、その重みは cohort ごとに
解き直すため stage の `config.yaml` に `[clusters, num_classes]` の実数として残る。
`iteration.cohort_checkpoint_selection` は `global_auroc_bacc` または `hidden_min_auroc` を選べる。
