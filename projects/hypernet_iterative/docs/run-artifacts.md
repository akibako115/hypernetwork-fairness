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

`run.json` は parent run の状態と stage ごとの result を保持する。状態は `running`、`succeeded`、`failed` のいずれかで、
最終的な選択 checkpoint は `selected_checkpoint` に記録する。既存 run の再開・上書きはしない。

`data_manifest.json` は train / val / test split のファイル hash、行数、画像集合 hash、target 分布と画像 root を持つ。
`preflight.json` は起動前に実行する project-local golden preflight のコマンド、出力、golden hash、Git commit を持つ。

各 `artifacts/cohorts/cohortNN/` は直前 stage の `val/auroc` checkpoint に基づく cohort である。
`cohort.json` に参照 checkpoint の絶対 path と SHA-256、KMeans の設定、group 数を記録し、`assignments.parquet` を
後続 stage の固定 cohort DataModule へ渡す。optimizer、scheduler、GroupDRO の内部状態は stage をまたいで引き継がない。

`iteration.cohort_training_strategy` は `group_dro`、`group_dro_balanced`、`uniform_group_iterative`、
`uniform_group` を選べる。`uniform_group` だけは warm-start を持たず、各 stage が backbone の初期化から
やり直す。反復の対照には `uniform_group_iterative` を使う。`group_dro_balanced` は `weighting=none` と
組み合わせる。
`iteration.cohort_checkpoint_selection` は `global_auroc_bacc` または `hidden_min_auroc` を選べる。
