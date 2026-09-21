# two-stage run artifact 契約

`hypernet_two_stage` の生成物は project 内の `runs/` にだけ保存する。`data/chexpert` は画像・split・
固定 metadata のような入力だけを持ち、学習生成物を置かない。

## 親 run

```text
projects/hypernet_two_stage/runs/<run-id>/
  config.yaml
  run.json
  data_manifest.json
  preflight.json
  logs/train.log
  stages/
    stage1/
      config.yaml
      metrics/fit.json
      checkpoints/
      wandb/
      result.json
    stage2/
      config.yaml
      metrics/fit.json
      checkpoints/
      wandb/
      result.json
```

`run.json` は `schema_version` / `run_id` / `kind` / `status` / `started_at` / `finished_at` /
`git_commit` / `seed` と、stage 計画、最終 `selected_checkpoint` を持つ。run-id は e2e と同じ
`<timestamp>-<experiment>-s<seed>-<hex>` 規約に従い、seed が未指定の run は `snone` とする。
親が成功するのは二つの stage がともに成功し、各 stage に最良 `val/auroc` checkpoint がある場合
だけとする。

`data_manifest.json` は train / val split の path、SHA-256、行数、画像集合の SHA-256、target 分布を
持つ。`preflight.json` は判定に使った golden ファイルの名前と SHA-256、実行した commit を持つ。

## stage の入力と出力

stage 固有の設定は config group `stage1` / `stage2` が所有し、CLI からは
`stage2.model.net.rank=8` のように stage を明示して上書きする。`workflow` は選ばれた stage の
`model` と `trainer` 差分から、その stage 一回分の config を組み立てる。

Stage 1 は ImageNet 初期化の ResNet を入力に、`val/auroc` が最大の checkpoint と last checkpoint を
出力する。Stage 2 は Stage 1 の最良 checkpoint の SHA-256 を load 直前に再計算して記録値と照合して
から、その共有 backbone と classifier を読み込み凍結する。照合に失敗した run は失敗として残す。
checkpoint の key が 1 つも一致しない部分ロードも失敗として扱い、ランダム初期化のまま学習した run
を成功として残さない。Stage 2 の最良 `val/auroc` checkpoint が親 run の最終選択となる。

各 stage の `config.yaml` は、その stage が実際に実行した解決済み設定だけを持つ（他 stage の
セクションは残さない）。`result.json` は scalar metrics、experiment logger の参照、checkpoint の
`path` / `role` / `score` / `sha256` を持つ。`score` は `ModelCheckpoint.best_model_score`、つまり
その checkpoint が選ばれた時点の `val/auroc` であり、最終 epoch の値である `metrics/fit.json` とは
一致しない。checkpoint と metrics は stage directory の外に複製しない。

## 実行ログと experiment logger

実行ログは親 run の `logs/train.log` が両 stage 分をまとめて持つ。Hydra 既定の file handler は
repository root に run をまたぐ共有 log を作るため、`configs/hydra/job_logging/console.yaml` で
console だけに制限し、`run_logging.text_log` が run directory 内の file handler を所有する。

epoch ごとの metric は `logger` group が指す experiment logger が持つ。既定は wandb で、project は
`fairness_hypernet_two_stage` とする。stage は別々の wandb run として記録し、run 名は
`<run-id>-stage1` / `<run-id>-stage2`、ローカル実体は各 stage の `wandb/` に入る。wandb run の
`name`・`id`・`url` は各 stage の `result.json` と親の `run.json` に残す。`logger=none` を指定した
run は experiment logger を作らない。`metrics/fit.json` は fit 終了時点の値だけなので、学習曲線は
experiment logger を、checkpoint に対応する値は `result.json` の `score` を正本とする。

fairness metric の key 集合はデータ依存で変わる。ある属性で観測される群が 1 つしかない batch では
`Eopp0` / `Eopp1` / `Eodds` が定義できず、その属性の key ごと出力されない。run をまたいで集計する
側は、key の欠損を前提に書く。

## 制約

- `runs/` は Git 管理しない。既存 run を上書き・再利用しない。
- Stage 2 に渡す checkpoint は同じ親 run の Stage 1 に記録された最良 checkpoint に限る。
- cohort の更新、GroupDRO、3回以上の stage、汎用 attempt/reconcile 機構は含めない。
