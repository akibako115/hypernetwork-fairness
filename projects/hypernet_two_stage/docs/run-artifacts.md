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

`run.json` は親 run の状態、stage 計画、Stage 1 から Stage 2 へ渡す checkpoint の path と SHA-256、
最終 `selected_checkpoint` を持つ。親が成功するのは二つの stage がともに成功し、各 stage に
最良 `val/auroc` checkpoint がある場合だけとする。

## stage の入力と出力

Stage 1 は ImageNet 初期化の ResNet を入力に、`val/auroc` が最大の checkpoint と last checkpoint を
出力する。Stage 2 は Stage 1 の最良 checkpoint の hash を検証してから、その共有 backbone と
classifier を読み込み凍結する。Stage 2 の最良 `val/auroc` checkpoint が親 run の最終選択となる。

各 `config.yaml` は実行に使った解決済み設定を持つ。`result.json` は scalar metrics、experiment
logger の参照、checkpoint の path、role、score、SHA-256 を持つ。checkpoint と metrics は stage
directory の外に複製しない。

## 実行ログと experiment logger

実行ログは親 run の `logs/train.log` が両 stage 分をまとめて持つ。Hydra 既定の file handler は
repository root に run をまたぐ共有 log を作るため、`configs/hydra/job_logging/console.yaml` で
console だけに制限し、`run_logging.text_log` が run directory 内の file handler を所有する。

epoch ごとの metric は `logger` group が指す experiment logger が持つ。既定は wandb で、project は
`fairness_hypernet_two_stage` とする。stage は別々の wandb run として記録し、run 名は
`<run-id>-stage1` / `<run-id>-stage2`、ローカル実体は各 stage の `wandb/` に入る。wandb run の
`name`・`id`・`url` は各 stage の `result.json` と親の `run.json` に残す。`logger=none` を指定した
run は experiment logger を作らない。`metrics/fit.json` は fit 終了時点の値だけなので、学習曲線は
experiment logger を正本とする。

## 制約

- `runs/` は Git 管理しない。既存 run を上書き・再利用しない。
- Stage 2 に渡す checkpoint は同じ親 run の Stage 1 に記録された最良 checkpoint に限る。
- cohort の更新、GroupDRO、3回以上の stage、汎用 attempt/reconcile 機構は含めない。
