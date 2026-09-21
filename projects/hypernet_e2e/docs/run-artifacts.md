# e2e run artifact 契約

`hypernet_e2e` の出力は project 内の `runs/` にだけ保存する。`data/chexpert` は、画像・split・
固定 metadata のような入力だけを持ち、学習や将来の cohort 生成で得た成果物を置かない。
path の解決基準と命名規則は [path・命名規約](path-conventions.md) を正本とする。

## run directory

1 回の `fit` は、次の1 directory を所有する。`artifacts/cohorts/` は cohort 機能を実装する際の予約領域である。

```text
projects/hypernet_e2e/runs/
  <run-id>/
    config.yaml
    run.json
    data_manifest.json
    preflight.json
    logs/
      train.log
    metrics/
    checkpoints/
    wandb/
    artifacts/
      cohorts/
```

`runs/` は Git 管理しない。生成済み run は不変であり、再実行や再開で既存 directory を使い回さない。
途中で失敗した run も `run.json` に `failed` を記録して残す。

## run ID

`<run-id>` は次の形式にする。

```text
<UTC時刻>-<experiment>-s<seed>-<suffix>
```

例: `20260918T101530Z-spatial-lora-erm-s42-a1b2`

- UTC時刻は `YYYYMMDDTHHMMSSZ` とする。
- `experiment` は解決済み設定の実験名を小文字ハイフン区切りで使う。
- `seed` は学習に使う整数である。cohort 生成だけで seed を持たない場合は `snone` とする。
- `suffix` は短いランダム値である。ID は比較・再現の根拠ではなく、衝突しない保存先を決めるためだけに使う。

ディレクトリ予約は作成時に行う。既存 ID と衝突した場合は suffix を作り直し、既存 run を上書きしない。

## 必須記録

| path | 内容 |
| --- | --- |
| `config.yaml` | 実行時に解決済みの全設定。後から defaults をたどらず同じ入力を読める形で保存する。 |
| `run.json` | `schema_version`、`run_id`、`kind`、状態、開始・終了時刻、Git commit、seed、experiment logger の参照、結果要約を持つ。`kind` は `fit` または `cohort_build`、状態は `running` / `succeeded` / `failed` とする。 |
| `data_manifest.json` | 使用した split の SHA-256、各 split の行数・画像集合 hash・target 分布、画像 root を持つ。実データ自体は複製しない。初回の `fit` は train / val を記録する。 |
| `preflight.json` | 実行前に通した golden の版・hash・結果と、その時点の Git commit を持つ。 |

`logs/` は人が追う実行ログ、`metrics/` は集計・比較に使う機械可読な metric を置く。
現在の `run.py` は fit 終了後の scalar callback metrics を `metrics/fit.json` に保存する。
有限でない metric は JSON の `null` として保存する。
fairness metric の key 集合はデータ依存で変わる。ある属性で観測される群が 1 つしかない batch では
`Eopp0` / `Eopp1` / `Eodds` が定義できず、その属性の key ごと出力されない。run をまたいで集計する
側は、key の欠損を前提に書く。
`checkpoints/` は ModelCheckpoint が出力し、`config.yaml` の `callbacks.model_checkpoint.dirpath` と一致する。

## 実行ログと experiment logger

run ごとの実行ログは `logs/train.log` が持つ。Hydra 既定の file handler は repository root に
run をまたぐ共有 log を作るため、`configs/hydra/job_logging/console.yaml` で console だけに
制限し、`run_logging.text_log` が run directory 内の file handler を所有する。
Lightning は root に handler が無い状態で import されると自分の logger の伝播を切るので、
その logger にも同じ handler を足して fit の経過を `train.log` に残す。

epoch ごとの metric は `logger` group が指す experiment logger が持つ。既定は wandb で、
project は `fairness_hypernet_e2e` とする。`metrics/fit.json` は fit 終了時点の値だけなので、
学習曲線はこちらを正本とする。`logger=none` を指定した run は experiment logger を作らない。

| 項目 | 扱い |
| --- | --- |
| `logger.wandb.save_dir` | run 予約時に run directory へ書き換える。ローカル実体は `wandb/` に入る。 |
| `logger.wandb.name` | 既定は `null`。run 予約時に run ID を入れ、dashboard と run directory を対応させる。 |
| `logger.wandb.log_model` | `False`。checkpoint は run directory だけが持ち、wandb へ複製しない。 |
| `run.json` の `loggers` | wandb run の `name`・`id`・`url`。fit の前に記録し、失敗した run からも辿れるようにする。 |

wandb run は成功・失敗のいずれでも `run_logging.experiment_loggers` が閉じる。

## cohort artifact（保留中の契約）

cohort は固定入力ではなく、生成した run の artifact である。名前を `k15` とした例は次のとおり。

```text
<run-id>/artifacts/cohorts/k15/
  assignments.parquet
  cohort.json
  kmeans.npz              # recipe が生成する場合だけ
```

`cohort.json` は少なくとも recipe 名、引数、入力 split hash、`num_groups`、生成 Git commit を持つ。
checkpoint に依存する recipe は、参照 checkpoint の path と SHA-256 も持つ。

後続の `fit` は cohort artifact を複製しない。使用した artifact の path と `cohort.json` の SHA-256 を
自分の `run.json` に記録する。これにより、どの群割り当てで学習したかを、入力データと独立して
たどれる。

## e2e 固有の制約

- `fit` は1 run につき1回だけ行い、stage や attempt の状態は持たない。
- cohort は `fit` の開始前に確定する。学習の途中で更新しない。
- `runs/` の外に `logs_chexpert/`、共有の experiment directory、cohort 専用 data directory を作らない。
- 本契約を実装する run 記録 module は、directory 予約 → `config.yaml` 保存 → preflight → fit →
  結果記録だけを担う薄い module とする。

## 実装済みの run 記録

`run_record.py` の `RunRecorder.prepare_fit` が directory の予約、`config.yaml`、train / val の
manifest、preflight の順に作成する。予約済みの `run_dir`、checkpoint 出力先、experiment logger の
保存先と run 名は `config.yaml` にも反映する。呼び出し側は学習後に `succeed`、例外時に `fail` を呼び、
既存 run を再開・上書きしない。`run.py` はこの recorder を使って model / data / callbacks /
trainer を作成し、1 回の Lightning `fit` を実行する。test、resume、stage 制御は行わない。
