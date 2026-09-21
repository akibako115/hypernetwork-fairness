# e2e path・命名規約

この文書は `hypernet_e2e` が読む入力と生成する成果物の path・名前の正本である。
入力、実装、run artifact の所有者を path だけで区別し、実行環境に固有の directory を
設定ファイルへ持ち込まない。

## path の基準

- 追跡する config・文書内の path は repo root 基準の相対 path とする。例:
  `data/chexpert/splits`、`projects/hypernet_e2e`。
- 実行時に予約する `run_dir` と checkpoint 出力先は絶対 path に解決して run の
  `config.yaml` に保存する。これらは実行した machine 上の保存先を一意に示すためである。
- `data_manifest.json` の画像 root・split path のような入力 provenance も絶対 path と
  hash を併記する。入力そのものは複製しない。
- 相対 path の解決基準は常に repo root とし、コマンドを起動した current working directory
  には依存しない。`runtime_paths.normalize_runtime_paths` が fit の開始時に正規化する。外部入力を
  指定する場合だけ絶対 path を許し、解決済み config にその path を残す。hash が必要な artifact は
  個別の artifact 契約で定める。

## directory の所有者

| path | 所有者 | 内容 |
| --- | --- | --- |
| `data/<dataset>/` | データ準備 | 画像、split、固定 metadata のみ。学習生成物は置かない。 |
| `projects/hypernet_e2e/` | e2e project | 実装、config、文書、テスト。 |
| `projects/hypernet_e2e/runs/<run-id>/` | 1 回の run | config、記録、metrics、checkpoint、artifact。既存 run は再利用しない。 |

`logs_chexpert/`、共有 experiment directory、cohort 専用の data directory は作らない。

## 名前

| 対象 | 形式 | 例 |
| --- | --- | --- |
| Python package・config file・実験 preset | `snake_case` | `spatial_lora_chexpert_erm` |
| run ID 内の experiment 部 | 小文字 `kebab-case` | `spatial-lora-chexpert` |
| run ID | `<UTC>-<experiment>-s<seed>-<suffix>` | `20260918T101530Z-spatial-lora-erm-s42-a1b2` |
| run 内の固定 directory | 小文字の複数形 | `logs/`、`metrics/`、`checkpoints/`、`artifacts/` |
| wandb run 名 | run ID をそのまま使う | `20260918T101530Z-spatial-lora-erm-s42-a1b2` |
| cohort artifact | recipe を表す短い `snake_case`、または意味のあるパラメータ名 | `demographic_groups`、`k15` |

run ID の UTC 時刻は `YYYYMMDDTHHMMSSZ`、seed がない生成処理は `snone` とする。ランダム
suffix は衝突回避だけに使い、比較・再現の根拠にはしない。

## 適用

run artifact の構造・必須記録は [run artifact 契約](run-artifacts.md) が定める。この規約を
変える場合は、config の path 解決、run record、該当テスト、run artifact 契約を同じ変更で
更新する。
