# runs

run 本体もログも Git 管理外なので、この表が参照の正本となる。

| run-id | 状態 | 備考 |
|---|---|---|
| `20260921T081141Z-iterative-s42-5752` | succeeded | 分析対象 |
| （run dir なし） | 中断 | 08:11 の run の前に KeyboardInterrupt で停止。074739Z のログにだけ残る |

## 20260921T081141Z-iterative-s42-5752

- git commit `5c84559`、seed 42、`status: succeeded`（08:11:41Z 開始 / 08:53:29Z 終了）
- W&B: `fairness_hypernet_iterative/1w44nmwo`
- 反復: warmup 2 epoch → stage01 → stage02（各 2 epoch）、cohort は 10 クラスタ、`n_init=50`
- `cohort_training_strategy=group_dro`、`group_dro_step_size=0.001`
- `cohort_checkpoint_selection=global_auroc_bacc`

epoch 数が warmup / stage とも 2 しかないため、推移というより到達点の比較になる。

## 数値の出どころ

`projects/hypernet_iterative/runs/<run-id>/` を run-id で開く。

- `run.json` — stage ごとの scalar metric と checkpoint
- `stages/*/metrics/metrics.csv` — epoch 推移の正本
- `artifacts/cohorts/cohortNN/cohort.json` — cohort の参照 checkpoint

## ログ

実行ログは `run_logs/` にあり、ファイル名は起動時に人が付けるので run-id とは対応しない。
必要になったら run-id で引く。ログ本文に run dir が出ているので一致する。

```bash
grep -l <run-id> run_logs/*.log
```

分析で読む数値はログではなく上の run artifact にある。ログを見るのは run が落ちた時に
理由を確かめる場合に限る。
