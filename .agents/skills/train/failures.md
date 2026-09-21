# 失敗した run の扱い

## 原則

**run directory を消さない。** 失敗した run も `run.json` に `status: failed` と
`error_type` / `error_message` を持ったまま残る契約であり、それが失敗の一次記録になる。
再開機構は無いので、やり直しは必ず新しい run directory になる。

## 診断の順序

1. `run.json` の `status`、`error_type`、`error_message`、`started_at` / `finished_at`
2. `preflight.json` の `exit_code`。0 以外なら golden 照合で落ちており、fit は始まっていない
3. controller log（`run_logs/<name>.log`）の末尾。`Traceback` / `RuntimeError` / `CUDA` /
   `out of memory` / `Killed` / `Error` を探す
4. `checkpoints/` の最終更新時刻。どこまで進んでいたかが分かる
5. `metrics/fit.json` の有無。存在すれば fit は正常終了しており、失敗は後段
6. ローカル run で Python の traceback が無い場合は、OS/session 側を見る

```bash
journalctl --since '<timestamp - 10 min>' --until '<timestamp + 10 min>' --no-pager
```

traceback が無く SSH/session の終了付近で止まっているなら中断として扱い、detached 起動で
該当 seed だけ回し直す。

## よくある切り分け

| 症状 | 見るところ |
| --- | --- |
| `golden preflight に失敗した` | `preflight.json` の stdout。`tests/golden/` の版と実装のずれ。移植中なら [migrate skill](../migrate/SKILL.md) の範囲 |
| `run config に paths.project_dir が必要` | override で `paths` を壊していないか |
| CUDA OOM | `data.batch_size`、`trainer.precision`、同じ GPU 上の他 run |
| `DataLoader worker ... exited unexpectedly` / `Bus error` | 共有メモリ。remote は `--ipc=host`、ローカルは `data.num_workers` |
| `train.csv の target は ... である必要がある` | split CSV と `data.num_classes` の不一致 |

## 掃除

- checkpoint を削除しない
- 成功・進行中の run directory を削除しない
- 失敗 run の掃除が必要になった場合も、run ID とエラーを報告してからユーザーの指示で行う。
  `projects/*/runs/` は Git 管理外なので、消すと復元できない
