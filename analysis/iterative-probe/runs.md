# runs

run 本体もログも Git 管理外なので、この表が参照の正本となる。同じ preset で値だけを override した
run は run-id では区別できないため、振った水準はここに残す。

## 本実験（2×2、seed 42）

warmup 2 epoch → stage01 5 epoch → stage02 5 epoch、cohort 10 クラスタ、`weighting=inverse`、
ResNet-50 ImageNet 初期化。ws11 の GPU 5〜8 で並列実行し、4 本とも `succeeded`。

| run-id | 変調範囲 | step size | 所要 |
|---|---|---|---|
| `20260921T103222Z-iterative-s42-d24d` | fc | 1e-3 | 54 分 |
| `20260921T103225Z-iterative-s42-3166` | stage4, fc | 1e-3 | 64 分 |
| `20260921T103219Z-iterative-s42-e992` | fc | 1e-2 | 54 分 |
| `20260921T103225Z-iterative-s42-fac5` | stage4, fc | 1e-2 | 64 分 |

コードは local の `8c33f20` に、`configs/experiment/spatial_lora_chexpert_{fc,stage4_fc}.yaml` の
2 ファイルを加えた状態。ws11 側は `.git` を除外して同期しているため、`run.json` の `git_commit` は
`null` である。

## 先行する探り run

| run-id | 変調範囲 | step size | epoch | 状態 |
|---|---|---|---|---|
| `20260921T081141Z-iterative-s42-5752` | stage3, stage4, fc | 1e-3 | stage 2 | succeeded |

step size が小さく GroupDRO が動かなかった run。変調範囲も epoch 数も本実験と違うので、直接の
対照にはならない。本実験の 1e-3 条件がその役割を引き継ぐ。

## 失敗した run

| run-id | 原因 |
|---|---|
| `20260921T102859Z-…-4fdb`, `…102900Z-…-99a7`, `…102902Z-…-ce17`, `…102903Z-…-e1e1` | container に W&B の API key が無く `wandb.init` で停止。host の `~/.netrc` を mount して再実行した |

これより前に、repo 内の `data` symlink が container 内で解決できず split CSV を読めなかった試行が
4 本あるが、run directory 予約の前に停止したため run-id は無い。

## 数値の出どころ

`projects/hypernet_iterative/runs/<run-id>/` を run-id で開く。

- `run.json` — stage ごとの scalar metric と checkpoint
- `stages/*/metrics/metrics.csv` — epoch 推移の正本
- `stages/*/config.yaml` — 群ごとの class weight `[clusters, num_classes]`
- `artifacts/cohorts/cohortNN/cohort.json` — cohort の参照 checkpoint

実行ログは `run_logs/` にあり、ファイル名は run-id と対応しない。必要なら run-id で引く。

```bash
grep -l <run-id> run_logs/*.log
```
