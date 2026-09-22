# runs

run 本体もログも Git 管理外なので、この表が参照の正本となる。同じ preset で値だけを override した
run は run-id では区別できないため、振った水準はここに残す。

run-id・状態・experiment・W&B URL の表は、`run.json` から生成できる。

```bash
uv run python analysis/common/studies.py iterative_probe
```

ただし `study` は本実験より後に入れた key なので、**下の run は `study` を持たない**。移行が済むまで、
下の手書きの表が正本になる。ここに人が書くのは「なぜこの条件なのか」のほうとする。

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

## 比較対象（baseline）

global 指標を並べるための通常 ResNet。`projects/hypernet_e2e/runs/` 側にある。

| run-id | 条件 | epoch |
|---|---|---|
| `20260921T103036Z-resnet-chexpert-s42-5538` | ResNet-50 全体を ERM、`weighting=inverse` | 30 |

`data_manifest.json` の train / val の sha256、optimizer（AdamW lr 1e-4 / wd 0.01）、batch size 128、
class weight `[0.201358, 1.798642]`、seed、transform が本実験と一致する。違うのは学習の中身
（全体 ERM 30 epoch か、Spatial LoRA + cohort GroupDRO 12 epoch か）だけになる。

この run は CSVLogger を付けずに回しているため、epoch 推移は `wandb/<run>/run-*.wandb` から読む。
`collect.py --baseline <run-id>` がそれを行う。

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

群ごと・交差群ごとの公平性指標は run artifact に無い（属性ごとの worst と gap までしか記録して
いない）。`analysis/common/predictions.py` が `selected_checkpoint`（iterative）と `best_val_auroc`（baseline）
を test split で推論し、`cache/<run-id>_test.npz` に予測を置く。群の切り方は `groups.py` が決める。

実行ログは `run_logs/` にあり、ファイル名は run-id と対応しない。必要なら run-id で引く。

```bash
grep -l <run-id> run_logs/*.log
```
