# runs

run 本体もログも Git 管理外なので、この表が参照の正本となる。同じ preset で値だけを override した
run は run-id では区別できないため、振った水準はここに残す。

run-id・状態・experiment・W&B URL の表は、`run.json` から生成できる。分析対象の表には、
`project` と repo root からの相対 `run path` も残す。

```bash
uv run python analysis/common/studies.py iterative_probe
```

本実験の 4 本は `study=iterative_probe` を持つ。baseline と先行の探り run は `study` を入れる前の run なので
持たない。加えて `study` が記録するのは「何のために回したか」だけなので、**どの run をこの分析が引用しているかは
この表が恒久的な正本**になる（baseline のように、別の study のために回した run も引用する）。
生成された表は補助として読む。ここに人が書くのは「なぜこの条件なのか」のほうとする。

## 本実験（2×2、seed 42）

warmup 2 epoch → stage01 5 epoch → stage02 5 epoch、cohort 10 クラスタ、`weighting=inverse`、
ResNet-50 ImageNet 初期化。ws11 で 4 本を並列実行し、4 本とも `succeeded`。

| run-id | project | run path | W&B | 変調範囲 | step size | 所要 |
|---|---|---|---|---|---|---|
| `20260924T112035Z-spatial-lora-iterative-chexpert-fc-s42-b81a` | `hypernet_iterative` | `analysis/iterative_probe/runs/20260924T112035Z-spatial-lora-iterative-chexpert-fc-s42-b81a` | [run](https://wandb.ai/kohki-akiba-kyushu-university/fairness_hypernet/runs/mw9f32gz) | fc | 1e-3 | 56 分 |
| `20260924T112045Z-spatial-lora-iterative-chexpert-fc-s42-4602` | `hypernet_iterative` | `analysis/iterative_probe/runs/20260924T112045Z-spatial-lora-iterative-chexpert-fc-s42-4602` | [run](https://wandb.ai/kohki-akiba-kyushu-university/fairness_hypernet/runs/0rnrgqua) | fc | 1e-2 | 56 分 |
| `20260924T112034Z-spatial-lora-iterative-chexpert-stage4-fc-s42-1caa` | `hypernet_iterative` | `analysis/iterative_probe/runs/20260924T112034Z-spatial-lora-iterative-chexpert-stage4-fc-s42-1caa` | [run](https://wandb.ai/kohki-akiba-kyushu-university/fairness_hypernet/runs/ctz9hw3a) | stage4, fc | 1e-3 | 66 分 |
| `20260924T112043Z-spatial-lora-iterative-chexpert-stage4-fc-s42-c755` | `hypernet_iterative` | `analysis/iterative_probe/runs/20260924T112043Z-spatial-lora-iterative-chexpert-stage4-fc-s42-c755` | [run](https://wandb.ai/kohki-akiba-kyushu-university/fairness_hypernet/runs/klvddrzc) | stage4, fc | 1e-2 | 66 分 |

stage01 / stage02 の `metrics.csv` は cohort ごとの `val/hidden_{loss,bacc,auroc,support}_NN` を持つ（warmup は
cohort が無いので持たない）。属性ごとの群については、worst と gap までしか記録していない。

ws11 側は `.git` を除外して同期しているため、`run.json` の `git_commit` は `null` である。

### 選択 checkpoint

分析に使うのは、各 run の `selected_checkpoint`（stage02 の中で val AUROC が最大の epoch）である。
12 epoch を通した best ではない。4 本とも、val AUROC の最大値は warmup の最終 epoch にある。
epoch は 0 始まりで、checkpoint 内の `epoch` / `global_step` と照合した。通し epoch は warmup 0–1、stage01 2–6、
stage02 7–11 と数える。

| run | stage02 の epoch | 通し epoch（全 12） | val AUROC |
|---|---|---|---|
| `…-fc-s42-b81a` | 4 | 11 | 0.8510 |
| `…-fc-s42-4602` | 2 | 9 | 0.8539 |
| `…-stage4-fc-s42-1caa` | 3 | 10 | 0.8482 |
| `…-stage4-fc-s42-c755` | 0 | 7 | 0.8495 |

baseline は `best_val_auroc_009.ckpt`（epoch 9、全 30）。

**`run.json` の score はこの表と一致しない。** `selected_checkpoint.score` と
`stages.*.checkpoints["val/auroc"].score` には、best の値ではなく各 stage の最終 epoch の値が入っている
（例: `4602` は best 0.8539 に対して記録は 0.8349。`b81a` は best が最終 epoch なので一致する）。
原因は `projects/hypernet_iterative/stage.py` が、score を `checkpoint.best_model_score` ではなく fit 後の `trainer.callback_metrics` から取っていること。
checkpoint の path は正しく、分析は score を読まないので、予測 cache には影響しない。
run 記録は不変なので書き換えていない。best の score は checkpoint 内の `ModelCheckpoint` の状態か、
`metrics.csv` から読む。

## 比較対象（baseline）

global 指標を並べるための通常 ResNet。e2e で回した run を、この package の `runs/` に取り込んである。

| run-id | project | run path | W&B | 条件 | epoch |
|---|---|---|---|---|---|
| `20260921T103036Z-resnet-chexpert-s42-5538` | `hypernet_e2e` | `analysis/iterative_probe/runs/20260921T103036Z-resnet-chexpert-s42-5538` | [run](https://wandb.ai/kohki-akiba-kyushu-university/fairness_hypernet/runs/41gsahly) | ResNet-50 全体を ERM、`weighting=inverse` | 30 |

`data_manifest.json` の train / val の sha256、optimizer（AdamW lr 1e-4 / wd 0.01）、batch size 128、
class weight `[0.201358, 1.798642]`、seed、transform が本実験と一致する。違うのは学習の中身
（全体 ERM 30 epoch か、Spatial LoRA + cohort GroupDRO 12 epoch か）だけになる。

この run は CSVLogger を付けずに回しているため、epoch 推移は `wandb/<run>/run-*.wandb` から読む
（`analysis/common/run_artifacts.read_wandb_history`。`global_training_curves.ipynb` が使う）。

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

`runs.md` の `run path` を開く。

- `run.json` — stage ごとの scalar metric と checkpoint
- `stages/*/metrics/metrics.csv` — epoch 推移の正本
- `stages/*/config.yaml` — 群ごとの class weight `[clusters, num_classes]`
- `artifacts/cohorts/cohortNN/cohort.json` — cohort の参照 checkpoint

群ごと・交差群ごとの公平性指標は run artifact に無い（属性ごとの worst と gap までしか記録して
いない）。`analysis/common/predictions.py` が `selected_checkpoint`（iterative）と `best_val_auroc`（baseline）
を test split で推論し、`cache/<run-id>_test.npz` に予測を置く。

実行ログは `run_logs/` にあり、ファイル名は run-id と対応しない。必要なら run-id で引く。

```bash
grep -l <run-id> run_logs/*.log
```
