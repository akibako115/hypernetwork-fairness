---
name: train
description: project 単位の学習をローカルまたはリモートで起動し、run directory 契約に沿って記録・監視する
allowed-tools: Bash(uv:*) Bash(nvidia-smi:*) Bash(ps:*) Bash(git:*) Bash(ssh:*) Bash(cat:*) Bash(ls:*) Bash(du:*) Bash(date:*) Bash(nohup:*) Bash(tmux:*) Bash(tail:*) Read Write Edit
---

対象 project の `README.md`、`AGENTS.md`、`docs/run-artifacts.md`、`docs/path-conventions.md` を
読み、起動方法と出力契約を特定する。**契約の正本は project 側の文書**であり、この skill は
起動・確認・承認の手順だけを持つ。

## この repo の前提（旧 repo との違い）

旧 repo `/home/akiba/workspace/fairness` の train skill をそのまま持ち込まない。機構ごと無い。

| | 旧 repo | この repo |
| --- | --- | --- |
| 起動 | `scripts/train.sh` + `CONDITION` / `SEED` / `DATE_TAG` env | `uv run python -m projects.<project>.run <hydra override>` |
| データ | `DATA_FOLDER=/data` | `paths.data_dir=data`（repo root 基準）。env で渡さない |
| 出力 | `logs_<dataset>/train/...` と `docs/experiments/*.md` | `projects/<project>/runs/<run-id>/` が正本。人手の実験記録 md は作らない |
| 再開 | `--resume` / `--reconcile` | **無い。** 再実行は必ず新しい run directory |
| 単位 | 親 run + stage + attempt | `hypernet_e2e` は 1 run = 1 fit、`hypernet_iterative` は 1 run = warmup + 任意回数の cohort stage |

## 条件の準備

project、experiment preset、seed、trainer、data override、実行先と GPU、logger、概算時間を特定する。
候補は `projects/<project>/configs/` を読んで確認する。

preset と override の使い分けは、**その条件が config の構造を変えるかどうか**で決める。

- **preset にする**: model や datamodule の差し替え、目的関数の系統、callback の増減、dataset の
  変更など、`defaults` が変わるもの。既存 preset に無い構造を override で捻じ込まない
- **override でよい**: seed、epoch 数、learning rate、`iteration.*` のような**既存の key の値だけを
  変える**もの。スカラの水準を振るたびに preset を増やすと `configs/experiment/` が組み合わせ爆発する

override で振った値は run directory の `config.yaml` が正本として記録する。run-id には experiment 名
までしか入らないので、**同じ preset で値だけを変えた run は run-id では区別できない**。どの水準を
振ったかは起動時に報告し、分析側の `runs.md` に run-id と対応付けて残す。

`run_dir` は指定しない。`run.py` が `runs/<run-id>/` を予約して config に注入する。

dry-run は Hydra の設定表示を使う。

```bash
uv run python -m projects.hypernet_e2e.run --cfg job --resolve <overrides>
```

fit も run directory 作成も行わない。ただし `weighting=inverse` の class weight は実行時に
train split から算出されるため、この出力には出ない（`model.loss_fn.class_weight` は null のまま）。

## 起動前の確認

```bash
uv run pytest projects/<project>/tests -m preflight -q
git status --short
```

- preflight は `run.py` 内でも走り、落ちると **directory だけ残った failed run** になる。先に潰す
- `git status --short` で今回反映される変更を確認する。未コミットの変更は `run.json` の
  `git_commit` と食い違うので、意図的かどうかを明示する
- GPU と実行中プロセスは [gpu skill](../gpu/SKILL.md)。旧 repo の学習が同じ GPU を使っていることがある

## 承認

**起動コマンド全文と全体計画を提示し、明示承認を取る。** 含めるもの:

project / experiment / 全 seed / trainer と主要 override / 実行先と GPU / 概算時間 /
既存プロセスへの影響 / 生成される run directory の数。

複数 seed はまとめて承認し、承認範囲内なら順次起動してよい。条件変更・後日の再開は再承認する。
改修依頼は学習起動の承認ではない。

## 起動と確認

ローカルは [launch.md](launch.md)、remote は [remote.md](remote.md)、
ws の新規/再構築 Docker は [ws_docker.md](ws_docker.md) を読む。

**回線をまたぐ転送（`rsync` / `scp`）はエージェントが実行しない。** コマンドを組み立てて
提示し、ユーザーが直接打つ。規約と提示の形は [transfer.md](transfer.md) が正本で、
`rsync` はこの skill の `allowed-tools` にも入れていない。承認済みの学習起動（ssh +
detached docker）と、`du` / `ls` / `ps` のような読み取りは転送ではないので対象外とする。

起動後、run directory を特定して次を確認する。

- `preflight.json` の `exit_code` が 0
- `run.json` の `status` が `running`、`git_commit` と `seed` が意図どおり
- `config.yaml` と `data_manifest.json` が生成され、split の行数・hash が想定どおり
- `callbacks.model_checkpoint.dirpath` が run の `checkpoints/` を指している
- GPU にメモリが載っている（プロセス存在・container `Up` だけで成功とみなさない）
- controller log に traceback が無く、最初の validation まで進んでいる

## 記録

**run directory が正本。** `run.json` / `config.yaml` / `data_manifest.json` / `preflight.json` /
`metrics/fit.json` の内容を人手の文書へ複製しない。人が残すのは run ID と「なぜこの条件か」だけで、
それも必要になった時点で書く。

`projects/*/runs/` は Git 管理外。生成済み run は不変で、上書き・改名・削除をしない。
失敗した run も `status: failed` のまま残す。

## 失敗と再実行

[failures.md](failures.md)。再開機構は無いので、途中失敗した条件をやり直す場合も新しい run になる。
前の run directory は消さず、どの run の再実行かを報告に残す。

## 完了条件

- 起動コマンドが project の公開入口（`python -m projects.<project>.run`）と一致している
- 親実行単位で明示承認を取っている
- detached で起動し、controller log と run directory の両方を特定できている
- 「起動済み」と「完了済み」を区別して報告している
- preflight・run.json・GPU を確認済みで、未確認の項目はそう明記している
