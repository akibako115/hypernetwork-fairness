---
name: gpu
description: GPU使用状況と実行中の学習プロセスを確認する
allowed-tools: Bash(nvidia-smi:*) Bash(ps:*) Bash(ssh:*)
---

GPU 使用状況、実行中の学習プロセス、リモート指定時のコンテナ状況を確認し、
新しい学習を開始できるか判定する。

このホストには旧 repo `/home/akiba/workspace/fairness` も置かれており、同じ GPU を使う。
**新 repo のプロセスだけを見て空きと判定しない。**

## ローカル

1. `nvidia-smi` で GPU 使用状況を表示する
2. 学習プロセスを確認する。`rg` はこのホストに入っていないので `grep -E` を使う

```bash
ps -eo pid,lstart,args | \
  grep -E '(projects[.][a-z_]+[.]run|src[.]experiments[.].*[.]train|src[.]training[.]worker|src/train[.]py)' | \
  grep -v grep
```

- `projects.<project>.run` … この repo の学習（`hypernet_e2e` / `hypernet_two_stage` 等）
- `src.experiments.*` / `src.training.worker` … 旧 repo の学習
- DataLoader worker は親と同じ引数で出るため、同一 run が複数行に見える。PID と開始時刻で親を判別する

3. 学習中プロセスがある場合は、PID、開始時刻、どちらの repo のものか、コマンド概要を報告する

`nvidia-smi` にプロセスが出ていても ps に該当が無い場合は、他ユーザーまたは Docker 内の
プロセスである。`docker ps` を確認し、勝手に kill しない。

## リモート（指定された場合）

ユーザーがサーバ番号（例: `ws3`）を指定した場合だけ、SSH 経由で確認する。

1. `ssh -o ConnectTimeout=10 kohkiakiba@192.168.1.[N+10] "nvidia-smi"`
2. `ssh -o ConnectTimeout=10 kohkiakiba@192.168.1.[N+10] "ps -eo pid,lstart,args | grep -E '(projects[.][a-z_]+[.]run|src[.]experiments[.].*[.]train|src[.]training[.]worker)' | grep -v grep || true"`
3. `ssh -o ConnectTimeout=10 kohkiakiba@192.168.1.[N+10] "docker ps --format 'table {{.Names}}\t{{.Status}}\t{{.Image}}'"`

SSH 接続に失敗した場合は再試行せず、ユーザーに ControlMaster セッションを確立するよう依頼する。
再試行の規約は [../train/remote.md](../train/remote.md) の Failure Handling を正本とする。

## 出力

結果を簡潔にまとめる。

- GPU ごとのメモリ使用量と空き
- 実行中のプロセス/コンテナの一覧（あれば。新 repo / 旧 repo / 不明 を区別する）
- 新しい学習を開始可能かどうかの判定（可能 / 注意して可能 / 不可）
- 不可または注意が必要な場合の理由
