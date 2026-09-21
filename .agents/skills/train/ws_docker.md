# ws Docker Training

ユーザーが `wsN` を明示したときだけ使う。`<...>` はすべて実際の値に置き換える。
接続と再試行の規約は [remote.md](remote.md) を正本とする。

## 1. baseline を調べる

推測で `docker run` を組まない。継続・再試行の前に、直近で成功した container の image・mount・
user・IPC mode を見る。この repo の remote container がまだ無い場合は、旧 repo の
`fairness-chexpert` が最も近い baseline になる。

```bash
ssh -o ConnectTimeout=10 -o ControlPath=~/.ssh/sockets/%r@%h-%p kohkiakiba@192.168.1.<N+10> \
  "docker inspect --format 'image={{.Config.Image}} user={{.Config.User}} ipc={{.HostConfig.IpcMode}} shm={{.HostConfig.ShmSize}} {{range .Mounts}}{{.Source}}:{{.Destination}}:{{.Mode}} {{end}}' <baseline_container>"
```

GPU の空きと、進行中の学習プロセスを確認する（[gpu skill](../gpu/SKILL.md)）。

```bash
ssh -o ConnectTimeout=10 -o ControlPath=~/.ssh/sockets/%r@%h-%p kohkiakiba@192.168.1.<N+10> \
  "nvidia-smi --query-gpu=index,name,memory.used,memory.total --format=csv,noheader; \
   ps -eo pid,lstart,args | grep -E '(projects[.][a-z_]+[.]run|src[.]experiments[.].*[.]train)' | grep -v grep || true"
```

## 2. image と mount

image は repo の `dockerfile` から作る。`docker/build.sh` が `UID`/`GID` を渡して
`hypernet-fairness` を tag する。ローカルで動いた image を remote でも同じ手順で作る。

この repo は **`data/` を repo tree の中に置く**。container 内の repo path を
`/workspaces/hypernet-fairness` としたとき、データはその下の `data/` に見えている必要がある。
`docker/run.sh` は host の `$DATA_FOLDER` を `/workspaces/hypernet-fairness/data` に mount する
設計なので、ws 側では host のデータ置き場（`chexpert/` を含む directory）を `DATA_FOLDER` に指定する。

コードを bind mount するか image に焼くかは baseline に合わせる。bind mount にする場合、
host 側の repo は [remote.md](remote.md) の rsync で同期したツリーを使う。その転送は
エージェントが実行せず、[transfer.md](transfer.md) に従ってコマンドを提示するだけにする。

## 3. GPU を固定して 1 run 起動する

baseline が root 実行かつ host 所有の bind mount なら `--user 0:0` を維持する。DataLoader worker を
使うので `--ipc=host`、または十分な `--shm-size` を明示する。`ipc=private` の既定 64 MiB は
worker を `Bus error` で殺す。

```bash
ssh -o ConnectTimeout=10 -o ControlPath=~/.ssh/sockets/%r@%h-%p kohkiakiba@192.168.1.<N+10> 'bash -s' <<'REMOTE_SCRIPT'
set -euo pipefail
cd <remote_path>
mkdir -p run_logs
docker run --user 0:0 --ipc=host --gpus 'device=<gpu>' -d \
  --name <container_name> \
  -v <remote_path>:/workspaces/hypernet-fairness \
  -v <ws_data_path>:/workspaces/hypernet-fairness/data:ro \
  <image> bash -lc '
    cd /workspaces/hypernet-fairness
    set -euo pipefail
    env UV_CACHE_DIR=/tmp/uv-cache \
      uv run python -m projects.hypernet_e2e.run experiment=<preset> seed=<seed> \
      > run_logs/<run_name>.log 2>&1
  '
REMOTE_SCRIPT
```

- `CONDITION` / `SEED` / `DATE_TAG` / `DATA_FOLDER` の env は使わない。条件は Hydra override で表す
- run directory は `run.py` が `projects/<project>/runs/` に予約する。`RUN_DIR` を外から与えない
- GPU が少ないまま複数 seed を回す場合は、条件ごとに 1 container とし、次の seed は前の seed が
  正常終了した後に `&&` で続ける。seed ごとに container 名・controller log 名を分ける
- `data` を `:ro` で mount すると、学習は問題ないが、将来 cohort 生成を remote で行う場合は
  書き込み先が run directory 側であることを確認する

## 4. 起動を検証する

10〜60 秒後に確認する。container が `Up` なだけでは不十分。

```bash
ssh -o ConnectTimeout=10 -o ControlPath=~/.ssh/sockets/%r@%h-%p kohkiakiba@192.168.1.<N+10> \
  "docker ps -a --filter name=<container_name> --format 'table {{.Names}}\t{{.Status}}'; \
   nvidia-smi --query-gpu=index,memory.used --format=csv,noheader; \
   tail -n 80 <remote_path>/run_logs/<run_name>.log; \
   ls -1dt <remote_path>/projects/hypernet_e2e/runs/*/ | head -3"
```

起動成功と報告する前に、すべて満たすことを確認する。

- container が `Up` のまま
- 選んだ GPU にメモリが載っている
- log に `GPU available: True` と、意図した weighting / preset が出ている
- 予約された run directory に `preflight.json`（`exit_code` 0）と `run.json`（`status: running`）がある
- `Bus error`、`DataLoader worker ... exited unexpectedly`、permission error、traceback が無い

worker が bus error を起こした場合、失敗 container と成功 container の `IpcMode` と `ShmSize` を
比べる。成功側の IPC mode で `retryN` の名前を付けて再起動し、失敗側の log は残す。
失敗が確定した exited container だけを、ID とエラーを記録してから削除する。
