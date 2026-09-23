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

**`--user` を指定せず、image の既定ユーザーで実行する。** `docker/build.sh` が host の UID/GID を
image の `app` ユーザーに焼くので、container が書いた run・log は host の `kohkiakiba` の所有になる。
`--user 0:0` で起動すると bind mount 先が root 所有になり、host からは消せず、次に root 以外で
起動した学習は `runs/` に run directory を作れなくなる（2026-09-23 に ws11 で実際に起きた）。
baseline の container が root で動いていても、それを引き継がない。

起動の前に、image の uid が host と一致することを確かめる。一致しなければ、その host で
`docker/build.sh` を実行し直してから起動する。

```bash
ssh -o ConnectTimeout=10 -o ControlPath=~/.ssh/sockets/%r@%h-%p kohkiakiba@192.168.1.<N+10> \
  "id -u; docker run --rm <image> id -u 2>/dev/null | tail -1"
```

DataLoader worker を使うので `--ipc=host`、または十分な `--shm-size` を明示する。`ipc=private` の
既定 64 MiB は worker を `Bus error` で殺す。W&B の認証は host の `~/.netrc` を `app` の home へ
read-only で mount して渡す。

```bash
ssh -o ConnectTimeout=10 -o ControlPath=~/.ssh/sockets/%r@%h-%p kohkiakiba@192.168.1.<N+10> 'bash -s' <<'REMOTE_SCRIPT'
set -euo pipefail
cd <remote_path>
mkdir -p run_logs
docker run --ipc=host --gpus 'device=<gpu>' -d \
  --name <container_name> \
  -v <remote_path>:/workspaces/hypernet-fairness \
  -v <ws_data_path>:/workspaces/hypernet-fairness/data:ro \
  -v /home/kohkiakiba/.netrc:/home/app/.netrc:ro \
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
   ls -1dt <remote_path>/projects/hypernet_e2e/runs/*/ | head -3; \
   stat -c '%U %n' \$(ls -1dt <remote_path>/projects/hypernet_e2e/runs/*/ | head -1)"
```

起動成功と報告する前に、すべて満たすことを確認する。

- container が `Up` のまま
- 選んだ GPU にメモリが載っている
- log に `GPU available: True` と、意図した weighting / preset が出ている
- 予約された run directory に `preflight.json`（`exit_code` 0）と `run.json`（`status: running`）がある
- `Bus error`、`DataLoader worker ... exited unexpectedly`、permission error、traceback が無い
- 予約された run directory の所有者が `kohkiakiba` である（`root` なら `--user` の指定が残っている）

worker が bus error を起こした場合、失敗 container と成功 container の `IpcMode` と `ShmSize` を
比べる。成功側の IPC mode で `retryN` の名前を付けて再起動し、失敗側の log は残す。
失敗が確定した exited container だけを、ID とエラーを記録してから削除する。

## root 所有のファイルが残ったとき

host の `kohkiakiba` では消せない・書けない。root の container を **1 回だけ**使い、対象を明示して
所有者を戻す（消す場合も同じ container で消す）。mount は repo だけにし、`rm` の対象は run-id を
1 本ずつ書く。glob や repo root への `rm -rf` を書かない。

```bash
ssh -o ConnectTimeout=10 -o ControlPath=~/.ssh/sockets/%r@%h-%p kohkiakiba@192.168.1.<N+10> \
  "docker run --rm --user 0:0 -v <remote_path>:/repo <image> \
     chown -R <remote_uid>:<remote_gid> /repo/projects/hypernet_e2e/runs /repo/projects/hypernet_iterative/runs /repo/run_logs; \
   find <remote_path> -user root | wc -l"
```

`<remote_uid>:<remote_gid>` は remote で `id -u` / `id -g` を打った値にする（ws11 では `10090:10091`）。
`$(id -u)` と書くと、ssh の引数を組み立てる時点でローカルの値に展開される。最後の件数が 0 に
なっていることを確かめる。
