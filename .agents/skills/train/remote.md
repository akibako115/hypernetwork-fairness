# Remote Training

ユーザーが `ws7` のようにサーバを明示した場合だけ remote で実行する。

**回線をまたぐ転送はエージェントが実行しない。** コマンドを提示し、ユーザーが打つ。
規約と提示の形は [transfer.md](transfer.md) を正本とする。この文書の `rsync` は
すべて**提示用のテンプレート**であり、エージェントが Bash で走らせるものではない。

## サーバのアドレス

`wsN` は `kohkiakiba@192.168.1.[N+10]` に接続する。例: `ws7` → `kohkiakiba@192.168.1.17`。

## SSH ControlMaster

対話的な SSH ログインを再利用できない場合は、ユーザーに ControlMaster socket の作成を依頼する。

```bash
mkdir -p ~/.ssh/sockets
ssh -MNf \
  -o ControlMaster=yes \
  -o ControlPath=~/.ssh/sockets/%r@%h-%p \
  -o ControlPersist=24h \
  kohkiakiba@192.168.1.17
```

以後の `ssh` と `rsync` は同じ `ControlPath` を使い、操作のたびに新しいセッションを張らない。

## Failure Handling

すべての `ssh` と `rsync` に `-o ConnectTimeout=10` を明示する。付けないと、落ちたホストや
飽和した回線で TCP timeout まで固まり、1 回の試行が数分を捨てて何も報告しない。

再試行は限定する。

- **認証・鍵・権限の失敗は再試行しない。** 停止して、上の ControlMaster socket の作成を
  ユーザーに依頼する。同じコマンドを繰り返しても資格情報は直らない
- **一時的な失敗（`ConnectTimeout`、`Connection reset`、切断）は 1 回だけ再試行する。**
  2 回目も失敗したら、ホスト・コマンド・エラーを報告して止まる。3 つ目の変種を試さない
- **副作用のある操作を盲目的に再実行しない。** `docker run` と転送は、接続が切れる前に
  効いている可能性がある。先に remote の実際の状態を見る（起動なら
  `docker ps -a --filter name=<container_name>`、転送なら `--dry-run --stats` の提示から）。
  名前を変えた再起動は、1 回の切断を 1 GPU 上の 2 つの学習に変える。確認済みの失敗に対する
  再起動は [ws_docker.md](ws_docker.md) の `retryN` 規則で命名する

失敗した転送を再提示するときも `-v` を付けない。追跡ツリー全体の verbose 出力は数千行を吐き、
それを繰り返すことが実際に session を食い潰す。

## データ本体の扱い

**この repo は `data/` を repo の中に持つ。** `data/chexpert` だけで 5.3 GB あり、
`rsync ./` を素朴に打つとそれを丸ごと送る。下の exclude を削らない。

remote 側に CheXpert が無い場合、その移送は学習の付随作業ではない。転送手段そのものを
ユーザーと決める（[transfer.md](transfer.md) の「規模による扱い」）。学習用の同期に
データを紛れ込ませない。

## コード同期（提示用テンプレート）

再現に必要な追跡ファイルだけを送る。`projects/`、`tests/`、`pyproject.toml`、`uv.lock`、
`docker/`、`dockerfile`、`.agents/` が対象で、実測で数 MB に収まる。

まず dry-run を提示する。

```bash
rsync -azn --stats \
  -e "ssh -o ConnectTimeout=10 -o ControlPath=~/.ssh/sockets/%r@%h-%p" \
  --bwlimit=12500 --timeout=60 --partial \
  --inplace --no-times --no-perms --omit-dir-times \
  --exclude='.git' \
  --exclude='.venv' \
  --exclude='__pycache__' \
  --exclude='/data' \
  --exclude='/.claude' \
  --exclude='/.direnv' \
  --exclude='/projects/*/runs' \
  --exclude='/analysis/*/runs' \
  --exclude='/analysis/*/cache' \
  --exclude='/run_logs' \
  --exclude='/remote_logs' \
  --exclude='/wandb' \
  --exclude='/references' \
  --exclude='*.log' \
  --exclude='.mypy_cache' \
  --exclude='.pytest_cache' \
  --exclude='.ruff_cache' \
  ./ kohkiakiba@192.168.1.[N+10]:<remote_path>/
```

`Total transferred file size` が数 MB であることを確認してから、`-azn` を `-az` に変えた
本番を提示する。GB 単位になっていたら exclude が効いていないので、本番を提示しない。

2026-09-21 にローカルで測った実測は 542 files / 3.9 MB。`/.claude` を外すと worktree の
checkout が丸ごと乗って数倍になるので、この行を消さない。exclude を変えたら、ネットワークを
使わないローカル dry-run で内訳を確認できる。

```bash
rsync -an --out-format='%n' <同じ exclude 群> ./ /tmp/rsync-probe/ | awk -F/ '{print $1}' | sort | uniq -c | sort -rn
```

metadata の権限・timestamp で `rsync` が code 23 を返すことがある。ファイル本体が届いていれば
失敗とみなす前に remote 側を確認する。この確認はエージェントが実行してよい。

```bash
ssh -o ConnectTimeout=10 -o ControlPath=~/.ssh/sockets/%r@%h-%p kohkiakiba@192.168.1.[N+10] \
  "cd <remote_path> && git status --short && ls projects/hypernet_e2e/configs/experiment"
```

## 起動

remote では detached container か detached `docker exec` を使う。SSH session と一緒に死ぬ
foreground Docker を使わない。新規 container は [ws_docker.md](ws_docker.md) のテンプレートを読む。
起動は転送ではないので、承認済みならエージェントが実行してよい。

```bash
ssh -o ConnectTimeout=10 -o ControlPath=~/.ssh/sockets/%r@%h-%p kohkiakiba@192.168.1.[N+10] \
  "docker exec -d <container> bash -lc '
    cd <container_repo_path> &&
    mkdir -p run_logs &&
    uv run python -m projects.hypernet_e2e.run experiment=<preset> seed=<seed> \
      > run_logs/<name>.log 2>&1
  '"
```

監視:

```bash
ssh -o ConnectTimeout=10 kohkiakiba@192.168.1.[N+10] \
  "docker exec <container> bash -lc 'tail -n 80 <container_repo_path>/run_logs/<name>.log'"
```

## 結果の回収（提示用テンプレート）

分析はローカルで行うので、回収は remote run の一部である。回収先は同じ
`projects/<project>/runs/` であり、ローカル run と同じツリーに並ぶ。

- **metadata を先、checkpoint を後。** `--exclude='*.ckpt'` の回収は数 MB で、
  `run.json`・`config.yaml`・`data_manifest.json`・`preflight.json`・`metrics/` を運ぶ。
  それで分析対象を決めてから、選んだ run の `checkpoints/` だけを 1 run ずつ提示する
- **両側を glob しない。** `runs/*/ ./runs/*/` は、ローカルに 2 つ目が存在した瞬間に最後の
  引数が destination になる。フルパスを書く
- **既にローカルにあるファイルを上書きしない。** run directory は不変なので、同じ run ID の
  ファイルに差分が出ること自体が異常であり、上書きは事故の隠蔽になる。`--ignore-existing` で
  機械的に防ぎ、差分の有無は `-ni` の dry-run 出力で確認する。
  途中で切れた転送が最終名のまま残ると `--ignore-existing` がそれを正しいファイルとして
  スキップするので、`--partial` ではなく `--partial-dir=.rsync-partial` を使う
  （`--partial-dir` は `--inplace` と併用できない。回収側では `--inplace` を付けない）

```bash
rsync -azni --stats --exclude='*.ckpt' \
  -e "ssh -o ConnectTimeout=10 -o ControlPath=~/.ssh/sockets/%r@%h-%p" \
  --bwlimit=12500 --timeout=60 --partial-dir=.rsync-partial --ignore-existing \
  kohkiakiba@192.168.1.[N+10]:<remote_path>/projects/hypernet_e2e/runs/<run-id>/ \
  ./projects/hypernet_e2e/runs/<run-id>/
```

`-n` を外した本番を提示する前に、dry-run の出力に既存ファイルへの更新
（`>f` で始まり、かつローカルに既にある path）が無いことを確認する。あれば転送せず、
どちらが正しいかを先に確定させる。

回収する run の checkpoint サイズは、提示の前に remote 側で測る。これはエージェントが実行してよい。

```bash
ssh -o ConnectTimeout=10 -o ControlPath=~/.ssh/sockets/%r@%h-%p kohkiakiba@192.168.1.[N+10] \
  "du -sh <remote_path>/projects/hypernet_e2e/runs/<run-id>/checkpoints"
```

## ws07 メモ

- アドレス: `kohkiakiba@192.168.1.17`
- 旧 repo の配置: `/home/file_server2/kohkiakiba/fairness`（container 内 `/workspaces/fairness`）
- **この repo の remote 配置・container 名・データ置き場は未確定。** 初回は ws_docker.md の
  baseline 確認から決め、決まった値をこの節に追記する
