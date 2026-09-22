# 実験の進め方

実験 1 本を、思いついてから結論にするまでの手順書。**何をどの順でやるか**だけを書き、
個々の契約（run artifact の中身、分析コードの規約、データの分割）は既存の文書を正本とする。

```text
仮説を作る  →  条件を決める  →  回す  →  記録を読む  →  分析する  →  結論を書く
analysis/<slug>/   dry_run=true    run/     runs/<run-id>/   results/ figures/   README.md
```

この repo には境界が 2 つあり、**交差する**。

| 境界 | 単位 | 意味 |
| --- | --- | --- |
| **project** | `projects/<project>/` | 実装の境界。fit 開始時に group が確定しているか否かで分かれる |
| **study（仮説）** | `analysis/<slug>/` | 問いの境界。1 つの仮説が両 project の run を並べることがある |

project は互いにコードを import しない。重複は許し、一致は `tests/golden/` の固定データで縛る。

---

## 1. 仮説を作る

**先に分析 package を作る。** 学習は `study=<slug>` を必須で要求し、`analysis/` に無い値を拒む。

```bash
mkdir -p analysis/my_new_study
```

- slug は **underscore**。`analysis/<slug>/` は import される package なので hyphen は使えない
- `README.md`（何を確かめたいのか）と `runs.md`（どの run を使うのか）を置く
- 結論を出すつもりが無い run（経路確認、当たり付け、1 epoch の動作確認）は package を作らず
  `study=scratch` で回す

---

## 2. 条件を決める（dry-run）

```bash
uv run python -m projects.hypernet_e2e.run \
  experiment=spatial_lora_chexpert_fc study=my_new_study seed=42 dry_run=true
```

fit も run directory 作成も行わず、**実行と同じ順序で**検証と class weight 解決を通してから
条件表を出す。`weighting=inverse` の class weight も実値で出る。表の末尾に CLI override の
一覧が出るので、preset の既定と今回振った値をここで分けて読む。

preset と override の使い分けは、**その条件が config の構造を変えるかどうか**で決める。

| | 例 |
| --- | --- |
| **preset にする** | model や datamodule の差し替え、目的関数の系統、callback の増減 |
| **override でよい** | seed、epoch 数、learning rate、`iteration.*` のような既存 key の値 |

---

## 3. 回す

長時間 run は detached で起動し、controller log を `run_logs/` に残す。

```bash
mkdir -p run_logs
NAME="e2e_fc_s42_$(date -u +%Y%m%dT%H%M%SZ)"
nohup uv run python -m projects.hypernet_e2e.run \
  experiment=spatial_lora_chexpert_fc study=my_new_study seed=42 \
  > "run_logs/${NAME}.log" 2>&1 < /dev/null &
```

### 複数 seed を振るとき

同じ条件を 2 本以上の seed で回すなら `logger.wandb.group=<条件の slug>` を渡す。W&B が
束ねて、chart で平均線と min/max の band になる。折れ線が N 本並ぶのと違い、**条件間の差が
seed の散らばりより大きいか**がそのまま読める。

```bash
GROUP="spatial_lora_fc_inverse"
for SEED in 42 43 44; do
  nohup uv run python -m projects.hypernet_e2e.run \
    experiment=spatial_lora_chexpert_fc study=my_new_study \
    seed="$SEED" logger.wandb.group="$GROUP" \
    > "run_logs/e2e_${GROUP}_s${SEED}.log" 2>&1 < /dev/null &
done
```

group 名に **seed を入れない**。入れると 1 run ずつ別 group になり、束ねる意味が消える。
単発 run には渡さない（1 run の group は band にならない）。別条件に使い回さない。

手順の詳細（tmux、GPU の分け方、二段学習、監視）は
[`.agents/skills/train/launch.md`](../.agents/skills/train/launch.md)。

---

## 4. 記録がどこに残るか

### run directory — `projects/<project>/runs/<run-id>/`

**不変。** 上書きも rename も削除もしない。再実行は必ず新しい run directory になる。

| file | 持つもの |
| --- | --- |
| `run.json` | 状態、seed、`study`、git commit、checkpoint、W&B への参照 |
| `config.yaml` | **解決済み**設定。振った水準の正本はここ |
| `checkpoints/` | `best_val_auroc_*.ckpt` と `last.ckpt` |
| `metrics/`, `stages/` | fit の結果 |

run-id には experiment 名までしか入らない。**同じ preset で値だけ変えた run は run-id で
区別できない**ので、条件の差は `config.yaml` の diff で読む。

```bash
diff -u projects/hypernet_e2e/runs/<前回>/config.yaml projects/hypernet_e2e/runs/<今回>/config.yaml
```

契約の正本は各 project の `docs/run-artifacts.md`。

### W&B

project は repo で 1 つ（`fairness_hypernet`）。1 つの仮説が両 project の run を並べるため。

| 欄 | 入るもの | 絞り方 |
| --- | --- | --- |
| `config.study` | 仮説 | dashboard で **group by `config.study`** |
| `job_type` | code project（`hypernet_e2e` / `hypernet_iterative`） | filter |
| `group` | **seed 反復・分散実行の束ね**（既定は空） | 束ねた run が band になる |
| `tags` | 人が付ける印。設定から読める条件は入れない | filter |

`group` を仮説に使わないのは、seed 反復という本来の用途を塞ぐため。

---

## 5. 分析する

`analysis/<slug>/` で行う。**生成物は一方向にしか流れない。**

```text
run artifact  →  cache/  →  results/  →  figures/  →  notebook
             predictions.py  collect.py   plots.py    主張と考察
                             groups.py
```

各 script は単独で走り、前段の生成物だけを入力に取る。1 辺が 1 コマンドになる。

```bash
# 1. 選択済み checkpoint の予測を cache する（古い cache は自動で作り直される）
uv run python analysis/common/predictions.py --study my_new_study --split test \
  --run-dir projects/hypernet_e2e/runs/<run-id> ...

# 2〜3. run artifact と cache から表を作る
uv run python analysis/my_new_study/collect.py <run-id> ... --baseline <run-id>
uv run python analysis/my_new_study/groups.py --split test

# 4. 表から図を作る
uv run python analysis/my_new_study/plots.py --split test
```

notebook は `results/` と `figures/` を読むだけにする。split CSV も checkpoint も直接読まない。
**kernel restart → 全実行が通る状態**で終える。通らない notebook は DAG のどこかを手で飛ばした
状態になっている。出力は Git 追跡しない（`nbstripout`）。

`cache/` `results/` `figures/` は再生成できるので Git 管理外。書き方の規約は
[`analysis/AGENTS.md`](../analysis/AGENTS.md)。

### `runs.md` に引用を書く

`study` が記録するのは「**何のために回したか**」だけで、1 run に 1 つしか付かない。
**どの分析がその run を引用したか**は多対多で、あとから増える（baseline は使い回すため）。
引用の正本は `analysis/<slug>/runs.md` が持つ。

run 一覧の表は生成できる。人が書くのは「**なぜこの条件なのか**」のほう。

```bash
uv run python analysis/common/studies.py my_new_study
```

---

## 6. 壊れていないことを確かめる

```bash
uv run pytest                      # 引数なしで repo 全体
uv run ruff check analysis projects
```

学習側と分析側で同じ意味を持つ指標は、**実装どうしを比べない**。`tests/golden/` の固定データと
一致することを見る。実装を突き合わせると、どちらが正しいのか分からないまま両方が動く。

---

## 落とし穴

| やらないこと | 理由 |
| --- | --- |
| `projects/*/runs/` を上書き・rename・削除する | run artifact は不変。結論の根拠が消える |
| project 間でコードを import する | 一致は golden で縛る。共有すると、片方の変更が黙ってもう片方の結論を変える |
| `study` を後から書き換える | 起動時の記録なので、書き換えた値はその意味を持たない。引用は `runs.md` へ |
| analysis の slug に hyphen を使う | package として import できなくなる |
| notebook に 20 行超の関数を置く | 出す先は `plots.py` か `groups.py` |
| `sys.path` を触る / `parents[2]` で root を数える | `rootutils` + `.project-root` に統一している |
| `config.yaml` を読まずに run-id で条件を語る | run-id は experiment 名までしか持たない |

---

## 関連文書

| 文書 | 正本とするもの |
| --- | --- |
| [`README.md`](../README.md) | repo 全体の構成、project の境界、再現性の方針 |
| [`analysis/README.md`](../analysis/README.md) / [`AGENTS.md`](../analysis/AGENTS.md) | 分析 package の構成と実装規約 |
| [`projects/*/docs/run-artifacts.md`](../projects/hypernet_e2e/docs/run-artifacts.md) | run directory の契約 |
| [`data_pipeline/README.md`](../data_pipeline/README.md) | 分割単位・比率・CV・最小セル枚数 |
| [`tests/golden/README.md`](../tests/golden/README.md) | golden の形式と更新規則 |
| [`.agents/skills/`](../.agents/skills/) | agent が従う作業手順（train / analyze / gpu / data-explore / migrate） |
