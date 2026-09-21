# hypernet_iterative

学習の途中で group（hidden cohort）を作り直しながら、複数の fit stage を 1 run として進める
project です。ほかの project を import せず、この directory 配下だけで学習・評価できる形を
目標にします。

project の境界は **group が fit の開始時に確定しているか** で引いています。demographic 属性から
決まる group は [`hypernet_e2e`](../hypernet_e2e/README.md) が持ちます。ここが持つのは、直前
stage の checkpoint に依存して定義が変わる group です。

## 1 run の構造

1 run は warmup 1 回と、cohort 生成 + cohort stage の組を `iteration.stages` 回持つ parent run です。

```text
warmup fit ──▶ best val/auroc checkpoint
                 │
                 ├─ cohort01  その checkpoint の metadata encoder 出力を train で KMeans → assignments.parquet
                 └─ stage01   固定 cohort で GroupDRO fit ──▶ best val/auroc checkpoint
                                │
                                ├─ cohort02 …
                                └─ stage02 …
```

cohort の入力は画像 backbone の feature ではなく metadata embedding です。親 process が
checkpoint を `net` にだけ読み込み、CSV を行順のまま処理するので、cohort artifact は data
augmentation や sampler に依存しません。KMeans は train split だけで fit し、val / test には
同じ cluster center を適用します。

stage 間で何を引き継ぐかは固定です。

| 引き継ぐ | 引き継がない |
| --- | --- |
| `net` の tensor（`model.warm_start_checkpoint_path`） | optimizer / scheduler の状態 |
| 固定した cohort の group 割り当て | GroupDRO の adversarial weight（`adv_probs`） |
| | cohort の定義そのもの（stage ごとに作り直す） |

`uniform_group` は warm-start を持ちません（`supports_warm_start: false`）。各 fit は独立した
子 process (`stage.py`) で走るため、stage が GPU メモリや global state を持ち越すこともありません。

## 実装範囲

```text
workflow.py     # parent run の予約、warmup → cohort → stage の反復、artifact 記録
stage.py        # 子 process 側の単一 fit runner
validation.py   # cohort / warm-start / checkpoint 選択の組み合わせ検証
cohorts/build.py# metadata embedding の抽出と KMeans cohort artifact の保存
loss.py         # TaskLoss と group 系目的関数（uniform group / Group DRO / class-balanced）
data/           # ImageDataModule と、cohort sidecar を結合する CohortImageDataModule
models/         # ResNet と SpatialLoRAResNet（metadata encoder / Spatial LoRA / HyperLinear）
callbacks/      # metrics、fairness、hidden cohort 指標、GroupDRO diagnostics、text progress
```

## Configs

[`configs/train.yaml`](configs/train.yaml) が Hydra の入口です。反復の計画は `iteration` が
1 箇所で持ちます。

| key | 内容 |
| --- | --- |
| `iteration.warmup_epochs` | warmup fit の epoch 数 |
| `iteration.stages` | cohort 生成 + cohort stage を繰り返す回数 |
| `iteration.stage_epochs` | 各 cohort stage の epoch 数 |
| `iteration.clusters` | KMeans の k。cohort の group 数の正本 |
| `iteration.n_init` | KMeans の初期化回数 |
| `iteration.cohort_training_strategy` | `group_dro` / `group_dro_balanced` / `uniform_group` |
| `iteration.cohort_checkpoint_selection` | `global_auroc_bacc` / `hidden_min_auroc` |

parent run が実際に使う cohort stage の設定は、`workflow.cohort_stage_config()` が warmup の
解決済み config と `iteration.*` から組み立てます。`configs/training_strategy/`、
`configs/cohort_definition/`、`configs/checkpoint_selection/` と
`experiment=spatial_lora_iterative_cohort_chexpert` は、その stage config が満たすべき形を
宣言として持ち、config test が突き合わせる対象です。**parent run の挙動を変えるときは
`iteration.*` を振ります。** これらの group を CLI で振っても warmup 側にしか効きません。

checkpoint 選択は 2 通りあります。どちらも stage の選択基準（`run.json` に載る checkpoint）は
global AUROC のままで、差は補助保存される checkpoint です。

| 選択 | 追加で保存する checkpoint |
| --- | --- |
| `global_auroc_bacc` | `val/bacc` 最良 |
| `hidden_min_auroc` | `val/hidden_min_auroc`（worst-group AUROC）最良 |

`stage.py` は `val/auroc` を monitor する checkpoint callback をちょうど 1 つ要求します。
ここで選ばれた checkpoint が次 stage の warm-start と cohort 生成の参照になるため、monitor を
差し替えた run や checkpoint callback を増やした run は、別基準の checkpoint が黙って使われる
前に失敗します。

## 学習起動

計画だけを確認するときは `dry_run=true` を付けます。fit は起こしません。

```bash
uv run python -m projects.hypernet_iterative.run dry_run=true iteration.stages=3
```

本番は parent run directory を標準出力へ出します。CPU で最小構成を確かめる例は次のとおりです。

```bash
uv run python -m projects.hypernet_iterative.run \
  trainer=cpu iteration.warmup_epochs=1 iteration.stages=1 iteration.stage_epochs=1 \
  iteration.clusters=2 iteration.n_init=2 \
  data.num_workers=0 data.persistent_workers=false data.prefetch_factor=null
```

epoch ごとの metric は parent run が持つ 1 つの W&B run（project `fairness_hypernet_iterative`）に
集約します。子 process は `logger=False` で走り、自分では W&B run を作りません。送信せずに
試すときは `logger=none` を付けます。実行ログは parent run の `logs/train.log` に残ります。

## 出力の保存先

実データは repo 直下の `data/chexpert/` に置き、学習出力は
`projects/hypernet_iterative/runs/<run-id>/` に置きます。両者を混ぜません。directory 構成と
必須記録は [run artifact 契約](docs/run-artifacts.md) を正本とします。

## 運用規約

- 他 project から import しません。他 project 用の共有 module をここに置きません。
- run は不変です。既存 run の再開・上書きをせず、再実行は必ず新しい run directory を作ります。
  途中で失敗した run も `run.json` に `failed` を残します。
- 出力は `runs/<run-id>/` 配下だけに書きます。`data/chexpert/` は固定入力だけを持ちます。
- cohort は固定入力ではなく、生成した run の artifact です。参照 checkpoint の絶対 path と
  SHA-256、KMeans の設定を `cohort.json` に残し、既存 artifact directory へは上書きしません。
- 固定 cohort を使う stage は `trainer.devices=1` かつ `trainer.num_nodes=1` に限ります。
  group ごとの集計を 1 process に閉じるためで、`validation.py` が強制します。
- stage config は parent が組み立てますが、子 process 側でも `validate_training_config` で
  必ず再検証します。parent の生成結果を無検証で信用しません。
- 学習前に project-local の golden preflight を実行し、結果と golden の hash を
  `preflight.json` に残します。失敗した場合は fit を始めません。
- 比較条件は preset として project 内に持たせます。存在しない条件を既存 preset へ override で
  無理に載せません。
- 数値挙動に関わる既存コードを移植する際は、実装を整理・書き換えません。説明の追記は
  許可しますが、挙動変更と同じ変更に混ぜません。docstring は日本語で書きます。

`weighting: inverse` のとき、`workflow.py` が parent run を予約する前に train split の target
頻度から class weight を解決し、warmup と全 cohort stage が同じ値を使います。

## テスト

```bash
uv run pytest projects/hypernet_iterative/tests -q
uv run ruff check projects/hypernet_iterative
```
