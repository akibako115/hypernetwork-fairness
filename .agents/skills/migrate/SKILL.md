---
name: migrate
description: 旧 fairness repo のコードを、実験ごとに完全独立した project 構成へ対話的に移植する
allowed-tools: Bash(uv:*) Bash(git:*) Bash(cat:*) Bash(ls:*) Bash(grep:*) Bash(find:*) Bash(diff:*) Bash(wc:*) Bash(cp:*) Bash(mkdir:*) Read Write Edit
---

移植元は `/home/akiba/workspace/fairness`（以下 旧 repo）。**読み取り専用**。
方針の根拠と実測値は [rationale.md](rationale.md) を参照する。

## 目的と完了条件

実験ごとに完全独立した project へコードを移す。重複は許容し、project 間の import と
共有ライブラリへの依存をやめる。移植はユーザーのコード理解を兼ねるので**対話的に進める**。

完了条件:

- `projects/hypernet_e2e` と `projects/hypernet_iterative` が単独で学習・評価できる
- 各 project の preflight が `tests/golden/` を再現する
- 同一 seed・同一 split・同一設定で 1 epoch 学習し、loss と val/auroc が旧実装と一致する

## 移植先の構造

```
projects/
  _template/           # models/data/callbacks 込みの実体ある雛形
  hypernet_e2e/        # e2e 学習 + サブグループ最適化。1 run = 1 fit
  hypernet_iterative/  # 反復で学習プロセスが入る系。1 run = N stage
    models/ data/ callbacks/ module.py losses.py cohorts/
    configs/ scripts/ tests/ run.py eval.py
analysis/              # 比較の場。1つだけ。新旧両方の logs を読む
tests/golden/          # 期待値データ。import されない
data_pipeline/         # preprocess.py + create_*.py（/data を作る）
docs/  docker/  .agents/skills/
```

- **`src/` を作らない。** この名前は旧 repo の過去 run 復元専用に温存する
  （97 個の保存済み config が `_target_: src.models.*` を指している）
- project 間 import と、project 外の共有実装を作らない。root の configs/ も作らない
- `logs_chexpert/` `experiments/` は作らない。過去ログの棚卸しは移植とは別作業

## コピーと書き直しの境界

**数値に効くものを「整理しながら」書き直すと、過去 run と比較不能になる。**

| 扱い | 対象 |
| --- | --- |
| 逐語コピー | `utils/metrics.py`、`data/{splits,attribute_utils,transform_utils,dataset}.py`、`models/{resnet,spatial_hypernet}/*`、`metadata_encoder.py`、`loss.py`、`pyproject.toml`、`uv.lock` |
| 書き直す | LitModule、run 記録、workflow、configs ツリー、scripts、validation、callbacks |
| 持っていかない | `film/`、`hyperadapt/`、`module_v2`、`src/train.py`、`src/eval.py`、旧互換 wrapper、`src/training/{execution,runs}.py` の汎用 stage 機構 |

`uv.lock` は逐語コピーする。依存の更新を移植と同時に行わない。torch が動くと数値が動く。

## project ごとの差

- **hypernet_e2e**: 群割り当ては学習前に確定した静的 artifact。1 run = 1 fit。
  run 記録は単段用の薄いもの（ディレクトリ予約 → config 保存 → fit → 結果記録、80 行程度）。
  旧 `execution.py`/`runs.py` の stage・attempt・reconcile 機構は持ち込まない
- **hypernet_iterative**: 群は checkpoint の関数として run の途中に生成される。1 run = N stage。
  stage は独立子プロセスで 1 回だけ fit する。**反復の制御は Lightning の外に置く**。
  `workflow.py` が stage 列挙・引き継ぎ・状態 reset を所有し、dry-run と実行が同じ plan を使う

両 project とも当面 Lightning を使う。「1 つの学習の途中で群を更新し optimizer 状態を維持したまま
続ける」設計に進んだ時点で、iterative だけ素の loop に落とす。独立しているので e2e に影響しない。

## 群（cohort）の与え方

e2e では 3 層に分け、どの層も 1 ファイルで読めるようにする。

1. **recipe**: `cohorts/recipes.py` に「split frame → group_id」の純関数を並べる。CLI は 1 本
2. **artifact**: `assignments.parquet` の隣に `cohort.json`（recipe 名、引数、参照 checkpoint hash、
   入力 split hash、num_groups、生成 commit）を必ず置く
2. **identity**: `configs/cohort_definition/<name>.yaml`。`num_groups` は artifact と突き合わせる

## 比較可能性の担保

共有コードをやめる代わりに、**複製が同じ結果を出すことをデータで固定する**。

- `tests/golden/<name>.vN.json` に入力と期待値を置く。各 project の `tests/test_preflight.py` は
  **自分の実装だけを import** して golden を再現する。他 project を import しない
- 固定する対象: 公平性指標、eval の画像前処理、split の意味、cohort sidecar の schema
- 指標は関数単体ではなく、**数バッチを fairness callback に流した出力**を golden にして経路まで固定する
- 学習起動前に `uv run pytest projects/<p>/tests -m preflight -q` を走らせ、
  golden の version・hash と commit を run 記録に入れる
- 指標定義を変えるときは golden を**新バージョンとして追加**し、旧版を消さない。
  更新コミットに docs の変更と旧値→新値の理由を同梱する

## 進め方（対話的）

**1 回に 1 単位（1 ファイルまたは 1 サブパッケージ）。** 各単位で次の順に進む。

1. 旧実装の該当箇所を提示し、何をしているコードかを読み合わせる
2. 逐語コピーか書き直しかを**宣言**する（境界表に従う）
3. 書き直す場合は設計意図を 1〜2 行述べてから書く
3. 対応するテストを同じ単位で書く
4. ユーザーの理解を確認してから次へ進む

まとめて大量にコピーしない。複数 project を並行して移植しない。

## 移植の順序

旧 repo は読み取り専用。移植元は working tree をそのまま読む（絶対に変更しない）。

1. 新 repo を `git init`。骨格（pyproject/uv.lock/.envrc/.gitignore/.pre-commit-config/.project-root/docker）を置き `uv sync` が通ることを確認
2. `tests/golden/` を旧実装から生成する。**複製を始める前に基準を固める**
3. `projects/hypernet_e2e/`: models → data → callbacks → module → configs → run 記録 → cohorts → scripts
4. **検収**: `DATA_FOLDER=/data` で 1 epoch 学習し、旧実装と loss・val/auroc が一致することを確認。合わなければ先へ進まない
5. `projects/hypernet_iterative/`
6. `analysis/`、`data_pipeline/`、`docs/`、skill のパス更新
7. 旧 repo の README に「新 repo へ移行済み。過去 run の再評価専用」と明記して凍結

## 禁止事項

- 旧 repo を変更しない（読み取りのみ）
- 依存（pyproject / uv.lock）を移植と同時に更新しない
- `logs_chexpert` `experiments` `checkpoints` を移動・コピーしない。棚卸しは別作業
- 数値に効くファイルを書き直さない（境界表の「逐語コピー」列）
- `src/` という名前のパッケージを作らない
- project 間で import しない。「将来使いそう」で共有へ出さない
- 過去 run の logs を改名・削除しない
