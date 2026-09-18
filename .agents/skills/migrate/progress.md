# 移植の現在地

方針の正本は [SKILL.md](SKILL.md)、根拠と実測値は [rationale.md](rationale.md)。
このファイルは**どこまで終わったか・次に何をするか・何が未決か**だけを持つ。
移植を再開するときは最初にこれを読む。

最終更新: 2026-09-18 / 新 repo HEAD `fe23c8d`

## 完了

### step 1 骨格（`07137b0`）

`pyproject.toml` / `uv.lock` / `.envrc` / `.gitignore` / `.pre-commit-config.yaml` /
`.project-root` / `.python-version` / `dockerfile` / `docker/{build,run}.sh`。

- `pyproject.toml` は依存と `[tool.uv.sources]` を逐語コピー。変えたのは
  `name`・`description`・cache パス（`~/.cache/*/hypernet-fairness`）・
  `testpaths = ["tests", "projects"]`・marker `preflight` 追加・
  `known-first-party = ["projects", "analysis", "data_pipeline", "tests"]` だけ
- `uv.lock` はコピー後に `uv lock` を実行。旧 lock との差分は root package block が
  `fairness` → `hypernet-fairness` で alphabetical に移動しただけで、**依存の version 差分はゼロ**。
  `uv sync --frozen` は lock が project 名を記録するため、コピー直後は必ず失敗する（想定内）
- `.python-version = 3.11` を**追跡する**。書かないと uv が 3.13 を選び、旧 venv（3.11.14）と
  ずれる。`.gitignore` に `!.python-version` の例外を入れてある
- 検証済み: Python 3.11.14 / torch 2.7.1+cu118 / lightning 2.6.0 / CUDA 利用可。旧 repo と一致

### step 2 golden（`d3c46ea`, `0ce4a8d`, `fe23c8d`）— 一部残

`tests/golden/README.md` が契約の正本。要点だけ再掲する。

- golden は **import されない**。各 project の `tests/test_preflight.py` が
  自分の実装だけを import して再現する
- 期待値を変えない case 追加は version を上げない。**定義を変えるときだけ新 version file**を
  追加し、`v1` は消さない
- `null` は NaN を表す

#### `fairness_metrics.v1.json`（8 case / 24 key）

`compute_fairness_metrics` 単体ではなく **`FairnessMetricsCallback` に数バッチ流した
ログ出力**を固定している。経路（buffer → epoch 末 concat → `evaluation_categorical` 優先 →
`{prefix}/{attr}/{metric}` のキー命名）まで込みで縛るため。

case: `binary_two_groups` / `binary_multi_batch` / `binary_with_missing` /
`binary_two_attributes` / `binary_single_group`（群 <2 → `{}`）/
`binary_group_without_negatives`（Eopp0・Eodds が NaN）/ `multiclass_three_classes` /
`evaluation_categorical_precedence`。

#### `eval_transform.v1.json`（6 case）

`val_transforms_chexpert` の入力 PNG → 出力 tensor を sha256 まで固定する。
`min/max/mean/std` も持つが、これは**落ちたときの切り分け用**で判定は sha256。

case とその選定理由:

| case | 押さえている経路 |
| --- | --- |
| `square_rgb` 64×64 | pad が no-op の基本形 |
| `landscape_pad_remainder_bottom` 38×23 | 奇数余りが**下**に落ちる pad の非対称性 |
| `portrait_pad_remainder_right` 23×38 | 奇数余りが**右**に落ちる pad の非対称性 |
| `grayscale_converted_to_rgb` 40×30 L | Dataset 側の `.convert("RGB")` |
| `already_target_size` 224×224 | Resize が no-op の境界 |
| `checkerboard_downscaled` 600×400 | **縮小**。実データは常にこの経路。PIL は縮小で別の reduce 経路を通る |

`train_transforms_chexpert` は**乱数を含むので出力を固定しない**。
`train_transform.composition` に 5 段の構成文字列（パラメータ込み）を記録するだけ。
`scale`・`degrees`・順序・段の欠落は検出できる。torchvision 実装側の変化は検出できず、
そこは `uv.lock` 固定で担保する、という切り分け。

**golden は「再現できること」ではなく「壊れたときに落ちること」で検証した。**
metrics・transform ともに意図的な改変を入れて検出率を測ってある。
`antialias=False` が検出できないのは golden の穴ではなく、PIL Image 入力では
この引数が無視される（torchvision が UserWarning を出す）ため。

## 次にやること

### step 2 の残り: split の意味の golden

`DATA_FOLDER=/data` が必要。固定する対象:

- 各 split の行数
- `image` 集合の hash（順序非依存）
- `target` の分布
- 必須列集合（`src/data/splits.py:required_split_columns` の結果）

実体は `/data/chexpert/splits/{train,val,test}.csv`。
config の `cv_splits_dir` は `${paths.data_dir}/chexpert/splits` を指す
（`/data/chexpert/cv_splits/fold_*` は**使われていない**。混同しないこと）。

cohort sidecar の schema golden は step 3 の cohort 移植と同時の方が自然なので、
ここでは作らない判断にしてある。

### step 3 以降

SKILL.md の「移植の順序」 3〜7 のとおり。step 3 は
models → data → callbacks → module → configs → run 記録 → cohorts → scripts の順に、
**1 回 1 単位・対話的**に進める。

## 未決事項

- **`class_imbalance` と `two_stage` をどこに置くか。** 現時点で
  `hypernet_e2e` / `hypernet_iterative` の 2 project しか決めていない。
  両者とも「学習前に条件が確定する 1 fit」なので e2e に寄せられるが、
  独立 project にする選択肢も残っている
- **過去ログの棚卸し**（ユーザーの指示で延期）。実測: logs 63GB。
  内訳は checkpoint 71GB（hardlink 済み）・`final_features.npz` 13GB（再生成可能）・
  **代替不能な metadata 207MB**。demographic の 96 run には checkpoint が無い。
  移すときは同一 filesystem（`/dev/sda2`）なので `cp -l` / `rsync -H` を使う
- 旧 repo README の凍結文言（step 7）はまだ書いていない

## 引き継ぎ時の注意

- 旧 repo `/home/akiba/workspace/fairness` は**読み取り専用**。working tree をそのまま読む
- `.env` は機械的にコピーせず、中身を確認して手で移す
- golden 生成に使った一時スクリプトは scratchpad にあり、新 repo には**入れていない**。
  一度きりの bootstrap なので再生成が要るなら書き直す
- pytest は `importmode=prepend` なので、project 間で同名 module を複製する場合は
  各階層に `__init__.py` を置かないと collision する
