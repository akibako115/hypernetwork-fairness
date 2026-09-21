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

### step 3 以降

SKILL.md の「移植の順序」 3〜7 のとおり。step 3 は
models → data → callbacks → module → configs → run 記録 → cohorts → scripts の順に、
**1 回 1 単位・対話的**に進める。

`hypernet_e2e` の model は以下を移植済み: ResNet、MetadataEncoder、HyperLinear、
Spatial LoRA block / network、model utils。data 側は `attribute_utils.py` / `splits.py` /
`transform_utils.py` / `dataset.py` を移植済み。評価 transform は project の preflight で golden と
bit 単位照合し、学習 transform は乱数を含むため構成のみ照合する。通常の `datamodule.py` は
移植済み。Callbacks は全体性能ログと公平性ログを移植済み。`module.py` は通常の `TaskLoss`、
単段 fit、Spatial LoRA のデータ依存初期化までを実装済み。cohort sidecar・GroupDRO objective・
`TextProgressLogger` は、現時点の e2e スコープ外として保留する。CheXpert の ResNet / Spatial LoRA、
ERM / inverse-weighted loss / inverse-frequency sampling の Hydra preset は移植済み。`run_record.py` は
新規 run directory、解決済み config、train/val manifest、golden preflight、成功/失敗状態を記録する。
`run.py` は inverse class weight を train split から解決し、Hydra の自動 output を
使わずに 1 回の Lightning fit を行う。scalar callback metrics は `metrics/fit.json` と `run.json` に記録する。
CheXpert の Spatial LoRA は、モデル入力に `sex` / `race` / `ethnicity` /
`frontal_lateral` / `ap_pa` / `age` を使い、公平性ログには raw age から作る
`age_group_65` を加えた `sex` / `race` / `ethnicity` / `age_group_65` を使う。
この設定で 1 train batch・1 validation batch の最小 fit が完走した。
e2e の cohort artifact と GroupDRO objective は、現時点のスコープ外として保留する。

e2e の生成物は `projects/hypernet_e2e/runs/<run-id>/` が所有する。`data/chexpert` は
固定入力だけを置き、cohort は生成 run の `artifacts/cohorts/` に保存する。詳細な出力契約は
`projects/hypernet_e2e/docs/run-artifacts.md` に固定した。

実験ログは旧 repo と同じく **wandb を既定**にし、`run_logging.py` が両 project で
保存先・run 名・終了処理を持つ。移植直後は `logger=False` が固定され、epoch ごとの metric が
どこにも残らず、Hydra の job log も repo 直下の共有 `run.log` に落ちていた。現在は
run ごとの `logs/train.log` と `runs/<run-id>/wandb/` に収め、wandb run の `name` / `id` / `url` を
run 記録に残す。project 名は旧 repo とは分け、`fairness_hypernet_e2e` /
`fairness_hypernet_two_stage` とする。旧 repo の `config_tree.log` / `tags.log` / `.hydra/` は
解決済み `config.yaml` と重複するため再現しない。

two-stage は `projects/hypernet_two_stage/` として独立させる。Stage 1 の最良
`val/auroc` checkpoint を入力に、共有 backbone / classifier を凍結した Stage 2 Spatial LoRA を
学習する固定2段の workflow を持つ。ResNet、MetadataEncoder、HyperLinear、Spatial LoRA、model utility
に加え、CheXpert data、callbacks、Lightning module、Hydra configs、workflow、学習 CLI を移植済みである。
各 stage を1 train batch・1 validation batch に制限した CheXpert 最小実行が完走した。cohort の再生成や
任意回数の反復は持たない。

`hypernet_iterative` は固定 cohort の単一 stage に必要な model / data / callbacks / module /
configs を移植済み。cohort sidecar の train 被覆、binary hidden metric、warm-start と resume の
組み合わせを検証する project-local validation を置いた。checkpoint 選択は global AUROC を主選択とし、
BAcc および hidden-min-AUROC の補助 checkpoint を選べる config を移植済み。cohort 再生成、
stage subprocess、run artifact、validation の workflow 呼び出しは未実装である。

## 未決事項

- **iterative の warm-start 前データ依存初期化。** 現在の `LitModule.setup()` は旧実装と同じく
  `initialize_with_dataloader()` の後に net 全体を warm-start で厳密一致ロードする。このため
  warm-start stage では Var(c) の初期化結果が直後に上書きされるが、初期化を省くと train
  DataLoader の sampler が消費する RNG を含め既存 run との互換性が変わり得る。現時点では
  **互換性を優先して維持**し、最適化する場合は数値挙動変更として独立に検証・記録する。

- **split の golden は作らない。** `attribute_names` は設定で変わるため、fixture CSV に対する
  `splits.py` の単体テストで列契約を検証する。実データの行数・画像集合 hash・target 分布は
  run 記録の data manifest に残し、データ更新を意図的に追跡する。

- **`class_imbalance` をどこに置くか。** `hypernet_e2e` に含めるか、独立 project にするかは未決。
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
