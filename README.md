# hypernet-fairness

医用画像分類における公平性を、hypernetwork 系のモデルで比較・検証するリポジトリです。
実験ごとに実装を project 内へ閉じ、project 間でコードを import しません。共通化よりも、各実験を
単独で再現・比較できることを優先します。

## Project

| project | 責務 | 現在地 |
| --- | --- | --- |
| [`hypernet_e2e`](projects/hypernet_e2e/README.md) | fit 開始時に group が確定している hypernetwork 系学習と性能比較 | model / data / callbacks / Lightning module / configs / run 記録 / 学習起動、demographic group への GroupDRO を実装済み |
| [`hypernet_iterative`](projects/hypernet_iterative/README.md) | run 中に複数 stage を進める反復学習 | warmup、cohort 再生成、独立 stage process、artifact・preflight 記録まで実装済み。実データの最小 end-to-end run も確認済み |

hypernetwork 系 2 project の境界は、**group が fit の開始時に確定しているか**で引いています。
demographic 属性から決まる group は `hypernet_e2e`、学習の途中で再生成される hidden cohort は
`hypernet_iterative` です。

ResNet を 1 段目、凍結した Spatial LoRA を 2 段目とする二段学習は、`hypernet_e2e` の run を
2 回起動して構成します。段ごとに run が分かれるので、trainer 設定も学習条件も段ごとに独立して
振れます。checkpoint の受け渡しは `run.json` の `checkpoints` と `parent_run` が記録します。

## 仮説 (study)

**1 つの run は必ず 1 つの仮説に属します。** 仮説の正本は `analysis/<slug>/` の分析 package で、
その directory 名を `study` として学習の起動時に渡します。

```bash
uv run python -m projects.hypernet_e2e.run experiment=spatial_lora_chexpert_fc study=iterative-probe seed=42
```

- `study` に既定はありません。未指定の run は起動できず、`analysis/<study>/` が無い値も拒否します。
- 結論を出すつもりが無い run（経路確認・当たり付け）は `study=scratch` を使います。
- `run.json` に `study` が残り、W&B では `group` に入ります。W&B project は repo で 1 つ
  (`fairness_hypernet`) にまとめ、code project は `job_type` が持ちます。1 つの仮説が
  `hypernet_e2e` の baseline と `hypernet_iterative` の run を並べることがあるためです。

project は「実装の境界」、study は「問いの境界」であり、両者は交差します。

## データ

`data/<dataset>/` の split CSV は、どの project からも読み取り専用の固定入力です。分割単位・比率・
CV 戦略・群別指標の最小セル枚数は [data_pipeline/README.md](data_pipeline/README.md) を正本とします。

| dataset | 分割単位 | 比率 | CV |
| --- | --- | --- | --- |
| CheXpert | 患者 | 80/10/10 | 行わない |
| ISIC 2019 | 病変（`lesion_id`） | 80/10/10 | 5-fold（fold 内 60/20/20）。稀少クラスの評価に必要 |

## 再現性の境界

- `tests/golden/` は project が再現すべき評価・前処理の期待値であり、コードから import しません。
- 各 project は自分の実装だけを使って golden を検証します。
- 同じデータ、seed、設定で旧実装と比較するため、数値挙動に関わるモデル・データ処理は逐語的に移植します。

golden の形式と更新規則は [tests/golden/README.md](tests/golden/README.md)、移植の
[方針](.agents/skills/migrate/SKILL.md) と [現在地](.agents/skills/migrate/progress.md) も記録しています。

## 開発

Python 3.11 を使います。依存は lock file に固定されています。

```bash
uv sync --frozen
uv run pytest projects/hypernet_e2e/tests -q
uv run ruff check projects/hypernet_e2e
```

最小学習の起動方法と出力契約は、[hypernet_e2e の README](projects/hypernet_e2e/README.md) を参照してください。

## Agent skills

`.agents/skills/` に作業手順を置いています。skill 名で参照されたら対応する `SKILL.md` を読みます。

| skill | 用途 |
| --- | --- |
| [`migrate`](.agents/skills/migrate/SKILL.md) | 旧 fairness repo からの対話的な移植 |
| [`train`](.agents/skills/train/SKILL.md) | 学習の起動・承認・監視。ローカルと ws Docker |
| [`gpu`](.agents/skills/gpu/SKILL.md) | GPU と実行中プロセスの確認 |
| [`data-explore`](.agents/skills/data-explore/SKILL.md) | データセット構造の調査と data config との突き合わせ |
| [`analyze`](.agents/skills/analyze/SKILL.md) | run 記録を読んだ仮説 (study) の分析と、results・figures・reports の更新 |
