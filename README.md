# hypernet-fairness

医用画像分類における公平性を、hypernetwork 系のモデルで比較・検証するリポジトリです。
実験ごとに実装を project 内へ閉じ、project 間でコードを import しません。共通化よりも、各実験を
単独で再現・比較できることを優先します。

## Project

| project | 責務 | 現在地 |
| --- | --- | --- |
| [`hypernet_e2e`](projects/hypernet_e2e/README.md) | 1 回の fit で完結する hypernetwork 系学習と性能比較 | model / data / callbacks / Lightning module / configs / run 記録 / 学習起動を実装済み。cohort と GroupDRO は保留 |
| `hypernet_iterative` | run 中に複数 stage を進める反復学習 | 未着手 |

`two_stage` の配置は未決です。現時点では project を作らず、学習過程の実装を読む段階で
`hypernet_e2e` に含めるかを判断します。

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

分析（旧 repo の `analyze` skill 相当）は、`analysis/` を移植した時点で追加します。
