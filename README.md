# hypernet-fairness

医用画像分類における公平性を、hypernetwork 系のモデルで比較・検証するリポジトリです。
実験ごとに実装を project 内へ閉じ、project 間でコードを import しません。共通化よりも、各実験を
単独で再現・比較できることを優先します。

## Project

| project | 責務 | 現在地 |
| --- | --- | --- |
| [`hypernet_e2e`](projects/hypernet_e2e/README.md) | 1 回の fit で完結する hypernetwork 系学習と性能比較 | model / data / callbacks / Lightning module / configs / run 記録 / 学習起動を実装済み。cohort と GroupDRO は保留 |
| [`hypernet_two_stage`](projects/hypernet_two_stage/README.md) | Stage 1 ResNet から凍結した Stage 2 Spatial LoRA へ引き継ぐ二段学習 | 骨格・artifact 契約を整備中 |
| `hypernet_iterative` | run 中に複数 stage を進める反復学習 | 未着手 |

two-stage は checkpoint を引き継ぐ固定2段の workflow を持つため、単段の `hypernet_e2e` と
cohort を反復更新する `hypernet_iterative` のどちらにも含めません。

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
