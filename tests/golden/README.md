# golden

複製された実装が同じ結果を出すことを固定するデータ。**コードではないので import されない。**

各 project の `tests/test_preflight.py` は、自分の project の実装だけを import してここの
期待値を再現する。他 project を import しない。この仕組みの理由は
`.agents/skills/migrate/rationale.md` を参照する。

## ファイル

| ファイル | 固定する対象 |
| --- | --- |
| `fairness_metrics.v1.json` | fairness callback を 1 epoch 分駆動したときのログキーと値 |

## 形式

```
schema     形式のバージョン
version    golden 自体のバージョン。v1 / v2 は別物として扱う
tolerance  値の一致判定に使う絶対許容差
source     生成元の repo・commit・対象ファイル
cases[]    name / description / attribute_names / batches / expected
```

`expected` の `null` は NaN を表す。JSON は NaN を表現できないため。
`batches` は step output（`logits` / `target` / `attributes`）と同じ形で、
callback の公開フック（`on_validation_epoch_start` → `on_validation_batch_end` →
`on_validation_epoch_end`）へ順に渡す。指標関数を直接呼ばない。集計経路ごと固定するため。

## 更新

指標の定義を意図的に変えたときだけ、**新しいバージョンを追加する**。`v1` は消さない。
過去 run は旧バージョンの定義で出た数値なので、混ぜて比較すると誤る。
更新コミットには docs の変更と、旧値から新値へ変わった理由を同梱する。
