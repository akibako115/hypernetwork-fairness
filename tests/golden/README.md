# golden

複製された実装が同じ結果を出すことを固定するデータ。**コードではないので import されない。**

各 project の `tests/test_preflight.py` は、自分の project の実装だけを import してここの
期待値を再現する。他 project を import しない。この仕組みの理由は
`.agents/skills/migrate/rationale.md` を参照する。

## ファイル

| ファイル | 固定する対象 |
| --- | --- |
| `fairness_metrics.v1.json` | fairness callback を 1 epoch 分駆動したときのログキーと値 |
| `eval_transform.v1.json` | PNG を読んで eval transform を適用した出力テンソル。pad-to-square → 224 の旧前処理（2026-10-17 より前の CheXpert run）。hypernet_iterative の preflight が使う |
| `eval_transform.v2.json` | 同じ入力 PNG に、MEDFAIR の前処理（縦横比を保たず 256x256 → CenterCrop 224）を適用した出力テンソル。256x256 への縮小に PIL のアンチエイリアスが掛かっている（2026-10-17 の commit `9bfd096`〜`a94568d` の間だけ使った） |
| `eval_transform.v3.json` | v2 と同じ構成で、256x256 への縮小を MEDFAIR の cv2 と同じくアンチエイリアスなしにした出力テンソル（`val_transforms_medfair`）。hypernet_e2e の preflight が使う。CheXpert と HAM10000 が同じ前処理を通る |

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

**ケースの追加は version を上げない。** 既存ケースの期待値が 1 つも変わらないなら、
固定した定義は変わっていないため。落ちる範囲を広げるだけの変更は同じバージョンに足す。

定義を意図的に変えたときだけ、**新しいバージョンを追加する**。`v1` は消さない。
過去 run は旧バージョンの定義で出た数値なので、混ぜて比較すると誤る。
更新コミットには docs の変更と、旧値から新値へ変わった理由を同梱する。

## eval_transform 固有の約束

入力は `image.png_base64` に埋め込んだ PNG。Dataset と同じ順序で
`Image.open(...)` → `.convert("RGB")` → `val_transform` を適用する。

`expected.sha256` は float32 の C 連続バイト列に対する**完全一致**で、tolerance を適用しない。
同じコード・同じ torchvision なら bit 単位で一致するため。`min` / `max` / `mean` / `std` は
不一致になったときにどれだけずれたかを見るための診断用で、こちらには tolerance を適用する。

train 側の transform は乱数を含むので**出力を固定しない**。`train_transform.composition` に
各段の型と主要パラメータだけを記録する。seed を固定して出力を固定すると、torchvision の
内部 RNG 消費が変わっただけで落ちる脆い golden になる。

v1 では、pad を通らない case（正方形・224x224）は pad の変更を検出しない。これは仕様で、
pad の挙動は `landscape_pad_remainder_bottom` と `portrait_pad_remainder_right` が担当する。
v2・v3 は pad をしないので、この 2 つの case は縦横比を保たずに引き伸ばす経路を担当する（名前は v1 から引き継ぐ）。

`checkerboard_downscaled` だけが縮小経路を踏む（v1〜v3 とも）。実データは必ず縮小されるが、他の case は
すべて拡大か恒等になるため。市松模様なのは、PNG で強く圧縮できて縮小時の折り返しに敏感だから。
v2 → v3 でこの case の std が 0.94 から 1.73 に上がるのは、アンチエイリアスを外して市松模様がぼけずに残るため。
