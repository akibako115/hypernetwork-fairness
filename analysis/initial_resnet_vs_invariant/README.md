# 初期探索: ResNet vs attribute-invariant ResNet

通常の ResNet と attribute-invariant ResNet の既存実行ログを比較し、invariant 化による性能・公平性指標の傾向を初期探索する。

## 分析対象

| 条件 | run ID | 主な入力 |
| --- | --- | --- |
| ResNet | `20260921T103036Z-resnet-chexpert-s42-5538` | `runs/20260921T103036Z-resnet-chexpert-s42-5538/metrics/fit.json`、`logs/train.log` |
| attribute-invariant ResNet | `20260921T125711Z-resnet-chexpert-attribute-invariant-s42-9fd3` | `runs/20260921T125711Z-resnet-chexpert-attribute-invariant-s42-9fd3/metrics/fit.json`、`logs/train.log` |

実行ログは変更せず、このディレクトリへ比較用の notebook、集計表、所見を保存する。

## 構成

| file | 入力 → 出力 |
| --- | --- |
| `groups.py` | 予測 cache → `results/classification_performance_test.csv`・`fairness_metrics_test.csv` |
| `classification_performance.ipynb` | `results/` → 主張と考察 |

評価は validation selection に使っていない `test` split を標準とする。`cache/` と `results/` は
再生成できるので Git 管理外。

## 作り直す

```bash
uv run python analysis/common/predictions.py --study initial_resnet_vs_invariant --split test \
  --run-dir projects/hypernet_e2e/runs/20260921T103036Z-resnet-chexpert-s42-5538 \
  --run-dir projects/hypernet_e2e/runs/20260921T125711Z-resnet-chexpert-attribute-invariant-s42-9fd3
uv run python analysis/initial_resnet_vs_invariant/groups.py --split test
```

`--split val` にすると val 側の表になる。公平性表は sex / race / ethnicity / age group (65 歳) ごとの
Eopp0・Eopp1・Eodds・worst-group AUROC / bACC・gap を持つ。
