# 初期探索: ResNet vs attribute-invariant ResNet

通常の ResNet と attribute-invariant ResNet の既存実行ログを比較し、invariant 化による性能・公平性指標の傾向を初期探索する。

## 分析対象

| 条件 | run ID | 主な入力 |
| --- | --- | --- |
| ResNet | `20260921T103036Z-resnet-chexpert-s42-5538` | `runs/20260921T103036Z-resnet-chexpert-s42-5538/metrics/fit.json`、`logs/train.log` |
| attribute-invariant ResNet | `20260921T125711Z-resnet-chexpert-attribute-invariant-s42-9fd3` | `runs/20260921T125711Z-resnet-chexpert-attribute-invariant-s42-9fd3/metrics/fit.json`、`logs/train.log` |

実行ログは変更せず、このディレクトリへ比較用の notebook、集計表、所見を保存する。

## 分類性能比較

最初に prediction cache を作る。標準は validation selection に使っていない `test` split とする。
cache は再生成できる中間物であり、Git 追跡しない。

```bash
uv run python analysis/hypernet_e2e/initial-resnet-vs-invariant/cache_predictions.py
```

その後、[classification_performance.ipynb](classification_performance.ipynb) を開く。標準は `test` split
で、accuracy・balanced accuracy・AUROC・cross entropy と、属性別の公平性指標を比較し、集計表を
`results/` に書き出す。公平性表は sex / race / ethnicity / age group (65歳) ごとの Eopp0・Eopp1・
Eodds・worst-group AUROC / BAcc・gap を含む。
validation 指標を見たい場合だけ、`cache_predictions.py --split val` を実行して notebook の `SPLIT` を
`"val"` に切り替える。
