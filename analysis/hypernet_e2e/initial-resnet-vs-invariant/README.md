# 初期探索: ResNet vs attribute-invariant ResNet

通常の ResNet と attribute-invariant ResNet の既存実行ログを比較し、invariant 化による性能・公平性指標の傾向を初期探索する。

## 分析対象

| 条件 | run ID | 主な入力 |
| --- | --- | --- |
| ResNet | `20260921T103036Z-resnet-chexpert-s42-5538` | `runs/20260921T103036Z-resnet-chexpert-s42-5538/metrics/fit.json`、`logs/train.log` |
| attribute-invariant ResNet | `20260921T125711Z-resnet-chexpert-attribute-invariant-s42-9fd3` | `runs/20260921T125711Z-resnet-chexpert-attribute-invariant-s42-9fd3/metrics/fit.json`、`logs/train.log` |

実行ログは変更せず、このディレクトリへ比較用の notebook、集計表、図、所見を保存する。
