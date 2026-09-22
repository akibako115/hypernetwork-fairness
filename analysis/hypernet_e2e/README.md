# hypernet_e2e analysis

`hypernet_e2e` の分析成果物はここに置く。`projects/hypernet_e2e/runs/` は学習実行ごとの不変な出力、ここはそれらを横断して読む分析用の領域である。

検証ごとに、このディレクトリ直下へ実験名を kebab-case にしたサブディレクトリを1つ作る。各サブディレクトリには、その検証の notebook、集計結果、図、レポートをまとめる。

現在の検証:

- `initial-resnet-vs-invariant/`: 通常の ResNet と attribute-invariant ResNet の初期比較
