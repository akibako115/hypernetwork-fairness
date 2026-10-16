# fairness_mechanism_probe

2026-10-06開始。属性条件付きhypernetworkで公平性をどう捉え、どこを改善できるかを決めるための探索的診断。
既存のrace-known比較（`race_known_architecture_objective_comparison`）を読んだうえで、
次の問いを凍結特徴・既存予測・小規模な追加学習で調べる。結論はreports/に置く。

1. 群間の性能差は、群ごとに異なる処理（決定境界）で縮む余地があるか（属性ごとの最適な処理の上限）。
2. 差のうち、動作点（閾値・切片）で説明される部分と、群内の順位（AUROC）の差はどれだけか。
3. 人口統計の8群より大きな性能差を持つ潜在サブグループがあるか。それは属性で説明されるか。
4. 明示的な公平性損失は、固定した表現の上でworst-group性能をどこまで動かせるか。
5. 属性情報はモデルの判断にどう入り込んでいるか（hypernetworkを解析の道具として使う）。

対象runは[runs.md](runs.md)。

## 結論

[機構診断の記録](reports/mechanism_findings.md)。CheXpert No Finding の8群差は、hypernetwork の属性別処理で
縮む種類の差ではなかった。差の大半は基底率による動作点のずれと、年齢で変わる「No Finding」ラベルの意味で
説明でき、race の寄与はほぼ無い。

## 構成

notebook はまだ無い。探索段階のため、各問いを script 1 本に閉じ、表を `results/<topic>/` に書く。

| script | 問い |
| --- | --- |
| `features.py` | M01 表現・属性・8群の読み込み（共有） |
| `heads.py` | 条件付けの形（群別切片・Δw・独立・年齢交互作用・hypernet・MLP）と群内 AUROC |
| `objectives_frozen.py` | 目的関数（ERM・群均等・DRO・worst-rank）と群内 AUROC |
| `erasure.py` / `attribute_contribution.py` | 属性の LEACE 消去と、画像経路の属性依存 |
| `operating_point.py` | 群別閾値と群内較正 |
| `residual_tpr_gap.py` | 群別閾値後に残る TPR 差の分解（患者単位の分け直し）と、属性から閾値を決める方式の比較 |
| `latent_groups.py` | 潜在サブグループと margin 分散の分解 |
| `label_construct.py` | 陽性のレポート種別と年齢別 AUROC |
| `context_heads.py` / `label_process_heads.py` | 撮影文脈・ラベル生成過程での条件付け |
| `paired_noise.py` | 既存比較ペアの paired bootstrap SE |
| `probes/train_probe.py` | 群別 fine-tune（run 契約の外） |
| `probes/fairvlmed_headroom.py` | FairVLMed での同じ関門診断 |
