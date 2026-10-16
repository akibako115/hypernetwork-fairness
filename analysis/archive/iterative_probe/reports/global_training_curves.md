# global 指標の epoch 推移

## 条件

- split: validation（学習中の記録が val しか無い）
- 対象: iterative 4 run（変調範囲 `fc` / `stage4+fc` × GroupDRO step size 1e-3 / 1e-2）と baseline（ResNet ERM、30 epoch）
- seed: 42 のみ（各条件 1 run）
- epoch: iterative は warmup 0–1、stage01 2–6、stage02 7–11 の通し番号（0 始まり）。baseline は同じ予算の 0–11 と、到達点として 30 epoch 全体
- 指標: val AUROC、val balanced accuracy、val loss（class weight も GroupDRO の重みも掛けない素の cross-entropy。両 project で同じ定義）
- 選択 checkpoint: iterative は stage02 の best val AUROC、baseline は 30 epoch の best val AUROC

notebook は [`global_training_curves.ipynb`](../global_training_curves.ipynb)、数値の正本は
`results/global_epoch_metrics.csv` と `results/global_epoch_snapshots.csv`、図は
`figures/global_training_curves_val.png` とする。

`train/loss` は比較しない。baseline は class weight 付き cross-entropy、iterative の stage は GroupDRO の
目的関数であり、値の意味が違う。

## 観察事実

val AUROC を、読み取りに使う点で並べる。

| 条件 | warmup 最終（epoch 1） | 12 epoch 目（epoch 11） | 全 epoch の best | 選択 checkpoint |
|---|---:|---:|---:|---:|
| fc / 1e-3 | 0.8580 | 0.8510 | 0.8580 | 0.8510（epoch 11） |
| fc / 1e-2 | 0.8580 | 0.8349 | 0.8580 | 0.8539（epoch 9） |
| stage4+fc / 1e-3 | 0.8545 | 0.8303 | 0.8545 | 0.8482（epoch 10） |
| stage4+fc / 1e-2 | 0.8545 | 0.8295 | 0.8545 | 0.8495（epoch 7） |
| ResNet ERM | 0.8559 | 0.8605 | 0.8675（epoch 9） | 0.8675（epoch 9） |

- warmup 最終 epoch の val AUROC は baseline と同じ水準にある。同じ変調範囲の 2 run は warmup の値が一致する。
- 4 run とも、全 12 epoch の best は warmup 最終 epoch にある。GroupDRO の stage に入った後、その値に届いた epoch は無い。
- 選択 checkpoint は warmup 最終 epoch より 0.004〜0.007 低い。
- 12 epoch 目では、iterative が 0.830〜0.851、baseline が 0.8605 になる。
- stage 内の val AUROC の振れ幅（max − min）は、stage02 で stage4+fc が 0.018 / 0.020、fc が 0.011 / 0.019 になる。baseline は同じ epoch の区間で 0.009 に収まる。
- stage4+fc の 2 本は、stage02 の頭（epoch 7）で 0.848 / 0.850 に上がり、epoch 8 で 0.838 / 0.836 に落ちる。step size が違っても、この形は揃っている。
- balanced accuracy は、stage01 以降の 40 点（4 run × 10 epoch）のうち 39 点で baseline を下回る。
- fc / 1e-2 は epoch 4 で balanced accuracy が 0.747 まで落ち、同じ epoch で val loss が 0.72 に跳ねる。
- val loss は baseline 自身も epoch 0〜11 で 0.36〜0.59 と振れる。

## 解釈候補

- GroupDRO の stage は、この設定では global AUROC を warmup より下げている可能性がある。ただし baseline も epoch 間で 0.01 前後振れるため、stage 内の個々の epoch の差を効果とは読めない。確かなのは、12 epoch 通して warmup の水準へ戻っていないことだけである。
- stage4+fc の振れが大きいのは、変調する範囲が広いぶん、cohort の重み付けで表現がより大きく動くためかもしれない。ただし stage02 では fc / 1e-2 も 0.019 と同程度に振れており、変調範囲だけで説明できるとは言えない。
- stage の境目では cohort の引き直しと AdamW の状態の初期化が同時に起きる（stage 間で引き継ぐのは net の重みだけ）。stage 頭の上昇と、その直後の落ち込みがどちらによるのかは、この推移からは分けられない。
- balanced accuracy の落ち込みと val loss の跳ねが同じ epoch に出るのは、順位（AUROC）より確率の水準、つまり閾値付近の較正が動いたことを示唆する。

## 追加確認

- cohort ごとの `q` と `val/hidden_{auroc,bacc,loss}_NN` の推移を見て、global が落ちた epoch に、どの cohort が持ち上がり、どの cohort が落ちたのかを確認する（トピック 2 の notebook）。
- test split で、属性群・交差群の worst / best / gap を baseline と並べる。global を払った分だけ worst が上がっているかを見る。
- optimizer の初期化の影響を分けるには、stage 間で optimizer の状態を引き継ぐ run か、cohort を引き直さずに stage を継ぐ run が要る。
- seed を足すまで、条件間の差（選択 checkpoint で最大 0.006）は比較に使わない。
- 選択 checkpoint を stage02 の中で選ぶ規則が妥当かは、群別の結果を見てから決める。warmup の best を選ぶと、GroupDRO を回していない model を選ぶことになる。
