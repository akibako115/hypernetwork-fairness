# iterative_probe

`hypernet_iterative` を本格的に回す前の探り分析。反復学習の**初期方針**、つまり次にどの設定で
実験を組むかを決めるために、最小構成で回した run を読む。結論そのものより、次に振るべき
パラメータを絞ることが目的になる。CheXpert / Spatial LoRA / seed 42。

## 決めたいこと

1. **GroupDRO の step size をどこに置くか。** 今回の `group_dro_step_size=0.001` は adversarial
   weight をほとんど動かさない。実際 stage01 / stage02 とも `train/group_dro/q_*` は 0.09〜0.11 に
   収まり、`weight_entropy` は一様分布の log(10)=2.303 付近に張り付いている。この run を
   「GroupDRO が効いていない状態の反復」の基準線として使えるか、次にどれだけ上げるかを決める。
2. **stage を重ねる価値があるか。** warmup → stage01 → stage02 で、global の `val/auroc` と
   hidden cohort の `val/hidden_min_auroc` がどちらへ動くか。cohort は stage ごとに引き直されるため
   stage 間で hidden group の対応が取れない点をどう扱うかも、ここで方針を決める。
3. **epoch 数と stage 数の最小ライン。** warmup 2 epoch / stage 2 epoch では推移ではなく到達点しか
   読めない。本番の実験で何 epoch 必要かの当たりを付ける。

## 対象 run

変調範囲 × GroupDRO step size の 2×2（`projects/hypernet_iterative/runs/`）と、global 指標の
比較対象にする通常 ResNet（`projects/hypernet_e2e/runs/20260921T103036Z-resnet-chexpert-s42-5538`）。
run-id の一覧と中断した試行は [runs.md](runs.md) を見る。

baseline は [initial_resnet_vs_invariant](../initial_resnet_vs_invariant/) が invariant 化の対照に
使っているものと同じ run になる。split の sha256・optimizer・batch size・class weight・seed が
iterative 側と揃っているので、global の val 指標はそのまま並べて読める。

## 公平性の評価軸

group ごとの公平性は **test split** で測る。checkpoint は 5 model すべて val AUROC で選んで
いるので、val で群別に比べると選択の効いた側へ寄る。epoch 推移の図が val なのは、学習中の記録が
val しか無いためで、節ごとに split が違う点は notebook 側にも書いてある。

群は age group（65 歳）× sex × race で、race は **White / Asian / Black** に絞る。残り 3 カテゴリは
test で n が小さく、交差させると評価が成立しない。単独属性から 3 属性の交差まで、どの粒度でも
同じ母集団（3 属性が非欠損で race が上の 3 つ、15,932 行）を使う。

gap だけでなく worst と best の値も出す。gap が縮んでも、worst が上がったのか best が下がったのかで
意味が逆になる。

## 構成

| file | 入力 → 出力 |
| --- | --- |
| `epoch_metrics.py` | run artifact → `results/epoch_metrics.csv`・`headline.csv` |
| `cohort_analysis.py` | run artifact → `results/cohort_groups.csv`・`cohort_correlations.csv` |
| `baseline_comparison.py` | W&B transaction log + iterative results → `results/baseline_epoch_metrics.csv`・`global_comparison.csv` |
| `collect.py` | 上記 3 script の一括 wrapper |
| `groups.py` | 予測 cache + `results/` → `results/group_metrics_test.csv`・`fairness_summary_test.csv` |
| `plots.py` | notebook から呼ぶ再利用可能な Figure 作成 helper |
| `probe.ipynb` | `results/` → 可視化・表・主張と考察 |

予測 cache は共有 CLI（`analysis/common/predictions.py`）が作る。群ごとの公平性指標は run artifact
に無いので、群の切り方を分析側で決めるために予測を持つ。

`cache/` `results/` `figures/` は再生成できるので Git 管理外。`reports/` は分析メモと、そこから
決まった次の実験方針を置く。

hidden cohort の epoch ログは cohort ごとの `AUROC`・`bACC`・`loss`（および support）を raw metric とし、
`epoch_metrics.py` が `min`・`max`・`gap` を後計算する。これにより logger 側と分析側で派生指標の定義が
重複しない。

## 作り直す

上から順に 1 回ずつ。run-id は条件順（fc 0.001 → fc 0.01 → stage4+fc 0.001 → stage4+fc 0.01）に
並べる。表と図の並びがこの順になる。

```bash
uv run python analysis/common/predictions.py --study iterative_probe --split test \
  --run-dir projects/hypernet_e2e/runs/20260921T103036Z-resnet-chexpert-s42-5538 \
  --run-dir projects/hypernet_iterative/runs/20260921T103222Z-iterative-s42-d24d \
  --run-dir projects/hypernet_iterative/runs/20260921T103219Z-iterative-s42-e992 \
  --run-dir projects/hypernet_iterative/runs/20260921T103225Z-iterative-s42-3166 \
  --run-dir projects/hypernet_iterative/runs/20260921T103225Z-iterative-s42-fac5
uv run python analysis/iterative_probe/collect.py \
  20260921T103222Z-iterative-s42-d24d 20260921T103219Z-iterative-s42-e992 \
  20260921T103225Z-iterative-s42-3166 20260921T103225Z-iterative-s42-fac5 \
  --baseline 20260921T103036Z-resnet-chexpert-s42-5538
uv run python analysis/iterative_probe/groups.py --split test
```

個別に作り直す場合は、分析の塊ごとに次を実行する。

```bash
uv run python analysis/iterative_probe/epoch_metrics.py \
  20260921T103222Z-iterative-s42-d24d 20260921T103219Z-iterative-s42-e992 \
  20260921T103225Z-iterative-s42-3166 20260921T103225Z-iterative-s42-fac5
uv run python analysis/iterative_probe/cohort_analysis.py \
  20260921T103222Z-iterative-s42-d24d 20260921T103219Z-iterative-s42-e992 \
  20260921T103225Z-iterative-s42-3166 20260921T103225Z-iterative-s42-fac5
uv run python analysis/iterative_probe/baseline_comparison.py \
  20260921T103222Z-iterative-s42-d24d 20260921T103219Z-iterative-s42-e992 \
  20260921T103225Z-iterative-s42-3166 20260921T103225Z-iterative-s42-fac5 \
  --baseline 20260921T103036Z-resnet-chexpert-s42-5538
```
