# iterative-probe

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

`20260921T081141Z-iterative-s42-5752`（`projects/hypernet_iterative/runs/`）。詳細と中断した
試行は [runs.md](runs.md) を見る。

## 構成

- `results/` — stage ごとの metric を集計した表
- `figures/` — レポートが参照する図
- `reports/` — 分析メモと、そこから決まった次の実験方針
