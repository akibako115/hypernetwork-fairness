# runs

この分析が引用する run の正本。run 本体は Git 管理外のため、分析コードが読む repo root からの相対
`run path` と、dashboard を開くための W&B URL をここに残す。

## 対象（seed 43、λ の水準だけを振る）

いずれも `experiment=resnet_chexpert_attribute_invariant`、`study=adversary_strength`、`seed=43`、
`weighting=inverse`、30 epoch。**違いは `model.loss_fn.attribute_adversary_weight` だけ**で、
`gradient_scale=1.0` は据え置く。run-id には experiment 名までしか入らないので、
**振った水準は run-id では区別できない**。正本は各 run の解決済み `config.yaml` と W&B の config になる。

| λ | run path | W&B | 状態 |
|---|---|---|---|
| 1.0 | `analysis/adversary_strength/runs/20260922T111111Z-resnet-chexpert-attribute-invariant-s43-f5cb` | [run](https://wandb.ai/kohki-akiba-kyushu-university/fairness_hypernet/runs/ezpviay3) | succeeded |
| 3.0 | `analysis/adversary_strength/runs/20260922T133206Z-resnet-chexpert-attribute-invariant-s43-2d1b` | [run](https://wandb.ai/kohki-akiba-kyushu-university/fairness_hypernet/runs/aie8tiei) | succeeded |
| 10.0 | `analysis/adversary_strength/runs/20260922T155241Z-resnet-chexpert-attribute-invariant-s43-78d1` | [run](https://wandb.ai/kohki-akiba-kyushu-university/fairness_hypernet/runs/y49k6u1l) | succeeded |

3 本は**逐次**で回す。A6000 1 枚で 1 本が GPU を使い切る（利用率 96%、6.7GB）ので、
並列にしても総時間は変わらず、途中結果が 1 つも読めなくなる。1 本 1.8 時間の見込み。

**W&B project が比較対象と違う。** この 3 本は `fairness_hypernet`、ws11 で回した比較対象は
`fairness_hypernet_e2e` に入る（repo 単位で 1 project にする方針が ws11 の checkout より後）。
**dashboard 上では並ばない。** 分析は run artifact を直接読むので結論には影響しない。

## adversary loss を記録した回し直し（ws11、seed 43）

adversary loss の成分（`attribute_adversary/{sex,race,age}`）を記録する変更（`a602497`）の後に、
λ=0 / 1 / 3 / 10 の 4 本を ws11 で**並列**に回した。上の 3 本と同じく
`experiment=resnet_chexpert_attribute_invariant`、`seed=43`、`weighting=inverse`、30 epoch で、
違いは `attribute_adversary_weight` だけになる。λ=0 は総損失が `task + 0·adv` なので、
adversary head は勾配を受けず学習されない（head を付けただけの control）。

**W&B は無い。** 4 本とも `logger: {}` で起動しており（W&B を必須にした `441b694` より前）、
記録は run directory の `metrics/metrics.csv` だけになる。

| λ | run path | W&B | 状態 |
|---|---|---|---|
| 0 | `analysis/adversary_strength/runs/20260923T075004Z-resnet-chexpert-attribute-invariant-s43-1bae` | なし | succeeded（best: epoch 16） |
| 1 | `analysis/adversary_strength/runs/20260923T075021Z-resnet-chexpert-attribute-invariant-s43-e41b` | なし | succeeded（best: epoch 10） |
| 3 | `analysis/adversary_strength/runs/20260923T075026Z-resnet-chexpert-attribute-invariant-s43-ac28` | なし | succeeded（best: epoch 25） |
| 10 | `analysis/adversary_strength/runs/20260923T075033Z-resnet-chexpert-attribute-invariant-s43-753c` | なし | succeeded（best: epoch 28） |

4 本とも val AUROC 最大の checkpoint（`best_val_auroc_<epoch>.ckpt`）を ws11 から回収し、`run.json` の
sha256 と照合した。`last.ckpt` は best と同じ sha256 なので回収していない。取り込みは hardlink なので、
取り込んだ後に回収した checkpoint は `runs.py import` を再実行しても package 側に現れない。
`projects/` 側の `checkpoints/` を `cp -al` で package 側へリンクしてある。

## DANN schedule（ws11、seed 43、最大 λ を振る）

`f879abd` で入れた DANN schedule で、GRL の scale を学習の進捗 p に応じて
`max_scale · (2 / (1 + exp(−γ p)) − 1)`（γ=10）と上げる。4 本を ws11 で**並列**に起動した。
いずれも `experiment=resnet_chexpert_attribute_invariant_dann`、`seed=43`、`weighting=inverse`、30 epoch、
`attribute_adversary_weight=1.0` で、**違いは `gradient_schedule.max_scale` だけ**になる。
run-id には max_scale が入らないので、正本は各 run の `config.yaml`。

backbone が受け取る逆向きの勾配は **λ = 1.0 × scale** で、最終的な λ は max_scale に一致する。
scale は max_scale に対する比で、epoch 0 の末に 0.08、epoch 4 で 0.63、epoch 9 で 0.92、epoch 14 以降は 0.98 以上になる
（記録された `train/adversary_scale` から）。**定数 λ = max_scale の run に約 10 epoch の warmup を付けたもの**として読む。
adversary head 自身の loss の重みは 1.0 で、定数 λ の run と変わらない。

4 本とも `run.json` の `git_commit` は null（ws11 の container に `.git` が無かったため）。

| 最大 λ | run path | W&B | 状態 |
|---|---|---|---|
| 0.1 | `analysis/adversary_strength/runs/20260924T112045Z-resnet-chexpert-attribute-invariant-dann-s43-6f67` | [run](https://wandb.ai/kohki-akiba-kyushu-university/fairness_hypernet/runs/xlinynnz) | succeeded（best: epoch 7） |
| 1 | `analysis/adversary_strength/runs/20260924T112044Z-resnet-chexpert-attribute-invariant-dann-s43-eeeb` | [run](https://wandb.ai/kohki-akiba-kyushu-university/fairness_hypernet/runs/3w9h2e9r) | succeeded（best: epoch 13） |
| 3 | `analysis/adversary_strength/runs/20260924T112044Z-resnet-chexpert-attribute-invariant-dann-s43-3ae9` | [run](https://wandb.ai/kohki-akiba-kyushu-university/fairness_hypernet/runs/w6agzr2r) | succeeded（best: epoch 16） |
| 10 | `analysis/adversary_strength/runs/20260924T112038Z-resnet-chexpert-attribute-invariant-dann-s43-3550` | [run](https://wandb.ai/kohki-akiba-kyushu-university/fairness_hypernet/runs/qt66ue6r) | succeeded（best: epoch 28） |

checkpoint は、最大 λ=0.1 / 1 / 3 の 3 本は val AUROC 最大のもの（`best_val_auroc_<epoch>.ckpt`）だけを回収し、
`run.json` の sha256 と照合した。`last.ckpt` は best と sha256 が違うが、分析は best しか使わないので回収していない。
最大 λ=10 の 1 本は `best_val_auroc_028.ckpt` と `last.ckpt` を回収してある（どちらも `run.json` の sha256 と一致）。学習中に回収した epoch 7 / 13 / 16 の best が残っていたが、ws11 側では消えており `run.json` も指さないので、2026-09-28 に削除した。

## 比較対象（別 study の run）

λ=0.1 の水準は [initial_resnet_vs_invariant](../initial_resnet_vs_invariant/runs.md) の
`20260922T063725Z-resnet-chexpert-attribute-invariant-s43-546c`（seed 43）を、
adversary 無しの baseline は同 `20260922T063725Z-resnet-chexpert-s43-efcd` を引く。
どちらもこの package の `runs/` に取り込んである（`analysis/adversary_strength/runs/<run-id>`）。
**どちらも ws11 で回した run で、この package の 3 本はローカル実行**になる。`trainer.deterministic`
は両者とも `False` なので、いずれにせよ seed を揃えても完全再現はしない。
