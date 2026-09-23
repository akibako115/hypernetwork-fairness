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

## 比較対象（別 study の run）

λ=0.1 の水準は [initial_resnet_vs_invariant](../initial_resnet_vs_invariant/runs.md) の
`20260922T063725Z-resnet-chexpert-attribute-invariant-s43-546c`（seed 43）を、
adversary 無しの baseline は同 `20260922T063725Z-resnet-chexpert-s43-efcd` を引く。
どちらもこの package の `runs/` に取り込んである（`analysis/adversary_strength/runs/<run-id>`）。
**どちらも ws11 で回した run で、この package の 3 本はローカル実行**になる。`trainer.deterministic`
は両者とも `False` なので、いずれにせよ seed を揃えても完全再現はしない。
