# adversary の強さと、属性が消えるまでの水準

[initial_resnet_vs_invariant](../initial_resnet_vs_invariant/) の設定
（`attribute_adversary_weight=0.1`、`gradient_scale=1.0`）では、**adversary が属性を消せていない**。
post-hoc probe を当てると、invariant 側でも sex は balanced accuracy 0.871、年齢は MAE 10.0 歳で読め、
chance からの上積みの 94〜98% が残る（[attribute_probe.md](../initial_resnet_vs_invariant/reports/attribute_probe.md)）。

この package は **「どこまで上げれば属性が消えるのか」と「消えたときに分類性能と gap がどう動くのか」**を見る。

## 振る量

backbone が受け取る逆向きの勾配は **λ = `attribute_adversary_weight` × `gradient_scale`** で効く
（`objectives/attribute_invariance.py`。総損失は `task + w · adv`、GRL が backbone へ渡す勾配は `−s · ∂adv/∂feat`）。
2 つの積でしか効かないので、**`attribute_adversary_weight` だけを振り、`gradient_scale=1.0` は据え置く**。

optimizer は AdamW なので、w を上げても **adversary head 自身の歩幅はほとんど変わらない**
（Adam は勾配の定数倍に対してほぼ不変で、効くのは weight decay との相対比だけ）。
w が実際に効くのは **backbone 側で task 勾配と逆向き勾配のどちらが勝つか**の比になる。

## 読み方

**属性が消えたかどうかを先に見る。** 公平性 gap は属性が消えて初めて動きうるので、
probe が chance へ落ちていない水準の gap を読んでも、[initial_resnet_vs_invariant](../initial_resnet_vs_invariant/)
と同じ「動かない」を繰り返すだけになる。

| 見るもの | 手順 |
| --- | --- |
| 属性が消えたか | `initial_resnet_vs_invariant/attribute_probe.py` と同じ手順の probe（train で fit → val で epoch 選択 → test で評価） |
| 払った分類性能 | test の balanced accuracy / AUROC |
| 公平性 gap | Eopp0 / Eopp1 / Eodds と worst / gap |

**collapse も結果として読む。** λ を上げると task loss が壊れることがある。壊れた水準は
「属性は消えるが使えない」として記録し、水準の上限にする。

## いまの段階

**水準を探る段階で、seed は 1 本（43）しか回さない。** 属性が消える λ の桁が分からないうちに
seed を積んでも、効いていない設定の分散が精密になるだけになる。水準が決まってから seed を足す。
したがって **この package の数値は、まだ手法の比較に使えない**（n=1）。

**adversary head 自身が属性を読めていない。** loss を記録して回し直した 4 本では、λ=1〜10 のどれでも
adversary loss が数 epoch で「事前分布だけを答える予測器」の水準に張り付く。λ=1 の head は、入力に sex の情報が
残っているのに AUROC 0.50 だった（[adversary_head.md](reports/adversary_head.md)）。λ を振る前に、
head が属性を追えるようにする必要がある。

対象 run は [runs.md](runs.md) を正本とする。
