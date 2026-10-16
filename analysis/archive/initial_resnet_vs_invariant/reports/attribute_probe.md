# 属性 probe — adversary は属性を消せたか（3 seed）

対象は [runs.md](../runs.md) の 6 run（seed 43 / 44 / 45 × 2 手法）。分類 head の直前の表現
（2048 次元）を凍結し、そこから属性を当てる probe を後から学習する。表と図は
[attribute_probe.ipynb](../attribute_probe.ipynb)、数値は `results/attribute_probe_test.csv` と
その seed 集約（`attribute_probe_by_method_test.csv`）から引いた。SD は標本 SD（`ddof=1`、n=3、自由度 2）。

[overall_comparison.md](overall_comparison.md) の「次に決めること 1」に答える分析になる。
学習ログには属性予測損失が残っていない（`metrics/fit.json` が持つのは `train/loss` の合計だけ）ので、
ログからではなく checkpoint から測った。

**手順は post-hoc attacker の慣行に合わせた**（Elazar & Goldberg 2018）。probe は **train で学習し、
val で止めどきを決め、test で報告する**。「消えた」と主張したい側の分析なので、probe は強いほうへ倒す。
probe は **linear**（線形層 1 枚）と **mlp**（hidden 256、学習時の adversary と同じ容量）の 2 本で、
層の深さ以外は同じ手順で学習する。

## 結論

**adversary は属性を消せていない。** invariant 化した表現からも、sex は balanced accuracy 0.871、
年齢は MAE 10.0 歳で読める。chance からの上積みで見ると **94〜98% が残っている**。
[overall_comparison.md](overall_comparison.md) の「性能は下がるのに gap が動かない」は、
公平性指標の問題ではなく、**そもそも表現が変わっていない**ためと読める。

## 1. 属性はほぼそのまま読める

mlp probe（adversary と同じ容量）の test balanced accuracy。`残存` は chance からの上積みが
どれだけ残ったか（`(invariant − chance) / (ResNet − chance)`）。

| 属性 | adversary | ResNet | invariant | Δ mean | chance | 残存 |
|---|---|---|---|---|---|---|
| sex | 対象 | 0.8863 ± 0.0106 | 0.8708 ± 0.0037 | −0.0155 | 0.500 | **96%** |
| age_group_65 | 対象 | 0.7559 ± 0.0093 | 0.7414 ± 0.0012 | −0.0145 | 0.500 | **94%** |
| race | 対象 | 0.5569 ± 0.0157 | 0.5514 ± 0.0046 | −0.0055 | 0.333 | **98%** |
| ethnicity | 対象外 | 0.6192 ± 0.0011 | 0.6155 ± 0.0058 | −0.0036 | 0.500 | 97% |
| frontal_lateral | 対象外 | 0.9991 ± 0.0001 | 0.9991 ± 0.0002 | −0.0001 | 0.500 | 100% |

age は回帰のまま測る（adversary も MSE で消しにいく）。chance は「fit split の中央値を常に答えた場合」。

| 指標 | ResNet | invariant | Δ mean | chance |
|---|---|---|---|---|
| MAE (years) | 9.71 ± 0.32 | 10.00 ± 0.03 | +0.29 | 14.38 |
| R2 | 0.5249 ± 0.0321 | 0.4973 ± 0.0033 | −0.0276 | 0.0 |

**消えた年齢情報は 0.29 歳ぶん**で、chance までの距離（4.67 歳）の 6% にあたる。

## 2. 下げ幅は対照属性と同じ桁

adversary の対象外である ethnicity も −0.0036 下がる。sex と age_group_65 の下げ幅はその 4 倍ほどあり、
**3 seed の範囲も重ならない**ので、adversary が「少しは」効いているとは言える。

| 属性 | ResNet の 3 seed | invariant の 3 seed | 範囲 |
|---|---|---|---|
| sex | 0.8742 / 0.8907 / 0.8940 | 0.8666 / 0.8738 / 0.8721 | 重ならない（0.8742 > 0.8738） |
| age_group_65 | 0.7479 / 0.7660 / 0.7539 | 0.7421 / 0.7401 / 0.7421 | 重ならない |
| race | 0.5388 / 0.5654 / 0.5665 | 0.5489 / 0.5567 / 0.5484 | 重なる。s43 は invariant のほうが高い |
| age (MAE) | 10.07 / 9.47 / 9.58 | 9.97 / 9.99 / 10.04 | 重なる。s43 は invariant のほうが当たらない |

**効いてはいるが、読めるかどうかは変わっていない。** sex を 87% で当てられる表現を「sex が消えた表現」
とは呼べない。linear probe でも向きは同じで、大きさは半分になる（sex −0.0080、age_group_65 −0.0071、
race は +0.0055 と逆符号）。**非線形にだけ消え残っているのではなく、線形にも残っている。**

## 3. probe が弱いせいではない

- **frontal_lateral は両手法とも 0.999。** probe 自体は機能している。probe が弱すぎて差が出ない、
  という説明は取れない。
- probe の学習行は 12.9 万〜17.8 万行（race だけ 129,857、他は 177,783）。2048 次元に対して足りている。
- **invariant 側の SD のほうが小さい**（sex 0.0037 対 0.0106）。3 本が揃って同じところに着いており、
  「たまたま消え残った 1 本」ではない。

## 4. overall_comparison との突き合わせ

| | 大きさ |
|---|---|
| 払った分類性能（balanced accuracy） | −0.0038 |
| 消せた属性（sex、balanced accuracy） | −0.0155（chance までの距離の 4%） |
| 動いた公平性 gap | SD 以下、符号も定まらない |

`attribute_adversary_weight=0.1` は、**性能を少し削り、属性はほとんど消さず、gap は動かさない**
設定になっている。gap が動かないことを adversarial debiasing の限界として読む前に、
まず adversary を効かせる必要がある。

## 限界

- **race は White / Asian / Black の 3 群に絞ってある**（分類性能側の表と同じ定義）。学習時の adversary は
  6 カテゴリを見ている。同じ問題ではないので、**この表の値を adversary の損失と突き合わせない。**
- **control task / selectivity（Hewitt & Liang 2019）は取っていない。** 同じ probe を両手法に当てた
  手法間の差は読めるが、絶対値（0.871 が「読めているうち」のどのあたりか）は読めない。
- mlp probe は hidden 256 の 1 本だけ。容量を上げれば読める分は増えうるが、ここでの結論は
  「消えていない」の向きなので、容量を上げても向きは変わらない。
- probe が読めることと、下流の分類器がその情報を使っていることは別。**probe の低下が公平性の改善を
  意味しない**ことは overall_comparison の 2 節がその実例になる。
- n=3。SD の自由度は 2 しかない。ここで言えるのは「範囲が重なるか」までで、検定はしていない。
- 6 本とも `trainer.deterministic=False`。同じ seed でも完全再現はしない。

## 次に決めること

1. **adversary を効かせる。** 重み `attribute_adversary_weight=0.1` を上げるか、GRL の
   `gradient_scale` を上げる。**どこまで上げれば probe が chance へ落ちるか**を先に見る実験にすると、
   「属性を消すと gap がどう動くか」を初めて 1 本の軸で読める。
2. **probe を回す split を固定する。** 今回 train 特徴量を 6 run 分（約 3.9GB）作った。重みを変えた
   run を追加するたびに同じ抽出が要るので、probe を回すのは **最終 checkpoint だけ**に決めておく。
3. **消えたかどうかの判定線を決める。** 「chance + SD 以内なら消えた」とするのか、
   「ResNet からの低下量」で見るのか。1 の実験を始める前に決めておかないと、後から都合よく読める。
