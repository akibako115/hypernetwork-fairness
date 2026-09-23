# 全体性能と公平性の比較 — ResNet vs attribute-invariant ResNet（3 seed）

対象は [runs.md](../runs.md) の 6 run（seed 43 / 44 / 45 × 2 手法）。評価は test split、
checkpoint は 6 本とも val AUROC で選んだもの。図と表は
[classification_performance.ipynb](../classification_performance.ipynb) にある。数値は
`results/classification_performance_test.csv`・`fairness_metrics_test.csv` と、その seed 集約
（`*_by_method_test.csv`）から引いた。SD はすべて標本 SD（`ddof=1`、n=3、自由度 2）。

## 結論

**この 3 seed では、invariant 化による公平性の改善は確認できない。** 代わりに、小さいが
読み取れる程度の分類性能の低下がある。

## 1. 全体性能は invariant のほうがわずかに低い

| 指標 | ResNet | invariant | Δ mean |
|---|---|---|---|
| balanced accuracy | **0.7968 ± 0.0014** | 0.7930 ± 0.0016 | −0.0038 |
| AUROC | **0.8639 ± 0.0011** | 0.8609 ± 0.0025 | −0.0030 |
| accuracy | 0.7719 ± 0.0061 | 0.7746 ± 0.0157 | +0.0027 |
| cross entropy | 0.4641 ± 0.0044 | 0.4663 ± 0.0137 | +0.0022 |

**balanced accuracy は 3 seed の範囲が重ならない**（ResNet の最小 0.7957 > invariant の最大 0.7948）。
AUROC もほぼ分離する（重なるのは invariant の s45 = 0.8638 が ResNet の s45 と並ぶ 1 点だけ）。

一方 accuracy と cross entropy は完全に重なり、**invariant の SD が 2.5〜3 倍**ある
（accuracy 0.0157 対 0.0061）。invariant の accuracy 平均が高く見えるのは s43 の 0.7926 が
引き上げているためで、残り 2 本（0.7669 / 0.7642）は ResNet より低い。

## 2. 公平性の gap は、差の向きすら seed で入れ替わる

主要な gap の Δ mean は、いずれも SD と同程度かそれ以下になる。

| 属性 | Eopp1 (ResNet) | Eopp1 (invariant) | Δ mean |
|---|---|---|---|
| sex | 0.0274 ± 0.0189 | 0.0356 ± 0.0065 | +0.0082 |
| race | 0.0671 ± 0.0013 | 0.0668 ± **0.0304** | −0.0002 |
| ethnicity | 0.0197 ± 0.0125 | 0.0308 ± 0.0042 | +0.0110 |
| age_group_65 | 0.2505 ± 0.0077 | 0.2428 ± **0.0329** | −0.0077 |

seed ごとに差を取ると、**4 属性すべてで 3 seed のうちに符号の入れ替わりがある**。race は
−0.033 / +0.025 / +0.008、age は +0.032 / −0.008 / −0.047。平均の符号は、たまたま揃った 2 本が
決めているに過ぎない。**Eopp 系については、差の大きさどころか向きも主張できない。**

## 3. 唯一一貫しているのは worst-group のわずかな低下

seed ごとに見て 3 本とも同じ符号になるのは、次の 3 つだけだった。

| 指標 | Δ mean | 3 seed の符号 |
|---|---|---|
| sex worst AUROC | −0.0035 | 3 本とも invariant が低い |
| sex worst bACC | −0.0046 | 3 本とも invariant が低い |
| ethnicity worst bACC | −0.0069 | 3 本とも invariant が低い |

いずれも 1 節の全体性能の低下と同じ向きで、大きさも同程度（0.003〜0.007）。**invariant 化が
worst-group を特別に持ち上げた形跡は無く、全体が少し下がった分だけ worst も下がっている。**

## 4. seed 42 の 1 本で見えたものは再現しなかった

以前この package が s42 の 1 対 1 比較から読み取っていた結論は、次のように変わる。

| 当時の読み取り（s42、1 本） | 3 seed での実際 |
|---|---|
| race の Eopp1 が 0.0739 → 0.0406（−0.0333）と改善 | Δ −0.0002、invariant の SD 0.0304 |
| 全体性能は落ちていない（AUROC +0.0023、CE −0.034） | AUROC −0.0030、CE +0.0022 |
| worst-group は 4 属性すべてで改善 | 4 属性すべてで低下または横ばい |

s42 の invariant は adversary が全属性にかかる別設定だったので、手法の違いも混ざる。
ただしそれだけでは説明が付かない点がある。**ResNet 側の絶対値も s42 は 3 seed の範囲から
外れている**（accuracy 0.7309 対 0.7680〜0.7790、cross entropy 0.5536 対 0.4603〜0.4689）。
設定差は `trainer.deterministic` だけなので、これは seed 由来の揺れになる。
**1 本の run では accuracy を 4 ポイント外し得る**、というのがこの比較から得られた最も実用的な教訓である。

## 限界

- **n=3。** SD の自由度は 2 しかなく、SD 自体が大きく揺れる。ここで言えるのは「範囲が重なるか」
  「符号が揃うか」までで、検定はしていない。
- checkpoint は 6 本とも **val AUROC** で選んでいる。公平性で選んだ checkpoint ではない。
- race は White / Asian / Black に絞った。落とした 3 カテゴリ（n=3,280 / 356 / 49）は
  どの行にも入っていない。この絞り込みのため race は学習側の `val/race/*` と別定義になる。
- ws11 で回した 6 本は `trainer.deterministic=False`。同じ seed でも完全再現はしない。
- 交差群（age × sex × race）は見ていない。単独属性で動かなくても交差群で動くことはある。

## 次に決めること

1. ~~**adversary が効いているかを確かめる。**~~ → [attribute_probe.md](attribute_probe.md) で確認した。
   **消せていない。** 表現から属性を当てる post-hoc probe を当てると、invariant 側でも sex は
   balanced accuracy 0.871、年齢は MAE 10.0 歳で読め、chance からの上積みの 94〜98% が残る。
   したがって 2 節の「gap が動かない」は adversarial debiasing の限界ではなく、
   **adversary が働いていない状態**を見ている。次は重み（`attribute_adversary_weight=0.1`）を
   上げる実験になる。
2. **seed を増やすか決める。** n=3 では「効果が無い」を示す側の根拠として弱い。ただし 1 と 2 は
   順番が逆にできない。効いていない設定で seed を積んでも、効いていないことの精度が上がるだけになる。
3. **閾値の扱いを決める。** Eopp 系は動作点に強く依存する。群ごとの閾値調整を入れるのか、
   固定閾値のまま読むのかで、2 節の読み方が変わる。
