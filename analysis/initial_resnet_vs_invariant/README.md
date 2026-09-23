# 初期探索: ResNet vs attribute-invariant ResNet

通常の ResNet と attribute-invariant ResNet を **seed 43 / 44 / 45 の 3 本ずつ**で比較し、
invariant 化による性能・公平性指標の傾向を初期探索する。

## 分析対象

対象 run の manifest は [runs.md](runs.md) を正本とする。6 本とも `weighting=inverse`・30 epoch で、
違いは **seed と第1段の loss だけ**になる。invariant 側の adversary は sex / race / age にかかる。

**seed 42 の pair は含めない。** s42 の invariant は adversary が全属性にかかる別設定で、
同じ系列として平均を取ると手法の差と設定の差が混ざる。詳細は [runs.md](runs.md)。

## 評価の切り方

**評価は val selection に使っていない `test` split で行う。** checkpoint は 6 本とも val AUROC で
選んでいるので、val で比べると選択の効いた側へ寄る。

**seed は手法ごとに平均と標準偏差へ畳む**（標本 SD、`ddof=1`、n=3 なので自由度 2）。
per-seed の表も必ず残し、notebook では平均±SD に **3 seed の点を重ねて**描く。
n=3 の平均は 1 本の外れで動くので、散らばりを見せずに平均だけ読ませない。
**SD は揺れの目安であって検定ではない。**

**全体性能は test 全行（22,268 行）で測る。** 属性の欠損で母集団を削ると、「invariant 化で
分類性能をいくら払ったか」が属性の欠損パターンに依存してしまう。

**公平性の母集団は属性ごとに取る。** sex / ethnicity / age_group_65 は各属性の非欠損行すべて、
race だけ **White / Asian / Black** に絞る。test split の race は 6 カテゴリあるが、残り 3 つは
n=3,280 / 356 / 49 で、陽性が 32 例・5 例しかない群の TPR が gap を決めてしまう。ここは属性を
交差させないので母集団を揃える必要が無く、**定義を変えたのが race だけ**になるこの取り方にする。
その代わり **race の値は学習側が記録した `val/race/*` とは別定義**になる。突き合わせない。

**gap だけでなく worst / best と各群の n も出す。** gap が縮んでも、worst が上がったのか
best が下がったのかで意味が逆になる。また gap は必ず一番外れた群が決めるので、その群が何人かを
知らないと差を読めない。

公平性指標（Eopp0 / Eopp1 / Eodds と worst / gap）は `projects/` から import せず、群ごとの
TPR / FPR から `groups.py` が組み立てる。学習側と同じ定義であることは
[`analysis/tests/test_fairness_agreement.py`](../tests/test_fairness_agreement.py) が
golden データで固定する。

## 構成

| file | 入力 → 出力 |
| --- | --- |
| `groups.py` | 予測 cache → `results/` の per-seed 表 3 つと seed 集約表 2 つ |
| `classification_performance.ipynb` | `results/` → 分類性能と公平性の表・図・考察 |
| `attribute_probe.py` | 特徴量 cache → `results/` の probe 表 2 つ（per-seed と seed 集約） |
| `attribute_probe.ipynb` | `results/` → 属性 probe の表・図・考察 |
| `reports/` | 読み取った結論 |

`results/` の内訳は、per-seed が `classification_performance_test.csv`・`group_metrics_test.csv`・
`fairness_metrics_test.csv`・`attribute_probe_test.csv`、seed 集約が
`classification_performance_by_method_test.csv`・`fairness_metrics_by_method_test.csv`・
`attribute_probe_by_method_test.csv`。集約を script 側に置くのは、`reports/` が notebook を
実行せずに数値を引用できるようにするため。

**問いが 2 つある。** 分類性能・公平性（`groups.py`）と、属性が表現から読めるか（`attribute_probe.py`）で、
同じ 6 run を別の角度から見る。前者は予測 cache（logits）を、後者は特徴量 cache（2048 次元の表現）を
入力に取る。notebook も `reports/` も問いごとに分ける。

**図はこの package では notebook が持つ**（`plots.py` を置かない）。初期探索なので、
問い → 表 → 図 → 読み取りが 1 本に並ぶほうが分析の流れと対応する。再利用する図・レポートが
PNG として引用する図・大量条件を一括出力する図が出てきた時点で `plots.py` へ出す
（[`analysis/AGENTS.md`](../AGENTS.md)）。

`cache/` と `results/` は再生成できるので Git 管理外。notebook の出力も `nbstripout` が落とす。

## 作り直す

```bash
uv run python analysis/common/predictions.py --study initial_resnet_vs_invariant --split test \
  --run-dir analysis/initial_resnet_vs_invariant/runs/20260922T063725Z-resnet-chexpert-s43-efcd \
  --run-dir analysis/initial_resnet_vs_invariant/runs/20260922T063727Z-resnet-chexpert-s44-6f0c \
  --run-dir analysis/initial_resnet_vs_invariant/runs/20260922T063727Z-resnet-chexpert-s45-fb77 \
  --run-dir analysis/initial_resnet_vs_invariant/runs/20260922T063725Z-resnet-chexpert-attribute-invariant-s43-546c \
  --run-dir analysis/initial_resnet_vs_invariant/runs/20260922T063728Z-resnet-chexpert-attribute-invariant-s44-8e6b \
  --run-dir analysis/initial_resnet_vs_invariant/runs/20260922T063728Z-resnet-chexpert-attribute-invariant-s45-404a
uv run python analysis/initial_resnet_vs_invariant/groups.py --split test
```

そのうえで `classification_performance.ipynb` を kernel restart → 全実行する。
`--split val` にすると val 側の表になる。

属性 probe はさらに **train / val / test の特徴量**が要る（`--features`。予測 cache とは別ファイルに
書かれる。6 run × 3 split で約 4.7GB、A6000 で 30 分ほど）。

```bash
for split in train val test; do
  uv run python analysis/common/predictions.py --study initial_resnet_vs_invariant \
    --split "$split" --features --run-dir analysis/initial_resnet_vs_invariant/runs/...   # 上と同じ 6 run
done
uv run python analysis/initial_resnet_vs_invariant/attribute_probe.py
```

そのうえで `attribute_probe.ipynb` を kernel restart → 全実行する。probe は **train で学習し、
val で止めどきを決め、test で報告する**（`--fit-split` / `--select-split` / `--eval-split` で変えられる）。

## 結論

正本は [reports/overall_comparison.md](reports/overall_comparison.md) と
[reports/attribute_probe.md](reports/attribute_probe.md)。要点は次の 4 つ。

1. **この 3 seed では、invariant 化による公平性の改善は確認できない。** Eopp1 の差は 4 属性とも
   SD 以下で、seed ごとに見ると符号まで入れ替わる。
2. **代わりに小さな性能低下がある。** balanced accuracy は 3 seed の範囲が重ならず −0.0038、
   AUROC も −0.0030。worst-group は sex と ethnicity で 3 seed とも invariant が低い。
3. **s42 の 1 本で見えた改善は再現しなかった。** ResNet 側の絶対値すら s42 は 3 seed の範囲から
   外れており（accuracy 0.731 対 0.768〜0.779）、**1 本では accuracy を 4 ポイント外し得る**。

4. **その adversary は属性を消せていない。** 表現から属性を当てる post-hoc probe を後から当てると、
   invariant 側でも sex は balanced accuracy 0.871、年齢は MAE 10.0 歳で読める。chance からの
   上積みの 94〜98% が残る。**gap が動かないのは、そもそも表現が変わっていないため**になる。

次に決めること: **adversary を効かせる**（`attribute_adversary_weight` か `gradient_scale` を上げ、
どこまで上げれば probe が chance へ落ちるかを見る）、そのうえで **seed を増やすか**、
**閾値の扱いをどうするか**。詳細は各レポートの「次に決めること」を見る。
