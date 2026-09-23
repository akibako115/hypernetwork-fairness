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

変調範囲 × GroupDRO step size の 2×2（`runs/` に取り込み済み）と、global 指標の
比較対象にする通常 ResNet（`analysis/iterative_probe/runs/20260921T103036Z-resnet-chexpert-s42-5538`）。
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

## 分析の実装方針

この package は、反復学習の初期方針を探るための探索的な分析である。分析の保守性や共通化を
最初から最大化することよりも、**仮説・入力・分析・図・解釈を一つの流れで読めること**を優先する。

分析トピックごとに notebook を作り、原則として次の順で上から実行できる形にする。

1. import、定数、対象 run・split・checkpoint の宣言
2. 実行ログ、checkpoint、split / 属性データ、必要な cache の読み込み
3. 分析トピックごとの処理
4. その処理に対応する表・可視化・観察メモ
5. notebook 全体のまとめ

分析と可視化は、可視化の対象ごとに近くへ置く。例えば、epoch 推移を集計した直後に epoch の図を
表示し、群別性能を集計した直後に worst / best / gap の図を表示する。notebook の読者が、各図が
どの問いに答えるものかを追えることを重視する。

### notebook の分け方

当面は、次の 3 つの分析トピックを独立した notebook として持つ。

1. **全体性能・公平性指標**
   - run 間の global 指標の比較
   - test split における属性群・交差群の性能
   - worst / best / gap を含む公平性指標
2. **epoch ごとの `q` 重み・サブグループ性能推移**
   - stage / epoch ごとの GroupDRO の `q`
   - hidden cohort やサブグループの性能推移
   - stage の切り替えと cohort の引き直しを踏まえた解釈
3. **特徴量分析**
   - checkpoint から取得した特徴量の読み込み
   - 特徴量と属性・cohort・予測性能の関係
   - 必要に応じた低次元可視化や probe

notebook のファイル名は、分析対象が一目で分かる名前を採用する。現在の `probe.ipynb` は既存結果を
まとめて読む notebook として残っているが、今後の整理で上の 3 トピックへ分割する対象とする。

既存の `epoch_metrics.py`、`cohort_analysis.py`、`baseline_comparison.py`、`groups.py`、
`collect.py` は現在の実装として残っているが、新しい分析を追加する際の必須の分割単位ではない。
まず notebook 内で分析を組み立て、同じ処理を複数の分析で使う、実行に時間がかかる、または処理が
長くなって読めない、と分かった段階で外部モジュールへ切り出す。

例えば `groups.py` のような群定義・群別集計は、群の切り方自体がこの仮説固有の判断なので、まずは
分析 notebook の中に置く。必要になった場合だけ `iterative_probe` 内の小さな module へ切り出し、
群定義・集計関数などの重い実装をまとめる。実行順序、分析の意図、採用した判断は notebook に残す。

予測 cache は共有 CLI（`analysis/common/predictions.py`）を使ってもよいが、cache の読み込みから
群別指標の計算・可視化までを別々の script に分けること自体は要求しない。群ごとの公平性指標は
run artifact に無いため、群の切り方と指標の意味を分析 notebook で明示する。

`cache/` `results/` `figures/` は再生成できるので Git 管理外。`reports/` は分析メモと、そこから
決まった次の実験方針を置く。

hidden cohort の epoch ログは cohort ごとの `AUROC`・`bACC`・`loss`（および support）を raw metric とし、
`epoch_metrics.py` が `min`・`max`・`gap` を後計算する。これにより logger 側と分析側で派生指標の定義が
重複しない。

## 再実行

分析 notebook は、冒頭に対象 run、split、checkpoint、使用する入力の場所を明記し、kernel restart
後に上から実行できる状態にする。run-id は条件順（fc 0.001 → fc 0.01 → stage4+fc 0.001 →
stage4+fc 0.01）に並べる。表と図の並びがこの順になる。

予測 cache が必要な場合は、notebook の分析を実行する前に共有 CLI で作成する。cache の作成自体を
分析 notebook に埋め込むかどうかは、実行時間と再利用性を見て決める。

```bash
uv run python analysis/common/predictions.py --study iterative_probe --split test \
  --run-dir analysis/iterative_probe/runs/20260921T103036Z-resnet-chexpert-s42-5538 \
  --run-dir analysis/iterative_probe/runs/20260921T103222Z-iterative-s42-d24d \
  --run-dir analysis/iterative_probe/runs/20260921T103219Z-iterative-s42-e992 \
  --run-dir analysis/iterative_probe/runs/20260921T103225Z-iterative-s42-3166 \
  --run-dir analysis/iterative_probe/runs/20260921T103225Z-iterative-s42-fac5
uv run python analysis/iterative_probe/collect.py \
  20260921T103222Z-iterative-s42-d24d 20260921T103219Z-iterative-s42-e992 \
  20260921T103225Z-iterative-s42-3166 20260921T103225Z-iterative-s42-fac5 \
  --baseline 20260921T103036Z-resnet-chexpert-s42-5538
uv run python analysis/iterative_probe/groups.py --split test
```

現在の script 群を使って既存の結果を再生成する場合は、次のコマンドを利用できる。ただし、これは
今後の分析を必ずこの DAG に分解することを意味しない。

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
