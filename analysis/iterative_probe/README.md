# iterative_probe

`hypernet_iterative` を本格的に回す前の探り分析。反復学習の**初期方針**、つまり次にどの設定で
実験を組むかを決めるために、最小構成で回した run を読む。結論そのものより、次に振るべき
パラメータを絞ることが目的になる。CheXpert / Spatial LoRA / seed 42。

## 決めたいこと

1. **GroupDRO の step size をどこに置くか。** 先行の探り run（[runs.md](runs.md)）では
   `group_dro_step_size=0.001` で `q` がほとんど動かなかった。本実験は 1e-3 と 1e-2 を並べ、
   `q` がどこまで偏り、その代わりに global と群の性能がどう動くかを見る。
2. **stage を重ねる価値があるか。** warmup → stage01 → stage02 で、global の `val/auroc` と
   hidden cohort の worst がどちらへ動くか。cohort は stage ごとに引き直されるので、stage をまたいで
   hidden group を対応させることはできない。この点をどう扱うかも、ここで方針を決める。
3. **変調範囲（`fc` / `stage4+fc`）を広げる価値があるか。**
4. **epoch 数と stage 数の最小ライン。** 本番の実験で何 epoch 必要かの当たりを付ける。

## 対象 run

変調範囲 × GroupDRO step size の 2×2（seed 42）と、global 指標の比較対象にする通常 ResNet
（ERM、30 epoch）を読む。5 本とも `runs/` に取り込み済みである。run-id と条件、選択 checkpoint の
epoch は [runs.md](runs.md) を正本とする。

baseline は [initial_resnet_vs_invariant](../initial_resnet_vs_invariant/) が invariant 化の対照に
使っている run と同じである。split の sha256・optimizer・batch size・class weight・seed が iterative 側と
揃っているので、global の val 指標はそのまま並べて読める。val loss も両 project で同じ定義
（重みを掛けない素の cross-entropy）である。

## 読み方

| 見るもの | 入力 | notebook / report |
|---|---|---|
| global 指標の epoch 推移（AUROC・bACC・loss） | `stages/*/metrics/metrics.csv`、baseline は W&B transaction log | [`global_training_curves.ipynb`](global_training_curves.ipynb) / [report](reports/global_training_curves.md) |
| `q` と cohort ごとの性能推移、属性群の worst / gap / Eopp の推移 | `stages/*/metrics/metrics.csv`、baseline は W&B transaction log | [`subgroup_training_curves.ipynb`](subgroup_training_curves.ipynb) / [report](reports/subgroup_training_curves.md) |
| cohort の属性・撮影方向の構成、cohort × class の class weight、`q` の上位・下位 cohort の構成 | `artifacts/cohorts/`、`stages/*/config.yaml`、train split CSV | [`cohort_composition.ipynb`](cohort_composition.ipynb) / [report](reports/cohort_composition.md) |
| cohort ごとの切片のずれと pooled AUROC の低下、test の属性群の worst AUROC と動作点での TPR・FPR の群間差 | 予測 cache（`cache/<run-id>_{val,test}.npz`）、`artifacts/cohorts/cohort02/`、`stages/stage02/config.yaml` | [`cohort_logit_correction.ipynb`](cohort_logit_correction.ipynb) / [report](reports/cohort_logit_correction.md) |
| test の交差群の worst / best / gap | 予測 cache（`cache/<run-id>_test.npz`） | 未着手 |
| 特徴量 | checkpoint から取る表現 | 未着手 |

**global を払った分だけ群が良くなったのかを見る。** GroupDRO は worst を持ち上げる代わりに global を
払いうる。global の推移だけで判断せず、群別の結果と組にして読む。

## いまの段階

**各条件 seed 1 本（42）の探り段階である。** 条件間の差は seed 間の揺れと区別できない。baseline 自身も
epoch 間で val AUROC が 0.01 前後振れる。したがって、**この package の数値は推移の形と桁を読むためのもの**で、
条件の優劣を決める根拠には使わない。

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
   - cohort の中身（属性の構成と class weight）は学習前に決まる性質なので、推移とは別の notebook に置く
3. **特徴量分析**
   - checkpoint から取得した特徴量の読み込み
   - 特徴量と属性・cohort・予測性能の関係
   - 必要に応じた低次元可視化や probe

notebook のファイル名は、分析対象が一目で分かる名前を採用する。いまある notebook は
`global_training_curves.ipynb`（トピック 1 の global 推移）、`subgroup_training_curves.ipynb`（トピック 2 の推移）、
`cohort_composition.ipynb`（トピック 2 の cohort の中身）、`cohort_logit_correction.ipynb`（トピック 1 の test での比較）。1 つの notebook には 1 つの問いの型（推移か、静的な構成か、
test での比較か）だけを置く。

まず notebook 内で分析を組み立て、同じ処理を複数の分析で使う、実行に時間がかかる、または処理が
長くなって読めない、と分かった段階で外部モジュールへ切り出す。いま package にある module は次の 2 つで、
どちらも notebook から import する（CLI は持たない）。

| module | 持つもの | 切り出した理由 |
|---|---|---|
| `_shared.py` | iterative run の stage をまたいだ epoch 表と cohort 表の読み取り | 複数の notebook が同じ読み方をする |
| `groups.py` | 属性群・交差群の切り方と、群別指標・worst / best / gap の定義 | 学習側と同じ定義であることを golden データで固定する（`analysis/tests/test_fairness_agreement.py`、`test_group_rows.py`） |

どの run を比べるか、表や図をどこへ書くかは notebook が決め、module には置かない。

予測 cache は共有 CLI（`analysis/common/predictions.py`）を使ってもよいが、cache の読み込みから
群別指標の計算・可視化までを別々の script に分けること自体は要求しない。群ごとの公平性指標は
run artifact に無いため、群の切り方と指標の意味を分析 notebook で明示する。

`cache/` `results/` `figures/` は再生成できるので Git 管理外。`reports/` は分析メモと、そこから
決まった次の実験方針を置く。

hidden cohort の epoch ログは cohort ごとの `AUROC`・`bACC`・`loss`（および support）を raw metric とし、
`_shared.add_hidden_summaries` が `min`・`max`・`gap` を後計算する。これにより logger 側と分析側で派生指標の定義が
重複しない。

## 再実行

分析 notebook は、冒頭に対象 run、split、checkpoint、使用する入力の場所を明記し、kernel restart
後に上から実行できる状態にする。run-id は条件順（fc 1e-3 → fc 1e-2 → stage4+fc 1e-3 →
stage4+fc 1e-2）に並べる。表と図の並びがこの順になる。

予測 cache が必要な場合は、notebook の分析を実行する前に共有 CLI で作成する。`cohort_logit_correction.ipynb` は
動作点の閾値を val で決めるので、`--split val` と `--split test` の両方を作る。

```bash
uv run python analysis/common/predictions.py --study iterative_probe --split test \
  --run-dir analysis/iterative_probe/runs/20260921T103036Z-resnet-chexpert-s42-5538 \
  --run-dir analysis/iterative_probe/runs/20260924T112035Z-spatial-lora-iterative-chexpert-fc-s42-b81a \
  --run-dir analysis/iterative_probe/runs/20260924T112045Z-spatial-lora-iterative-chexpert-fc-s42-4602 \
  --run-dir analysis/iterative_probe/runs/20260924T112034Z-spatial-lora-iterative-chexpert-stage4-fc-s42-1caa \
  --run-dir analysis/iterative_probe/runs/20260924T112043Z-spatial-lora-iterative-chexpert-stage4-fc-s42-c755
```

notebook は repo root の kernel（`hypernet-fairness`）で開くか、次のように通しで実行する。

```bash
cd analysis/iterative_probe && uv run jupyter nbconvert --to notebook --execute --inplace \
  --ExecutePreprocessor.kernel_name=hypernet-fairness global_training_curves.ipynb subgroup_training_curves.ipynb cohort_composition.ipynb \
  cohort_logit_correction.ipynb
```
