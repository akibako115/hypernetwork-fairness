# 公平性損失の実装監査（2026-10-13）

> 後続の公式実装調査とユーザー方針に基づく更新：DANN/ALFR の欠測時 KeyError は修正済み。
> GroupDRO は e2e・iterative とも、欠損群の q を据え置く独自更新を廃止し、
> `q *= exp(eta * group_loss); q /= q.sum()` に戻した。自然抽出とクラス重みは維持し、対数保持は導入していない。
> 自然抽出で欠損群が生じること自体は公式実装にもある手法上の性質であり、以下の P2 評価をそのまま実装バグとは扱わない。
> 以下は修正前の監査記録。今回の関連テストは108件成功、ruff成功。実データでの再学習は未実施。


現行実装には、DANN/ALFR の欠測処理で停止する不具合がある。GroupDRO は欠損群の q を保つ修正後も、自然抽出で希少群の寄与が弱まる問題が残る。一方、EO・Minimax AUC・抽出補正・FairDRO の主要な計算と更新順序は、確認した仕様と整合する。既存テストの成功だけを根拠に、全手法が公平性を改善すると判断することはできない。

対象は現在の作業ツリー。学習コード、設定、既存の実験成果物は変更していない。既存の修正記録 `20261013_fairness_objective_fixes.md` を読み、独立にコードと数値例を確認した。実データでの再学習は行っていない。

## 実装・規約の確認：欠測処理に再現可能な停止条件がある

### P2：選択した属性の一部だけが全件欠測だと DANN/ALFR が停止する

対象：`projects/hypernet_e2e/objectives/attribute_invariance.py:339`。

内部の `component_losses()` は、バッチ全件が欠測の属性について損失を作らない。しかし `loss_components()` の診断値作成は、選択した全属性を辞書から無条件に取り出す。sex が観測され、race が全件欠測だと `KeyError: 'categorical_1'` になる。学習だけでなく validation でもこの経路を通る。

CheXpert の attribute_invariant / DANN / ALFR preset は sex・race・age を選択するため、この条件の対象になる。通常の大きいランダムバッチでは発現しにくくても、末尾の小さいバッチや欠測が偏ったデータで起き得る。過去の run で発現したかは確認していない。

最小再現（リポジトリ直下で `uv run python`）：

```python
import torch
from projects.hypernet_e2e.objectives.attribute_invariance import AttributeInvariantTaskLoss
from projects.hypernet_e2e.objectives.input import ObjectiveInput
from projects.hypernet_e2e.objectives.task import TaskLoss

objective = AttributeInvariantTaskLoss(
    TaskLoss(), 3,
    {"categorical": ["sex", "race"]},
    {"categorical": ["sex", "race"]},
    categorical_cardinalities=[2, 3],
)
inputs = ObjectiveInput(
    logits=torch.zeros(2, 2, requires_grad=True),
    target=torch.tensor([0, 1]),
    features=torch.randn(2, 3, requires_grad=True),
    attributes={
        "categorical": torch.zeros(2, 2, dtype=torch.long),
        "categorical_missing": torch.tensor([[False, True], [False, True]]),
    },
)
objective(inputs)  # KeyError: 'categorical_1'
```

修正方針は、存在する属性損失だけを診断値に移すこと。欠測属性をゼロ損失として平均に混ぜると、観測された属性への学習圧力まで弱くなるため、現在の「観測された属性だけで平均する」計算は維持する。

規約上の軽微な指摘として、`cohort.py` の公開 `forward()` には、project の AGENTS.md が必須とする日本語の `Args:` / `Returns:` docstring が不足する。数値挙動には影響しない。

## 仕様・数理の確認：GroupDRO の欠損群対策は部分的

### P2：自然抽出では、群が現れる頻度が目的関数に残る

対象：e2e `objectives/group.py:75`、`:130` と iterative `loss.py:201`、`:322`、`:393`。

現在の実装は、バッチにいない群の損失を 0 にして `sum_g q_g * loss_g` を返す。q を一定として勾配の推定だけを切り出すと、群 g の期待寄与は `q_g * P(g がバッチに存在) * grad(R_g)` となる。欠損群の q を据え置いても、この出現確率の係数は消えない。

独立復元抽出なら出現確率は `1 - (1 - p_g)^B`。2群の比率が 0.9 / 0.1、B=2、q=(0.5, 0.5)、群損失=(1, 2) の例では、群が両方観測されるときの目的値は 1.5 だが、欠損群を0として集計した期待値は 0.685。群別の係数は 0.495 / 0.095 となり、希少群が相対的に弱まる。この値は q を固定した比較であり、学習中の q の適応を含む最終性能の予測ではない。

さらに、新しい観測群内だけの q 更新では、1群しか現れないバッチの q は必ず不変になる。損失の異なる2群を1件ずつ交互に20バッチ渡しても、q は初期値 (0.5, 0.5) のままだった。この極限では最悪群を追う更新として機能しない。

手元の `data/chexpert/splits_race_known/train.csv` は157,852件。年齢65歳以上×race_group=2 は2,808件で、B=128 の非復元ランダム抽出でこの群が欠ける確率は **10.041968%**。これは超幾何分布から計算した値で、実バッチの観測頻度ではない。q を固定した場合、この群の勾配寄与には約0.89958が掛かる。HAM10000 の GMM8 最小群では同じ問題は小さく、B=128 の復元抽出近似で欠損確率は約0.71%。影響の大きさはデータとバッチサイズに依存する。

推奨する対応は、群ごとの損失を毎バッチ観測できる抽出方式にするか、自然抽出を維持するなら推定する目的関数と補正を明示すること。q の更新だけを変更して「欠損群問題を解決した」とは扱わない。原典と同一の再現が目的なら、独自の欠損群更新を原典の更新と区別する必要もある。[GroupDRO 公式実装](https://github.com/kohpangwei/group_DRO/blob/master/loss.py)も、空群の平均を0とする処理と全群を正規化する指数更新を持つ。

### 数値ストレス試験：q が0になった後の再観測で NaN

`_observed_exponentiated_update()` は `updated.sum()` が0の場合を処理していない。float32、q=(0.5,0.5)、損失=(0,20000)、step_size=0.01、全群観測で q=(0,1) になり、次のバッチで群0だけを観測すると q=(NaN,1) になった。e2e と iterative に同じ式がある。

これは極端な損失を与えた境界試験であり、通常の学習で発生した証拠はない。長期の数値安定性を確保するなら log-space の状態管理とゼロ質量時の方針が必要。既定の実験を直ちに無効と判断する根拠にはしない。

## 確認できた計算と、バグに分類しない注意点

| 手法 | 確認した点 | 残る注意点 |
| --- | --- | --- |
| EO rate | TPR/FPR の全有向群ペア、hinge 上下界、旧 λ によるモデル損失、hard 制約による列確率行列更新 | `active_fraction` は旧 smoothstep の帯域の割合。hinge の勾配がある標本の割合ではない |
| Minimax AUC | 陽性群×陰性群の pairwise softplus、母集団由来 q、旧 q で forward 後に一度 commit | 最悪の順位損失を狙う手法で、固定閾値の EO 改善を保証しない |
| Population AUC | 同じ順位損失と固定された母集団由来 q | 公平性の双対更新を持たない対照 |
| 抽出補正 CE / 群リスク | P/Q、固定正規化、群内の母集団クラス比への復元 | セル均等抽出が前提。起動時に DataModule を検証している |
| FairDRO | classwise q、0–1誤りに対する閉形式応答、epoch 単位更新、平滑化スケジュール、held-out 全件への切替 | 負の q は原典の仕様。既定 rho=1 を全群数で安定と見なさない |
| DANN / ALFR | GRL 符号、adversary の最小化、ALFR の交互更新 | 上記の欠測停止条件。LeakyReLU 化だけで属性情報が消えるとは保証できない |
| FIS | 個体重み・群重みの和がそれぞれ1、混合比の整合 | detach とバッチ内群で距離を測る点は公式実装との差分 |
| FaMI / Worst cohort | 記載された KDE score 差・群最大損失との対応 | FaMI の KDE は代替実装。原著の完全再現と呼ばない |

条件付き objective は起動時に single device、accumulation=1、bf16/float32、セル均等 DataModule を要求する。held-out の統計で pending を置き換える処理と、モデル更新後の commit が配線されている。これらの制限外での動作は今回保証していない。通常の GroupDRO の複数 device による分散学習も検証対象外。

FairDRO の G=4、rho=1、誤り率 (0,1,1,1) では q=(-0.183013,0.394338,0.394338,0.394338) を再現した。負の重みのセルでは CE を増やす方向に勾配が働く。これは[原論文の quasi-probability と最適化手順](https://arxiv.org/html/2303.00442)に含まれる挙動で、負値を禁止すれば別の手法になる。rho の調整と q・セル別性能の記録で判断すべきである。

FIS は重み和が1の状態でさらに `.mean()` を取る。全標本の損失が等しい B=128 の例では、CE 0.693147 に対して FIS は0.005415、logits の勾配L1は1に対して0.0078125。この集約は[FairVision 公式実装](https://github.com/Harvard-Ophthalmology-AI-Lab/FairVision/blob/f16fa2738ac9b325d584607ca7fa0c74b4d6861d/src/modules.py#L757-L769)にもあるため、バグには分類しない。ただし損失と勾配の尺度はバッチサイズに依存する。Adam の更新量まで単純に1/128になるという意味ではない。

## 検証範囲

次の既存テストは **131件成功、8.61秒**。損失値・勾配・双対更新・Lightning 統合・held-out の切替・DataModule を含む。

```sh
uv run pytest \
  projects/hypernet_e2e/tests/test_loss.py \
  projects/hypernet_e2e/tests/test_conditional_objectives.py \
  projects/hypernet_e2e/tests/test_cohort_objectives.py \
  projects/hypernet_e2e/tests/test_module.py \
  projects/hypernet_e2e/tests/test_group_datamodule.py \
  projects/hypernet_iterative/tests/test_loss.py \
  -q --durations=5
```

別途、欠測属性の KeyError、1群バッチでの q 不変、q のゼロ質量による NaN、FairDRO の負の q を数値例で確認し、train CSV から欠損群確率を計算した。独立レビューで一部のテストを再実行したが、上記の131件に重複加算していない。

実装・規約軸は実害1件と軽微な文書規約違反、仕様・数理軸はGroupDROの部分的な欠損群対策1件と数値ストレス条件1件。優先対応は欠測停止の修正と、GroupDRO の抽出方式・目的関数の整合である。実データでの公平性改善、EO 定常分布計算の長期安定性、GPUでの長期学習は未確認。
