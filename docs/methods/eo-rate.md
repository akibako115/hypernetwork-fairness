# EO rate（Cotter et al. の proxy-Lagrangian）：原論文・実装・使い方

2026-10-14 作成。原論文 2 本の本文を読み直し、この repo の実装と突き合わせた結果をまとめる。
EO rate の run を設計・起動・評価するセッションは、先にこれを読む。

## 先に押さえること

- **学習して、checkpoint を 1 つ選んで終わり。** 原論文の理論は「学習中の iterate を確率で混ぜた分類器」に
  しか保証を与えないが、実用上は途中の 1 点（best iterate）で十分という結果が同じ論文に載っている。
  混合分類器（shrinking）は作らない。
- **best val AUROC でも last でも選ばない。** best val AUROC は制約が効く前の epoch 0 を選びうる。
  last は過学習が進んだ末の 1 点で、原論文も危ないと書いている。下の「checkpoint の選び方」に従う。
- **既定の制約（全ペア・TPR と FPR の両方）は原論文の実験の形ではない。** 原論文に揃えるなら
  `constraint_form=overall` と `rates=[fpr]`（または `[tpr]`）を明示する。

## 原論文

| 論文 | 中身 | この repo との関係 |
| --- | --- | --- |
| Cotter et al. *Optimization with Non-Differentiable Constraints with Applications to Fairness, Recall, Churn, and Other Goals.* JMLR 20, 2019（[link](https://jmlr.org/papers/v20/18-616.html)） | proxy-Lagrangian。モデル側は微分可能な代理制約、λ 側は元の 0-1 の率で更新する。λ 側は swap regret を最小化する | 最適化の本体 |
| Cotter et al. *Training Well-Generalizing Classifiers for Fairness Metrics and Other Data-Dependent Constraints.* ICML 2019（[arXiv:1807.00028](https://arxiv.org/abs/1807.00028)） | λ を train と別のデータで更新する 2 データセット方式（dual holdout） | `*_dual_holdout` preset |

### 原論文の実験条件（本文で確認済み）

| 項目 | 原論文 | 出典 |
| --- | --- | --- |
| アルゴリズム | θ と λ を同時に確率的に更新。M は M_ij = 1/(m+1) の一様で初期化し、λ は M の定常分布 | ICML Algorithm 3 |
| 目的関数と代理制約 | どちらも hinge の上界 | ICML 5 節、JMLR 5 節 |
| 制約の形 | **各群の率を全体の率と比べる片側の制約。かける率は 1 種類** | JMLR 5.6、ICML 5.2 |
| 例 | COMPAS：TPR_g ≤ TPR_全体 + 0.05（4 群：Black・White・Female・Male、重なりあり）。Communities：FPR_g ≤ FPR_全体（8 群、ε = 0） | 同上 |
| 制約の本数 | 1〜19 本（単調性制約は別） | JMLR 表 5 |
| モデル | 線形モデル、隠れ層 1 層の小さな NN（5〜100 unit）、lattice。表データで数千〜約 130 万件 | JMLR 表 4、ICML 表 3 |
| dual holdout の分け方 | train を半分に分け、片方を λ 側に使う | ICML 5.2 |
| optimizer | θ は Adam の既定値（hyperparameter search を避けるため） | ICML 5 節 |
| 学習の長さ | 制約なしで性能が良い反復回数に固定 | JMLR 5.2 |
| 出力 | ICML：100 個の iterate から LP（shrinking）で m+1 個以下を混ぜた確率的分類器。JMLR：T-stoch・m-stoch・Last・Best の 4 種を比較 | ICML 5 節・付録 B、JMLR 5.3 |
| Best iterate の規則 | 目的関数の順位と最大違反の順位のうち、悪いほうが最小の iterate（式 18、同順位は損失で決める） | JMLR 5.2・5.3 |
| Best と混合の比較 | 「much of the time the best iterate works just as well in practice」。last は 7 つのうち 4 つで best より test が悪い。Bank の Hinge Last は常に多数クラスを出す退化解 | JMLR 6.3.3・6.3.4・5.4 |
| 代償 | test 誤差の増分は多くて数ポイント（例：COMPAS 0.311 → 0.313〜0.32）。dual holdout では Communities 0.142 → 0.199 | JMLR 表 8、ICML 表 4 |

## この repo の実装

`projects/hypernet_e2e/objectives/equalized_odds.py` の `EqualizedOddsRateTaskLoss`。
既定値は `configs/training_strategy/equalized_odds_rate.yaml`。

| 引数 | 既定 | 意味 |
| --- | --- | --- |
| `constraint_form` | `pairwise` | `pairwise`：全有向群ペアで rate_g − rate_h ≤ ε（最大 − 最小 ≤ ε と同値、評価指標に直接対応）。`overall`：原論文の各群 vs 全体の片側制約 |
| `rates` | `[tpr, fpr]` | 制約をかける率。原論文は 1 種類 |
| `epsilon_pos` / `epsilon_neg` | 0.1 / 0.1 | TPR 側 / FPR 側の許容値 |
| `initial_task_weight` | null（一様、原論文どおり） | CheXpert の 8 群 preset は 0.9（下記） |
| `dual_step_size` | 0.01 | M の指数勾配の step。原論文は validation で選ぶ |
| `tau` | 1.0 | hinge の幅。代理は TFCO の既定 HingeLoss と同じ上下界に固定（2026-10-13 より前の run が使った smoothstep は閾値を越える勾配を持たないため 2026-10-16 に削除） |

- **制約の向き（overall）**：TPR は TPR_全体 − TPR_g ≤ ε_pos（低すぎない）、FPR は FPR_g − FPR_全体 ≤ ε_neg（高すぎない）。
  FPR は原論文の Communities と同じ向き。TPR は原論文の COMPAS（TPR_g ≤ TPR_全体 + 0.05、高すぎない）と**逆向き**で、陽性の見落としを抑えるために意図的に変えている。
  CheXpert は陽性 = No Finding なので、FPR は所見のある人の見落としで、害の大きい側にあたる。
- **全体の率**：train の群構成で重み付けした群の率の和。セル均等のバッチの単純平均ではない。
- **制約の本数**：pairwise は |rates| × G(G−1)（8 群・両方で 112 本）、overall は |rates| × G（8 群・FPR だけなら 8 本）。
- **バッチ**：`GroupClassImageDataModule` が群 × クラスを均等に引く（8 群なら 1 セル 8 件、計 128 件）。
  率は 1 バッチで測るので、8 件の率の標準誤差は 0.15 前後ある。
- **task loss**：母集団の比率に重み付けし直したクラス重み付き CE（原論文は hinge）。
- **dual holdout**：`*_dual_holdout` preset は `holdout_split: holdout` を持つ。毎 step、held-out のセル均等バッチで
  hard な違反を測り直して λ を更新する（`module.py` の `_measure_on_held_out`）。split は
  `data/chexpert/splits_race_known_dual_holdout*`（train の 80% / 20%）。val・test は race_known と同じ。
- **checkpoint**：`_dual_holdout_chexpert.yaml` が `save_top_k: -1` で全 epoch を残す。
- **`initial_task_weight: 0.9`**：1 段で ImageNet から学習すると、一様では task の重みが 1/(m+1) しかなく、
  全予測を片側へ寄せる自明解に落ちたため入れた（112 本のときの経験）。原論文の形（8 本）なら一様でよいか試す。

### 診断値（`metrics/objective_diagnostics.jsonl`、1 行 = 1 epoch）

| key | 中身 |
| --- | --- |
| `epoch_mean` | train バッチの平均。`lambda`（先頭が task の重み）、`hard_g`（制約の値 − ε、正なら違反）、`proxy_g`、`rates`（[G,2]、列 0 が FPR、列 1 が TPR）、`task` |
| `held_out_epoch_mean` | λ の更新に使った held-out バッチの平均。制約が効いているかはこちらで見る |
| それ以外の同名 key | epoch 最後の 1 バッチの値（旧形式との互換） |

`rates` はバッチごとの率の平均で、母集団の率ではない。違反の割合にはバッチの雑音が入る。

## 起動

原論文の Communities 型（FPR だけ・各群 vs 全体・一様から始める）：

```bash
uv run python -m projects.hypernet_e2e.run experiment=<..._eo_rate_dual_holdout> seed=<seed> study=<slug> \
  model.loss_fn.constraint_form=overall 'model.loss_fn.rates=[fpr]' model.loss_fn.epsilon_neg=0.05 \
  model.loss_fn.initial_task_weight=null trainer.max_epochs=10
```

`trainer.max_epochs` は、同じ構成の ERM の best val epoch を目安にする（CheXpert の Spatial LoRA なら epoch 8 前後）。
30 epoch 回すと、held-out の task loss が上がり続け、退化解へ寄っていく（下の失敗例）。

## checkpoint の選び方と予測の使い方

全 epoch の val 予測を作り、選んだ 1 つで test を評価する。

```bash
for e in $(seq -w 0 9); do
  uv run python analysis/common/predictions.py --study <slug> --split val \
    --run-dir analysis/<slug>/runs/<run-id> --checkpoint checkpoints/best_val_auroc_0$e.ckpt
done
```

全 epoch を残した run では `run_artifacts.selected_checkpoint` が 1 つに定まらず失敗するので、`--checkpoint` を必ず渡す。
cache 名には checkpoint の hash が入るので、epoch ごとの cache は並べて置ける。

選び方（val で測る。holdout は λ の更新に使ったので、選択にも使うと二重に使うことになる）：

- **主（規則 A）**：学習でかけた制約を val の hard な率で測り、全制約を満たす epoch のうち val の損失が最小のもの。
  満たす epoch が無ければ、最大違反が最小のもの（`infeasible` と明記する）。年齢 2 群の
  `callbacks/conditional_fairness.py` の `select_candidate` と同じ考え方。
- **補足（規則 B）**：JMLR 式 18。損失の順位と最大違反の順位のうち、悪いほうが最小の epoch。
- 制約は**学習に使った群**（例：dw8）で測る。評価の 8 群（年齢 × race）とは別に扱う。

## 既知の失敗例（2026-10-13、`analysis/latent_group_fairness`）

run `…dw8-eo-rate-dual-holdout-s42-9bb9`：pairwise・両方の率・ε 0.1・初期 task 重み 0.9・30 epoch。

- held-out では制約がほぼ満たされた（FPR 差 0.26 → 0.05、TPR 差 0.26 → 0.13）。ただし、それは全群の TPR が
  0.80 → 0.22 に落ちた結果だった。test では TPR 0.87 → 0.38、AUROC 0.862 → 0.805。
- train の TPR は 0.78 のまま、train の TPR 差は 0.62 まで広がった。held-out の task loss は 0.51 → 0.92 と上がり続けた。
- grid の 8 群 EO rate（8b3d）も同じ形で、best val AUROC は epoch 0 だった。
- 原因の候補：原論文と違う制約の形（112 本・両側）、last と best val AUROC による選択、30 epoch の過学習、
  1 セル 8 件のバッチで測る λ 更新の雑音。原論文の条件に揃えた run での検証は 2026-10-14 に開始した。
