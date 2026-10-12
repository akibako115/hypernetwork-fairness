# 公平性 objective の修正記録

2026-10-13。[全体レビュー](20261013_experiment_project_review.md)の後に、公平性 objective のコードを監査し、
過去の実験で意図どおりに動かなかった記録と突き合わせた。そのうえで以下を直した。学習は回していない。
修正前の run directory は書き換えていない。

**修正前の run で使っていた挙動は、設定を指定すれば再現できるように残した。** 既定値だけを新しい挙動にしている。

## 原典の手法から外れていた実装

### EO rate の代理制約（`objectives/equalized_odds.py`）

- **修正前**：自前の smoothstep 上下界を使っていた。上界は s≥0 で 1、下界は s≤0 で 0 に張り付く。
  そのため代理制約は、予測を閾値の反対側へ動かす勾配を持たなかった。勾配がかかるのは、すでに閾値の同じ側にある標本を
  さらに遠ざける方向だけだった。代理制約だけを下げても hard rate は 1 件も変わらないことを、テストで確認した。
- **修正後**：TFCO の既定 `HingeLoss`（margin 1）と同じ hinge 上下界を既定にした（`proxy: hinge`）。
  上界は `(1+s/τ)_+`、下界は `1-(1-s/τ)_+`。下界は、負の係数がかかる rate を TFCO が評価する形と同じ。
  validation 側の proxy 記録（`callbacks/conditional_fairness.py`）も同じ関数を使う。
- **旧挙動の再現**：`model.loss_fn.proxy=smoothstep`。
- **影響**：修正前の EO rate の run は、全条件で「EO 制約の効果」として読めない。
  対象は conditional_fairness_one_stage、minimax_auc_vs_eo、latent_bias_hypernet（HAM10000 の EO、dual-holdout を含む）。

### GroupDRO の群重みの更新（e2e `objectives/group.py`、iterative `loss.py`）

- **修正前**：バッチに現れなかった群を損失 0 として、Sagawa et al. の指数勾配更新に通していた。
  その群だけ係数が 1 倍に留まるので、正規化すると q が系統的に縮む。希少な群ほど縮む。
  B=128 の自然抽出では、CheXpert の ≥65×Black（1.8%）はバッチの約 10% で 1 件も入らない。
- **修正後**：欠けた群の q は据え置き、観測した群の q の合計を保ったまま、観測群の中だけで更新する。
  全群がそろうバッチでは原典の更新と一致する。
- **影響**：自然抽出の data config で回した GroupDRO の run。対象は race_known、two_stage_vs_e2e、iterative_probe、
  HAM10000 の GroupDRO。group-class 均等抽出の run は、全群が毎バッチそろうので影響しない。
  「q が弱い群に向かない」という過去の観察には、この偏りが混ざっている可能性がある（未検証）。

## 設計上の問題

### FairDRO（`objectives/fair_dro.py`、`module.py`）

- **ρ の範囲**：閉形式の q* の最小成分は `(1-√(ρ(G-1)))/G`。`ρ ≥ 1/(G-1)` では、誤り率のわずかな差でも重みが 0 以下になる。
  この範囲の ρ を指定すると起動時に警告し、毎 epoch の `q_min` を CSV と `fair_dro_q.jsonl` に記録する。
  負の重みは原著の定義域内なので禁止はしない。preset の既定 ρ=1 は変えていない。
  G=2 では端点、HAM の 4 群・8 群では負の重みになる値なので、次に回すときは ρ を選び直す必要がある。
- **held-out**：DataModule が held-out split を持つとき、q を held-out 全件の誤り率で更新する。
  `fair_dro_q.jsonl` の `source` に、どちらで測ったかを残す。held-out がなければ原著どおり train 全件で測る。
- **配線**：各 objective が、双対を更新する単位（`dual_update`：step / epoch / なし）を宣言するようにした。
  held-out の毎 step の forward は step 単位の objective（EO、Minimax AUC、抽出補正 DRO）にだけ流す。
  双対を持たない objective に held-out を設定すると、起動時に止める。

### FIS（`objectives/cohort.py`）

- **修正前**：群の重みを群について softmax してから標本に配っていた。合計が約 B/G になり、c=0.5 でも群の項が支配していた。
- **修正後**：公式実装（FairVision `src/modules.py` の `Fair_Loss_Scaler`）と同じく、標本ごとに並べた群の距離を
  バッチについて softmax する。標本の重みと群の重みは、どちらも合計が 1 になる。
- **診断値**：`group_weight` を `group_weight_mass`（群ごとの重みの合計）に改名した。旧 run の `group_weight` とは意味が違う。
- **未対応**：公式実装は重みを detach しないが、こちらは detach したまま残した。

### DANN の adversary（`objectives/attribute_invariance.py`、`module.py`）

- **原因と考えたこと（推測）**：GRL で反転した勾配を受ける backbone は、ReLU の trunk の pre-activation を全標本で負にすれば、
  head への情報と head の勾配を同時に断てる。adversary_strength では 256 unit 中 206 が死に、head は事前分布だけを答えていた。
- **修正後**：trunk の活性化を LeakyReLU（傾き 0.2）にした。parameter を持たないので checkpoint の互換は保つ。
  adversary の学習率を backbone と別にする `adversary_lr_scale` も追加した（既定 1、DANN の同時更新の形は変えない）。
  ALFR では `adversary_optimizer` で指定するので、この倍率を指定すると止める。
- **旧挙動の再現**：`model.loss_fn.activation=relu`。

### 診断値の記録（`callbacks/objective_diagnostics.py`）

- **修正前**：epoch 最後の 1 batch の値だけを記録していた。held-out を使う run では、その値も held-out の値に上書きされていた。
- **修正後**：従来の key（最後の batch の値）に加えて、全 train batch の平均 `epoch_mean` を記録する。
  held-out を使う run では、双対の更新に使った held-out の統計の平均 `held_out_epoch_mean` も分けて記録する。

### その他

- held-out の毎 step の loader が `num_workers=0` で失敗する問題を直した（前回レビューの P2）。

## 検証

- `uv run pytest projects/hypernet_e2e/tests projects/hypernet_iterative/tests analysis/tests`：1,016 件成功、1 件失敗。
  失敗は修正前から落ちていた `07_two_stage_operating_point.ipynb` の出力保存で、今回の変更とは関係しない。
- 追加したテスト：
  - hinge は代理制約だけで TPR 差を縮め、smoothstep は縮めない。
  - GroupDRO は欠けた群の q を縮めない。全群がそろうバッチでは原典の更新と一致する。
  - FairDRO の ρ の上限と警告、q_min の記録、held-out 全件で測った誤り率の使用。
  - FIS の 2 つの重みの合計がそろう。
  - DANN の LeakyReLU と学習率倍率。
  - 診断値の epoch 平均と、held-out の分離。
- `ruff check`：指摘なし。
- 実データでの学習・再実行はしていない。修正が実験結果をどう変えるかは未確認。
