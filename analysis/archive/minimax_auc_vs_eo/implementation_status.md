# 実装と検証の記録

2026-10-06。worktree: `.claude/worktrees/minimax-auc-eo-age`、branch: `feat/minimax-auc-eo-age`。
元worktreeの未コミット計画・DRO診断など、今回の前提に必要な変更を引き継いだ。

## 実装済み

- 年齢2群のMinimax AUC、固定母集団qのAUCMax、hard EOによるproxy-Lagrangian。
- trainの4セル均等復元抽出、seed+epoch、最後も128件のフルバッチ。
- 母集団CEへのimportance補正と固定Z、群内クラス比を戻したDRO・群等重み対照。
- モデル更新後の双対更新。validationでplain objectiveを呼ぶ既存経路も修正。
- C1〜C8の同seed親ERMからの全層継続学習preset。親なし・seed不一致を起動前に拒否。
- CSV/W&Bへの診断、全epoch候補とSHA、自然分布validationによる単一checkpoint選択。
- 明示checkpointの予測cacheとcheckpoint/split SHA照合。
- EO mixture LP、hard判定の確率的混合、患者単位paired bootstrap。
- validation選択・test比較・動作点参照の3 notebook。run未取り込み時は結果を生成しない。

学習は未実行。実画像1epoch smoke、seed42探索、43/44再検証は実験計画に従う後続作業。
notebookのFINAL_RUN_IDSはvalidationで設定を決めた後に記入する。
原著との一致と変更点は [source_alignment.md](source_alignment.md) に記録した。

## 起動例

```bash
python -m projects.hypernet_e2e.run \
  experiment=resnet_chexpert_from_resnet_age_minimax_auc_race_known \
  study=minimax_auc_vs_eo seed=42 \
  model.backbone_checkpoint_path=<同seedの親ERMのbest_val_auroc_checkpoint> \
  dry_run=true
```

EOはexperimentを `resnet_chexpert_from_resnet_age_equalized_odds_rate_race_known` に変更する。
探索値は `model.loss_fn.dual_step_size=`、EOのproxy幅は `model.loss_fn.tau=`。
通常のfitはW&BとCSVの両方を使い、group名は手法・eta・tauから自動設定する。seedはgroupへ含めない。

## 実装上の補足

親評価に全件sanityを使うため、既存presetよりfit開始前のvalidationが長い。
各epochの候補は `weights_only=True` で保存し、optimizer状態を含めない。双対状態はstate_dictに残る。
best AUROCとlastは従来の保存設定を維持する。途中再開は未対応として明示的に拒否する。
C1〜C7の候補選択もMinimax規約にそろえ、従来のbest AUROCは別に残す。
EO不成立時はselectionにinfeasibleを残し、最小違反の候補は成立した候補として扱わない。

## 検証結果

- `pytest projects/hypernet_e2e/tests analysis/tests`: **460 passed**（41.94秒）。
- 小さなCPU学習を2epoch実行し、Minimax・EOの更新回数、評価時の状態保持、候補checkpoint内の双対状態を確認。
- C1〜C8をfitせず合成・解決・instantiate。trainのセル件数 `[[75145,12385],[66453,3869]]`、各セル32件、1234フルバッチを確認。
- 3 notebookはrun未取り込みの状態で全code cellを実行。実runの推論・最終分析は未実行。
- 変更したPythonファイルのRuff、`git diff --check`、`uv lock --check --offline`が通過。
- 新規の対応文書・実装記録は日本語lintで指摘なし。

## レビューへの対応（2026-10-07）

- 候補checkpointを重みのみの保存に変更。小規模fitでAdamW状態が含まれず、双対bufferが残ることを確認した。lastにはoptimizer状態が残る。
- `continuation.require_parent: true` で親の検証を有効にした。実験名を上書きしても、親checkpoint・configの存在、同seed、ERM、precision、deviceの検査は外れない。親なしのdry-runは構成確認に限って許可する。
- recorderの候補一覧・親評価を復元する機能は実装せず、途中再開をエラーにした。条件付きobjectiveのfitではcheckpoint読み込み時に拒否する。評価目的のcheckpoint読み込みは許可する。
- 患者ごとの行番号を一度まとめ、同じ患者再抽出で5指標を計算するbootstrapに変更。rateのみの比較ではAUCを計算しない。旧単一指標APIも維持する。
- SHA-256を1回のcache生成処理内で再利用する。別の処理では必ず読み直し、同サイズ・同mtimeの書き換えも検出する。ファイルのstatだけによるprocess全体のmemo化は採用しない。
- data/checkpointsのsymlinkをignore対象に追加。新規objective・sampler・callbackの型注釈をそろえた。
- notebook 02にseed別の4ペアAUCとTPR/FPR gap、03に親ERMの共通閾値・群別閾値のTPR/TNR図を追加した。数値はCSVにも残す。

再検証: **471 passed**（43.91秒）。8 presetのdry-runとsamplerの実CSV検算、3 notebookのrunなし全cell実行、合成結果を使った3図の描画を確認した。
bootstrapの合成データ比較（20,000行、3 seed、20反復、5指標）では、点推定・区間が一致し、12.048秒から2.429秒に短縮した（4.96倍）。実データでの速度は未測定。
Ruff、`git diff --check`、`uv lock --check --offline`も通過。実画像の学習・実runの分析は引き続き未実行。

## FairDROの追加（2026-10-07）

- C9 `FairDROTaskLoss`（`objectives/fair_dro.py`）を追加した。
- FairDROのq更新は `LitModule.on_train_epoch_end` で行う。trainの全件を評価transform・CSV順・non-persistent workerで推論し、セル別hard誤り率を渡す。
- 候補選択はC8と同じEO規約。C9は最終epochも `fairness_selection.json` に併記する。W&B groupは `maeo_age_fair_dro_rho1` の形。
- preset：`resnet_chexpert_from_resnet_age_fair_dro_race_known`。

テストでは次を確認した。
- 閉形式解がχ²球上の無作為な境界点のどれにも負けないこと（G=2,3）。値域がProp.2の範囲に入ること。
- 年齢2群で負の重みが出ること。偏差0で一様になること。
- 平滑化スケジュールの値。
- 損失の式。
- 小さなCPU学習2epochでq更新が2回、各回の誤り集計がtrain全行であること。

検証結果：
- `pytest projects/hypernet_e2e/tests analysis/tests`: **487 passed**。
- presetのdry-runで、trainセル件数 `[[75145,12385],[66453,3869]]`、radius、group名を確認した。
- 実画像の学習は未実行。
