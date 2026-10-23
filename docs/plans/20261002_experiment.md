# 2026-10-02 実験計画・実行記録

## 全18条件・54 fitが完了し、同期・分析フォルダへの取り込みも完了

2026-10-05に全runの正常終了・30epoch記録を確認した。
ws11の48runをcheckpoint込みで同期し、localの6runと合わせて分析packageへhardlinkで取り込んだ。
best / last計108 checkpointのSHA256を照合済み。DRO42runのCSVに全30epoch分・8群のqを確認済み。
仮説別の独立した5冊のnotebookを実装し、主比較の9条件×3seed＝27runで全セル実行を確認した。
対象と実行状態は[分析README](../../analysis/archive/race_known_architecture_objective_comparison/README.md#notebook)を参照。

主比較は13条件 × seed 42・43・44 = 39 fit。
追加の全stage Spatial LoRAは5条件 × 3seed = 15 fit（2026-10-03 23:28 JSTに起動）。合計18条件・54 fit。
1-stageの9条件（27 fit）を起動し、2026-10-03に2-stageのM06・M10（6 fit）を追加起動した。
各GPUでseed 42 → 43 → 44を逐次実行し、正常終了後に次へ進む。
2026-10-03 22:50 JSTに残りM13・M15（6 fit）を追加起動した。後続seedは正常終了後に逐次実行する。

定期監視・空きGPUへの自動割り当て・失敗時の自動再試行は行わない。
`overnight.py`は停止済みで、再起動しない。固定したseedループで実行し、現在は全て終了している。

記録済みstudy：`two_stage_vs_e2e_age_race_group_dro`（来歴として変更しない）。
今回の全条件は新しい分析packageへ集約する：
[README](../../analysis/archive/race_known_architecture_objective_comparison/README.md)、
run参照：[runs.md](../../analysis/archive/race_known_architecture_objective_comparison/runs.md)。
設定・状態・数値の正本は各runのartifactとし、本文へ複製しない。

## 条件とGPU割り当て

presetはすべて作成済み。M05・M09（2-stage ERM → ERM）は除外し、欠番を維持する。

| ID | モデル | 学習 | 目的関数 / class weight | preset | 実行先 |
| --- | --- | --- | --- | --- | --- |
| M01 | ResNet | 1-stage | ERM / global | `resnet_chexpert_race_known` | ws11 GPU2 |
| M02 | ResNet | 1-stage | DRO / global | `resnet_chexpert_age_race_group_dro_race_known` | local GPU0 |
| M03 | Spatial LoRA | 1-stage | ERM / global | `spatial_lora_chexpert_race_known` | ws11 GPU3 |
| M04 | Spatial LoRA | 1-stage | DRO / global | `spatial_lora_chexpert_age_race_group_dro_race_known` | ws11 GPU4 |
| M06 | Spatial LoRA | 2-stage | ERM → DRO / global | `spatial_lora_chexpert_from_resnet_age_race_group_dro_race_known` | ws11 GPU2 |
| M07 | HyperAdapt | 1-stage | ERM / global | `hyperadapt_chexpert_race_known` | ws11 GPU5 |
| M08 | HyperAdapt | 1-stage | DRO / global | `hyperadapt_chexpert_age_race_group_dro_race_known` | ws11 GPU6 |
| M10 | HyperAdapt | 2-stage | ERM → DRO / global | `hyperadapt_chexpert_from_resnet_age_race_group_dro_race_known` | ws11 GPU3 |
| M11 | ResNet | 1-stage | DRO / subgroup | `resnet_chexpert_age_race_group_dro_subgroup_race_known` | local GPU0、M02の3 seed終了後 |
| M12 | Spatial LoRA | 1-stage | DRO / subgroup | `spatial_lora_chexpert_age_race_group_dro_subgroup_race_known` | ws11 GPU7 |
| M13 | Spatial LoRA | 2-stage | ERM → DRO / subgroup | `spatial_lora_chexpert_from_resnet_age_race_group_dro_subgroup_race_known` | ws11 GPU4 |
| M14 | HyperAdapt | 1-stage | DRO / subgroup | `hyperadapt_chexpert_age_race_group_dro_subgroup_race_known` | ws11 GPU8 |
| M15 | HyperAdapt | 2-stage | ERM → DRO / subgroup | `hyperadapt_chexpert_from_resnet_age_race_group_dro_subgroup_race_known` | ws11 GPU7 |

ws11 GPU0・1は他の学習が使用中で対象外。各GPUの同時fitは1本。
localはM02の3 seed → M11の3 seed、ws11は1条件の3 seedを1 containerで逐次実行する。

## 共通条件

- データ：`chexpert/splits_race_known`のtrain / val / test。元の分割を維持し、race欠損のみ除外。
- 群：年齢`<65 / ≥65` × race `White / Asian / Black / Others`の8群。encoderには元のrace 6カテゴリと連続ageを入力する。
- 各fit：batch 128、30 epoch、AdamW（lr `1e-4`、weight decay `1e-2`）、schedulerなし、uniform sampling、bf16-mixed。
- 初期化：1-stageはImageNet ResNet-50からbackbone全体を学習。分類用FCとadapter等は新規初期化。
- 2-stage：同seedのM01のbest `val/auroc`を共有し、backboneと元classifierを凍結。属性条件付き部分だけ学習する。
- class weight：train splitから計算。ERMはglobal inverse、DROはglobal / subgroupを比較し、step sizeは`0.01`。重みなしは実行しない。
- checkpoint選択：best `val/auroc`。予測は`argmax(logits)`に固定し、testで閾値・条件を選ばない。

確認済み：全13 presetのsplit参照、3 splitのrace欠損0件、local / ws11のCSVとImageNet重みのSHA256一致。
1-stage全9モデルに実際の読み込み関数を適用し、backboneの265テンソルがImageNet重みと完全一致。
関連テスト35件と起動前preflight 3件が成功。設定・読み込みmoduleのlocal / ws11一致も確認済み。
新preset等は未コミットの状態で起動しているため、再現にはrunの解決済み設定と実行コードの保存が必要。

## 比較・評価

- DROの効果：M01–M02、M03–M04、M07–M08。
- class weightの範囲：M02–M11、M04–M12、M06–M13、M08–M14、M10–M15。
- 学習方式：M04–M06、M08–M10、M12–M13、M14–M15。
- アーキテクチャ：目的関数・class weight・学習方式を揃えて比較。

主指標は全体AUROCと8群のworst-group AUROC。balanced accuracy、群間gap、
Eopp0 / Eopp1 / Eodds、単独属性（sex・race_group・ethnicity・age_group_65）も確認する。
2-stageは計60 epochと凍結を伴い、1-stageは30 epochなので、学習方式だけの効果とは解釈しない。
Spatial LoRAは`[stage3,stage4,fc]`・`[conv2]`・rank 4。
HyperAdaptはstem以外のstage1〜4のconv1〜3とFC・rank 4であり、位置・adapter数の差も含む。

## loggerと分析上の制約

W&B（`fairness_hypernet`、online）とCSVを全fitで有効化。
groupは`20261002-<preset>`でseedを含めず、各条件の3 seedを束ねる。
実値class weight・設定・split provenance・checkpoint来歴はrun artifactとW&B configに残る。

初回runには群別学習損失とDRO重みの時系列がないため、DRO条件は修正版でseed 42から再実行する。
修正版は群別`q_end`・`q_mean`・class weight適用後の学習loss・countをepochごとにCSV/W&Bへ記録する。
群IDとラベルは`metrics/group_dro_groups.json`に保存する。更新式・学習条件は変更しない。
再実行のstudyは`race_known_architecture_objective_comparison`、W&B groupは`20261003-qdiagnostics-<preset>`。
群ごとの個別性能・クラス別件数は別途分析する。
`run.py`はfitのみでtestを自動実行しない。後続でbest checkpointを評価し、
fit終了時の`metrics/fit.json`をbest checkpointの結果と同一視しない。

## 起動履歴・操作

- 2026-10-03 23:28 JST、全stage Spatial LoRAのM16〜M20をGPU2・3・5・6・8で各3seedの固定ループとして起動。
  起動コマンドは`run_logs/stage1_4_ws11_launch.sh`、ws11の生dry-run表は`run_logs/stage1_4_ws11_dry_run.txt`。
  containerは`e2e_<mID>_stage1_4_3seed_20261003`。ログは`run_logs/stage1_4_<preset>_s<seed>.log`。
  初期化は1-stageがImageNet、2-stageがM01の同seed best。各seedは成功終了後のみ次へ進む。
  seed 42の全5本でpreflight・W&B・指定GPU学習・設定・split manifestを確認済み。
  DRO4本のq群定義も生成済み。初epochのvalidation・q時系列出力は確認待ち。

- 2026-10-03 22:50 JST、承認済みM13・M15をws11 GPU4・7で各seed 42→43→44の固定ループとして起動。
  親はM06・M10と同じM01の同seed best `val/auroc`で、3本ともSHA256を再照合済み。
  起動コマンドは`run_logs/two_stage_subgroup_20261003_launch.sh`、生dry-run表は`run_logs/two_stage_subgroup_20261003_dry_run.txt`。
  containerは`e2e_m13_qdiagnostics_3seed_20261003`・`e2e_m15_qdiagnostics_3seed_20261003`。
  seed 42のpreflight・W&B・GPU使用・凍結・split・親checkpoint来歴・q群定義を確認済み。初epochのq出力は確認待ち。

- 2026-10-03 07:50 JST、承認済みのM06・M10をws11 GPU2・3で各seed 42→43→44の固定ループとして起動。
  親は正常終了したM01の同seed best `val/auroc`。3本ともSHA256を実体と照合済み。
  起動コマンドは`run_logs/two_stage_20261003_launch.sh`、生dry-run表は`run_logs/two_stage_20261003_dry_run.txt`。
  containerは`e2e_m06_qdiagnostics_3seed_20261003`・`e2e_m10_qdiagnostics_3seed_20261003`。
  ログは`run_logs/qdiagnostics_<preset>_s<seed>.log`。M13・M15は未起動、既存の1-stageは継続。

- q診断修正のためM02・M04・M08・M12・M14の進行中runとseedループを停止。M11は未開始。
- 停止した5本のテキストログとW&B remote run記録はユーザー指示で削除。checkpoint・metrics・設定は保持。
- ERM（M01・M03・M07）は継続。旧DRO完了runは診断欠損の暫定結果であり、主比較には修正版の再実行を使う。
- 2026-10-03 01:02 JST、修正版DROの3seed固定ループを新規起動。localはM02→M11、ws11はM04/GPU4・M08/GPU6・M12/GPU7・M14/GPU8。2-stageは引き続き別途依頼待ち。
- local tmuxは`e2e_qdiagnostics_local_3seed_20261003`、起動スクリプトは`run_logs/qdiagnostics_local_3seed.sh`。
  ws11 containerは`e2e_<mID>_qdiagnostics_3seed_20261003`。各seedログは`run_logs/qdiagnostics_<preset>_s<seed>.log`。
  正常終了した場合のみ次seedへ進み、失敗時には停止する。旧runの再開・自動再試行はしない。

- 当初のseed 42の8 runをユーザー指示で中断。比較対象・2-stageの親には使わない。IDは[runs.md](../../analysis/archive/two_stage_vs_e2e_age_race_group_dro/runs.md)に保持。
- 中断runの起動ログと`logs/train.log`は元の場所から除去し、各環境の`run_logs/discarded_20261002_interrupted/`へ復旧用に退避。metrics・checkpoint・W&B記録は保持。
- 2026-10-02 21:49 JST頃、今回の3 seed固定ループを新規起動。M11はlocalのM02全seed完了待ち。旧runの再開ではない。
- 22:12 JSTに全8 runの初回validation成功とCSV出力を確認。preflight・W&B・設定・manifest・GPUメモリ確保も確認済みで、ログに例外なし。全8 runともseed 42を学習中。
- 本文整理前の詳細な操作履歴はlocalの`run_logs/experiment_history_before_compaction_20261002.md`へ保存。

起動コマンド全文：`run_logs/three_seed_launch_20261002T1249Z.sh`。
localループ：`run_logs/three_seed_local_20261002T1249Z.sh`。
各条件の生dry-run表：`run_logs/three_seed_<mID>_dry_run_20261002.txt`（local、9条件）。
各seedの起動ログ：`run_logs/e2e_<mID>_s<seed>_20261002T1249Z.log`（実行先repo内）。

local tmux：`e2e_local_3seed_20261002T1249Z`。
localループ履歴：`run_logs/three_seed_local_20261002T1249Z.log`。
ws11 container：`e2e_<mID>_3seed_20261002T1249Z`。
remoteループ履歴は`docker logs <container>`、学習ログは上記seed別ファイル。
新規run IDとW&B参照は[runs.md](../../analysis/archive/race_known_architecture_objective_comparison/runs.md)に記載する。

remote repo：`/mnt/fast/kohkiakiba/hypernet-fairness`。
container repo：`/workspaces/hypernet-fairness`。
データ：`/mnt/fast/kohkiakiba/fairness_data`をcontainerの`data/`へread-only mount。
image：`hypernet-fairness`、IPC：host、W&B認証は`.netrc`をread-only mount。

## 明日の2-stage起動手順

1. 1-stageの実際の進捗とGPU空きを確認する。旧中断runと新runを混同しない。
2. 新M01のseed 42・43・44それぞれが成功終了したことを`run.json`で確認する。
3. 同seedのM01が記録したbest `val/auroc` checkpointを選び、実体・SHA256を照合する。last・元split・別seed・暫定bestを使わない。
4. M06・M10・M13・M15を各3 seedの固定ループで実行する。親checkpointはseedごとに切り替える。全条件でbackbone・元classifierを凍結する。
5. checkpoint pathを含むコマンドとdry-runを提示し、ユーザー依頼・承認後に起動する。GPU割り当ては当日の空きで決め、段階間のcheckpoint受け渡しはws11内で完結させる。
6. 新runを記録し、preflight・設定・W&B・GPU・最初のvalidationを確認する。

## 追加検証（今回は未起動）

### 全stage Spatial LoRA（追加15 fitを起動済み）

変調範囲を`[stage1,stage2,stage3,stage4,fc]`へ拡張する。
conv2のみ・rank 4・metadata encoder・race-known・学習条件・q記録は既存Spatial LoRAと共通。
2-stageは既存M01の同seed bestを使い、backbone・元classifierを凍結する。
新しい第1段runの完了を待つ必要はない。各条件は1GPUに固定しseed 42→43→44。

| ID | 学習 / class weight | 対照 | preset | GPU案 |
| --- | --- | --- | --- | --- |
| M16 | 1-stage ERM / global | M03 | `spatial_lora_chexpert_stage1_4_race_known` | ws11 GPU2 |
| M17 | 1-stage DRO / global | M04 | `spatial_lora_chexpert_stage1_4_age_race_group_dro_race_known` | ws11 GPU3 |
| M18 | 1-stage DRO / subgroup | M12 | `spatial_lora_chexpert_stage1_4_age_race_group_dro_subgroup_race_known` | ws11 GPU5 |
| M19 | 2-stage ERM→DRO / global | M06 | `spatial_lora_chexpert_stage1_4_from_resnet_age_race_group_dro_race_known` | ws11 GPU6 |
| M20 | 2-stage ERM→DRO / subgroup | M13 | `spatial_lora_chexpert_stage1_4_from_resnet_age_race_group_dro_subgroup_race_known` | ws11 GPU8 |

比較対象・結果の取り込み先は同じ`race_known_architecture_objective_comparison`。
W&B groupは`20261003-stage1-4-<preset>`（seedを含めない）。
既存HyperAdaptとの差にはconv種別・generator構造の違いも残るため、位置だけの比較は上表のSpatial LoRA対照で行う。
5条件のpreset作成・関連178テスト・preflight 3件・local/ws11 dry-runが成功。
実際のモデルで全16 Bottleneckのconv2 adapterと元classifier凍結を検証済み。
remoteコード同期・起動前確認・ユーザーの実行指示を受けて開始済み。

### その他の候補

Spatial LoRAの位置比較はM06に固定し、`[fc]`（P01）、
`[stage4,fc]`（P02）、主比較の`[stage3,stage4,fc]`（P03）を比較する。
conv2・rankを固定し、追加は2条件 × 3 seed = 6 fit。採用判断にはvalidationだけを使う。

その後、必要ならconv1〜3への拡張、M02を親にするDRO → DRO
（Spatial LoRA / HyperAdapt、global、凍結維持）、1-stage 60 epochを検討する。
一度に複数の軸を変更せず、今回の39 fitとは分けて承認・実行する。
