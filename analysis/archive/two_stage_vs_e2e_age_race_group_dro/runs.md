# runs

2026-10-02のrace-known実験の今後の引用・分析は、
[新packageのruns.md](../race_known_architecture_objective_comparison/runs.md)に集約する。
以下のrace-known区分は移行前の履歴として保持する。元splitの過去runは従来どおりここで扱う。

この表を、この study が引用する run の正本とする。run 本体は Git 管理外で、完了済み run は
`analysis/archive/two_stage_vs_e2e_age_race_group_dro/runs/` に取り込んである。

## 2026-10-02 race-known・8群

### 完了runの取り込み（2026-10-03）

M02 seed 42（末尾`58a1`）の取り込みはq時系列欠損のため除去済み。比較には使わない。
修正版は新packageで扱い、学習側の元artifactと以下の起動履歴だけ保持する。
M01 seed 42はws11から同期し、best / last checkpointのSHA256照合後に取り込み済み。
共通のResNet ERM基準として使用する。
M03 seed 42もws11から同期し、best / last checkpointのSHA256照合後に取り込み済み。
Spatial LoRA / ERMの対照として使用する。2 runともcheckpointを含む全artifactを保持する。

| run-id | project | run path |
|---|---|---|
| `20261002T125005Z-spatial-lora-chexpert-race-known-s42-a21c` | `hypernet_e2e` | `analysis/archive/two_stage_vs_e2e_age_race_group_dro/runs/20261002T125005Z-spatial-lora-chexpert-race-known-s42-a21c` |
| `20261002T124957Z-resnet-chexpert-race-known-s42-d6d3` | `hypernet_e2e` | `analysis/archive/two_stage_vs_e2e_age_race_group_dro/runs/20261002T124957Z-resnet-chexpert-race-known-s42-d6d3` |

### 3 seed固定ループの新規起動（2026-10-02 21:49 JST頃）

1-stage 9条件 × seed 42・43・44の固定ループを起動した。
下表は現在開始済みのseed 42 run。43・44は各runの正常終了後に同じGPUで開始され、
M11はlocalのM02全3 seed終了後に開始する。まだ27 runすべてが開始したわけではない。
2-stage・定期監視・動的割り当ては起動していない。
22:12 JSTに下表の全8 runで初回validation・CSV出力を確認し、例外がないことを確認した。
実行条件・現在の状態は実行先のrun artifactが正本。
未回収のremote runのhost側rootは
`/mnt/fast/kohkiakiba/hypernet-fairness/projects/hypernet_e2e/runs/`。
local runは`projects/hypernet_e2e/runs/`にある。

| 条件 | 実行先 | seed | run ID | W&B |
| --- | --- | --- | --- | --- |
| M01 | ws11 GPU2 | 42 | `20261002T124957Z-resnet-chexpert-race-known-s42-d6d3` | [run](https://wandb.ai/kohki-akiba-kyushu-university/fairness_hypernet/runs/rswjo11h) |
| M03 | ws11 GPU3 | 42 | `20261002T125005Z-spatial-lora-chexpert-race-known-s42-a21c` | [run](https://wandb.ai/kohki-akiba-kyushu-university/fairness_hypernet/runs/cnsy8b78) |
| M04 | ws11 GPU4 | 42 | `20261002T124957Z-spatial-lora-chexpert-age-race-group-dro-race-known-s42-1609` | [run](https://wandb.ai/kohki-akiba-kyushu-university/fairness_hypernet/runs/sytmrl7v) |
| M07 | ws11 GPU5 | 42 | `20261002T125004Z-hyperadapt-chexpert-race-known-s42-b901` | [run](https://wandb.ai/kohki-akiba-kyushu-university/fairness_hypernet/runs/i7qef91a) |
| M08 | ws11 GPU6 | 42 | `20261002T124954Z-hyperadapt-chexpert-age-race-group-dro-race-known-s42-49d3` | [run](https://wandb.ai/kohki-akiba-kyushu-university/fairness_hypernet/runs/2izc0cvt) |
| M12 | ws11 GPU7 | 42 | `20261002T125001Z-spatial-lora-chexpert-age-race-group-dro-subgroup-race-known-s42-a83d` | [run](https://wandb.ai/kohki-akiba-kyushu-university/fairness_hypernet/runs/cg79goa5) |
| M14 | ws11 GPU8 | 42 | `20261002T125004Z-hyperadapt-chexpert-age-race-group-dro-subgroup-race-known-s42-c814` | [run](https://wandb.ai/kohki-akiba-kyushu-university/fairness_hypernet/runs/1cfw4n3z) |
| M02 | local GPU0 | 42 | `20261002T124930Z-resnet-chexpert-age-race-group-dro-race-known-s42-58a1` | [run](https://wandb.ai/kohki-akiba-kyushu-university/fairness_hypernet/runs/badxbm30) |

後続seedのrun IDは条件ごとのW&B group `20261002-<preset>`と
解決済みconfigで対応付ける。明日の2-stageには新M01の同seedのbest checkpointだけを使う。
詳細な操作は[実験計画](../../../docs/plans/20261002_experiment.md)を参照。

### 起動済み8 runはユーザー指示で中断（比較・親checkpointとして使用しない）

以下に掲載した2026-10-02のseed 42 runは、3 seedの固定ループで新規実行し直す方針により中断した。
runtime側のstatusが成功またはrunningのままでも、今回の比較や2-stageの親として採用しない。
run artifactは保持する。新しいrunは別IDで記録し、旧runを上書きしない。

### 中断した6条件の旧起動履歴（remote、未取り込み）

各runは当初の独立した1-stage条件で、現在は中断扱い。主比較には取り込まない。
状態の正本はremoteのrun.json。GPU割り当てとcontroller logは実験計画を参照。

| 条件 | run ID | remote host側run path | W&B |
| --- | --- | --- | --- |
| M03 | `20261002T121938Z-spatial-lora-chexpert-race-known-s42-bb39` | `/mnt/fast/kohkiakiba/hypernet-fairness/projects/hypernet_e2e/runs/20261002T121938Z-spatial-lora-chexpert-race-known-s42-bb39` | [run](https://wandb.ai/kohki-akiba-kyushu-university/fairness_hypernet/runs/hjih2ork) |
| M04 | `20261002T121938Z-spatial-lora-chexpert-age-race-group-dro-race-known-s42-6577` | `/mnt/fast/kohkiakiba/hypernet-fairness/projects/hypernet_e2e/runs/20261002T121938Z-spatial-lora-chexpert-age-race-group-dro-race-known-s42-6577` | [run](https://wandb.ai/kohki-akiba-kyushu-university/fairness_hypernet/runs/1np4ezxk) |
| M07 | `20261002T121927Z-hyperadapt-chexpert-race-known-s42-e756` | `/mnt/fast/kohkiakiba/hypernet-fairness/projects/hypernet_e2e/runs/20261002T121927Z-hyperadapt-chexpert-race-known-s42-e756` | [run](https://wandb.ai/kohki-akiba-kyushu-university/fairness_hypernet/runs/bfdt0cha) |
| M08 | `20261002T121938Z-hyperadapt-chexpert-age-race-group-dro-race-known-s42-c888` | `/mnt/fast/kohkiakiba/hypernet-fairness/projects/hypernet_e2e/runs/20261002T121938Z-hyperadapt-chexpert-age-race-group-dro-race-known-s42-c888` | [run](https://wandb.ai/kohki-akiba-kyushu-university/fairness_hypernet/runs/pb5dfakm) |
| M12 | `20261002T121941Z-spatial-lora-chexpert-age-race-group-dro-subgroup-race-known-s42-19d4` | `/mnt/fast/kohkiakiba/hypernet-fairness/projects/hypernet_e2e/runs/20261002T121941Z-spatial-lora-chexpert-age-race-group-dro-subgroup-race-known-s42-19d4` | [run](https://wandb.ai/kohki-akiba-kyushu-university/fairness_hypernet/runs/wmznx1s4) |
| M14 | `20261002T121941Z-hyperadapt-chexpert-age-race-group-dro-subgroup-race-known-s42-0001` | `/mnt/fast/kohkiakiba/hypernet-fairness/projects/hypernet_e2e/runs/20261002T121941Z-hyperadapt-chexpert-age-race-group-dro-subgroup-race-known-s42-0001` | [run](https://wandb.ai/kohki-akiba-kyushu-university/fairness_hypernet/runs/1z4uowu7) |


13条件の対応と起動依存は[実験計画](../../../docs/plans/20261002_experiment.md)を参照。
実行中runは学習側の正本への参照だけを記録し、完了後に`analysis/common/runs.py import`で取り込む。
旧splitの取り込み済みrunは今回の主比較へ混ぜない。

起動履歴は以下に追記する。

当初はM02を先行起動したが、現在は中断扱いで未取り込み。

当初のM01はws11で起動したが、現在は中断扱いで親checkpointに使用しない。
run IDは`20261002T121238Z-resnet-chexpert-race-known-s42-33cc`。
remote正本は`/mnt/fast/kohkiakiba/hypernet-fairness/projects/hypernet_e2e/runs/20261002T121238Z-resnet-chexpert-race-known-s42-33cc/`。
W&Bは[run](https://wandb.ai/kohki-akiba-kyushu-university/fairness_hypernet/runs/w582niyv)。
未回収・未取り込み。以下の旧runではなく、新M01の同seedの成功終了を待つ。
以下はrun.jsonから取得した参照であり、状態の正本は学習側run.json。

| run-id | project | 学習側run path（未取り込み） | W&B | 条件 |
| --- | --- | --- | --- | --- |
| `20261002T120237Z-resnet-chexpert-age-race-group-dro-race-known-s42-bde6` | `hypernet_e2e` | `projects/hypernet_e2e/runs/20261002T120237Z-resnet-chexpert-age-race-group-dro-race-known-s42-bde6` | [run](https://wandb.ai/kohki-akiba-kyushu-university/fairness_hypernet/runs/zmsc5sok) | M02: race-known ResNet / global Group DRO |

## 旧実験の取り込み済み run（元split・14群）

Spatial LoRA を ImageNet 初期化から E2E で age_group_65 × race の intersectional GroupDRO により最適化した対照条件。seed 42、30 epoch。

| run-id | project | run path | W&B | 条件 | 状態 |
|---|---|---|---|---|---|
| `20260928T025502Z-spatial-lora-chexpert-age-race-group-dro-s42-9502` | `hypernet_e2e` | `analysis/archive/two_stage_vs_e2e_age_race_group_dro/runs/20260928T025502Z-spatial-lora-chexpert-age-race-group-dro-s42-9502` | [run](https://wandb.ai/kohki-akiba-kyushu-university/fairness_hypernet/runs/44c62t7k) | Spatial LoRA、age × race GroupDRO をE2E最適化 | succeeded |

第1段の ResNet checkpoint を初期値として、第2段の Spatial LoRA を
`age_group_65 × race` の intersectional GroupDRO で最適化した条件。seed 42、14群。

| run-id | project | run path | W&B | 条件 | 状態 |
|---|---|---|---|---|---|
| `20260924T122518Z-spatial-lora-chexpert-from-resnet-age-race-group-dro-s42-985c` | `hypernet_e2e` | `analysis/archive/two_stage_vs_e2e_age_race_group_dro/runs/20260924T122518Z-spatial-lora-chexpert-from-resnet-age-race-group-dro-s42-985c` | [run](https://wandb.ai/kohki-akiba-kyushu-university/fairness_hypernet/runs/56wee8ew) | ResNet checkpoint → Spatial LoRA、age × race GroupDRO | succeeded |

親 checkpoint:

`projects/hypernet_e2e/runs/20260921T103036Z-resnet-chexpert-s42-5538/checkpoints/best_val_auroc_009.ckpt`

test split の予測 cache は同じ分析 package に作成し、5条件を同じ notebook で比較する。

## 比較対象: 同じアーキテクチャの E2E 条件

GroupDRO の効果を、アーキテクチャの差と分けて読むための対照。`E2E Spatial LoRA + GroupDRO` に
対しては同じ Spatial LoRA を CE で学習した run を、baseline ResNet に対しては同じ ResNet を
最初から同じ age × race GroupDRO で学習した run を置く。どちらも seed 42、30 epoch、inverse class
weight。

| run-id | project | run path | W&B | 条件 | 状態 |
|---|---|---|---|---|---|
| `20260929T082054Z-spatial-lora-chexpert-s42-b15c` | `hypernet_e2e` | `analysis/archive/two_stage_vs_e2e_age_race_group_dro/runs/20260929T082054Z-spatial-lora-chexpert-s42-b15c` | [run](https://wandb.ai/kohki-akiba-kyushu-university/fairness_hypernet/runs/nrultp11) | Spatial LoRA を ImageNet 初期化から CE で E2E 最適化 | succeeded |
| `20260929T113619Z-resnet-chexpert-age-race-group-dro-s42-5474` | `hypernet_e2e` | `analysis/archive/two_stage_vs_e2e_age_race_group_dro/runs/20260929T113619Z-resnet-chexpert-age-race-group-dro-s42-5474` | [run](https://wandb.ai/kohki-akiba-kyushu-university/fairness_hypernet/runs/cjaxqm9i) | ResNet-50 全体を age × race GroupDRO で E2E 最適化 | succeeded |

## 比較対象: baseline ResNet

2段階学習の第1段で用いた、ImageNet 初期化の ResNet-50 を全体 ERM で学習した baseline。
class weight は本実験と同じ inverse weighting、seed 42、30 epoch。第2段の親 checkpoint は
この run の `best_val_auroc_009.ckpt` である。

| run-id | project | run path | W&B | 条件 | 状態 |
|---|---|---|---|---|---|
| `20260921T103036Z-resnet-chexpert-s42-5538` | `hypernet_e2e` | `analysis/archive/two_stage_vs_e2e_age_race_group_dro/runs/20260921T103036Z-resnet-chexpert-s42-5538` | [run](https://wandb.ai/kohki-akiba-kyushu-university/fairness_hypernet_e2e/runs/41gsahly) | ResNet-50 全体を ERM、inverse class weight | succeeded |
