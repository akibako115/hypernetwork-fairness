# hypernet_two_stage

Stage 1 の ResNet を学習し、その最良 validation AUROC checkpoint を入力に、共有 backbone と
classifier を凍結した Stage 2 Spatial LoRA を学習する project です。**1 run = 固定された2回の
fit** であり、cohort を更新する反復学習は扱いません。

## workflow

```text
Stage 1: ResNet + inverse-weighted loss
  ↓ best val/auroc checkpoint（SHA-256 を照合して受け渡す）
Stage 2: checkpoint を読み込み、共有 backbone / classifier を凍結
         metadata-conditioned Spatial LoRA を学習
  ↓ best val/auroc checkpoint
親 run の selected_checkpoint
```

`workflow.py` がこの順序と artifact の接続を所有します。各 stage は独立した Lightning fit であり、
Stage 2 は Stage 1 の checkpoint を warm-start ではなく凍結対象の共有重みとして読み込みます。

## 実装状況

ResNet、MetadataEncoder、HyperLinear、Spatial LoRA block / network、model utility、CheXpert data、
callbacks、Lightning module、Hydra configs、二段 workflow、学習 CLI を移植済みです。親 run は Stage 1 / 2
の解決済み config、metrics、checkpoint、入力 split manifest、golden preflight を記録します。

## 設定の構造

この project の実験対象は **stage1 の条件 × stage2 の条件 × 両者に共通の条件**の組です。そのため
stage 固有の設定は config group `stage1` / `stage2` が所有し、data・trainer・callbacks・logger は
共有します。

```text
configs/
  stage1/resnet50.yaml               # cfg.stage1 = {model, trainer}
  stage2/spatial_lora_resnet50.yaml  # cfg.stage2 = {model, trainer}
  experiment/chexpert_*.yaml         # dataset と weighting 戦略（stage の構成は選ばない）
  trainer/ callbacks/ logger/ data/ paths/
```

`stage{N}.trainer` は共有 `trainer` への差分で、`null` は共有値をそのまま使う意味です。
`weighting=inverse` のときの class weight は親 config の `class_weight` 1 箇所で解決し、両 stage の
`loss_fn` がそれを参照します。

## 学習起動

```bash
uv run python -m projects.hypernet_two_stage.run
```

stage を明示して条件を振ります。

```bash
uv run python -m projects.hypernet_two_stage.run \
  experiment=chexpert_inverse_weighted_loss \
  stage1.model.optimizer.lr=1e-3 \
  stage2.model.net.modulation_stages=[stage4,fc] \
  stage2.model.net.rank=8 \
  stage2.trainer.max_epochs=10
```

Stage 2 の backbone 凍結は既定の実験条件として `stage2/spatial_lora_resnet50.yaml` に書いてあり、
凍結しない条件を試すときは `stage2.model.freeze_backbone=false` で明示的に外します。

epoch ごとの metric は既定で wandb（project `fairness_hypernet_two_stage`）へ stage ごとに
別 run として送ります。送信せずに試すときは `logger=none` を付けます。実行ログは親 run の
`logs/train.log` に両 stage 分をまとめて残すため、repository root に共有の log file は作りません。

出力の path と必須記録は [run artifact 契約](docs/run-artifacts.md) を参照してください。
