# hypernet_two_stage

Stage 1 の ResNet を学習し、その最良 validation AUROC checkpoint を入力に、共有 backbone と
classifier を凍結した Stage 2 Spatial LoRA を学習する project です。**1 run = 固定された2回の
fit** であり、cohort を更新する反復学習は扱いません。

## workflow

```text
Stage 1: ResNet + inverse-weighted loss
  ↓ best val/auroc checkpoint
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

## 学習起動

```bash
uv run python -m projects.hypernet_two_stage.run
```

Stage 2 の変調箇所は `model.net.modulation_stages=[stage4,fc]` のように指定します。

出力の path と必須記録は [run artifact 契約](docs/run-artifacts.md) を参照してください。
