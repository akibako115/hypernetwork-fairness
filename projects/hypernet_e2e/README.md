# hypernet_e2e

静的に確定した属性を使い、**1 run = 1 fit** で完結する hypernetwork 系モデルの
学習・性能比較を担う project です。cohort に基づく学習は保留しています。ほかの project を import せず、この directory 配下だけで
学習・評価できる形を目標にします。

## 現在の実装範囲

Spatial LoRA ResNet のモデル群、CheXpert DataModule、評価 callbacks、Lightning module、
Hydra preset を移植済みです。

```
models/
├── resnet/                 # ResNetBackbone と通常の ResNet
├── spatial_hypernet/
│   ├── metadata_encoder.py # 属性辞書を condition embedding に変換
│   ├── blocks.py           # 3x3 conv2 への sample-wise Spatial LoRA
│   ├── modulation.py       # FC 用 HyperLinear LoRA
│   └── network.py          # stage3 / stage4 / fc を束ねる SpatialLoRAResNet
└── utils.py                # checkpoint 部分ロードと embedding 分散推定
```

`SpatialLoRAResNet` は `attributes` から metadata encoder が作る condition を使い、`stage3`、
`stage4`、`fc` の任意の部分集合を変調します。Spatial LoRA は Bottleneck の `conv2` にだけ
低ランク差分を加え、初期状態では元の ResNet と同じ出力になります。

属性辞書は以下の key を持ちます。

| key | shape | 内容 |
| --- | --- | --- |
| `categorical` | `(B, num_categorical)` | カテゴリ値 |
| `categorical_missing` | `(B, num_categorical)` | カテゴリ欠損フラグ |
| `continuous` | `(B, num_continuous)` | 連続値 |
| `continuous_missing` | `(B, num_continuous)` | 連続値の欠損フラグ |

## Configs

[`configs/train.yaml`](configs/train.yaml) が Hydra の入口です。`run.py` が設定を合成して
model・data・callbacks・trainer を作成し、1 回の `fit` を実行します。CheXpert の比較条件は次から選べます。

- `experiment=resnet_chexpert_erm`
- `experiment=spatial_lora_chexpert_erm`
- 各モデルの `inverse_weighted_loss` / `inverse_weighted_sampling`
- `experiment=spatial_lora_chexpert_from_resnet`（2 段学習の 2 段目）

`inverse_weighted_loss` の class weight は、`run.py` が train split の target 頻度から算出して
解決済み config に記録します。cohort artifact / GroupDRO objective は未移植です。

CheXpert の model 入力は `sex`、`race`、`ethnicity`、連続 `age` と撮影条件の
`frontal_lateral`、`ap_pa` です。公平性 logging は別の評価属性 `sex`、`race`、`ethnicity`、
`age_group_65` だけを使います。`age_group_65` は raw age の `<65` / `≥65` であり、model に渡す
標準化済み age とは別に生成します。より細かい年齢群は分析時に split CSV の age から定義します。

## 2 段学習

ResNet を 1 段目、backbone と classifier を凍結した Spatial LoRA を 2 段目とする学習は、
`run.py` を 2 回起動して構成します。1 段目の run は通常の単段 run です。

```bash
uv run python -m projects.hypernet_e2e.run experiment=resnet_chexpert_erm
```

2 段目は 1 段目の `run.json` の `checkpoints` から best checkpoint の path を選び、
`model.backbone_checkpoint_path` に渡します。`from_resnet` preset が `freeze_backbone: true` を
持つため、CLI で指定するのは checkpoint path だけです。

```bash
uv run python -m projects.hypernet_e2e.run \
  experiment=spatial_lora_chexpert_from_resnet \
  model.backbone_checkpoint_path=projects/hypernet_e2e/runs/<1段目 run-id>/checkpoints/<best>.ckpt
```

2 段目の `run.json` は、読み込んだ checkpoint の path・SHA-256 と、それを出力した run の ID を
`parent_run` に記録します。1 段目の記録と SHA-256 が食い違う checkpoint は、fit を始めずに
失敗させます。段ごとに別 run なので、`trainer.max_epochs` のような設定は段ごとに独立して振れます。

## 学習起動

`run.py` は `fit` を一度だけ実行し、Hydra 自身の `outputs/` を作りません。run の入力と出力は
`projects/hypernet_e2e/runs/<run-id>/` に集約されます。例えば CPU で 1 epoch だけ確認するには、
次を実行します。

```bash
uv run python -m projects.hypernet_e2e.run trainer=cpu trainer.max_epochs=1 data.num_workers=0 data.persistent_workers=false data.prefetch_factor=null
```

デフォルトは GPU trainer と Spatial LoRA ERM です。`experiment=` で比較条件を選べます。

epoch ごとの metric は既定で wandb（project `fairness_hypernet_e2e`）へ送ります。送信せずに
試すときは `logger=none` を付けます。実行ログは run ごとに `logs/train.log` へ残るため、
repository root に共有の log file は作りません。

## 出力の保存先

実データは repo 直下の `data/chexpert/` に置き、学習出力は
`projects/hypernet_e2e/runs/<run-id>/` に置きます。両者を混ぜません。cohort を実装する場合も、
生成した run が `artifacts/cohorts/` 配下で所有します。必須ファイルと cohort 参照の契約は
[run artifact 契約](docs/run-artifacts.md) を参照してください。

実装済みの model 単体テストは次で実行できます。

```bash
uv run pytest projects/hypernet_e2e/tests -q
uv run ruff check projects/hypernet_e2e
```

project 内の docstring・コメント規約は [AGENTS.md](AGENTS.md) を参照してください。
