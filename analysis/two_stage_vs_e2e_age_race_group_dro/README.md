# two_stage_vs_e2e_age_race_group_dro

既存の ResNet checkpoint から第2段だけ Spatial LoRA を age_group_65 × race の
intersectional GroupDRO で最適化する条件と、Spatial LoRA を最初から同じ GroupDRO で
最適化する条件を比較する。

## 仮説

第2段の既存条件では ERM の局所的な停滞により目的関数の更新が弱かった。第2段だけに
intersectional GroupDRO を与えることで、同じ task loss の再最適化とは異なる勾配方向が
Spatial LoRA 系パラメータへ入り、少なくとも群別性能の worst / gap が変化するかを検証する。
E2E 条件は、同じ fairness objective を初期学習から与えた対照である。

## 群定義

`age_group_65`（age < 65 / age >= 65）× `race`（6カテゴリ + missingカテゴリ）の14群。
train split では全群が非空で、最小セルは106件。

## 対象 run

既存の第1段 ResNet checkpoint は次の run の best `val/auroc` を使う。

`projects/hypernet_e2e/runs/20260921T103036Z-resnet-chexpert-s42-5538/checkpoints/best_val_auroc_009.ckpt`

比較は次の5条件で行う（run-id は [runs.md](runs.md)）。GroupDRO の効果をアーキテクチャの差と
分けるため、E2E の GroupDRO 条件には同じアーキテクチャの ERM を対照に置く。

| 条件 | アーキテクチャ | 目的関数 | 初期値 |
|---|---|---|---|
| ResNet (ERM) | ResNet-50 | CE | ImageNet |
| ResNet + GroupDRO | ResNet-50 | age × race GroupDRO | ImageNet |
| E2E Spatial LoRA (ERM) | Spatial LoRA | CE | ImageNet |
| E2E Spatial LoRA + GroupDRO | Spatial LoRA | age × race GroupDRO | ImageNet |
| 2-stage Spatial LoRA + GroupDRO | Spatial LoRA（第2段のみ学習） | age × race GroupDRO | ResNet (ERM) の checkpoint |

test split の比較は [reports/test_comparison.md](reports/test_comparison.md)。
