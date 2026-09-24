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

`age_group_65`（age < 65 / age >= 65）× `race`（6カテゴリ）の12群。train split では全群が
非空で、最小セルは106件。

## 対象 run

既存の第1段 ResNet checkpoint は次の run の best `val/auroc` を使う。

`projects/hypernet_e2e/runs/20260921T103036Z-resnet-chexpert-s42-5538/checkpoints/best_val_auroc_009.ckpt`

学習完了後に2本の run ID と test split の intersectional 指標を `runs.md` と reports に追記する。
