# runs

この package が読む run。設定・状態・数値の正本は各 run の artifact で、ここには run ID と、なぜこの run かだけを書く。

## CheXpert race-known、1 段・ERM の Spatial LoRA（study=two_stage_vs_e2e_age_race_group_dro で回したもの）

柱 B の当初案（1 段・ERM で学習した Spatial LoRA の metadata encoder・生成パラメータの空間）をそのまま調べるための run。
柱 A の格子（`hypernet_conditioning_grid`）でも 1 段の ERM として再利用している同じ 3 本で、評価の split・class weight
（全群共通の inverse）・変調段（stage3・stage4・fc）がそろっている。checkpoint は best val AUROC。

| run-id | project | run path |
|---|---|---|
| `20261002T125005Z-spatial-lora-chexpert-race-known-s42-a21c` | `hypernet_e2e` | `analysis/latent_subgroup_spaces/runs/20261002T125005Z-spatial-lora-chexpert-race-known-s42-a21c` |
| `20261002T145233Z-spatial-lora-chexpert-race-known-s43-424e` | `hypernet_e2e` | `analysis/latent_subgroup_spaces/runs/20261002T145233Z-spatial-lora-chexpert-race-known-s43-424e` |
| `20261002T165551Z-spatial-lora-chexpert-race-known-s44-eb13` | `hypernet_e2e` | `analysis/latent_subgroup_spaces/runs/20261002T165551Z-spatial-lora-chexpert-race-known-s44-eb13` |

空間の cache の生成（2026-10-13、local GPU0）:

```bash
uv run python -m analysis.latent_subgroup_spaces.extract --device cuda:0 \
  --run-dir analysis/latent_subgroup_spaces/runs/<上の 3 run>
```

## run 以外の入力

- `cache/raddino/{val,test}.npz`：RAD-DINO の埋め込み（[embed.py](embed.py)）
- `cache/context/{val,test}.csv`：報告書由来の文脈ラベル（[context_labels.py](context_labels.py)）

どちらも race-known の val・test と同じ行順であることを `subgroups.load_split` が検算する。
2026-10-17 まで archive の `latent_bias_hypernet/cache/` から読んでいたものを、同じ file の hardlink としてここへ移した。
画像の前処理は RAD-DINO 自身のもの（短辺 518 に bicubic → CenterCrop）で、学習側の前処理の変更（MEDFAIR への統一）の影響を受けない。
