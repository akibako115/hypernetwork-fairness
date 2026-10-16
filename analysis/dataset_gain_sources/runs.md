# 対象run

このパッケージは学習も推論もしない。下の run の予測 cache（`analysis/<package>/cache/<run-id>_{val,test}*.npz`）を、
それを書いた package から読む。run の来歴（W&B・checkpoint・SHA256）は各 package の `runs.md` が正本。

- 予測 cache は `analysis/common/predictions.py` の形式で、行は split CSV と `image` 列で検算する。
- 同じ run の cache が 2 つの package にある場合は、`data.py` の `caches` で後に並べた package のものを読む
  （同じ checkpoint の推論なので値は同じ）。
- 2026-10-12 に足した cache：FairVLMed ResNet-50 の `val`・`val_features`・`test_features`（3 seed）、
  CheXpert の ResNet-50 s43・s44・Spatial LoRA・HyperAdapt の `val`（計 8 run）。
- HAM の `meta_fixed`・`hyper_fixed` は `ham10000_architecture_erm/counterfactual.py` の `_test_cf-constant_typical`。

| データ | 分析上の名前 | seed | run-id | cache の package |
| --- | --- | --- | --- | --- |
| ham10000 | hyperadapt | 42 | `20261006T155759Z-hyperadapt-ham10000-s42-f78c` | `ham10000_architecture_erm` |
| ham10000 | hyperadapt | 43 | `20261006T161735Z-hyperadapt-ham10000-s43-4db2` | `ham10000_architecture_erm` |
| ham10000 | hyperadapt | 44 | `20261006T163711Z-hyperadapt-ham10000-s44-15e7` | `ham10000_architecture_erm` |
| ham10000 | meta | 42 | `20261006T155802Z-spatial-lora-ham10000-s42-d782` | `latent_bias_hypernet` |
| ham10000 | meta | 43 | `20261006T160741Z-spatial-lora-ham10000-s43-bdd1` | `latent_bias_hypernet` |
| ham10000 | meta | 44 | `20261006T161713Z-spatial-lora-ham10000-s44-513e` | `latent_bias_hypernet` |
| ham10000 | resnet | 42 | `20261006T155816Z-resnet-ham10000-s42-8a40` | `latent_bias_hypernet` |
| ham10000 | resnet | 43 | `20261006T160730Z-resnet-ham10000-s43-1338` | `latent_bias_hypernet` |
| ham10000 | resnet | 44 | `20261006T161647Z-resnet-ham10000-s44-d3bc` | `latent_bias_hypernet` |
| ham10000 | const | 42 | `20261009T022144Z-spatial-lora-ham10000-const-s42-0c47` | `latent_bias_hypernet` |
| ham10000 | const | 43 | `20261009T031843Z-spatial-lora-ham10000-const-s43-650f` | `latent_bias_hypernet` |
| ham10000 | const | 44 | `20261009T041512Z-spatial-lora-ham10000-const-s44-59c0` | `latent_bias_hypernet` |
| ham10000 | random8 | 42 | `20261009T023300Z-spatial-lora-ham10000-random8-s42-f76c` | `latent_bias_hypernet` |
| ham10000 | random8 | 43 | `20261009T033003Z-spatial-lora-ham10000-random8-s43-9eae` | `latent_bias_hypernet` |
| ham10000 | random8 | 44 | `20261009T042635Z-spatial-lora-ham10000-random8-s44-7f00` | `latent_bias_hypernet` |
| ham10000 | fc | 42 | `20261009T104332Z-spatial-lora-ham10000-fc-s42-b253` | `latent_bias_hypernet` |
| ham10000 | fc | 43 | `20261009T105230Z-spatial-lora-ham10000-fc-s43-6ded` | `latent_bias_hypernet` |
| ham10000 | fc | 44 | `20261009T110118Z-spatial-lora-ham10000-fc-s44-36f3` | `latent_bias_hypernet` |
| fairvlmed | meta | 42 | `20261006T162710Z-spatial-lora-fairvlmed-s42-804d` | `latent_bias_hypernet` |
| fairvlmed | meta | 43 | `20261006T163449Z-spatial-lora-fairvlmed-s43-bd2b` | `latent_bias_hypernet` |
| fairvlmed | meta | 44 | `20261006T164233Z-spatial-lora-fairvlmed-s44-f49c` | `latent_bias_hypernet` |
| fairvlmed | resnet | 42 | `20261006T162713Z-resnet-fairvlmed-s42-00c2` | `latent_bias_hypernet` |
| fairvlmed | resnet | 43 | `20261006T163427Z-resnet-fairvlmed-s43-f4ad` | `latent_bias_hypernet` |
| fairvlmed | resnet | 44 | `20261006T164136Z-resnet-fairvlmed-s44-269c` | `latent_bias_hypernet` |
| fairvlmed | hyperadapt | 42 | `20261006T162714Z-hyperadapt-fairvlmed-s42-92a4` | `fairvlmed_architecture_erm` |
| fairvlmed | hyperadapt | 43 | `20261006T164344Z-hyperadapt-fairvlmed-s43-f14d` | `fairvlmed_architecture_erm` |
| fairvlmed | hyperadapt | 44 | `20261006T170013Z-hyperadapt-fairvlmed-s44-1d27` | `fairvlmed_architecture_erm` |
| fairvlmed | const | 42 | `20261009T061134Z-spatial-lora-fairvlmed-const-s42-ba35` | `latent_bias_hypernet` |
| fairvlmed | const | 43 | `20261009T070045Z-spatial-lora-fairvlmed-const-s43-173a` | `latent_bias_hypernet` |
| fairvlmed | const | 44 | `20261009T074844Z-spatial-lora-fairvlmed-const-s44-9091` | `latent_bias_hypernet` |
| fairvlmed | random8 | 42 | `20261009T062241Z-spatial-lora-fairvlmed-random8-s42-2c4b` | `latent_bias_hypernet` |
| fairvlmed | random8 | 43 | `20261009T071014Z-spatial-lora-fairvlmed-random8-s43-f088` | `latent_bias_hypernet` |
| fairvlmed | random8 | 44 | `20261009T075815Z-spatial-lora-fairvlmed-random8-s44-ac1b` | `latent_bias_hypernet` |
| chexpert | resnet | 42 | `20261002T124957Z-resnet-chexpert-race-known-s42-d6d3` | `race_known_architecture_objective_comparison` |
| chexpert | resnet | 43 | `20261002T142200Z-resnet-chexpert-race-known-s43-bbfc` | `race_known_architecture_objective_comparison` |
| chexpert | resnet | 44 | `20261002T155222Z-resnet-chexpert-race-known-s44-3d73` | `race_known_architecture_objective_comparison` |
| chexpert | hyperadapt | 42 | `20261002T125004Z-hyperadapt-chexpert-race-known-s42-b901` | `race_known_architecture_objective_comparison` |
| chexpert | hyperadapt | 43 | `20261002T181025Z-hyperadapt-chexpert-race-known-s43-4ab0` | `race_known_architecture_objective_comparison` |
| chexpert | hyperadapt | 44 | `20261002T233012Z-hyperadapt-chexpert-race-known-s44-a7ba` | `race_known_architecture_objective_comparison` |
| chexpert | meta | 42 | `20261002T125005Z-spatial-lora-chexpert-race-known-s42-a21c` | `race_known_architecture_objective_comparison` |
| chexpert | meta | 43 | `20261002T145233Z-spatial-lora-chexpert-race-known-s43-424e` | `race_known_architecture_objective_comparison` |
| chexpert | meta | 44 | `20261002T165551Z-spatial-lora-chexpert-race-known-s44-eb13` | `race_known_architecture_objective_comparison` |
