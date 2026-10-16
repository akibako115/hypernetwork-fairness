# test split の5条件比較

## 対象

| 表示名 | run-id | 学習 |
|---|---|---|
| `ResNet (ERM)` | `20260921T103036Z-resnet-chexpert-s42-5538` | ResNet-50 全体を CE。2段階条件の第1段 |
| `ResNet + GroupDRO` | `20260929T113619Z-resnet-chexpert-age-race-group-dro-s42-5474` | ResNet-50 全体を最初から age × race GroupDRO |
| `E2E Spatial LoRA (ERM)` | `20260929T082054Z-spatial-lora-chexpert-s42-b15c` | Spatial LoRA を最初から CE |
| `E2E Spatial LoRA + GroupDRO` | `20260928T025502Z-spatial-lora-chexpert-age-race-group-dro-s42-9502` | Spatial LoRA を最初から age × race GroupDRO |
| `2-stage Spatial LoRA + GroupDRO` | `20260924T122518Z-spatial-lora-chexpert-from-resnet-age-race-group-dro-s42-985c` | `ResNet (ERM)` の checkpoint から、第2段だけ age × race GroupDRO |

- 評価 split: `test`
- checkpoint: 各 run の best validation AUROC
- seed: 42 の1本のみ。class weight はすべて inverse、30 epoch
- race: White / Asian / Black。gender は sex の Male / Female
- 交差群: age_group_65 × race（上の3 race）の6群。群サイズは `<65` 側が White 6162 / Asian 1166 /
  Black 892、`>=65` 側が White 6326 / Asian 1061 / Black 325

`ResNet (ERM)` と `E2E Spatial LoRA (ERM)` を加えたことで、GroupDRO の効果を同じアーキテクチャの
ERM と比べられるようになった。3条件の時点では、`E2E Spatial LoRA + GroupDRO` と `ResNet (ERM)` の
差に、アーキテクチャの差と目的関数の差が混ざっていた。

数値は `results/` の3つの CSV、図は `figures/` にあり、どちらも
[classification_and_fairness.ipynb](../classification_and_fairness.ipynb) から再生成できる。

## 観察事実

### 全体性能

| 条件 | AUROC | BACC | recall_0 | F1 |
|---|---|---|---|---|
| ResNet (ERM) | 0.8597 | 0.7873 | 0.7167 | 0.3906 |
| ResNet + GroupDRO | 0.8471 | 0.7828 | 0.7608 | 0.4081 |
| E2E Spatial LoRA (ERM) | **0.8625** | **0.7947** | **0.7944** | **0.4375** |
| E2E Spatial LoRA + GroupDRO | 0.8485 | 0.7766 | 0.7095 | 0.3798 |
| 2-stage Spatial LoRA + GroupDRO | 0.8603 | 0.7784 | 0.6774 | 0.3691 |

- `E2E Spatial LoRA (ERM)` が4指標すべてで最も高い。
- 最初から GroupDRO で学習すると、同じアーキテクチャの ERM より AUROC が下がった。ResNet では
  0.8597 → 0.8471（−0.0126）、Spatial LoRA では 0.8625 → 0.8485（−0.0140）。
- 2段階条件は、親の `ResNet (ERM)` と AUROC がほぼ同じ（0.8603）で、recall_0 と F1 は5条件で最も低い。
- recall_0 は条件によって 0.677 から 0.794 まで動いている。固定閾値での動作点が条件ごとに違う。

### age × race 交差群の公平性

| 条件 | AUROC gap | BACC gap | Eopp0 | Eopp1 |
|---|---|---|---|---|
| ResNet (ERM) | 0.0740 | 0.0634 | 0.1707 | 0.2314 |
| ResNet + GroupDRO | **0.0670** | 0.0715 | 0.1407 | 0.2030 |
| E2E Spatial LoRA (ERM) | 0.0795 | 0.0913 | 0.2029 | 0.3006 |
| E2E Spatial LoRA + GroupDRO | 0.0769 | 0.0733 | **0.0949** | 0.1707 |
| 2-stage Spatial LoRA + GroupDRO | 0.0671 | **0.0588** | 0.1281 | **0.1683** |

- `E2E Spatial LoRA (ERM)` は全体性能が最も高い一方、交差群の gap は4指標すべてで最も大きい。
- 同じアーキテクチャの ERM と比べた GroupDRO の変化:
  - Spatial LoRA（E2E）: 4指標すべて縮小。特に Eopp0 は 0.2029 → 0.0949、Eopp1 は 0.3006 → 0.1707。
  - ResNet（E2E）: AUROC gap・Eopp0・Eopp1 は縮小したが、BACC gap は 0.0634 → 0.0715 と拡大。
  - 2段階（親の ResNet (ERM) との比較）: 4指標すべて縮小。
- AUROC gap の縮小がどちら側から来ているか（交差群内の最大 − 最小）:
  - `ResNet + GroupDRO`: 最小群（>=65 × White）は 0.8046 → 0.7972、最大群（<65 × White）は
    0.8786 → 0.8642。**両端とも下がり**、最大側の低下が大きかったので gap が縮んだ。
  - `E2E Spatial LoRA + GroupDRO`: ERM の最大群は n=325 の >=65 × Black（0.8893）で、GroupDRO では
    0.8413 に下がった。最小群の >=65 × White も 0.8098 → 0.7983 と下がった。
  - `2-stage`: 最小群の >=65 × White が 0.8046 → 0.8106 と上がり、最大群の <65 × White は
    0.8786 → 0.8777 とほぼ同じ。GroupDRO の3条件のうち、最小群を上げて gap を縮めたのはこの条件だけ。

### 単独属性の公平性

| 属性 | 条件 | AUROC gap | BACC gap | Eopp0 | Eopp1 |
|---|---|---|---|---|---|
| age | ResNet (ERM) | 0.0622 | 0.0358 | 0.1396 | 0.2112 |
| | ResNet + GroupDRO | 0.0524 | 0.0528 | 0.0981 | 0.2038 |
| | E2E Spatial LoRA (ERM) | 0.0545 | 0.0742 | 0.1343 | 0.2827 |
| | E2E Spatial LoRA + GroupDRO | 0.0574 | 0.0441 | 0.0532 | 0.1415 |
| | 2-stage Spatial LoRA + GroupDRO | 0.0551 | 0.0294 | 0.0985 | 0.1573 |
| race | ResNet (ERM) | 0.0403 | 0.0475 | 0.0580 | 0.0739 |
| | ResNet + GroupDRO | 0.0222 | 0.0314 | 0.0450 | 0.0549 |
| | E2E Spatial LoRA (ERM) | 0.0328 | 0.0251 | 0.0451 | 0.0566 |
| | E2E Spatial LoRA + GroupDRO | 0.0233 | 0.0284 | 0.0530 | 0.0345 |
| | 2-stage Spatial LoRA + GroupDRO | 0.0357 | 0.0344 | 0.0380 | 0.0621 |
| gender | ResNet (ERM) | 0.0175 | 0.0166 | 0.0087 | 0.0419 |
| | ResNet + GroupDRO | 0.0037 | 0.0102 | 0.0067 | 0.0138 |
| | E2E Spatial LoRA (ERM) | 0.0150 | 0.0162 | 0.0057 | 0.0380 |
| | E2E Spatial LoRA + GroupDRO | 0.0176 | 0.0088 | 0.0231 | 0.0407 |
| | 2-stage Spatial LoRA + GroupDRO | 0.0190 | 0.0134 | 0.0058 | 0.0326 |

- age では `E2E Spatial LoRA (ERM)` の BACC gap（0.0742）と Eopp1（0.2827）が大きい。群別 BACC は
  <65 が 0.7976、>=65 が 0.7233 で、>=65 群の BACC は5条件で最も低い。
- race の AUROC gap は、E2E の GroupDRO で両アーキテクチャとも縮小した（ResNet 0.0403 → 0.0222、
  Spatial LoRA 0.0328 → 0.0233）。2段階は 0.0357。
- gender では `ResNet + GroupDRO` の gap が4指標とも最も小さい。

## 解釈候補

- 3条件の比較で「E2E GroupDRO は age / race の gap が小さい」と見えた差の一部は、Spatial LoRA という
  アーキテクチャの差ではなく GroupDRO によるものと読める。ERM の Spatial LoRA は、むしろ age × race
  の gap が最も大きい。
- E2E で GroupDRO を入れると、アーキテクチャによらず AUROC が約 0.013〜0.014 下がった。2段階条件は
  AUROC を保ったまま、交差群の BACC gap と Eopp1 で最小、AUROC gap で `ResNet + GroupDRO` と同程度の値になった。GroupDRO を入れる
  時期として、第2段だけに入れる方が全体性能を損ないにくい可能性がある。
- 2段階条件だけが、交差群の最小群（>=65 × White）を上げて AUROC gap を縮めていた。E2E の GroupDRO は
  両アーキテクチャとも最大群を下げる方向で gap を縮めていた。
- Eopp0 / Eopp1 は固定閾値での recall の差なので、recall_0 が条件間で大きく動いている今回の比較では、
  群間の公平性の差と動作点のずれが混ざっている。Eopp の大小は、AUROC gap ほど直接には比べられない。

## 留保

- seed は1本なので、効果の再現性や有効性はまだ主張できない。条件間の差が seed の散らばりより大きい
  かどうかは不明。
- 交差群の >=65 × Black は n=325 と小さい。`E2E Spatial LoRA (ERM)` の交差群 AUROC gap はこの群が
  最大側を決めている。
- E2E GroupDRO の2本は best epoch が 28（ResNet）と 29（Spatial LoRA）で、30 epoch の終盤でも
  val AUROC が伸びていた。学習が収束しきっていない可能性があり、AUROC の低下幅は epoch 数に依存する
  かもしれない。
- `ResNet (ERM)` だけ `trainer.deterministic: true` で学習しており、他の4本は `false`。
- race は6カテゴリ全体ではなく、既存分析と同じ3カテゴリに限定した。
