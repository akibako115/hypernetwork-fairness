# runs

学習 run として引用するのは M01（ResNet ERM、race-known、global class weight）の 3 seed だけ。
best validation AUROC checkpoint の表現（分類 head 直前、2048 次元）を train / val / test で書き出し、
凍結特徴の実験に使う。既存の主比較の予測は `race_known_architecture_objective_comparison/cache/` を
読み取り専用で参照する（paired_noise.py）。

| run-id | project | run path |
|---|---|---|
| `20261002T124957Z-resnet-chexpert-race-known-s42-d6d3` | `hypernet_e2e` | `analysis/archive/fairness_mechanism_probe/runs/20261002T124957Z-resnet-chexpert-race-known-s42-d6d3` |
| `20261002T142200Z-resnet-chexpert-race-known-s43-bbfc` | `hypernet_e2e` | `analysis/archive/fairness_mechanism_probe/runs/20261002T142200Z-resnet-chexpert-race-known-s43-bbfc` |
| `20261002T155222Z-resnet-chexpert-race-known-s44-3d73` | `hypernet_e2e` | `analysis/archive/fairness_mechanism_probe/runs/20261002T155222Z-resnet-chexpert-race-known-s44-3d73` |

特徴量 cache の生成（2026-10-06、local GPU0）:

```bash
uv run python analysis/common/predictions.py --study fairness_mechanism_probe --split <train|val|test> --features \
  --run-dir analysis/archive/fairness_mechanism_probe/runs/<上の3 run>
```

## 探索用 fine-tune probe（run 契約の外）

`probes/train_probe.py` で回した当たり付けの短い学習。`projects/*/runs` には載らず、W&B にも記録しない。
毎 epoch の val / test score と `log.json` を `cache/probes/<name>/` に置く。結論の根拠にするときは、
同じ条件を `hypernet_e2e` の run として回し直す。

| name | 初期化 | 学習データ | epoch | ログ |
| --- | --- | --- | --- | --- |
| `spec_old_m01_s42` | M01 s42 best | train の ≥65 歳（70,322 件） | 6 | `run_logs/fmp_spec_old_m01_s42.log` |
| `cont_all_m01_s42` | M01 s42 best | train 全体 | 3 | `run_logs/fmp_cont_all_m01_s42.log` |
| `in_old_s42` | ImageNet | train の ≥65 歳（70,322 件） | 10 | `run_logs/fmp_in_old_s42.log` |
| `in_random_s42` | ImageNet | train 全体から無作為 70,322 件 | 10 | `run_logs/fmp_in_random_s42.log` |
| `in_young_s42` | ImageNet | train の <65 歳から無作為 70,322 件 | 10 | `run_logs/fmp_in_young_s42.log` |

共通：batch 128、AdamW（lr 1e-4、wd 1e-2）、bf16、CXR_Fairness と同じ augmentation、
class weight は train 全体の global inverse（0.206 / 1.794）。

2026-10-06 01:53、ImageNet 初期化の `in_old_s42`・`in_random_s42` を並列で回したところ、評価 DataLoader の共有メモリ
（1 プロセス約 25GB）で OOM が起き、両 run と FairVLMed の seed 42 が強制終了された（他プロセスも巻き込んだ）。
DataLoader を worker 8・prefetch 2・評価 loader 非 persistent に下げ、3 本を逐次で最初から回し直した。表の値は再実行分。
