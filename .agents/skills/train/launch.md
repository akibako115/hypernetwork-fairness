# ローカル起動と監視

Python・pytest・学習はすべて `uv run` 経由。長時間 run は detached で起動し、controller log を
`run_logs/` に残す（`run_logs/` は .gitignore 済み）。

## 起動

```bash
mkdir -p run_logs
NAME="e2e_spatial_lora_fc_s42_$(date -u +%Y%m%dT%H%M%SZ)"
nohup uv run python -m projects.hypernet_e2e.run \
  experiment=spatial_lora_chexpert_fc study=iterative_probe seed=42 \
  > "run_logs/${NAME}.log" 2>&1 < /dev/null &
```

- trailing `&` だけの foreground command を長時間 run に使わない。`nohup` と
  `< /dev/null`、明示的な log 先を必ず付ける
- tmux を使う場合: `tmux new -d -s "$NAME" "uv run python -m projects.hypernet_e2e.run ... 2>&1 | tee run_logs/${NAME}.log"`
- controller log 名は run ID とは別物。衝突しない名前を選び、複数 seed で使い回さない
- `DATA_FOLDER` や `CONDITION` のような env は渡さない。条件はすべて Hydra override で表す
- `experiment=` と `study=` は必須。どちらも既定が無く、抜けると Hydra が起動前に止める

CPU で経路だけ確認する最小実行:

```bash
uv run python -m projects.hypernet_e2e.run \
  experiment=spatial_lora_chexpert study=scratch \
  trainer=cpu trainer.max_epochs=1 \
  data.num_workers=0 data.persistent_workers=false data.prefetch_factor=null
```

複数 seed を 1 つの実験として束ねる（W&B の chart で平均と band になる）:

```bash
mkdir -p run_logs
GROUP="spatial_lora_fc_inverse"
for SEED in 42 43 44; do
  NAME="e2e_${GROUP}_s${SEED}_$(date -u +%Y%m%dT%H%M%SZ)"
  nohup uv run python -m projects.hypernet_e2e.run \
    experiment=spatial_lora_chexpert_fc study=iterative_probe \
    seed="$SEED" logger.wandb.group="$GROUP" \
    > "run_logs/${NAME}.log" 2>&1 < /dev/null &
done
```

group 名に seed を入れない。入れると 1 run ずつ別 group になり、束ねる意味が無くなる。GPU を
分けるなら `trainer.devices=[<index>]` を seed ごとに変える。

二段学習は `hypernet_e2e` の run を 2 回起動する。2 段目は
`experiment=spatial_lora_chexpert_from_resnet` を選び、1 段目 run の `run.json` が記録した best
checkpoint の path を `model.backbone_checkpoint_path=` で渡す。Stage 2 の変調箇所は
`model.net.modulation_stages=[stage4,fc]` のように指定する。

## run directory の特定

`run.py` は完了時に run directory を標準出力へ出すので、成功すれば controller log の末尾に残る。
実行中は作成時刻で引く。

```bash
ls -1dt projects/hypernet_e2e/runs/*/ | head -3
```

並行起動した場合は、各 run の `config.yaml` の `seed` と `experiment_name` で対応を取る。
run ID の suffix はランダムであり、対応付けの根拠にしない。

同じ study の直前の run との差は、解決済み `config.yaml` の diff で読む。承認した条件だけが
変わっているかを、これで確かめる。

```bash
diff -u projects/hypernet_e2e/runs/<前回 run-id>/config.yaml projects/hypernet_e2e/runs/<今回 run-id>/config.yaml
```

## 監視

```bash
tail -f run_logs/<name>.log
cat projects/hypernet_e2e/runs/<run-id>/run.json
python -c 'import json,sys; print(json.load(open(sys.argv[1]))["exit_code"])' \
  projects/hypernet_e2e/runs/<run-id>/preflight.json
```

- `metrics/fit.json` は fit が正常終了した後にだけ書かれる。進行中は存在しない
- `logs/` は run artifact 契約上の予約領域で、現在の `run.py` は書かない。実行中のログは controller log を見る
- `checkpoints/` に `best_val_auroc_*.ckpt` と `last.ckpt` が出る。best は `val/auroc` が最良を更新したときに
  差し替わり、`last.ckpt` は毎 epoch 上書きされる。2026-09-24 の修正より前の run では、`last.ckpt` が best と同じ中身
- 二段学習は段ごとに別 run directory を持つ。2 段目の `run.json` の `parent_run` が、読み込んだ
  checkpoint の path・SHA-256 と 1 段目の run ID を持つ
