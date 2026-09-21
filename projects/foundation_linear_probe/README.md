# foundation_linear_probe

CheXpert の No Finding 分類で、凍結済み foundation model の特徴を線形 probe で比較する PoC です。
Hydra や PyTorch Lightning は使いません。

比較対象は ImageNet ResNet50、DINOv2-base、RAD-DINO です。各 encoder の特徴を一度 NPZ に保存し、
train で logistic regression を学習、val AUROC で正則化強度を選び、test の全体・属性群別 AUROC を出します。

## 実行

`DATA_FOLDER` は `chexpert/splits/{train,val,test}.csv` と `chexpert/images/` を持つディレクトリです。

```bash
# まず全 split の特徴を抽出する。--limit で小さく試せる。
DATA_FOLDER=/data uv run python -m projects.foundation_linear_probe.extract \
  --encoder dinov2_base --splits train val test

# Jupyter Lab で notebooks/probe.ipynb を開く。
uv run jupyter lab
```

特徴は `outputs/features/<encoder>/<split>.npz`、簡易な再現情報（モデル revision、split CSV hash、行数）は
同じ場所の `metadata.json` に保存されます。これらの生成物は Git 管理しません。
Notebook の `ENCODER` を選ぶと、結果は `results/probe/<encoder>/` に保存されます。
群別 AUROC、画像ごとの予測値、選んだ `C` と全体 AUROC をそれぞれ CSV / JSON として残します。

欠損済み属性は補完先の群へ混ぜず、`Unknown` 群として集計します。小さい群の推定、bootstrap CI、
キャッシュの厳密な再利用検証は、この PoC の範囲外です。

## 確認

```bash
uv run pytest projects/foundation_linear_probe/tests -q
uv run ruff check projects/foundation_linear_probe
```
