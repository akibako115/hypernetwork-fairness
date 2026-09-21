---
name: data-explore
description: データセットの構造・統計情報を調査し、project の data config と突き合わせる
allowed-tools: Bash(uv:*) Bash(find:*) Bash(ls:*) Bash(wc:*) Bash(head:*) Bash(cat:*) Bash(grep:*) Bash(du:*) Read
---

データセットの構造、統計情報、project の data config との整合性を調査する。
引数でデータセット名（例: `chexpert`）を指定する。

## データの場所

この repo の学習入力は **repo 直下の `data/<dataset>/`**（`paths.data_dir=data`、repo root 基準）。
旧 repo の `DATA_FOLDER=/data` とは別で、config は `${paths.data_dir}/...` だけを見る。

| 場所 | 中身 |
| --- | --- |
| `data/chexpert/` | この repo が学習に使う CheXpert（画像 5.3 GB） |
| `/data/` | ホスト共有のデータ置き場。`isic2019` / `pad_ufes_20` / `ODIR-5K` / `preprocessed_images`。**この repo にはまだ持ち込んでいない** |

パスを決め打ちせず、まず `ls data/` と `ls /data/` で実際の配置を確認する。`/data` にしかない
データセットを調べる場合は、その旨と、repo へ持ち込む単位が未決であることを明示する。

## 手順

### 1. ディレクトリ構造の把握

```bash
find data/<dataset>/ -maxdepth 2 -type d | head -30
find data/<dataset>/ -maxdepth 2 -type f | head -30
du -sh data/<dataset>/*
```

CheXpert の場合の構成:

| path | 中身 |
| --- | --- |
| `images/` | 画像本体 |
| `splits/{train,val,test}.csv` | 学習が読む split（`data.cv_splits_dir` の既定） |
| `cv_splits/fold_*/` | 交差検証用の split |
| `label/*.json` | レポート由来のラベル |
| `df_chexpert_plus_240401.csv` | 元 metadata（405 MB）。全読みせず `usecols` / `nrows` を使う |
| `cohorts/` | 旧 repo 由来の群割り当て。**現在の契約では cohort は生成 run の `artifacts/cohorts/` が持つ**ので、扱いは未決として報告する |

### 2. split / metadata の分析

split CSV について報告する。

- カラム一覧（`image`、`target`、属性列、`<attr>_missing` の欠損フラグ列）
- レコード数
- `target` の分布
- 公平性に関わる属性（`sex` / `race` / `ethnicity` / `age` 等）の分布
- `<attr>_missing` の割合。欠損は別列で表現されるので、NaN 率だけを見ない

```bash
uv run python -c "
import pandas as pd
frame = pd.read_csv('data/chexpert/splits/train.csv')
print(frame.shape)
print(frame['target'].value_counts(dropna=False))
print(frame['sex'].value_counts(dropna=False))
print(frame.filter(like='_missing').mean().sort_values(ascending=False).head(10))
"
```

### 3. split 構造の確認

`splits/` と `cv_splits/fold_*/` の両方について、行数と `image` 列の重複・重なりを確認する。
train/val/test 間で `image` が重複していないことは明示的に確かめる。

```bash
wc -l data/<dataset>/splits/*.csv data/<dataset>/cv_splits/fold_*/*.csv
```

### 4. config との整合性確認

`projects/<project>/configs/data/<dataset>.yaml` を読み、次を突き合わせる。

- `data_dir` / `cv_splits_dir` が実在するか（`${paths.data_dir}` 展開後）
- `num_classes` と `target` の実際の値域
- `attribute_names.categorical` / `.continuous` の列が split CSV にあるか
- `attribute_spec.categorical_cardinalities` が各属性の実際の水準数と一致するか。
  **ここがずれると embedding の index error か、静かな誤対応になる**
- `fairness_attribute_names` の属性が、`attribute_names` とは別に評価用として揃っているか
- `fairness_age_groups` の派生属性（例: `age_group_65`）が raw 列から作れるか
- `_target_` の DataModule が該当 project に存在するか

project ごとに config は独立している。`hypernet_e2e` と `hypernet_iterative` の
`configs/data/<dataset>.yaml` は別物なので、調査対象の project を明示する。

### 5. 既存 run との突き合わせ（あれば）

`projects/<project>/runs/*/data_manifest.json` は、その run が使った split の SHA-256・行数・
画像集合 hash・target 分布を持つ。過去 run があるなら、現在のデータが同じかを比較する。
違う場合は「データが更新された」であって、どちらかが壊れているとは限らない。

### 6. 結果の報告

表形式でまとめる。

- データセットの概要（サンプル数、クラス数、画像サイズ）
- クラス分布
- 属性分布と欠損率（公平性の観点で重要）
- config との整合性（一致・不一致・未確認を区別する）
- 追加確認が必要な点

### 7. 完了条件

- データの実体パスと主要ファイルが特定されている
- クラス分布と属性分布、欠損率が報告されている
- split の行数と重なりが確認されている
- 対象 project の data config との整合性が、項目ごとに確認済み・不一致・未確認で区別されている
