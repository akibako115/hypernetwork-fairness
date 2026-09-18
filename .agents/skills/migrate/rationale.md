# 移植方針の根拠

2026-09-18 に旧 repo `/home/akiba/workspace/fairness` を調査して決めた。実測値は調査時点のもの。

## なぜ共有をやめるのか

**(1) `src/` が「現役の共通部品」と「過去 run の復元先」を兼ねていて矛盾している。**

保存済み config の `_target_` を数えると、既に解決できないものがある。

```
97 src.models.module_v2                          互換 wrapper として生存
20 src.cohorts.datamodule.CohortImageDataModule  存在しない
 3 src.data.chexpert_datamodule.CheXpertDataModule 存在しない
```

過去の再編で 23 run が厳密復元できなくなっている。共有コードを import パスとして公開している
限り、整理のたびに過去 run が静かに壊れる。旧 `src/` を凍結して触らないことがこの再発を止める。

**(2) 汎用化のコストが実際に効いている。**

`src/training/execution.py`（352 行）+ `runs.py`（288 行）は、単段の e2e と 14 段の iterative を
同じ protocol で捌くために汎用化されている。e2e だけなら 80 行で足りる。分けた方が
合計行数は増えても 1 つあたりは小さくなる。

**(3) 複製の規模は小さい。**

`src/models` は 2506 行あるが、2 project が実際に使うのは resnet + spatial_hypernet +
metadata_encoder + loss ≈ 1166 行だけ。`film/`（275 行）と `hyperadapt/`（1060 行）は
旧 run 専用で、使いもしないコードが常に視界に入っている状態だった。

1 project が抱える量は models ~1,170 / data ~735 / callbacks ~490 / module+fit+eval ~600 /
run 記録 ~150 で **約 3,100 行**。2 project で ~6,200 行、増分は 2 千行程度。

**(4) 名前だけの subclass が既に発生していた。**

`src/experiments/{class_imbalance,two_stage,demographic,iterative}/module.py` の 4 本は
docstring 以外同一で、しかも共有 `src/training/module.py` の `LitModule` と挙動が同じだった。
`demographic/validation.py` と `iterative/validation.py` は完全一致。
共有と実験所有の線引きが実態として機能していなかった。

## 複製の唯一の実害と、その対処

害は「気づかないうちに比較不能になる」こと。import による共有はこれを防ぐが、意図的な変更まで
止めて結合を生む。**欲しいのは禁止ではなく信号**なので、複製を許した上で golden データとの
一致をテストで固定する。

旧 `tests/analysis/test_metrics_agreement.py` は 2 実装を**両方 import して突き合わせる**形で、
独立化すると書けない。比較相手をコードではなく固定データに置き換える。

golden が保証するのは「指標関数が同じ数値を出すこと」だけで、callback の集計経路が違えば
通ってしまう。だから指標は関数単体ではなく callback 出力を固定する。

## 10 project 規模でのコスト

- 新規作成は `projects/_template/` の複製から
- バグ修正は active な project だけでよい。**凍結した project は直さない方が正しい** —
  過去 run の数値と実装が一致したまま保たれ、「この結果はこのコードで出た」と言える
- preflight が「golden vN 準拠」を教えるので、追随が必要な project は常に区別できる

税は読むときではなく直すときに払い、凍結済みには課されない。

## 過去 run との関係

- 移した run は新 repo では**厳密復元できない**（`_target_: src.*` を解決できるのは旧 repo だけ）
- ただし**重みだけ引き継ぐ経路は成立する**。model コードを逐語コピーするので、
  config を instantiate せず state_dict を直接読めば cohort reference にも warm-start 元にも使える

## 過去ログの棚卸し（移植とは別作業。2026-09-18 時点の実測）

| 種類 | 容量 | 性質 |
| --- | --- | --- |
| `*.ckpt` | 71GB（281 個、hardlink 共有あり） | 再学習が必要 |
| `*final_features.npz` | 13GB（22 個） | 推論 1 回で再生成できるキャッシュ |
| `*predictions.npz` | 0.2GB（334 個） | 再生成に GPU 推論が要る |
| 純 metadata（config/result/log/wandb） | **207MB** | 再生成不可 |
| experiments の非 ckpt | 1.4GB | 再生成不可 |

- 63GB のうち再生成できないテキスト・数値は 207MB。**metadata は選別せず全部持っていけばよい**
- `~/workspace` は同一 filesystem（/dev/sda2）。checkpoint は `cp -l`（ハードリンク）で運べば
  ディスク消費ゼロで新 repo に実体として現れ、旧 repo からも消えない。
  `rsync`/`cp -a` は hardlink を壊して実容量が倍増する（`rsync` なら `-H` 必須）
- 2026-09-07/09-14 の demographic 系 96 run は checkpoint も `run.json` も持たない（各 700KB）。
  完走していない可能性があり、棚卸し時に状態確認が要る
