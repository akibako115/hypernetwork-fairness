---
name: analyze
description: run 記録を読んで仮説 (study) の分析を進め、results・figures・reports を更新する
allowed-tools: Bash(uv:*) Bash(find:*) Bash(ls:*) Bash(cat:*) Bash(grep:*) Bash(git:*) Read Write Edit
---

学習の run 記録を読み、その run が属する仮説 (study) について何が言えるかを出す。

**正本は仮説 package 側の文書**である。[analysis/README.md](../../../analysis/README.md) が置き場と
追跡境界を、[analysis/AGENTS.md](../../../analysis/AGENTS.md) が層の分け方と書き方を持つ。
この skill はその上の手順だけを持つ。

## この repo の前提（旧 repo との違い）

旧 repo `/home/akiba/workspace/fairness` の analyze skill を持ち込まない。機構ごと無い。

| | 旧 repo | この repo |
| --- | --- | --- |
| 対象の特定 | `logs_*` を `find` で探す | `run.json` の `study` から引く（`analysis/common/studies.py`） |
| 置き場 | `experiments/<slug>/outputs/` から `figures/` へ昇格 | `analysis/<study>/` の `results/` と `figures/` へ直接書く |
| 入力の run | `logs_*` を直接読む | `analysis/<study>/runs/` に hardlink で取り込んだものだけを読む（`analysis/common/runs.py`） |
| 実験記録 | `docs/experiments/YYYY-MM-DD/*.md` | 無い。run directory と `analysis/<study>/runs.md` が持つ |
| remote 回収 | `src.training.sync` | 無い。回線をまたぐ転送はユーザーが打つ（train skill の transfer.md） |
| 共有 CLI | `src/analysis` の module 群 | `analysis/common/`（run 記録の読み方と予測 cache だけ） |

## 1. 対象の特定

仮説の名前が分かっているなら、その study に属する run を出す。

```bash
uv run python analysis/common/studies.py <study>
uv run python analysis/common/studies.py --all   # study を持たない過去 run も含めて見る
```

`study` は途中から入れた key なので、それ以前の run は持たない。その場合は
`analysis/<study>/runs.md` の手書きの表が対象の正本になる。

対象が複数あってどれを見るべきか決まらないときは、ユーザーに確認する。

決まった run を package へ取り込む。別の study のために回した run（baseline など）も、引用する
package ごとに取り込む。実行中の run は拒まれるので、終わるまで待つ。

```bash
uv run python analysis/common/runs.py import <study> projects/<project>/runs/<run-id> ...
```

出力された表の行を `runs.md` に貼る。表の `run path` 列には `analysis/<study>/runs/<run-id>` が入る。
取り込んだ file は正本と同じ inode なので、書き換えない。

## 2. 仮説 package を読む

`analysis/<study>/README.md`（仮説・対象 run・結論）と `runs.md`（run の一覧と状態）を読む。
既に `results/` と `figures/` に何があるかを見て、**作り直しが要るのか、読むだけでよいのか**を
先に決める。生成物は再生成できるので、迷ったら作り直してよい。

新しい仮説なら package を先に作る。学習側が `study=<slug>` を受け付けないので、
package が無い run は起動できない。構成は `analysis/README.md` の形に揃える。

## 3. 分析候補の提示

既存の `results/` と run 記録から実行できる候補を挙げ、**ユーザーが選んだものだけ**を実行する。
選ばれていない分析へ勝手に広げない。候補の例:

- epoch 推移（global 指標・worst group・群間の広がり）
- baseline との突き合わせ（同じ split・optimizer・class weight で回した run があるか先に確かめる）
- 群ごと・交差群ごとの性能（予測 cache が要る）
- 条件間の比較（変調範囲・step size・目的関数）
- 失敗した run の理由（`run.json` の `result_summary` と controller log）

## 4. 実行

DAG の向きに沿って、必要な段だけを走らせる。

```bash
uv run python analysis/common/predictions.py --study <study> --split test --run-dir analysis/<study>/runs/<run-id> ...
uv run python analysis/<study>/collect.py <run-id> ... [--baseline <run-id>]
uv run python analysis/<study>/groups.py --split test
uv run python analysis/<study>/plots.py --split test
```

script が無い、あるいは形が合わないなら、`analysis/AGENTS.md` の層の分け方に従って足す。
**notebook へ処理を書き足して済ませない。** notebook に置くのは主張と、その根拠の表示だけとする。

run 記録の読み方（`run.json` / `metrics.csv` / W&B transaction log / checkpoint 選択）を書く前に
`analysis/common/run_artifacts.py` を見る。同じ読み方が既にあるなら、そちらへ寄せる。
逆に、**群の切り方・指標の定義・図は `analysis/common/` へ上げない**。仮説ごとの判断であり、
共有すると別の仮説がその判断を暗黙に引き継ぐ。

## 5. 報告

**観察事実・解釈候補・追加確認が必要な点を分けて**報告する。数値は `results/` の表を根拠にし、
図は `figures/` の file 名で示す。結論の採用はユーザーの確認後にする。

seed が 1 本しかない比較で「効いた」と書かない。n が小さい群の指標を、群の大きさを添えずに
並べない。gap が縮んだと書くときは、worst が上がったのか best が下がったのかまで見る。

## 6. 記録

- `analysis/<study>/reports/<name>.md` — 使った checkpoint と split、観察事実、解釈候補、追加確認
- `analysis/<study>/README.md` — 仮説と対象 run、そこから出た結論
- `analysis/<study>/runs.md` — run の一覧と状態、なぜその条件なのか

run directory の内容（run-id・状態・experiment・W&B URL）を手で写さない。表は
`analysis/common/studies.py` が出す。`runs.md` に人が書くのは **なぜこの条件なのか**のほうになる。

## 完了条件

- 対象 run が study で特定されている（`study` を持たない run はその旨を明示している）
- 対象 run が `analysis/<study>/runs/` に取り込まれ、`runs.md` の `run path` がそこを指している
- 実行した分析がユーザーの選んだものに一致している
- 生成物が `results/` と `figures/` にあり、notebook からも script からも再生成できる
- 観察事実と解釈が分かれて報告されている
- 読み取れなかったこと・追加で要る run が明示されている
