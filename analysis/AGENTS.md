# analysis の実装規約

## 読み手

このディレクトリのコードは、**数か月後の本人が読み返して判断を検算するため**にある。
学習コード（`projects/`）は動けば良い部分があるが、ここは結論の根拠なので、
読めないコードは根拠として機能しない。エージェントが書きやすい形ではなく、
人が上から読める形を優先する。

拠り所は外部の規約に置く。ここで採る主張と出典は次のとおり。

| 出典 | 採る主張 |
| --- | --- |
| Cookiecutter Data Science, *Opinions* | notebook は探索と伝達、script は反復。分析は一方向の DAG |
| Rule et al., *Ten simple rules for writing and sharing computational analyses in Jupyter Notebooks* (PLOS CB 2019) | notebook は読者に向けた物語。1 cell = 1 step。再利用するコードは module へ |
| Wilson et al., *Good enough practices in scientific computing* (PLOS CB 2017) | 関数は短く引数は少なく、重複を排し、ファイル名は内容を表す |

## 層と DAG

生成物は一方向にしか流れない。逆流させると「図を 1 枚描き直すために notebook を全部
実行する」状態になる。

```text
run artifact  →  cache/      →  results/     →  figures/     →  notebook
(projects/*/runs)  predictions.py  collect.py    plots.py       主張と考察
                                   groups.py
```

- `collect.py` — run artifact を読んで `results/*.csv` を書く。run の形（`run.json`、
  `stages/*/metrics/metrics.csv`）を知ってよいのはここと `analysis/common/` だけ。
- `groups.py` — 群の定義と群別指標。予測 cache と `results/` を入力に、`results/*.csv` を書く。
- `plots.py` — `results/*.csv` だけを入力に `figures/*.png` を書く。run artifact を読まない。
- `*.ipynb` — 主張・表の表示・考察。`results/` と `figures/` を読むだけで、
  split CSV も checkpoint も直接読まない。

各 script は単独で走り、前段の生成物だけを入力に取る。DAG の 1 辺が 1 コマンドになる。

## notebook

- **1 cell = 1 つの意味のある step**。cell の前に markdown を置き、何を見るのか、
  なぜそれを見るのかを書く。結果だけの cell を並べない。
- `import` は先頭 1 cell に集める。cell の途中で `import` しない。
- 20 行を超える関数を notebook に置かない。出す先は `plots.py` か `groups.py`。
- `sys.path` を直接触らない。path 解決は下の 1 行だけを使う。
- 出力は Git 追跡しない（`nbstripout`）。**kernel restart → 全実行が通る状態**で終える。
  通らない notebook は、DAG のどこかを手で飛ばして作った状態になっている。

## script

- module docstring に、その script が何を入力に何を書くのかと、実行例を書く。
- 関数は 60 行・引数 6 個を上限の目安にする。超えたら段を分ける。
- 内包表記をネストしない。中間結果に名前を付ける。1 行に詰め込まない（`line-length = 120`）。
- 書き出し先は `results/` と `figures/` に限る。`cache/` を書くのは
  `analysis/common/predictions.py` だけとする。

## path と import

repo root の解決は次の 1 行に統一する。`.project-root` は repo root にある。

```python
ROOT = rootutils.setup_root(Path.cwd(), indicator=".project-root", pythonpath=True)   # notebook
ROOT = rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)     # script
```

これで `analysis.common` と `projects.*` を import できる。以降の path は
`analysis/common/paths.py` が持つものを使い、`parents[2]` のような相対指定を書かない。

## docstring

`projects/` と同じ Google 形式（日本語、`Args:` / `Returns:` 必須、必要なら `Raises:`）。
ただし引数の型を言い換えるだけの docstring は書かない。分析側で書く価値があるのは、
**その群の切り方を選んだ理由**、**その列を見る理由**、**その指標が何を意味しないか**である。

## 追跡境界

`analysis/README.md` の表を正本とする。ここでは繰り返さない。
