# analysis の実装規約

## 読み手

このディレクトリのコードは、**数か月後の本人が読み返して判断を検算するため**にある。
学習コード（`projects/`）は動けば良い部分があるが、ここは結論の根拠なので、
読めないコードは根拠として機能しない。エージェントが書きやすい形ではなく、
人が上から読める形を優先する。

拠り所は外部の規約に置く。ここで採る主張と出典は次のとおり。

| 出典 | 採る主張 |
| --- | --- |
| Cookiecutter Data Science, *Opinions* | notebook は探索と伝達、反復・再利用する処理は script / module。依存関係は一方向に保つ |
| Rule et al., *Ten simple rules for writing and sharing computational analyses in Jupyter Notebooks* (PLOS CB 2019) | notebook は読者に向けた物語。1 cell = 1 step。再利用するコードは module へ |
| Wilson et al., *Good enough practices in scientific computing* (PLOS CB 2017) | 関数は短く引数は少なく、重複を排し、ファイル名は内容を表す |

## notebook-first と依存関係

探索段階の分析では、notebook を分析の主単位にしてよい。入力の読み込みから分析・可視化・考察までを
一つの notebook に置くことで、仮説と根拠の対応を読み手が追いやすくなる。分析トピックが複数ある
場合は notebook を分け、各 notebook の中では次の流れを基本とする。

1. import・定数・対象 run / split / checkpoint の宣言
2. 実行ログ、checkpoint、split / 属性データ、必要な cache の読み込み
3. 分析トピックごとの処理と、それに対応する可視化・観察メモ
4. 全体のまとめ

生成物の依存関係は一方向に保つ。notebook から run artifact や checkpoint を読むこと自体は禁止しない。
一方、図を描き直すために後段の結果から run を読み直すなど、依存関係を逆流させてはいけない。

```text
run artifact  →  notebook  →  figures/（必要な場合）
(<slug>/runs)               ↘ cache / results（必要な場合）
```

run artifact は package の `runs/` に取り込んだものだけを読む（`common/runs.py import`、
[README](README.md#run-の取り込み)）。`projects/*/runs/` を直接読まない。

外部 module / script への切り出しは、複数の notebook / package で共有する、実行に時間がかかる、
notebook 内で長くなりすぎる、または固定してテストしたいロジックがある場合に行う。切り出す場合も、
分析固有の判断と実行順序は notebook に残す。`common/` には run 契約の読み方や予測 cache など、
複数 package で実際に共有する処理だけを置く。群の切り方・指標の定義・図は原則として package または
notebook に閉じる。

## notebook

- **1 cell = 1 つの意味のある step**。cell の前に markdown を置き、何を見るのか、
  なぜそれを見るのかを書く。結果だけの cell を並べない。
- `import` は先頭 1 cell に集める。cell の途中で `import` しない。
- notebook 内の関数が長くなり、分析の流れを隠すようになったら外部 module へ出す。短い補助関数や
  その notebook のための集計処理は notebook 内に置いてよい。
- `sys.path` を直接触らない。path 解決は下の 1 行だけを使う。
- 出力は Git 追跡しない（`nbstripout`）。**kernel restart → 全実行が通る状態**で終える。

## script

- module docstring に、その script が何を入力に何を書くのかと、実行例を書く。
- 関数は 60 行・引数 6 個を上限の目安にする。超えたら段を分ける。
- 内包表記をネストしない。中間結果に名前を付ける。1 行に詰め込まない（`line-length = 120`）。
- 書き出し先は `results/` と `figures/` を基本とする。時間のかかる中間結果を再利用する必要が
  ある場合は `cache/` を使ってよい。notebook の図は inline 表示を基本とし、レポート等で再利用する
  図だけ `figures/` に保存する。

## テスト

`analysis/tests/` に置き、引数なしの `uv run pytest` で走る。**`common/` に手を入れるときは
テストを伴う**。2 つ以上の package が依存するので、ここが静かに間違うと結論だけが変わる。

対象にするのは「落ちずに間違う」経路とする。epoch の束ね直し、checkpoint の選び方、cache と
split CSV の行対応、群をまたいだ worst / gap の扱い。型を確かめるだけのテストは書かない。

学習側と同じ意味を持つ指標は、`projects/` の実装を import して突き合わせるのではなく
`tests/golden/` の固定データと一致することを見る。理由は
[`.agents/skills/migrate/rationale.md`](../.agents/skills/migrate/rationale.md) と同じで、
実装どうしを比べると、どちらが正しいのか分からないまま両方が動く。

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
