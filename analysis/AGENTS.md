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
| Wilson et al., *Good enough practices in scientific computing* (PLOS CB 2017) | 重複を排し、ファイル名は内容を表す |
| Ousterhout, *A Philosophy of Software Design* (2018) | 中身の薄い関数は読み手の負担を増やす。コメントにはコードから読めないことを書く |
| Google Python Style Guide §3.8, §3.18 | docstring が要るのは公開・非自明な関数だけ。コードを言い換えない。関数長に上限は置かない |

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
- 関数の分け方・検証・docstring は下の 3 節に従う。
- 内包表記をネストしない。中間結果に名前を付ける。1 行に詰め込まない（`line-length = 120`）。
- 書き出し先は `results/` と `figures/` を基本とする。時間のかかる中間結果を再利用する必要が
  ある場合は `cache/` を使ってよい。notebook の図は inline 表示を基本とし、レポート等で再利用する
  図だけ `figures/` に保存する。

## 関数の分け方

関数を切り出すのは、**名前を付ける価値のある段**ができたときだけにする。行数は分ける理由にならない。
読み手は関数に出会うたびに定義へ飛ぶので、中身が 1〜2 行で 1 か所からしか呼ばれない関数は、
名前から得るものより飛ぶ手間のほうが大きい。そういう処理は呼び出し元にそのまま書く。

切り出してよいのは次のどれかに当たるときに限る。

- 書く側と読む側など、複数の場所が**同じ結果を作らなければならない**
- 呼び出し元から見て 1 つの意味を持つ段になり、分けると呼び出し元が上から読めるようになる
- テストで単独に固定したいロジックがある（「落ちずに間違う」経路。[テスト](#テスト)）

悪い例（`paths.py`）。1 行の式を包んで docstring とテストまで付けているが、`run_dir` を呼ぶのは
`runs.py` の 1 か所だけで、`split_csv` はテスト以外のどこからも呼ばれていない。

```python
def run_dir(study: str, run_id: str) -> Path:
    """（Args / Returns を並べた 9 行の docstring）"""
    return runs_root(study) / run_id

def split_csv(dataset: str, split: str) -> Path:
    """（Args / Returns を並べた 12 行の docstring）"""
    return REPOSITORY_ROOT / "data" / dataset / "splits" / f"{split}.csv"
```

`runs_root(study) / run_id` は呼び出し元に書いたほうが何をしているか読める。使われていない関数は
消す。

良い例（`predictions.py`）。どちらも上の条件に当たる。

- `cache_path` は 1 行だが、書く側（`write_cache`）と読む側（各 package の notebook・script）が
  同じ file 名を作る約束なので、1 か所に置く意味がある。
- `_forward_split` は loader を 1 周して logits と表現を集める段で、`write_cache` から見ると
  「推論する」という 1 つの意味になる。

## 検証

検証は**外から来るものを読む境界**にだけ置く。境界とは、run artifact・split CSV・cache を
読み込むところで、ここで食い違うと数値が「それらしく」出たまま結論が変わる。境界の内側の関数は、
呼び出し元が渡す引数と、Lightning・PyTorch・OmegaConf の挙動を信じる。

書かないもの：

- repo の契約上起こらないことの検査。run directory は不変なので、取り込んだ checkpoint が
  後から書き換わることは無い
- 古い形式との互換処理。形式を変えたら、古い cache は作り直す（`--overwrite`）
- 次の行が同じ失敗で落ちる場合の事前検査。どうせ落ちるなら、元の例外のほうが原因に近い

悪い例（`predictions.py`）。

```python
# load_network: torch.load の結果を型で確かめている。state_dict が無ければ次の
# weights の内包表記で落ちるし、重みがずれていれば load_state_dict(strict=True) が落ちる。
state_dict = checkpoint.get("state_dict")
if not isinstance(state_dict, Mapping):
    raise TypeError(f"{run_dir} の checkpoint に state_dict がない")

# stale_reason: 不変な run directory の checkpoint を、cache を使うたびに SHA-256 で照合する。
# この hash を使い回すために、source_hashes という辞書を 4 つの関数に引き回している。
if metadata.get("checkpoint_sha256") != _sha256(_checkpoint_path(run_dir, checkpoint), source_hashes):
    return "checkpoint SHA-256 が一致しない"

# _from_run_id: 古い記録の path 形式を、新しい形式へ読み替えてから照合する。
```

良い例（`predictions.py` の `_load_npz`）。cache の i 番目と split CSV の i 行目が対応している
ことを、読み込むたびに image 列で確かめる。ずれていても例外は出ず、群の割り当てが全部ずれた表が
できるので、ここは境界の検証として残す。

```python
if len(result["image"]) != len(images) or not (result["image"] == images).all():
    raise ValueError(f"{path} の行が split CSV と対応していない（群の割り当てが全部ずれる）")
```

## docstring とコメント

`projects/` とは規約が違う。analysis では `Args:` / `Returns:` を必須にしない。

docstring を書くのは、ほかの file から呼ばれる関数で、名前と型注釈だけでは分からないことが
あるときだけにする。書くのは次のことで、引数や返り値を言い換えるだけの行は書かない。

- **なぜそうしたか**：その群の切り方を選んだ理由、その列を見る理由、その形式を選んだ理由
- **何を保証しないか**：その指標が何を意味しないか、その関数が確かめていないこと

`Args:` は、型と名前から意味が読めない引数があるときだけ、その引数について書く。`Raises:` は、
呼び出し側が捕まえて分岐する例外があるときだけ書く。

悪い例（`common/`）。

```python
def cache_path(cache_dir: Path, run_id: str, split: str) -> Path:
    """run と split に対応する cache path を返す。

    Args:
        cache_dir: cache を置く directory
        run_id: run ID
        split: 評価した split 名

    Returns:
        Path: `<cache_dir>/<run_id>_<split>.npz`
    """
    return cache_dir / f"{run_id}_{split}.npz"

def main() -> None:
    """（中略）

    Args:
        なし

    Returns:
        None
    """
```

`cache_path` は関数としては残してよい（[関数の分け方](#関数の分け方)）が、docstring は関数名・引数名・
本体の 1 行がすでに言っていることの繰り返しなので要らない。`main()` の `Args: なし` /
`Returns: None` は何も伝えていない。

良い例（`run_artifacts.py` の `read_epoch_metrics`）。コードを読んでも分からない、ログの形と
除く理由を書いている。実物はこの後に `Args:` / `Returns:` も続くが、この規約ではそこは要らない。

```python
def read_epoch_metrics(csv_path: Path) -> list[dict[str, float]]:
    """`CSVLogger` の出力を epoch ごとの 1 行へまとめる。

    train と val は別行に書かれるので epoch で束ね直す。空セルはその行が記録しなかった
    指標を表すので落とす。`step` は optimizer step であり epoch と混ざるため除く。
    """
```

コメントも同じ考え方で書く。コードが何をしているかではなく、なぜそうしたかを書く
（例：`# wandb の import は重い。transaction log を読む run だけが払えばよい。`）。同じ理由を
module docstring とコメントに 2 回書かない。

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

## 追跡境界

`analysis/README.md` の表を正本とする。ここでは繰り返さない。
