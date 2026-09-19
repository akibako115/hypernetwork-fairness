# hypernet_two_stage の実装規約

この project は Stage 1 ResNet と Stage 2 Spatial LoRA の固定二段学習を所有する。
他 project から import せず、他 project 用の共有 module もここに置かない。

## workflow

- `workflow.py` が Stage 1 の fit、最良 `val/auroc` checkpoint の選択、Stage 2 への入力、親 run の
  状態確定を所有する。呼び出し側は stage の順序や checkpoint path を操作しない。
- Stage 2 は Stage 1 の選択 checkpoint から共有 backbone と classifier を読み込み、両者を凍結する。
  metadata encoder と Spatial LoRA だけを更新する。
- cohort の再生成・GroupDRO・任意回数の stage 反復はこの project の責務に含めない。

## docstring とコメント

- docstring は日本語で書く。公開 module・class・関数には、その **interface** を利用者が
  理解するための docstring を付ける。自明な private helper や test には不要。
- class の docstring は1行の責務要約を必須とする。入出力 shape、属性辞書のキー、設定値の
  制約、初期化・状態更新など、呼び出し側が知る必要がある不変条件は続けて明記する。
- コメントはコードの逐語訳ではなく、**なぜこの実装・順序・数値設定なのか**を説明する場合に
  限る。数値再現性のために変えてはいけない処理、非自明な初期化、意図的な例外はコメントに残す。
- モデルの数値挙動に関わる既存コードを移植する際は、実装を整理・書き換えない。説明の追記は
  許可するが、挙動変更と同じ変更に混ぜない。
