# hypernet_iterative の実装規約

この project は、学習の途中で hidden cohort を作り直しながら複数 stage を進める反復学習を
所有する。他 project から import せず、他 project 用の共有 module もここに置かない。

## workflow

- `workflow.py` が parent run の予約、warmup と cohort stage の実行順、cohort artifact の生成、
  parent run の状態確定を所有する。呼び出し側は stage の順序や checkpoint path を操作しない。
- 反復の計画は `iteration.*` が正本とする。`training_strategy` / `cohort_definition` /
  `checkpoint_selection` の config group は stage config が満たすべき形の宣言であり、parent run の
  挙動はそこからは変わらない。片方だけを変えない。
- 各 fit は `stage.py` の独立 process が実行する。parent が組み立てた stage config でも、子 process
  は `validate_training_config` で必ず再検証する。生成結果を無検証で信用しない。
- stage 間で引き継ぐのは `net` の tensor だけとする。optimizer、scheduler、GroupDRO の
  adversarial weight を stage をまたいで持ち越さない。
- cohort は固定入力ではなく、生成した run が所有する artifact とする。参照 checkpoint の絶対 path と
  SHA-256 を `cohort.json` に残し、既存 artifact directory を上書きしない。
- 次 stage が参照する checkpoint は選択基準まで照合して1つに絞る。`best_model_path` を持つ先頭の
  callback を使わない。

## docstring とコメント

- docstring は日本語で書く。公開 module・class・関数には、その **interface** を利用者が
  理解するための docstring を付ける。自明な private helper や test には不要。
- class の docstring は1行の責務要約を必須とする。入出力 shape、属性辞書のキー、設定値の
  制約、初期化・状態更新など、呼び出し側が知る必要がある不変条件は続けて明記する。
- コメントはコードの逐語訳ではなく、**なぜこの実装・順序・数値設定なのか**を説明する場合に
  限る。先行研究との対応、数値再現性のために変えてはいけない処理、非自明な初期化、意図的な
  例外はコメントに残す。
- モデルの数値挙動に関わる既存コードを移植する際は、実装を整理・書き換えない。説明の追記は
  許可するが、挙動変更と同じ変更に混ぜない。
- stale になった docstring・コメントはコードと同じ変更で更新または削除する。将来の利用を
  仮定した一般論は書かず、現在の project の責務だけを記述する。
