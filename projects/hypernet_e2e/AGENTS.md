# hypernet_e2e の実装規約

この project は hypernetwork 系モデルの学習・性能比較を所有する。他 project から import
せず、他 project 用の共有 module もここに置かない。

## 実験 logger

- 通常の実験 fit では、外部 logger に `WandbLogger`（project `fairness_hypernet`）を必ず使う。
- `CSVLogger` は `run_fit` が常に追加するため、`metrics/metrics.csv` として必ず残る。W&Bを無効化してはならない。
- `logger=none` は preflight、単体テスト、依存関係確認など、結果を分析対象にしない短い動作確認に限る。
  分析対象の学習でW&B認証や接続に問題がある場合は、loggerを黙って無効化せず起動を止めて確認する。

## docstring とコメント

- docstring は日本語で書く。公開 module・class・関数には、その **interface** を利用者が
  理解するための docstring を付ける。自明な private helper や test には不要。公開関数・公開
  メソッドの docstring は Google 形式の `Args:` と `Returns:` を必須とし、引数がない場合は
  `Args: なし`、戻り値がない場合は `Returns: None` と明記する。送出し得る例外が利用者の
  分岐や復旧に必要な場合は `Raises:` も付ける。
- class の docstring は1行の責務要約を必須とする。入出力 shape、属性辞書のキー、設定値の
  制約、初期化・状態更新など、呼び出し側が知る必要がある不変条件は続けて明記する。
- コメントはコードの逐語訳ではなく、**なぜこの実装・順序・数値設定なのか**を説明する場合に
  限る。先行研究との対応、数値再現性のために変えてはいけない処理、非自明な初期化、意図的な
  例外はコメントに残す。
- モデルの数値挙動に関わる既存コードを移植する際は、実装を整理・書き換えない。説明の追記は
  許可するが、挙動変更と同じ変更に混ぜない。
- stale になった docstring・コメントはコードと同じ変更で更新または削除する。将来の利用を
  仮定した一般論は書かず、現在の project の責務だけを記述する。
