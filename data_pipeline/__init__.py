"""固定入力（split CSV など）を生成する data 準備層。

学習 run は `data/` 配下を読み取り専用の固定入力として扱う。ここはその固定入力を
作る側であり、run からは呼ばれない。

分割単位・比率・CV 戦略の根拠は `data_pipeline/README.md` を正本とする。
"""
