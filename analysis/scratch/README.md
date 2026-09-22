# scratch

まだ仮説に紐づいていない run の置き場。経路確認、設定の当たり付け、1 epoch の動作確認など、
**結論を出すつもりが無い run** の `study` にこれを指定する。

```bash
uv run python -m projects.hypernet_e2e.run experiment=spatial_lora_chexpert study=scratch trainer.max_epochs=1
```

`study` は必須なので、指定できないからといって未設定で回すことはできない。ここは
「どの仮説にも属さない」と明示するための逃げ道であり、結論の根拠にする run は置かない。
結論を出す段階になったら `analysis/<slug>/` を新しく作り、その slug で回し直す。

この package は `README.md` だけを持つ。`runs.md` も `results/` も作らない
（scratch の run 一覧を維持する意味が無いため）。
