# hypernet-fairness（エージェント向け入口）

医用画像分類の公平性を hypernetwork 系モデルで検証する研究 repo。構成と方針の正本は [README.md](README.md)、
実験 1 本の進め方は [docs/experiment-workflow.md](docs/experiment-workflow.md) にある。ここには、作業を始める前に
押さえる前提と、どこを読めばよいかだけを書く。

## 先に読むもの

| 作業 | 読むもの |
| --- | --- |
| 研究の現在地（いまの問い・どの文書が正本か） | [docs/README.md](docs/README.md) |
| 研究の前提・評価の定義・手法選定・結論の書き方 | [docs/research-premises.md](docs/research-premises.md) |
| 公平性損失（EO rate・GroupDRO）の run の設計・起動・checkpoint の選び方 | [docs/methods/eo-rate.md](docs/methods/eo-rate.md)、[docs/methods/group-dro.md](docs/methods/group-dro.md)。原論文の条件との違いと、best val AUROC で選ばない理由 |
| 学習コード | `projects/<project>/AGENTS.md`（`CLAUDE.md` は同じ file への symlink）と `README.md` |
| 分析コード・notebook | [analysis/AGENTS.md](analysis/AGENTS.md)、[analysis/README.md](analysis/README.md) |
| データ split・群別指標の最小セル | [data_pipeline/README.md](data_pipeline/README.md) |

個々の計画・実験記録は日付とともに古くなるので、ここには名指しで書かない。今の正本は docs/README.md が指す。

作業手順は `.agents/skills/` が正本で、`.claude/skills/` から symlink している（train / gpu / analyze / data-explore / migrate）。
学習の起動・ws での実行は train、run 記録の分析は analyze を使う。

## 守ること

- **回線をまたぐ転送（rsync / scp）はエージェントが実行しない。** コマンドを提示してユーザーが打つ。
  規約は [.agents/skills/train/transfer.md](.agents/skills/train/transfer.md)。
- 作業ツリーは複数セッションで共有される。git のブランチ操作の前に `git branch --show-current` と
  `git status --short` を確認し、`reset --hard` より一時 worktree を使う。
- run directory（`projects/*/runs/`）は不変。分析は `analysis/<study>/runs/` に取り込んだものだけを読む。
  失敗した run は、原因を報告したら消す（[.agents/skills/train/failures.md](.agents/skills/train/failures.md)）。
- 学習は必ず `study=<slug>` を付けて起動し、承認を取ってから回す（train skill）。
- `rg` は入っていない。`grep` を使う。
