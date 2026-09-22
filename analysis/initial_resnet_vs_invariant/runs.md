# runs

この分析が引用する run の正本。run 本体は Git 管理外のため、分析コードが読む repo root からの相対
`run path` と、dashboard を開くための W&B URL をここに残す。

## 対象（seed 43 / 44 / 45 × 2 手法）

6 本とも ws11 で 2026-09-22 に実行し、`status=succeeded`。`weighting=inverse`、
class weight `[0.201358, 1.798642]`、AdamW lr 1e-4 / wd 0.01、batch size 128、30 epoch。
train / val split の sha256（`ad5356a7...` / `59b71453...`）も 6 本で一致する。
違いは **seed と第1段の loss だけ**になる。

| condition | seed | project | run-id | run path | W&B |
|---|---|---|---|---|---|
| ResNet | 43 | `hypernet_e2e` | `20260922T063725Z-resnet-chexpert-s43-efcd` | `projects/hypernet_e2e/runs/20260922T063725Z-resnet-chexpert-s43-efcd` | [run](https://wandb.ai/kohki-akiba-kyushu-university/fairness_hypernet_e2e/runs/8e0x1wmq) |
| ResNet | 44 | `hypernet_e2e` | `20260922T063727Z-resnet-chexpert-s44-6f0c` | `projects/hypernet_e2e/runs/20260922T063727Z-resnet-chexpert-s44-6f0c` | [run](https://wandb.ai/kohki-akiba-kyushu-university/fairness_hypernet_e2e/runs/i5he4orn) |
| ResNet | 45 | `hypernet_e2e` | `20260922T063727Z-resnet-chexpert-s45-fb77` | `projects/hypernet_e2e/runs/20260922T063727Z-resnet-chexpert-s45-fb77` | [run](https://wandb.ai/kohki-akiba-kyushu-university/fairness_hypernet_e2e/runs/9w7uotjc) |
| attribute-invariant | 43 | `hypernet_e2e` | `20260922T063725Z-resnet-chexpert-attribute-invariant-s43-546c` | `projects/hypernet_e2e/runs/20260922T063725Z-resnet-chexpert-attribute-invariant-s43-546c` | [run](https://wandb.ai/kohki-akiba-kyushu-university/fairness_hypernet_e2e/runs/6pgqjafu) |
| attribute-invariant | 44 | `hypernet_e2e` | `20260922T063728Z-resnet-chexpert-attribute-invariant-s44-8e6b` | `projects/hypernet_e2e/runs/20260922T063728Z-resnet-chexpert-attribute-invariant-s44-8e6b` | [run](https://wandb.ai/kohki-akiba-kyushu-university/fairness_hypernet_e2e/runs/awdtcal9) |
| attribute-invariant | 45 | `hypernet_e2e` | `20260922T063728Z-resnet-chexpert-attribute-invariant-s45-404a` | `projects/hypernet_e2e/runs/20260922T063728Z-resnet-chexpert-attribute-invariant-s45-404a` | [run](https://wandb.ai/kohki-akiba-kyushu-university/fairness_hypernet_e2e/runs/23loskog) |

invariant 側の adversary は **sex / race（categorical）と age（continuous）** にかかる。入力として
渡す属性は sex / race / ethnicity / frontal_lateral / ap_pa / age の 6 つで、そのうち上の 3 つだけを
GRL で消す（`attribute_adversary_weight=0.1`、`gradient_scale=1.0`、`hidden_dim=256`）。

## 混ぜてはいけない run

**seed 42 の pair（`20260921T103036Z-...-s42-5538` と `20260921T125711Z-...-s42-9fd3`）は
この分析に含めない。** s42 の invariant は `adversarial_attribute_names` が入る前のコードで回っており、
adversary が **全属性**（sex / race / ethnicity / frontal_lateral / ap_pa / age）にかかる。上の 3 本とは
別の手法なので、同じ系列として平均を取ると手法の差と設定の差が混ざる。

s42 の invariant run は削除済みで、再生成できない。s42 の ResNet run は
[iterative_probe](../iterative_probe/) が baseline として使うため残してある。

## ws11 について

ws11（`kohkiakiba@192.168.1.21`）の `/mnt/fast/kohkiakiba/hypernet-fairness/` で実行し、
`best_val_auroc_*.ckpt` を回収した。転送の規約は
[`.agents/skills/train/transfer.md`](../../.agents/skills/train/transfer.md) に従う。
`run.json` の checkpoint path は container の `/workspaces/...` を指すが、`selected_checkpoint` が
run directory からの相対位置で解決する。同じ理由で `config.yaml` の `cv_splits_dir` /
`data_dir` も container の path なので、`analysis/common/paths.py` の `local_data_path` が
この repo の `data/` へ読み替える。`.git` を除外して同期しているため `git_commit` は `null`。

入力は各 run path 配下の `config.yaml`、`run.json`、`metrics/fit.json`、checkpoint を使う。
再生成物はこの package の `cache/`、`results/` に置く。図は notebook の出力として持つ。
