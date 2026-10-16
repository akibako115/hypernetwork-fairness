"""M01（race-known の train 全体で学習した ResNet-50 ERM、seed 42）を、2 段目のスクリプトが読む 1 段目の形式で書き出す。

Q2 の 2 段 in-sample は、train 全体で学習した 1 段目の上に、同じ train で 2 段目を学習する。1 段・cross-fit と
全体で使うデータの量（train 全体）をそろえるためで、1 段目には M01 を使う。Lightning の checkpoint から
`net.` を外した state_dict を、`probes/train_stage1.py` の `best.pt` と同じ形で保存する。

入力: `analysis/archive/race_known_architecture_objective_comparison/runs/<M01 s42>/checkpoints/best_val_auroc_008.ckpt`
出力: `analysis/archive/latent_bias_hypernet/cache/stage1/m01_s42/best.pt`

    uv run python analysis/archive/latent_bias_hypernet/probes/export_m01_stage1.py
"""

from __future__ import annotations

import rootutils
import torch

ROOT = rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

RUN = "20261002T124957Z-resnet-chexpert-race-known-s42-d6d3"
SOURCE = ROOT / "analysis" / "archive" / "race_known_architecture_objective_comparison" / "runs" / RUN / "checkpoints"
TARGET = ROOT / "analysis" / "archive" / "latent_bias_hypernet" / "cache" / "stage1" / "m01_s42" / "best.pt"


def main() -> None:
    checkpoint = torch.load(SOURCE / "best_val_auroc_008.ckpt", map_location="cpu", weights_only=False)
    state = {k.removeprefix("net."): v for k, v in checkpoint["state_dict"].items() if k.startswith("net.")}
    reference = torch.load(TARGET.parent.parent / "a_s42" / "best.pt", map_location="cpu", weights_only=False)
    if set(state) != set(reference):
        raise ValueError("M01 の重みの名前が 1 段目の形式と一致しない")
    torch.save(state, TARGET)
    print(f"wrote {TARGET.relative_to(ROOT)} ({len(state)} tensors)")


if __name__ == "__main__":
    main()
