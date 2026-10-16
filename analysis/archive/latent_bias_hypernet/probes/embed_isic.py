"""ISIC 2019 の施設間 split の画像を DINOv2 で埋め込む（e(x)）。

embed.py の ISIC 版。facebook/dinov2-base をローカルの HF cache から読み、224 で CLS と patch 平均を連結した
1536 次元を float16 で保存する。正規化は ImageNet。

出力: `cache/embed/dinov2/isic_<split>.npz`（image、embedding、marked）

    HF_HUB_OFFLINE=1 uv run python analysis/archive/latent_bias_hypernet/probes/embed_isic.py
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import rootutils
import torch
from torch.utils.data import DataLoader
from transformers import AutoModel

ROOT = rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from analysis.archive.latent_bias_hypernet.probes.embed import MODELS, build_transform  # noqa: E402
from analysis.archive.latent_bias_hypernet.probes.train_stage1_isic import PACKAGE, SPLITS, Images  # noqa: E402


def main() -> None:
    out = PACKAGE / "cache" / "embed" / "dinov2"
    out.mkdir(parents=True, exist_ok=True)
    model = AutoModel.from_pretrained(MODELS["dinov2"][0]).to("cuda").eval()
    transform = build_transform("dinov2", 224)
    for split in SPLITS:
        frame = pd.read_csv(PACKAGE / "cache" / "context" / f"{split}.csv")
        dl = DataLoader(Images(frame, transform), batch_size=128, num_workers=8, prefetch_factor=2)
        parts = []
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            for image, _y, _i in dl:
                hidden = model(pixel_values=image.to("cuda", non_blocking=True)).last_hidden_state
                pooled = torch.cat([hidden[:, 0], hidden[:, 1:].mean(dim=1)], dim=1)
                parts.append(pooled.float().cpu().numpy().astype(np.float16))
        np.savez(
            out / f"{split}.npz",
            image=frame.image.to_numpy(dtype=str),
            embedding=np.concatenate(parts),
            marked=np.zeros(len(frame), dtype=bool),
        )
        print(split, len(frame), flush=True)


if __name__ == "__main__":
    main()
