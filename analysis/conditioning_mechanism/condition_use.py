"""学習済みの属性条件付きモデルが、条件をどれだけ使っているかを測る。

test の画像はそのままに、条件（属性辞書の全列）だけを行の間で入れ替えて推論し直す。条件を使っていない
モデルなら予測は変わらない。属性の入れ替えで AUROC が下がらないなら、条件は使われていないか、画像と同じ情報しか
運んでいない。指標は次の 3 つ。
- `shift`：ロジット差（陽性 − 陰性）の変化の絶対値の平均を、元のロジット差の SD で割ったもの
- `auroc_true`・`auroc_perm`：元の条件と入れ替えた条件での AUROC
- `flip`：val の Youden しきい値で判定が反転した行の割合

入れ替えは行全体の置換（seed 0）。定数の条件では入れ替えても何も変わらないので、0 の対照になる。
2026-10-17 に `archive/latent_bias_hypernet/condition_use.py` の `measure` をここへ移した。

入力: `runs/<run-id>/`（属性条件付きの run）
出力: `results/<dataset>_condition_use.csv`（GPU で 1 run 約 10 分。既にある run は飛ばす）

    uv run python -m analysis.conditioning_mechanism.condition_use --dataset chexpert \
        --run-dir analysis/conditioning_mechanism/runs/<run-id> ...
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import rootutils
import torch
from hydra.utils import instantiate
from omegaconf import OmegaConf
from sklearn.metrics import roc_auc_score, roc_curve

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from analysis.common.paths import local_data_path  # noqa: E402
from analysis.common.predictions import load_network  # noqa: E402
from analysis.conditioning_mechanism.data import PACKAGE  # noqa: E402


def collect_attributes(loader) -> dict[str, torch.Tensor]:
    parts: dict[str, list[torch.Tensor]] = {}
    for _, attributes, _ in loader:
        for key, value in attributes.items():
            parts.setdefault(key, []).append(value)
    return {key: torch.cat(v) for key, v in parts.items()}


def forward(network, loader, attrs, device, rows_of) -> tuple[np.ndarray, np.ndarray]:
    out, target, offset = [], [], 0
    with torch.inference_mode():
        for image, attributes, y in loader:
            rows = rows_of[offset : offset + len(y)]
            batch = {key: attrs[key][rows].to(device) for key in attributes}
            logits = network(image.to(device, non_blocking=True), batch).float().cpu().numpy()
            out.append(logits[:, 1] - logits[:, 0])
            target.append(y.numpy())
            offset += len(y)
    return np.concatenate(out), np.concatenate(target)


def measure(run_dir: Path, device: torch.device) -> dict:
    config = OmegaConf.load(run_dir / "config.yaml")
    for key in ("cv_splits_dir", "data_dir"):
        config.data[key] = str(local_data_path(config.data[key]))
    datamodule = instantiate(config.data)
    network = load_network(run_dir, device)
    val_loader, test_loader = datamodule.evaluation_dataloader("val"), datamodule.evaluation_dataloader("test")
    val_attrs = collect_attributes(val_loader)
    zv, yv = forward(network, val_loader, val_attrs, device, np.arange(len(val_loader.dataset)))
    fpr, tpr, t = roc_curve(yv, zv)
    thr = t[np.argmax(tpr - fpr)]
    attrs = collect_attributes(test_loader)
    n = len(test_loader.dataset)
    z, y = forward(network, test_loader, attrs, device, np.arange(n))
    zp, _ = forward(network, test_loader, attrs, device, np.random.default_rng(0).permutation(n))
    return {
        "shift": np.abs(zp - z).mean() / z.std(),
        "auroc_true": roc_auc_score(y, z),
        "auroc_perm": roc_auc_score(y, zp),
        "flip": ((z >= thr) != (zp >= thr)).mean(),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", required=True, choices=("chexpert", "ham10000", "fairvlmed"))
    parser.add_argument("--run-dir", type=Path, nargs="+", required=True)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    dest = PACKAGE / "results" / f"{args.dataset}_condition_use.csv"
    done = pd.read_csv(dest) if dest.exists() else pd.DataFrame(columns=["run_id"])
    rows = done.to_dict("records")
    for run_dir in args.run_dir:
        if run_dir.name in set(done.run_id):
            continue
        rows.append({"run_id": run_dir.name, **measure(run_dir, torch.device(args.device))})
        print(rows[-1], flush=True)
        pd.DataFrame(rows).to_csv(dest, index=False)


if __name__ == "__main__":
    main()
