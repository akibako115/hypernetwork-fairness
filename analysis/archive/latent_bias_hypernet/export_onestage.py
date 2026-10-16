"""Q2 の 1 段の run（hypernet_e2e の `*_dual_holdout` preset）から、epoch ごとの val・test の予測と群の重みを書き出す。

CheXpert（Spatial LoRA）と HAM10000（ResNet-50・Spatial LoRA）の run を受け付ける。どちらかは run の
`experiment_name` で決まる。

2 段の run（`probes/train_adapter_fair.py`）と同じ形にそろえ、`q2_fairness_loss.py` が 3 つの手順を
同じ規則で読めるようにする。
1 段の run は全 epoch の重みを残しているので、各 epoch の checkpoint で val・test を推論する。
epoch の選び方は分析側で決める（全体 AUROC と、損失自身の狙い）。

群の重みは `metrics/objective_diagnostics.jsonl`（各 epoch の最後の step の診断値）から読む。2 段は epoch 中の
held-out の率を平均して残すのに対し、1 段は最後の 1 バッチ（16 セル × 8 件）の値なので、率の読みは粗い。
λ と q は学習の状態そのものなので、どちらも同じ意味で読める。

入力: `projects/hypernet_e2e/runs/<run-id>/`（`config.yaml`・`checkpoints/best_val_auroc_XXX.ckpt`・`metrics/`）
出力（`epoch_XX_{val,test}.npy` は z1 − z0、`log.json`）:
- CheXpert: `analysis/archive/latent_bias_hypernet/cache/adapter_fair/onestage/<loss>_meta_s<seed>/`
- HAM10000: `analysis/archive/latent_bias_hypernet/cache/ham10000_q2/<model>_<loss>_s<seed>/`（model は resnet・meta）

    uv run python -m analysis.archive.latent_bias_hypernet.export_onestage --run-dir projects/hypernet_e2e/runs/<run-id> ...
"""

from __future__ import annotations

import argparse
import inspect
import json
from pathlib import Path

import numpy as np
import torch
from hydra.utils import instantiate
from omegaconf import OmegaConf

from analysis.archive.latent_bias_hypernet.data import PACKAGE
from analysis.common.paths import local_data_path
from analysis.common.predictions import _forward_split, load_network
from projects.hypernet_e2e.data.datamodule import ImageDataModule

OUT = PACKAGE / "cache" / "adapter_fair" / "onestage"
OUT_HAM = PACKAGE / "cache" / "ham10000_q2"
# experiment_name → (出力先の親, model, loss)。CheXpert は既存の出力名（`<loss>_meta_s<seed>`）を変えない。
RUNS = {
    "spatial-lora-chexpert-dual-holdout-erm": (OUT, "meta", "erm"),
    "spatial-lora-chexpert-age-race-eo-rate-dual-holdout": (OUT, "meta", "eo"),
    "spatial-lora-chexpert-age-race-minimax-auc-dual-holdout": (OUT, "meta", "minimax_auc"),
    "resnet-ham10000-dual-holdout-erm": (OUT_HAM, "resnet", "erm"),
    "resnet-ham10000-sex-age-eo-rate-dual-holdout": (OUT_HAM, "resnet", "eo"),
    "resnet-ham10000-sex-age-minimax-auc-dual-holdout": (OUT_HAM, "resnet", "minimax_auc"),
    "spatial-lora-ham10000-dual-holdout-erm": (OUT_HAM, "meta", "erm"),
    "spatial-lora-ham10000-sex-age-eo-rate-dual-holdout": (OUT_HAM, "meta", "eo"),
    "spatial-lora-ham10000-sex-age-minimax-auc-dual-holdout": (OUT_HAM, "meta", "minimax_auc"),
}


def epoch_checkpoints(run_dir: Path) -> dict[int, Path]:
    """`best_val_auroc_XXX.ckpt`（save_top_k = -1 なので全 epoch）を epoch 番号で引く。"""
    return {int(p.stem.rsplit("_", 1)[1]): p for p in (run_dir / "checkpoints").glob("best_val_auroc_*.ckpt")}


def loaders(config) -> dict:
    """val・test を CSV の順で返す DataLoader。2 段の run と同じ行・同じ順になる。"""
    for key in ("cv_splits_dir", "data_dir"):
        config.data[key] = str(local_data_path(config.data[key]))
    if "group_attribute_names" in config.data:
        # 公平性損失の run の datamodule は群 ID を作るため、群の属性が欠けた行（HAM10000 の test の外部施設）で
        # 止まる。予測に群 ID は要らないので、群を作らない ImageDataModule で同じ前処理・同じ属性の入力を読む。
        accepted = inspect.signature(ImageDataModule.__init__).parameters
        plain = {k: v for k, v in OmegaConf.to_container(config.data).items() if k in accepted}
        # train の抽出（群 × クラス均等）は評価の loader に効かないので、ImageDataModule が受け付ける既定に戻す。
        plain |= {"_target_": f"{ImageDataModule.__module__}.ImageDataModule", "train_sampling": "uniform"}
        config.data = OmegaConf.create(plain)
    datamodule = instantiate(config.data)
    return {split: datamodule.evaluation_dataloader(split) for split in ("val", "test")}


def diagnostics_log(run_dir: Path, loss: str) -> dict:
    """群の重みを 2 段の `log.json` と同じ名前で並べる（EO は λ と held-out の率、minimax AUC は q と順位損失）。"""
    rows = [json.loads(line) for line in (run_dir / "metrics" / "objective_diagnostics.jsonl").read_text().splitlines()]
    epochs = []
    for row in rows:
        if loss == "eo":
            epochs.append({"epoch": row["epoch"], "lambda": row["lambda"], "holdout_rates": row["rates"]})
        elif loss == "minimax_auc":
            epochs.append({"epoch": row["epoch"], "q": row["q_after"], "holdout_risk": row["risk"]})
    return {"run_dir": str(run_dir), "epochs": epochs}


def export(run_dir: Path, device: torch.device) -> Path:
    # checkpoint は run_dir からの glob で引く。相対の run_dir のままだと `load_network` が run_dir を前に
    # 重ねて二重の path になるので、先に絶対 path にする。
    run_dir = run_dir.resolve()
    config = OmegaConf.load(run_dir / "config.yaml")
    parent, model, loss = RUNS[config.experiment_name]
    out = parent / (f"{loss}_meta_s{config.seed}" if parent == OUT else f"{model}_{loss}_s{config.seed}")
    out.mkdir(parents=True, exist_ok=True)
    use_attributes = bool(config.model.get("use_attributes", False))
    split_loaders = loaders(config)
    for epoch, checkpoint in sorted(epoch_checkpoints(run_dir).items()):
        network = load_network(run_dir, device, checkpoint)
        for split, loader in split_loaders.items():
            logits, _target, _ = _forward_split(network, loader, device, use_attributes, False)
            np.save(out / f"epoch_{epoch:02d}_{split}.npy", (logits[:, 1] - logits[:, 0]).astype(np.float32))
        print(f"{out.name} epoch {epoch}", flush=True)
    if loss != "erm":
        (out / "log.json").write_text(json.dumps(diagnostics_log(run_dir, loss)))
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, nargs="+", required=True)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    for run_dir in args.run_dir:
        export(run_dir, torch.device(args.device))


if __name__ == "__main__":
    main()
