"""e2e の Hydra preset が独立して合成・インスタンス化できることを検証する。"""

from pathlib import Path

import pytest
from hydra import compose, initialize_config_dir
from hydra.utils import instantiate


@pytest.mark.parametrize("architecture", ["resnet_chexpert", "spatial_lora_chexpert"])
@pytest.mark.parametrize("strategy", ["erm", "inverse_weighted_loss", "inverse_weighted_sampling"])
def test_chexpert_presets_select_only_their_intended_weighting(architecture: str, strategy: str) -> None:
    config_dir = Path(__file__).parent.parent / "configs"
    with initialize_config_dir(version_base="1.3", config_dir=str(config_dir)):
        cfg = compose(config_name="train", overrides=[f"experiment={architecture}_{strategy}"])

    assert cfg.weighting == ("inverse" if strategy == "inverse_weighted_loss" else "none")
    assert cfg.data.train_sampling == ("inverse_frequency" if strategy == "inverse_weighted_sampling" else "uniform")
    assert instantiate(cfg.data) is not None
    assert instantiate(cfg.model) is not None


def test_default_callbacks_include_metrics_and_fairness_without_text_progress() -> None:
    config_dir = Path(__file__).parent.parent / "configs"
    with initialize_config_dir(version_base="1.3", config_dir=str(config_dir)):
        cfg = compose(config_name="train")

    assert set(cfg.callbacks) == {"model_checkpoint", "model_summary", "metrics_logger", "fairness_metrics"}
    assert cfg.callbacks.model_checkpoint.dirpath is None
