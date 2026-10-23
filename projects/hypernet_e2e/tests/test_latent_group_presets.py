"""潜在群（dw8・attrk8）の preset が、8 群の preset と群の定義だけ違うことを検証する。"""

from pathlib import Path

import pandas as pd
import pytest
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

CONFIGS = Path(__file__).parents[1] / "configs"
DATA = Path(__file__).parents[3] / "data" / "chexpert"
# 潜在群の preset と、群の定義だけを差し替えた元の 8 群の preset。
PARENTS = {
    f"{model}_chexpert_{part}_{objective}": f"{model}_chexpert_age_race_{objective}"
    for model in ("resnet", "spatial_lora")
    for part in ("dw8", "attrk8")
    for objective in ("group_dro_race_known", "eo_rate_dual_holdout")
}
# 群の差し替えで変わってよい key（data は下で個別に見る）。
CHANGED = {"experiment_name", "tags", "latent_partition", "data"}


def _compose(preset: str, seed: int):
    with initialize_config_dir(version_base="1.3", config_dir=str(CONFIGS)):
        config = compose(config_name="train", overrides=[f"experiment={preset}", f"seed={seed}", "study=scratch"])
    return OmegaConf.to_container(config, resolve=True)


@pytest.mark.parametrize("preset", sorted(PARENTS))
def test_only_group_definition_differs(preset):
    latent, parent = _compose(preset, 43), _compose(PARENTS[preset], 43)
    part = preset.split("_chexpert_")[1].split("_")[0]
    assert latent["data"]["group_attribute_names"] == [f"{part}_s43"]
    assert latent["data"]["group_cardinalities"] == [8]
    assert latent["data"]["num_groups"] == 8
    assert latent["data"]["cv_splits_dir"] == parent["data"]["cv_splits_dir"] + "_latent"
    for key in set(parent["data"]) - {"cv_splits_dir", "group_attribute_names", "group_cardinalities"}:
        assert latent["data"][key] == parent["data"][key], key
    assert latent["callbacks"]["group_dro_diagnostics"]["group_labels"] == [f"latent {i}" for i in range(8)]
    latent["callbacks"]["group_dro_diagnostics"] = parent["callbacks"]["group_dro_diagnostics"]
    for key in set(parent) | set(latent):
        if key not in CHANGED:
            assert latent.get(key) == parent.get(key), key
    # 評価は年齢 × race の 8 群のまま。
    assert latent["callbacks"]["fairness_metrics"]["intersectional_groups"]["age_race_group"]["attributes"] == [
        "age_group_65",
        "race_group",
    ]


@pytest.mark.parametrize("preset", sorted(PARENTS))
def test_group_columns_exist_in_splits(preset):
    config = _compose(preset, 44)
    directory = DATA / Path(config["data"]["cv_splits_dir"]).name
    if not directory.exists():
        pytest.skip(f"{directory} が無い（latent_groups.py で作る）")
    column = config["data"]["group_attribute_names"][0]
    for csv in directory.glob("*.csv"):
        header = pd.read_csv(csv, nrows=1).columns
        assert column in header and f"{column}_missing" in header, csv.name
