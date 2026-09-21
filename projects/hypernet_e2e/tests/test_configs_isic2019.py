"""ISIC 2019 の preset が dermoscopy 用の data・transform に繋がることを検証する。"""

from pathlib import Path

from hydra import compose, initialize_config_dir
from hydra.utils import instantiate


def test_isic2019_preset_connects_the_dermoscopy_data_and_transforms() -> None:
    """ISIC は 8 クラスで、CheXpert とは属性も transform も別物になる。"""
    config_dir = Path(__file__).parent.parent / "configs"
    with initialize_config_dir(version_base="1.3", config_dir=str(config_dir)):
        config = compose(config_name="train", overrides=["experiment=spatial_lora_isic2019"])

    assert config.dataset == "isic2019"
    assert config.data.num_classes == config.model.net.num_classes == 8
    assert config.data.cv_splits_dir.endswith("isic2019/splits_fairness")
    assert "no_shades_of_gray" in config.data.data_dir
    assert config.data.train_transform._target_.endswith("train_transforms_isic2019")
    assert list(config.data.attribute_spec.categorical_cardinalities) == [2, 8]
    # attribution は元データセットを表す共変量。モデル入力にも fairness の評価軸にも入れない。
    assert "attribution" not in config.data.fairness_attribute_names.categorical
    assert "attribution" not in config.data.attribute_names.categorical
    assert instantiate(config.model) is not None


def test_isic2019_cv_preset_points_each_fold_at_its_own_split_directory() -> None:
    """fold ごとに別ディレクトリを読まないと、全 run が同じ test を評価してしまう。"""
    config_dir = Path(__file__).parent.parent / "configs"
    for fold in range(5):
        with initialize_config_dir(version_base="1.3", config_dir=str(config_dir)):
            config = compose(config_name="train", overrides=["experiment=spatial_lora_isic2019_cv", f"fold={fold}"])

        assert config.data.cv_splits_dir.endswith(f"isic2019/splits_fairness_cv/fold{fold}")
        assert config.experiment_name.endswith(f"fold{fold}")
        # CV でも 8 クラス・属性・transform は単一 split の preset と同じ
        assert config.data.num_classes == 8
        assert config.data.train_transform._target_.endswith("train_transforms_isic2019")
