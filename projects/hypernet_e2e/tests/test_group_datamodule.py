"""demographic 属性の組から group ID を作る DataModule を検証する。"""

from pathlib import Path

import pandas as pd
import pytest
import torch
from PIL import Image
from torchvision.transforms import transforms

from projects.hypernet_e2e.data.group_datamodule import GroupImageDataModule, demographic_group_ids

_AGE_GROUPS = {"age_group_65": {"source": "age", "boundaries": [65]}}


def _write_fixture(tmp_path: Path, rows_by_split: dict[str, list[tuple[str, int, int, float]]]) -> tuple[Path, Path]:
    image_dir = tmp_path / "images"
    split_dir = tmp_path / "splits"
    image_dir.mkdir(exist_ok=True)
    split_dir.mkdir(exist_ok=True)
    for split, rows in rows_by_split.items():
        frame_rows = []
        for image_name, target, sex, age in rows:
            Image.new("RGB", (8, 8), color=128).save(image_dir / image_name)
            frame_rows.append({"image": image_name, "target": target, "sex": sex, "sex_missing": False, "age": age, "age_missing": False})
        pd.DataFrame(frame_rows).to_csv(split_dir / f"{split}.csv", index=False)
    return image_dir, split_dir


def _all_four_groups(tmp_path: Path) -> tuple[Path, Path]:
    return _write_fixture(
        tmp_path,
        {
            "train": [("t0.png", 0, 0, 40.0), ("t1.png", 1, 0, 70.0), ("t2.png", 0, 1, 40.0), ("t3.png", 1, 1, 70.0)],
            "val": [("v0.png", 1, 0, 30.0), ("v1.png", 0, 1, 80.0)],
        },
    )


def _datamodule(image_dir: Path, split_dir: Path, **kwargs: object) -> GroupImageDataModule:
    defaults: dict[str, object] = {
        "group_attribute_names": ["sex", "age_group_65"],
        "group_cardinalities": [2, 2],
        "num_groups": 4,
    }
    defaults.update(kwargs)
    return GroupImageDataModule(
        data_dir=str(image_dir),
        cv_splits_dir=str(split_dir),
        num_classes=2,
        batch_size=2,
        num_workers=0,
        pin_memory=False,
        persistent_workers=False,
        attribute_names={"categorical": ["sex"], "continuous": ["age"]},
        fairness_age_groups=_AGE_GROUPS,
        train_transform=transforms.Compose([transforms.ToTensor()]),
        val_transform=transforms.Compose([transforms.ToTensor()]),
        **defaults,
    )


def test_group_ids_combine_the_attributes_in_mixed_radix_order() -> None:
    frame = pd.DataFrame(
        {
            "image": ["a", "b", "c", "d"],
            "sex": [0, 0, 1, 1],
            "sex_missing": False,
            "age_group_65": [0, 1, 0, 1],
            "age_group_65_missing": False,
        }
    )

    ids = demographic_group_ids(frame, ["sex", "age_group_65"], [2, 2], group_key="group_id")

    assert ids.tolist() == [0, 1, 2, 3]


def test_group_ids_refuse_missing_attributes_instead_of_folding_them_into_group_zero() -> None:
    frame = pd.DataFrame({"image": ["a", "b"], "sex": [0, 1], "sex_missing": [False, True]})

    with pytest.raises(ValueError, match="欠損"):
        demographic_group_ids(frame, ["sex"], [2], group_key="group_id")


def test_group_ids_refuse_values_outside_the_declared_cardinality() -> None:
    frame = pd.DataFrame({"image": ["a", "b"], "sex": [0, 2], "sex_missing": False})

    with pytest.raises(ValueError, match=r"\[0, 2\)"):
        demographic_group_ids(frame, ["sex"], [2], group_key="group_id")


def test_train_batch_carries_the_group_id_the_objective_reads(tmp_path: Path) -> None:
    dm = _datamodule(*_all_four_groups(tmp_path))
    dm.setup("fit")

    assert sorted(dm.data_train.df["group_id"]) == [0, 1, 2, 3]
    _, attributes, _ = next(iter(dm.val_dataloader()))
    assert attributes["group_id"].dtype == torch.long
    assert sorted(attributes["group_id"].tolist()) == [0, 3]


def test_num_groups_must_match_the_product_of_the_cardinalities(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="積 4"):
        _datamodule(*_all_four_groups(tmp_path), num_groups=3)


def test_setup_refuses_a_train_split_that_leaves_a_group_empty(tmp_path: Path) -> None:
    fixture = _write_fixture(
        tmp_path,
        {
            "train": [("t0.png", 0, 0, 40.0), ("t1.png", 1, 0, 70.0), ("t2.png", 0, 1, 40.0)],
            "val": [("v0.png", 1, 0, 30.0)],
        },
    )
    dm = _datamodule(*fixture)

    with pytest.raises(ValueError, match=r"\[3\] が空"):
        dm.setup("fit")


def test_group_dro_reaches_its_group_ids_through_one_lightning_fit(tmp_path: Path) -> None:
    """DataModule が付けた group ID が training_step の目的関数まで届くことを確かめる。"""
    import lightning as L
    import torch.nn as nn

    from projects.hypernet_e2e.loss import GroupDROTaskLoss
    from projects.hypernet_e2e.module import LitModule

    class _Net(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.fc = nn.Linear(3, 2)

        def forward(self, image: torch.Tensor) -> torch.Tensor:
            return self.fc(image.mean(dim=(2, 3)))

    dm = _datamodule(*_all_four_groups(tmp_path))
    objective = GroupDROTaskLoss(num_groups=4, step_size=1.0)
    module = LitModule(net=_Net(), loss_fn=objective, optimizer=lambda params: torch.optim.SGD(params, lr=0.1), scheduler=None)

    L.Trainer(max_epochs=1, accelerator="cpu", logger=False, enable_checkpointing=False, enable_progress_bar=False).fit(module, datamodule=dm)

    # 4 群それぞれが train batch に現れるので、adversarial weight は一様初期値から動く。
    assert objective.adv_probs.sum().item() == pytest.approx(1.0)
    assert not torch.allclose(objective.adv_probs, torch.full((4,), 0.25))
