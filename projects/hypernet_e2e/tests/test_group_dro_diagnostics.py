import csv
import json
from types import SimpleNamespace

import lightning as L
import pytest
import torch
from lightning.pytorch.loggers import CSVLogger, WandbLogger
from torch.utils.data import DataLoader, TensorDataset

from projects.hypernet_e2e.callbacks.group_dro_diagnostics import GroupDRODiagnostics
from projects.hypernet_e2e.loss import GlobalClassWeightedGroupDROTaskLoss, ObjectiveInput, SubgroupClassWeightedGroupDROTaskLoss, TaskLoss


class _Module(L.LightningModule):
    def __init__(self, objective):
        super().__init__()
        self.net = torch.nn.Linear(2, 2)
        self.training_objective = objective
        self.snapshots = []

    def training_step(self, batch, batch_idx):
        x, target, groups = batch
        loss = self.training_objective(ObjectiveInput(self.net(x), target, {"group_id": groups}))
        self.snapshots.append({key: value.clone() for key, value in self.training_objective.batch_diagnostics().items()})
        return loss

    def configure_optimizers(self):
        return torch.optim.SGD(self.parameters(), lr=0.01)


@pytest.mark.parametrize("subgroup", [False, True])
def test_two_epochs_emit_actual_q_and_sample_weighted_loss(tmp_path, subgroup):
    objective = SubgroupClassWeightedGroupDROTaskLoss(3, [[1.0, 2.0], [2.0, 1.0], [1.0, 1.0]]) if subgroup else GlobalClassWeightedGroupDROTaskLoss(3, [1.0, 2.0])
    module = _Module(objective)
    data = TensorDataset(torch.tensor([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]]), torch.tensor([0, 1, 0]), torch.tensor([0, 1, 0]))
    logger = CSVLogger(str(tmp_path), name="", version="")
    remote_records = []
    wandb_logger = WandbLogger(project="test", save_dir=str(tmp_path))
    wandb_logger._experiment = SimpleNamespace(log=lambda values: remote_records.append(dict(values)))
    trainer = L.Trainer(
        accelerator="cpu",
        max_epochs=2,
        callbacks=[GroupDRODiagnostics(["a", "b", "c"])],
        logger=[logger, wandb_logger],
        default_root_dir=tmp_path,
        enable_checkpointing=False,
        enable_model_summary=False,
        enable_progress_bar=False,
        log_every_n_steps=1,
    )
    trainer.fit(module, train_dataloaders=DataLoader(data, batch_size=2))
    with open(tmp_path / "metrics.csv") as stream:
        rows = list(csv.DictReader(stream))
    rows = [row for row in rows if row.get("train/group_dro/q_end/g0")]
    assert len(rows) == 2
    remote_rows = [row for row in remote_records if "train/group_dro/q_end/g0" in row]
    assert len(remote_rows) == 2
    for epoch, row in enumerate(rows):
        assert int(row["epoch"]) == epoch
        assert remote_rows[epoch]["train/group_dro/q_end/g0"] == pytest.approx(float(row["train/group_dro/q_end/g0"]))
        snapshots = module.snapshots[epoch * 2 : epoch * 2 + 2]
        for group in range(3):
            assert float(row[f"train/group_dro/q_end/g{group}"]) == pytest.approx(snapshots[-1]["q"][group].item())
            assert float(row[f"train/group_dro/q_mean/g{group}"]) == pytest.approx(torch.stack([s["q"] for s in snapshots]).mean(0)[group].item())
            count = sum(s["count"][group].item() for s in snapshots)
            assert float(row[f"train/group_dro/count/g{group}"]) == count
            if count:
                expected = sum(s["loss_sum"][group].item() for s in snapshots) / count
                assert float(row[f"train/group_dro/loss/g{group}"]) == pytest.approx(expected)
            else:
                assert row[f"train/group_dro/loss/g{group}"].lower() == "nan"
    metadata = json.loads((tmp_path / "metrics/group_dro_groups.json").read_text())
    assert metadata["groups"] == [{"id": i, "label": label} for i, label in enumerate(["a", "b", "c"])]
    assert list(objective.state_dict()) == ["adv_probs"]
    assert torch.equal(objective.adv_probs, module.snapshots[-1]["q"])


def test_erm_noop_and_reject_ambiguous_labels_or_distributed(tmp_path):
    callback = GroupDRODiagnostics()
    trainer = SimpleNamespace(world_size=2, default_root_dir=tmp_path)
    callback.on_fit_start(trainer, SimpleNamespace(training_objective=TaskLoss()))
    assert not callback._active
    objective = GlobalClassWeightedGroupDROTaskLoss(2, [1.0, 2.0])
    module = SimpleNamespace(training_objective=objective)
    with pytest.raises(ValueError, match="単一device"):
        callback.on_fit_start(trainer, module)
    trainer.world_size = 1
    with pytest.raises(ValueError, match="group_labels"):
        GroupDRODiagnostics(["same", "same"]).on_fit_start(trainer, module)
