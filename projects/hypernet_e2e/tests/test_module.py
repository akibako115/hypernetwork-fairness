from __future__ import annotations

import logging
from types import SimpleNamespace

import lightning as L
import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from projects.hypernet_e2e.loss import ObjectiveInput, TaskLoss
from projects.hypernet_e2e.models.attribute_adversary import AttributeAdversary, gradient_reverse
from projects.hypernet_e2e.module import LitModule


class _Net(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.backbone = nn.Sequential(nn.Linear(1, 2), nn.BatchNorm1d(2))
        self.fc = nn.Linear(1, 2)
        self.last_attributes = None

    def forward(self, image: torch.Tensor, attributes=None) -> torch.Tensor:
        self.last_attributes = attributes
        return self.fc(image.mean(dim=(1, 2, 3)).unsqueeze(-1))


def _module(**kwargs) -> LitModule:
    kwargs.setdefault("net", _Net())
    kwargs.setdefault("loss_fn", TaskLoss())
    kwargs.setdefault("optimizer", None)
    kwargs.setdefault("scheduler", None)
    return LitModule(**kwargs)


def _batch() -> tuple[torch.Tensor, dict[str, torch.Tensor], torch.Tensor]:
    return torch.randn(2, 3, 4, 4), {"sex": torch.tensor([0, 1])}, torch.tensor([0, 1])


def test_training_step_exposes_the_callback_contract_and_keeps_loss_gradients() -> None:
    module = _module()
    output = module.training_step(_batch(), 0)

    assert set(output) == {"loss", "logits", "preds", "target", "attributes"}
    assert output["loss"].requires_grad
    assert not output["logits"].requires_grad
    assert output["attributes"]["sex"].tolist() == [0, 1]


def test_train_uses_injected_objective_but_evaluation_uses_plain_cross_entropy() -> None:
    class _Objective(nn.Module):
        def forward(self, inputs: ObjectiveInput) -> torch.Tensor:
            return inputs.logits.sum() * 0 + 7.0

    module = _module(loss_fn=_Objective())
    batch = _batch()
    train = module.training_step(batch, 0)
    validation = module.validation_step(batch, 0)

    assert train["loss"].item() == 7.0
    assert torch.allclose(validation["loss"], F.cross_entropy(validation["logits"], validation["target"]))


def test_forward_requires_attributes_only_when_enabled() -> None:
    module = _module(use_attributes=True)
    with pytest.raises(ValueError, match="attributes が必要"):
        module(torch.randn(2, 3, 4, 4))

    plain = _module(use_attributes=False)
    plain(torch.randn(2, 3, 4, 4), {"unused": torch.tensor([0, 1])})
    assert plain.net.last_attributes is None


def test_configure_optimizers_uses_only_trainable_net_parameters() -> None:
    module = _module(optimizer=lambda params: torch.optim.SGD(params, lr=0.1))
    module._trainer = SimpleNamespace(model=module)
    config = module.configure_optimizers()
    assert isinstance(config["optimizer"], torch.optim.SGD)


def test_freeze_backbone_keeps_it_in_eval_mode_at_epoch_boundary(tmp_path) -> None:
    path = tmp_path / "backbone.ckpt"
    torch.save({"state_dict": {f"net.{key}": value for key, value in _Net().state_dict().items()}}, path)
    module = _module(freeze_backbone=True, backbone_checkpoint_path=str(path))
    module.setup("fit")
    assert all(not parameter.requires_grad for parameter in module.net.backbone.parameters())
    assert not module.net.backbone.training

    module.net.backbone.train()
    module.on_train_epoch_start()
    assert not module.net.backbone.training


def test_backbone_checkpoint_strips_lightning_net_prefix(tmp_path) -> None:
    source = _Net()
    target = _Net()
    with torch.no_grad():
        source.backbone[0].weight.fill_(2.0)
    path = tmp_path / "backbone.ckpt"
    torch.save({"state_dict": {f"net.{key}": value for key, value in source.state_dict().items()}}, path)

    module = _module(net=target)
    module.load_backbone_checkpoint(str(path))
    assert torch.equal(target.backbone[0].weight, source.backbone[0].weight)


def test_trainer_runs_one_epoch_and_updates_the_classifier(tmp_path) -> None:
    module = _module(optimizer=lambda params: torch.optim.SGD(params, lr=0.1))
    before = module.net.fc.weight.detach().clone()
    loader = DataLoader([_batch()], batch_size=None)
    trainer = L.Trainer(
        accelerator="cpu",
        devices=1,
        max_epochs=1,
        logger=False,
        enable_checkpointing=False,
        enable_model_summary=False,
        enable_progress_bar=False,
        default_root_dir=tmp_path,
    )

    trainer.fit(module, train_dataloaders=loader)

    assert not torch.equal(module.net.fc.weight, before)


def test_setup_fails_when_the_backbone_checkpoint_matches_nothing(tmp_path) -> None:
    """key が 1 つも一致しない部分ロードは、成功した run として残してはならない。"""
    path = tmp_path / "unrelated.ckpt"
    torch.save({"state_dict": {"net.unrelated.weight": torch.zeros(2, 2)}}, path)

    module = _module(backbone_checkpoint_path=str(path))

    with pytest.raises(RuntimeError, match="一致する parameter が 1 つも無い"):
        module.setup("fit")


def test_setup_logs_how_many_backbone_tensors_were_loaded(tmp_path, caplog) -> None:
    source = _Net()
    path = tmp_path / "backbone.ckpt"
    torch.save({"state_dict": {f"net.{key}": value for key, value in source.state_dict().items()}}, path)

    module = _module(net=_Net(), backbone_checkpoint_path=str(path))
    with caplog.at_level(logging.INFO):
        module.setup("fit")

    assert "Loaded" in caplog.text


def test_optimizer_factory_stays_out_of_the_checkpoint_hparams() -> None:
    """lambda / Hydra partial を checkpoint に pickle させない。"""
    module = _module(optimizer=lambda params: torch.optim.SGD(params, lr=0.1))

    assert "optimizer" not in module.hparams
    assert "scheduler" not in module.hparams
    # 残る hparams は checkpoint に安全に書ける値だけであること。
    assert not [name for name, value in module.hparams.items() if callable(value)]


def test_freezing_the_backbone_requires_a_checkpoint_to_freeze() -> None:
    with pytest.raises(ValueError, match="backbone_checkpoint_path"):
        _module(freeze_backbone=True)


def test_gradient_reverse_negates_only_the_backbone_gradient() -> None:
    features = torch.tensor([[1.0, -2.0]], requires_grad=True)
    weights = torch.tensor([[3.0, 4.0]])
    (gradient_reverse(features, 0.5) * weights).sum().backward()

    assert torch.equal(features.grad, -0.5 * weights)


def test_attribute_adversary_ignores_missing_values_and_updates_its_parameters() -> None:
    adversary = AttributeAdversary(feature_dim=2, categorical_cardinalities=[2], num_continuous=1, hidden_dim=3)
    features = torch.randn(3, 2, requires_grad=True)
    attributes = {
        "categorical": torch.tensor([[0], [1], [0]]),
        "categorical_missing": torch.tensor([[False], [True], [False]]),
        "continuous": torch.tensor([[0.0], [0.3], [-0.2]]),
        "continuous_missing": torch.tensor([[False], [True], [False]]),
    }

    loss = adversary(features, attributes)
    loss.backward()

    assert loss.requires_grad
    assert features.grad is not None
    assert all(parameter.grad is not None for parameter in adversary.parameters())
