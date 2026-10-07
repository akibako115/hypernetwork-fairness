"""原著の更新式・条件付き統計・sampling の意味を固定する。"""

import math
from pathlib import Path

import pytest
import torch
from hydra import compose, initialize_config_dir
from hydra.utils import instantiate
from torch.nn import functional as F

from projects.hypernet_e2e.callbacks.conditional_fairness import select_candidate
from projects.hypernet_e2e.data.group_class_sampler import GroupClassBatchSampler
from projects.hypernet_e2e.objectives.equalized_odds import EqualizedOddsRateTaskLoss, smooth_step
from projects.hypernet_e2e.objectives.fair_dro import FairDROTaskLoss, chi_square_best_response
from projects.hypernet_e2e.objectives.input import ObjectiveInput
from projects.hypernet_e2e.objectives.minimax_auc import MinimaxAUCTaskLoss, PopulationAUCTaskLoss
from projects.hypernet_e2e.objectives.sampling_corrected import SamplingCorrectedGroupTaskLoss, SamplingCorrectedTaskLoss

COUNTS = [[30, 10], [10, 50]]


def inputs(scores=(-2.0, 1.0, 3.0, -1.0)):
    s = torch.tensor(scores, dtype=torch.float32, requires_grad=True)
    return ObjectiveInput(torch.stack([torch.zeros_like(s), s], dim=1), torch.tensor([0, 1, 0, 1]), {"group_id": torch.tensor([0, 0, 1, 1])})


def test_minimax_four_risks_priors_old_gradient_and_model_first_update():
    obj = MinimaxAUCTaskLoss(2, COUNTS, dual_step_size=0.1)
    batch = inputs()
    q = torch.tensor([1 / 6 * 3 / 4, 1 / 6 * 1 / 4, 5 / 6 * 3 / 4, 5 / 6 * 1 / 4])
    risks = F.softplus(torch.tensor([-3.0, 2.0, -1.0, 4.0]))
    loss = obj(batch)
    assert torch.allclose(obj.log_q.exp(), q)
    assert torch.allclose(obj.batch_diagnostics()["risk"], risks)
    assert torch.allclose(loss, q @ risks)
    expected_gradient = torch.autograd.grad(
        q @ torch.stack([F.softplus(-(batch.logits[g, 1] - batch.logits[g, 0] - batch.logits[h, 1] + batch.logits[h, 0])) for g, h in [(1, 0), (1, 2), (3, 0), (3, 2)]]),
        batch.logits,
        retain_graph=True,
    )[0]
    actual_gradient = torch.autograd.grad(loss, batch.logits, retain_graph=True)[0]
    assert torch.allclose(actual_gradient, expected_gradient)
    loss.backward()
    assert obj.update_count == 0
    obj.commit_update()
    assert torch.allclose(obj.log_q.exp(), torch.softmax(q.log() + 0.1 * risks, 0))
    assert expected_gradient.isfinite().all()
    obj.commit_update()
    assert obj.update_count == 1


def test_eo_four_directed_constraints_and_column_swap_update():
    obj = EqualizedOddsRateTaskLoss(2, COUNTS, dual_step_size=0.1)
    batch = inputs()
    loss = obj(batch)
    hard = torch.tensor([0.9, -1.1, -1.1, 0.9])
    assert torch.allclose(obj.batch_diagnostics()["hard_g"], hard)
    assert (obj.batch_diagnostics()["proxy_g"] >= hard - 1e-6).all()
    assert torch.allclose(obj.batch_diagnostics()["lambda"], torch.full((5,), 0.2))
    loss.backward()
    obj.commit_update()
    expected = torch.softmax(torch.outer(torch.cat([torch.zeros(1), hard]), torch.full((5,), 0.2)).double() * 0.1, dim=0)
    assert torch.allclose(obj.log_M.exp(), expected)
    assert torch.allclose(obj.log_M.exp().sum(dim=0), torch.ones(5, dtype=torch.float64))
    distribution, residual = obj._stationary()
    assert residual <= 1e-8
    assert torch.allclose(obj.log_M.exp() @ distribution, distribution, atol=1e-8)


def test_proxy_bounds_including_strict_zero_tie():
    score = torch.cat([torch.linspace(-100, 100, 10001), torch.tensor([0.0])]).requires_grad_()
    upper = smooth_step(1 + score)
    lower = smooth_step(score)
    hard = (score > 0).float()
    assert (upper >= hard).all() and (lower <= hard).all()
    (upper + lower).sum().backward()
    assert score.grad[-1] == 0


@pytest.mark.parametrize("kind", [MinimaxAUCTaskLoss, EqualizedOddsRateTaskLoss, PopulationAUCTaskLoss, FairDROTaskLoss])
def test_eval_pure_and_state_restore(kind):
    obj = kind(2, COUNTS)
    obj(inputs())
    obj.commit_update()
    state = {key: value.clone() for key, value in obj.state_dict().items()}
    obj.eval()
    obj(inputs())
    obj.commit_update()
    assert all(torch.equal(value, obj.state_dict()[key]) for key, value in state.items())
    restored = kind(2, COUNTS)
    restored.load_state_dict(state)
    assert torch.allclose(obj(inputs()), restored.eval()(inputs()))


def test_population_corrected_ce_fixed_normalizer():
    obj = SamplingCorrectedTaskLoss(2, COUNTS, class_weight=[1.0, 2.0])
    batch = inputs()
    loss = obj(batch)
    ce = F.cross_entropy(batch.logits, batch.target, reduction="none")
    expected = (ce * torch.tensor([0.3, 0.1, 0.1, 0.5]) * torch.tensor([1.0, 2.0, 1.0, 2.0])).sum() / 1.6
    assert torch.allclose(loss, expected)
    uniform = SamplingCorrectedGroupTaskLoss(2, COUNTS, class_weight=[1.0, 2.0])
    expected_group = ((ce[0] * 0.75 + ce[1] * 0.25 * 2) + (ce[2] / 6 + ce[3] * 5 / 6 * 2)) / 2
    assert torch.allclose(uniform(batch), expected_group)


def test_cell_duplication_keeps_conditional_risks_and_rates():
    batch = inputs()
    index = torch.tensor([0, 0, 1, 2, 2, 2, 3])
    duplicate = ObjectiveInput(batch.logits[index], batch.target[index], {"group_id": batch.attributes["group_id"][index]})
    for kind, field in [(MinimaxAUCTaskLoss, "risk"), (EqualizedOddsRateTaskLoss, "hard_g")]:
        obj = kind(2, COUNTS)
        obj(batch)
        a = obj.batch_diagnostics()[field].clone()
        obj(duplicate)
        assert torch.allclose(a, obj.batch_diagnostics()[field])


def test_sampler_cells_seed_epoch_and_full_final_batch():
    groups = torch.tensor([0] * 40 + [1] * 60)
    labels = torch.tensor([0] * 30 + [1] * 10 + [0] * 10 + [1] * 50)
    sampler = GroupClassBatchSampler(groups, labels, 2, 128, 42)
    first = list(sampler)
    assert len(sampler) == math.ceil(100 / 128)
    assert first == list(sampler)
    assert torch.equal(torch.bincount(groups[first[0]] * 2 + labels[first[0]], minlength=4), torch.full((4,), 32))
    sampler.set_epoch(1)
    assert first != list(sampler)
    with pytest.raises(ValueError, match="空"):
        GroupClassBatchSampler([0, 1], [0, 1], 2, 128)


def test_selection_refuses_nan_and_infeasible_eo():
    parent = dict(epoch=0, auroc=0.8, worst_age_auroc=0.7, min_pair_auroc=0.6, weighted_ce=0.4, tpr=[0.8, 0.8], tnr=[0.8, 0.8])
    parent["tpr"] = [0.8, 0.6]
    candidate = dict(parent, epoch=1, tpr=[0.7, 0.9])
    assert select_candidate([candidate], parent, "equalized_odds_rate")["status"] == "infeasible"
    assert select_candidate([dict(parent, epoch=2, min_pair_auroc=float("nan"))], parent, "minimax_auc")["selected"]["epoch"] == 0


def test_all_age_presets_compose_and_objectives_instantiate():
    config_dir = Path(__file__).parents[1] / "configs"
    methods = ["erm", "group_dro", "sampling_corrected_erm", "sampling_corrected_group_dro", "sampling_corrected_uniform_group", "population_auc", "minimax_auc", "equalized_odds_rate"]
    methods += ["fair_dro"]
    for method in methods:
        with initialize_config_dir(version_base="1.3", config_dir=str(config_dir)):
            config = compose(config_name="train", overrides=[f"experiment=resnet_chexpert_from_resnet_age_{method}_race_known", "study=minimax_auc_vs_eo"])
        assert config.data.num_groups == 2
        if "population_cell_counts" in config.model.loss_fn:
            config.model.loss_fn.population_cell_counts = COUNTS
        if "class_weight" in config.model.loss_fn and method not in ("minimax_auc", "population_auc", "fair_dro"):
            config.model.loss_fn.class_weight = [1.0, 2.0]
        instantiate(config.model.loss_fn)
        instantiate(config.data)


@pytest.mark.parametrize("kind", [MinimaxAUCTaskLoss, EqualizedOddsRateTaskLoss, SamplingCorrectedTaskLoss, FairDROTaskLoss])
def test_lightning_commits_once_per_step_and_records_candidates(tmp_path, kind):
    import json

    import lightning as L
    import pandas as pd
    from lightning.pytorch.callbacks import ModelCheckpoint
    from torch.utils.data import DataLoader, Dataset

    from projects.hypernet_e2e.callbacks.conditional_fairness import ConditionalFairnessRecorder
    from projects.hypernet_e2e.module import LitModule

    class Data(Dataset):
        def __init__(self):
            self.df = pd.DataFrame({"target": [0, 1, 0, 1] * 4, "group_id": [0, 0, 1, 1] * 4})

        def __len__(self):
            return len(self.df)

        def __getitem__(self, index):
            row = self.df.iloc[index]
            return torch.tensor([float(index % 4), 1.0]), {"group_id": torch.tensor(int(row.group_id))}, int(row.target)

    class DM(L.LightningDataModule):
        def __init__(self):
            super().__init__()
            self.data_train = Data()
            self.hparams.num_groups = 2

        def train_dataloader(self):
            return DataLoader(self.data_train, batch_size=8)

        def val_dataloader(self):
            return DataLoader(self.data_train, batch_size=4)

        def evaluation_dataloader(self, split, persistent_workers=None):
            assert split == "train" and persistent_workers is False
            return DataLoader(self.data_train, batch_size=4)

    parent = tmp_path / "parent.ckpt"
    net = torch.nn.Linear(2, 2)
    torch.save({"state_dict": {f"net.{key}": value for key, value in net.state_dict().items()}}, parent)
    objective = kind(2, COUNTS)
    module = LitModule(net, objective, optimizer=lambda params: torch.optim.AdamW(params, lr=0.001), scheduler=None, backbone_checkpoint_path=str(parent))
    methods = {EqualizedOddsRateTaskLoss: "equalized_odds_rate", FairDROTaskLoss: "fair_dro"}
    recorder = ConditionalFairnessRecorder(methods.get(kind, "minimax_auc"))
    trainer = L.Trainer(
        accelerator="cpu",
        max_epochs=2,
        logger=False,
        enable_checkpointing=True,
        enable_model_summary=False,
        enable_progress_bar=False,
        callbacks=[recorder, ModelCheckpoint(dirpath=tmp_path / "checkpoints", filename="full_{epoch}", save_top_k=-1, save_last=True)],
        num_sanity_val_steps=-1,
        default_root_dir=tmp_path,
    )
    trainer.fit(module, datamodule=DM())
    assert objective.update_count == {SamplingCorrectedTaskLoss: 0, FairDROTaskLoss: 2}.get(kind, 4)
    if kind == FairDROTaskLoss:
        rows = [json.loads(line) for line in (tmp_path / "metrics/fair_dro_q.jsonl").read_text().splitlines()]
        assert [row["update"] for row in rows] == [1, 2] and all(sum(sum(r) for r in row["count"]) == 16 for row in rows)
        assert "last_epoch" in json.loads((tmp_path / "metrics/fairness_selection.json").read_text())
        # 最後の q 更新は最終モデルによる train 全件の hard 誤り率を使う。
        data = Data()
        with torch.no_grad():
            logits = net(torch.stack([data[i][0] for i in range(len(data))]))
        target, group = torch.tensor(data.df.target.values), torch.tensor(data.df.group_id.values)
        miss = ((logits[:, 1] - logits[:, 0] > 0).long() != target).double()
        expected = torch.tensor([[miss[(group == g) & (target == c)].mean().item() for c in range(2)] for g in range(2)], dtype=torch.float64)
        assert torch.allclose(torch.tensor(rows[-1]["error"], dtype=torch.float64), expected)
        q_star = torch.stack([chi_square_best_response(expected[:, c], objective.radius) for c in range(2)])
        assert torch.allclose(torch.tensor(rows[-1]["q_star"], dtype=torch.float64), q_star)
    assert recorder.parent["epoch"] == 0 and len(recorder.candidates) == 2
    assert len(json.loads((tmp_path / "metrics/fairness_candidates.json").read_text())) == 2
    candidate = torch.load(tmp_path / "checkpoints/candidate_002.ckpt", weights_only=False)
    assert "optimizer_states" not in candidate and "callbacks" not in candidate
    full = torch.load(tmp_path / "checkpoints/last.ckpt", weights_only=False)
    assert full["optimizer_states"]
    state = candidate["state_dict"]
    # FairDRO の q は validation 後の epoch 末に更新されるので、候補はその epoch の学習に使った q を持つ。
    assert state["loss_fn.update_count"] == objective.update_count - int(kind == FairDROTaskLoss)
    before = objective.update_count.clone()
    trainer.validate(module, datamodule=DM(), verbose=False)
    assert torch.equal(before, objective.update_count)
    resumed = L.Trainer(accelerator="cpu", max_epochs=3, logger=False, enable_checkpointing=False, enable_progress_bar=False, enable_model_summary=False)
    with pytest.raises(ValueError, match="途中再開"):
        resumed.fit(module, datamodule=DM(), ckpt_path=tmp_path / "checkpoints/last.ckpt")


def test_stationary_distribution_of_asymmetric_markov_matrix():
    obj = EqualizedOddsRateTaskLoss(2, COUNTS)
    expected = torch.tensor([0.5, 0.2, 0.1, 0.15, 0.05], dtype=torch.float64)
    matrix = 0.7 * torch.eye(5, dtype=torch.float64) + 0.3 * expected[:, None].expand(5, 5)
    obj.log_M.copy_(matrix.log())
    distribution, residual = obj._stationary()
    assert torch.allclose(distribution, expected, atol=1e-8)
    assert residual <= 1e-8


def test_minimax_large_risk_update_stays_in_log_space():
    obj = MinimaxAUCTaskLoss(2, COUNTS, dual_step_size=100)
    obj(inputs([10000, -10000, 20000, -20000]))
    obj.commit_update()
    assert torch.isfinite(obj.log_q).all()
    assert torch.allclose(obj.log_q.exp().sum(), torch.tensor(1.0))


def test_eo_near_tie_selection_and_parent_candidate():
    parent = dict(epoch=0, auroc=0.8, worst_age_auroc=0.7, min_pair_auroc=0.6, weighted_ce=0.4, tpr=[0.8, 0.8], tnr=[0.8, 0.8])
    candidate = dict(parent, epoch=1, weighted_ce=0.4 - 1e-10)
    assert select_candidate([candidate], parent, "equalized_odds_rate")["selected"]["epoch"] == 0


def test_resume_is_rejected_before_parent_or_candidate_output(tmp_path):
    from types import SimpleNamespace

    from projects.hypernet_e2e.callbacks.conditional_fairness import ConditionalFairnessRecorder

    callback = ConditionalFairnessRecorder("erm")
    with pytest.raises(ValueError, match="途中再開"):
        callback.on_fit_start(SimpleNamespace(ckpt_path="last.ckpt", default_root_dir=tmp_path), None)
    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize("size", [2, 3])
@pytest.mark.parametrize("radius", [0.1, 1.0, 3.0])
def test_fair_dro_closed_form_maximizes_over_chi_square_ball(size, radius):
    losses = torch.tensor([0.1, 0.4, 0.25][:size], dtype=torch.float64)
    best = chi_square_best_response(losses, radius)
    # Q_rho = {q : sum q = 1, (1/2|A|) sum (|A| q - 1)^2 <= rho}（式 4）。境界上の候補と比べる。
    assert torch.allclose(best.sum(), torch.tensor(1.0, dtype=torch.float64))
    assert torch.allclose(((size * best - 1).square().sum() / (2 * size)), torch.tensor(radius / 2, dtype=torch.float64)) or size * ((best - 1 / size).square().sum()) <= radius + 1e-9
    generator = torch.Generator().manual_seed(0)
    for _ in range(2000):
        direction = torch.randn(size, generator=generator, dtype=torch.float64)
        direction -= direction.mean()
        candidate = 1 / size + math.sqrt(radius / size) * direction / direction.norm()
        assert torch.dot(candidate, losses) <= torch.dot(best, losses) + 1e-12
    bound = math.sqrt(radius * (size - 1)) / size
    assert ((best >= 1 / size - bound - 1e-12) & (best <= 1 / size + bound + 1e-12)).all()


def test_fair_dro_two_groups_bang_bang_negative_weight_and_tie():
    assert torch.allclose(chi_square_best_response(torch.tensor([0.3, 0.1], dtype=torch.float64), 4.0), torch.tensor([1.5, -0.5], dtype=torch.float64))
    assert torch.equal(chi_square_best_response(torch.tensor([0.2, 0.2], dtype=torch.float64), 4.0), torch.tensor([0.5, 0.5], dtype=torch.float64))


def test_fair_dro_smoothed_ibr_schedule_and_classwise_q():
    obj = FairDROTaskLoss(2, COUNTS, radius=0.25, max_epochs=4)
    errors = torch.tensor([[0.3, 0.1], [0.1, 0.2]])
    first = obj.update_from_errors(errors)
    assert first["eta"] == 1 and torch.allclose(obj.q, torch.tensor([[0.75, 0.25], [0.25, 0.75]], dtype=torch.float64))
    second = obj.update_from_errors(-errors)
    flipped = torch.tensor([[0.25, 0.75], [0.75, 0.25]], dtype=torch.float64)
    assert second["eta"] == 0.75 and torch.allclose(obj.q, 0.25 * flipped.flip(1) + 0.75 * flipped)
    obj.update_from_errors(errors)
    obj.update_from_errors(errors)
    with pytest.raises(RuntimeError, match="max_epochs"):
        obj.update_from_errors(errors)
    with pytest.raises(ValueError, match="class_weight"):
        FairDROTaskLoss(2, COUNTS, class_weight=[1.0, 2.0])


def test_fair_dro_loss_is_classwise_q_weighted_cell_ce():
    obj = FairDROTaskLoss(2, COUNTS, radius=1.0)
    obj.q.copy_(torch.tensor([[0.2, 0.8], [1.3, -0.3]], dtype=torch.float64))
    batch = inputs()
    loss = obj(batch)
    ce = F.cross_entropy(batch.logits, batch.target, reduction="none")
    # inputs() の並びは (g0,y0), (g0,y1), (g1,y0), (g1,y1)。q は [y][g]。
    expected = (0.2 * ce[0] + 0.8 * ce[2] + 1.3 * ce[1] - 0.3 * ce[3]) / 2
    assert torch.allclose(loss, expected)
    loss.backward()
    obj.commit_update()
    assert obj.update_count == 0 and torch.allclose(obj.batch_diagnostics()["rates"], torch.tensor([[0.0, 1.0], [1.0, 0.0]]))


def test_fair_dro_uses_eo_selection_and_reports_last_epoch():
    parent = dict(epoch=0, auroc=0.8, worst_age_auroc=0.7, min_pair_auroc=0.6, weighted_ce=0.4, tpr=[0.8, 0.6], tnr=[0.8, 0.8])
    candidates = [dict(parent, epoch=1, tpr=[0.79, 0.7], weighted_ce=0.5), dict(parent, epoch=2, tpr=[0.9, 0.6])]
    result = select_candidate(candidates, parent, "fair_dro")
    assert result["selected"]["epoch"] == 1 and result["last_epoch"]["epoch"] == 2
    assert "last_epoch" not in select_candidate(candidates, parent, "equalized_odds_rate")
