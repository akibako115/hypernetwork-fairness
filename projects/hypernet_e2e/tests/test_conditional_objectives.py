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
from projects.hypernet_e2e.objectives.equalized_odds import EqualizedOddsRateTaskLoss, proxy_bounds
from projects.hypernet_e2e.objectives.fair_dro import FairDROTaskLoss, chi_square_best_response, nonnegative_radius_limit
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


def test_eo_overall_constraints_compare_each_group_with_the_population_rate():
    """各群 vs 全体の片側制約。全体の率は train の群構成で重み付けし、セル均等のバッチの単純平均にしない。"""
    obj = EqualizedOddsRateTaskLoss(2, COUNTS, constraint_form="overall")
    obj(inputs())
    # 群 0 は TPR 1・FPR 0、群 1 は TPR 0・FPR 1。全体の TPR は 1/6、FPR は 0.25（COUNTS の構成）。
    hard = torch.tensor([1 / 6 - 1, 1 / 6 - 0, 0 - 0.25, 1 - 0.25]) - 0.1
    assert torch.allclose(obj.batch_diagnostics()["hard_g"], hard)
    assert (obj.batch_diagnostics()["proxy_g"] >= hard - 1e-6).all()
    assert torch.allclose(obj.batch_diagnostics()["lambda"], torch.full((5,), 0.2))
    assert "overall_coefficients" not in EqualizedOddsRateTaskLoss(2, COUNTS).state_dict()
    with pytest.raises(ValueError, match="constraint_form"):
        EqualizedOddsRateTaskLoss(2, COUNTS, constraint_form="two_sided")


def test_eo_overall_fpr_only_keeps_the_communities_form():
    """Cotter et al. の Communities と同じく FPR だけにかける。λ は task ＋ G 本。"""
    obj = EqualizedOddsRateTaskLoss(2, COUNTS, constraint_form="overall", rates=["fpr"])
    obj(inputs())
    assert torch.allclose(obj.batch_diagnostics()["hard_g"], torch.tensor([0 - 0.25, 1 - 0.25]) - 0.1)
    assert torch.allclose(obj.batch_diagnostics()["lambda"], torch.full((3,), 1 / 3))
    pairwise = EqualizedOddsRateTaskLoss(2, COUNTS, rates=["fpr"])
    pairwise(inputs())
    assert torch.allclose(pairwise.batch_diagnostics()["hard_g"], torch.tensor([-1.1, 0.9]))
    for rates in ([], ["fpr", "fpr"], ["tnr"]):
        with pytest.raises(ValueError, match="rates"):
            EqualizedOddsRateTaskLoss(2, COUNTS, rates=rates)


def test_eo_overall_proxy_bounds_the_hard_constraints_on_random_batches():
    generator = torch.Generator().manual_seed(0)
    counts = [[50, 5], [20, 9], [70, 30]]
    obj = EqualizedOddsRateTaskLoss(3, counts, constraint_form="overall", epsilon_pos=0.05, epsilon_neg=0.05)
    group, target = torch.arange(3).repeat_interleave(8), torch.tensor([0, 1]).repeat_interleave(4).repeat(3)
    for _ in range(50):
        score = torch.randn(24, generator=generator) * 3
        batch = ObjectiveInput(torch.stack([torch.zeros_like(score), score], dim=1), target, {"group_id": group})
        obj(batch)
        diagnostics = obj.batch_diagnostics()
        assert len(diagnostics["hard_g"]) == 6
        assert (diagnostics["proxy_g"] >= diagnostics["hard_g"] - 1e-6).all()


def test_proxy_bounds_including_strict_zero_tie():
    score = torch.cat([torch.linspace(-100, 100, 10001), torch.tensor([0.0])])
    upper, lower = proxy_bounds(score, 1.0)
    hard = (score > 0).float()
    assert (upper >= hard).all() and (lower <= hard).all()


def _rate_gap_after_minimizing_proxy() -> tuple[float, float]:
    """代理制約だけを score について下げ、陽性セルの TPR 差の前後を返す。"""
    generator = torch.Generator().manual_seed(0)
    score_g = (torch.randn(32, generator=generator) + 0.8).requires_grad_()
    score_h = (torch.randn(32, generator=generator) - 0.5).requires_grad_()
    before = float((score_g > 0).float().mean() - (score_h > 0).float().mean())
    optimizer = torch.optim.SGD([score_g, score_h], lr=0.5)
    for _ in range(500):
        upper, _ = proxy_bounds(score_g, 1.0)
        _, lower = proxy_bounds(score_h, 1.0)
        optimizer.zero_grad()
        (upper.mean() - lower.mean()).backward()
        optimizer.step()
    after = float((score_g > 0).float().mean() - (score_h > 0).float().mean())
    return before, after


def test_hinge_proxy_can_flip_predictions_to_close_the_rate_gap():
    before, after = _rate_gap_after_minimizing_proxy()
    assert before > 0.3 and after < 0.05


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
    methods = ["erm", "group_dro", "sampling_corrected_erm", "sampling_corrected_uniform_group", "population_auc", "minimax_auc", "equalized_odds_rate"]
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


@pytest.mark.parametrize("reference", ["parent", "deferred"])
@pytest.mark.parametrize("kind", [MinimaxAUCTaskLoss, EqualizedOddsRateTaskLoss, SamplingCorrectedTaskLoss, FairDROTaskLoss])
def test_lightning_commits_once_per_step_and_records_candidates(tmp_path, kind, reference):
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
    recorder = ConditionalFairnessRecorder(methods.get(kind, "minimax_auc"), reference=reference)
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
    selection = json.loads((tmp_path / "metrics/fairness_selection.json").read_text())
    if reference == "deferred":
        # 1 stage では sanity が初期値なので選ばず、候補と初期値の記録だけを残す。
        assert selection["status"] == "deferred" and selection["selected"] is None and selection["last_epoch"]["epoch"] == 2
        assert (tmp_path / "metrics/fairness_initial.json").is_file() and not (tmp_path / "metrics/fairness_parent.json").exists()
    else:
        assert selection["status"] in {"selected", "infeasible"} and (tmp_path / "metrics/fairness_parent.json").is_file()
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
    # 公式実装の単位では Q_rho = {q : sum q = 1, (1/2|A|) sum (|A| q - 1)^2 <= rho}（φ(t) = (t-1)²/2 の χ² 球）。
    # 閉形式解はこの球の境界上にあり、境界上の他の候補より大きい。
    assert torch.allclose(best.sum(), torch.tensor(1.0, dtype=torch.float64))
    assert torch.allclose((size * best - 1).square().sum() / (2 * size), torch.tensor(radius, dtype=torch.float64))
    generator = torch.Generator().manual_seed(0)
    for _ in range(2000):
        direction = torch.randn(size, generator=generator, dtype=torch.float64)
        direction -= direction.mean()
        candidate = 1 / size + math.sqrt(2 * radius / size) * direction / direction.norm()
        assert torch.dot(candidate, losses) <= torch.dot(best, losses) + 1e-12
    bound = math.sqrt(2 * radius * (size - 1)) / size
    assert ((best >= 1 / size - bound - 1e-12) & (best <= 1 / size + bound + 1e-12)).all()


def test_fair_dro_two_groups_bang_bang_negative_weight_and_tie():
    assert torch.allclose(chi_square_best_response(torch.tensor([0.3, 0.1], dtype=torch.float64), 2.0), torch.tensor([1.5, -0.5], dtype=torch.float64))
    assert torch.equal(chi_square_best_response(torch.tensor([0.2, 0.2], dtype=torch.float64), 2.0), torch.tensor([0.5, 0.5], dtype=torch.float64))


def test_fair_dro_smoothed_ibr_schedule_and_classwise_q():
    obj = FairDROTaskLoss(2, COUNTS, radius=0.125, max_epochs=4)
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


def test_eo_initial_task_weight_sets_the_initial_lambda():
    obj = EqualizedOddsRateTaskLoss(2, COUNTS, initial_task_weight=0.9)
    distribution, _ = obj._stationary()
    assert torch.allclose(distribution, torch.tensor([0.9, 0.025, 0.025, 0.025, 0.025], dtype=torch.float64))
    assert torch.allclose(obj.log_M.exp().sum(dim=0), torch.ones(5, dtype=torch.float64))
    uniform, _ = EqualizedOddsRateTaskLoss(2, COUNTS)._stationary()
    assert torch.allclose(uniform, torch.full((5,), 0.2, dtype=torch.float64))
    with pytest.raises(ValueError):
        EqualizedOddsRateTaskLoss(2, COUNTS, initial_task_weight=1.0)


def test_lightning_updates_duals_from_held_out_batches(tmp_path):
    """held-out を持つ DataModule では、双対の更新に学習バッチではなく held-out の順位損失を使う。

    モデルの学習率を 0 にしてスコアを固定し、q の推移が held-out のリスクだけから決まることを確かめる。
    """
    import lightning as L
    from torch.utils.data import DataLoader, TensorDataset

    from projects.hypernet_e2e.module import LitModule

    class Data(TensorDataset):
        def __getitem__(self, index):
            x, group, target = super().__getitem__(index)
            return x, {"group_id": group}, int(target)

    groups, targets = torch.tensor([0, 0, 1, 1] * 2), torch.tensor([0, 1, 0, 1] * 2)
    train = Data(torch.tensor([[0.0, 1.0], [1.0, 1.0]] * 4), groups, targets)
    # held-out では群 1 の陽性のスコアを陰性より低くし、学習バッチと違うペアが最悪になるようにする。
    held = Data(torch.tensor([[0.0, 1.0], [1.0, 1.0], [1.0, 1.0], [0.0, 1.0]] * 2), groups, targets)
    held_batch = next(iter(DataLoader(held, batch_size=8)))

    class DM(L.LightningDataModule):
        def __init__(self):
            super().__init__()
            self.data_holdout = held

        def train_dataloader(self):
            return DataLoader(train, batch_size=8)

        def held_out_batches(self):
            while True:
                yield held_batch

    net = torch.nn.Linear(2, 2)
    objective = MinimaxAUCTaskLoss(2, COUNTS)
    start = objective.log_q.clone()
    module = LitModule(net, objective, optimizer=lambda params: torch.optim.SGD(params, lr=0.0), scheduler=None, backbone_checkpoint_path=None)
    trainer = L.Trainer(
        accelerator="cpu", max_epochs=3, logger=False, enable_checkpointing=False, enable_model_summary=False, enable_progress_bar=False, limit_val_batches=0, default_root_dir=tmp_path
    )
    trainer.fit(module, datamodule=DM())

    with torch.no_grad():
        x, attributes, target = held_batch
        expected_risk = MinimaxAUCTaskLoss(2, COUNTS).train()
        expected_risk(ObjectiveInput(net(x), target, attributes))
        risk = expected_risk.batch_diagnostics()["risk"]
    expected = start
    for _ in range(3):
        expected = torch.log_softmax(expected + objective.dual_step_size * risk, dim=0)
    assert objective.update_count == 3
    assert torch.allclose(objective.log_q, expected, atol=1e-6)


@pytest.mark.parametrize("num_groups", [2, 4, 8])
def test_fair_dro_radius_limit_is_where_the_closed_form_first_hits_zero(num_groups):
    limit = nonnegative_radius_limit(num_groups)
    # 1 群だけが他と逆向きに外れる偏差が、q* の最小成分を最も下げる。
    losses = torch.zeros(num_groups, dtype=torch.float64)
    losses[0] = 1.0
    assert torch.isclose(chi_square_best_response(-losses, limit).min(), torch.tensor(0.0, dtype=torch.float64), atol=1e-12)
    assert chi_square_best_response(-losses, 0.9 * limit).min() > 0
    counts = [[10, 10]] * num_groups
    with pytest.warns(UserWarning, match="radius"):
        FairDROTaskLoss(num_groups, counts, radius=limit)


def test_fair_dro_update_reports_minimum_weight():
    obj = FairDROTaskLoss(4, [[10, 10]] * 4, radius=1.0, max_epochs=2)
    update = obj.update_from_errors(torch.tensor([[0.4, 0.9], [0.4, 0.9], [0.4, 0.9], [0.1, 0.2]], dtype=torch.float64))
    assert update["q_min"] < 0 and torch.isclose(update["q_min"], obj.q.min())


def test_lightning_fair_dro_measures_errors_on_held_out_when_available(tmp_path):
    """held-out を持つ DataModule では、FairDRO の q を train ではなく held-out 全件の誤り率で更新する。"""
    import json

    import lightning as L
    from torch.utils.data import DataLoader, TensorDataset

    from projects.hypernet_e2e.module import LitModule

    class Data(TensorDataset):
        def __getitem__(self, index):
            x, group, target = super().__getitem__(index)
            return x, {"group_id": group}, int(target)

    groups, targets = torch.tensor([0, 0, 1, 1] * 2), torch.tensor([0, 1, 0, 1] * 2)
    # 恒等の線形層で score = x1 - x0。train は全件正解、held-out は群 1 の両クラスを全件誤る。
    train = Data(torch.tensor([[1.0, 0.0], [0.0, 1.0]] * 4), groups, targets)
    held = Data(torch.tensor([[1.0, 0.0], [0.0, 1.0], [0.0, 1.0], [1.0, 0.0]] * 2), groups, targets)

    class DM(L.LightningDataModule):
        def __init__(self):
            super().__init__()
            self.data_holdout = held

        def train_dataloader(self):
            return DataLoader(train, batch_size=8)

        def evaluation_dataloader(self, split, persistent_workers=None):
            return DataLoader(train, batch_size=8)

        def held_out_evaluation_dataloader(self):
            return DataLoader(held, batch_size=8)

        def held_out_batches(self):
            raise AssertionError("epoch 単位の objective に step ごとの held-out は流さない")

    net = torch.nn.Linear(2, 2, bias=False)
    with torch.no_grad():
        net.weight.copy_(torch.eye(2))
    objective = FairDROTaskLoss(2, COUNTS, radius=0.1, max_epochs=2)
    module = LitModule(net, objective, optimizer=lambda params: torch.optim.SGD(params, lr=0.0), scheduler=None, backbone_checkpoint_path=None)
    trainer = L.Trainer(
        accelerator="cpu", max_epochs=1, logger=False, enable_checkpointing=False, enable_model_summary=False, enable_progress_bar=False, limit_val_batches=0, default_root_dir=tmp_path
    )
    trainer.fit(module, datamodule=DM())

    (row,) = [json.loads(line) for line in (tmp_path / "metrics/fair_dro_q.jsonl").read_text().splitlines()]
    assert row["source"] == "holdout"
    assert row["error"] == [[0.0, 0.0], [1.0, 1.0]]
    assert row["q_min"] == pytest.approx(min(min(q) for q in row["q"]))


def test_objective_diagnostics_average_train_batches_and_keep_held_out_separate(tmp_path):
    """診断値は最後の 1 batch ではなく epoch 平均で残し、held-out の統計は学習バッチと混ぜない。"""
    import json

    import lightning as L
    from torch.utils.data import DataLoader, TensorDataset

    from projects.hypernet_e2e.callbacks.objective_diagnostics import ObjectiveDiagnosticsRecorder
    from projects.hypernet_e2e.module import LitModule

    class Data(TensorDataset):
        def __getitem__(self, index):
            x, group, target = super().__getitem__(index)
            return x, {"group_id": group}, int(target)

    groups, targets = torch.tensor([0, 0, 1, 1] * 2), torch.tensor([0, 1, 0, 1] * 2)
    # 2 つの train batch でスコアの並びを変え、batch ごとの risk を違う値にする。
    train = Data(torch.tensor([[0.0, 1.0], [1.0, 1.0]] * 2 + [[0.0, 2.0], [2.0, 0.0]] * 2), groups, targets)
    held = Data(torch.tensor([[0.0, 1.0], [1.0, 1.0], [1.0, 1.0], [0.0, 1.0]] * 2), groups, targets)
    held_batch = next(iter(DataLoader(held, batch_size=8)))

    class DM(L.LightningDataModule):
        def __init__(self):
            super().__init__()
            self.data_holdout = held

        def train_dataloader(self):
            return DataLoader(train, batch_size=4)

        def held_out_batches(self):
            while True:
                yield held_batch

    net = torch.nn.Linear(2, 2)
    objective = MinimaxAUCTaskLoss(2, COUNTS)
    module = LitModule(net, objective, optimizer=lambda params: torch.optim.SGD(params, lr=0.0), scheduler=None, backbone_checkpoint_path=None)
    trainer = L.Trainer(
        accelerator="cpu",
        max_epochs=1,
        logger=False,
        enable_checkpointing=False,
        enable_model_summary=False,
        enable_progress_bar=False,
        limit_val_batches=0,
        default_root_dir=tmp_path,
        callbacks=[ObjectiveDiagnosticsRecorder()],
    )
    trainer.fit(module, datamodule=DM())

    with torch.no_grad():

        def risk(batch):
            reference = MinimaxAUCTaskLoss(2, COUNTS).train()
            x, attributes, target = batch
            reference(ObjectiveInput(net(x), target, attributes))
            return reference.batch_diagnostics()["risk"].double()

        train_batches = list(DataLoader(train, batch_size=4))
        expected_train = (risk(train_batches[0]) + risk(train_batches[1])) / 2
        expected_held = risk(held_batch)
    (row,) = [json.loads(line) for line in (tmp_path / "metrics/objective_diagnostics.jsonl").read_text().splitlines()]
    assert row["batches"] == 2 and row["held_out_batches"] == 2
    assert torch.allclose(torch.tensor(row["epoch_mean"]["risk"], dtype=torch.float64), expected_train, atol=1e-6)
    assert torch.allclose(torch.tensor(row["held_out_epoch_mean"]["risk"], dtype=torch.float64), expected_held, atol=1e-6)
    assert torch.allclose(torch.tensor(row["risk"], dtype=torch.float64), risk(train_batches[1]), atol=1e-6)
