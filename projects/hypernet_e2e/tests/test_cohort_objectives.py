"""LHCF の比較に使う目的関数（ℓ_wst・FIS・FaMI）と SWAD の谷の判定のテスト。"""

import pytest
import torch
import torch.nn.functional as F

from projects.hypernet_e2e.callbacks.swad import SWADCallback, _Segment
from projects.hypernet_e2e.objectives.cohort import FaMITaskLoss, FISTaskLoss, WorstCohortTaskLoss, polynomial_log_density_gradient, sinkhorn_divergence
from projects.hypernet_e2e.objectives.input import ObjectiveInput


def _inputs(logits: torch.Tensor, target: list[int], groups: list[int]) -> ObjectiveInput:
    return ObjectiveInput(logits=logits, target=torch.tensor(target), attributes={"group_id": torch.tensor(groups)})


def test_worst_cohort_adds_the_largest_group_mean() -> None:
    logits = torch.tensor([[2.0, 0.0], [0.0, 2.0], [2.0, 0.0], [0.0, 0.0]])
    inputs = _inputs(logits, [0, 1, 1, 1], [0, 0, 1, 1])
    losses = F.cross_entropy(logits, inputs.target, reduction="none")
    expected = losses.mean() + 0.5 * max(losses[:2].mean(), losses[2:].mean())
    assert torch.isclose(WorstCohortTaskLoss(3, fairness_weight=0.5)(inputs), expected)


def test_worst_cohort_with_zero_weight_is_plain_cross_entropy() -> None:
    logits = torch.randn(6, 2)
    inputs = _inputs(logits, [0, 1, 0, 1, 0, 1], [0, 1, 2, 0, 1, 2])
    assert torch.isclose(WorstCohortTaskLoss(3, fairness_weight=0.0)(inputs), F.cross_entropy(logits, inputs.target))


def test_sinkhorn_divergence_is_zero_for_identical_sets_and_grows_with_shift() -> None:
    x = torch.linspace(0, 1, 20)
    assert sinkhorn_divergence(x, x).abs() < 1e-4
    near, far = sinkhorn_divergence(x, x + 0.2), sinkhorn_divergence(x, x + 1.0)
    # 1 次元の平行移動では W2²/2 = shift²/2 に近い
    assert 0 < near < far
    assert far == pytest.approx(0.5, rel=0.1)


def _with_group_samples(objective: FISTaskLoss, logits: torch.Tensor, target: list[int], groups: list[int]) -> FISTaskLoss:
    samples = _inputs(logits, target, groups)
    objective.set_group_sample_source(lambda: samples)
    return objective


def test_fis_matches_equations_three_to_five() -> None:
    """s^G は B^t と B^t_a の損失分布の OT を、群について取った softmax（論文 Eq. 5）。"""
    torch.manual_seed(0)
    logits, sample_logits = torch.randn(8, 2), torch.randn(12, 2)
    target, groups = [0, 1] * 4, [0] * 6 + [1] * 2
    sample_target, sample_groups = [0, 1] * 6, [0] * 6 + [1] * 6
    inputs = _inputs(logits, target, groups)
    loss = _with_group_samples(FISTaskLoss(2, fusion=0.5), sample_logits, sample_target, sample_groups)(inputs)
    losses = F.cross_entropy(logits, inputs.target, reduction="none")
    sample_losses = F.cross_entropy(sample_logits, torch.tensor(sample_target), reduction="none")
    sg = torch.tensor(sample_groups)
    distance = torch.stack([sinkhorn_divergence(losses, sample_losses[sg == k], 0.1) for k in range(2)])
    weight = 0.5 * torch.softmax(losses, 0) + 0.5 * torch.softmax(distance, 0)[torch.tensor(groups)]
    assert torch.isclose(loss, (weight * losses).mean(), atol=1e-6)


def test_fis_group_weights_sum_to_one_over_groups() -> None:
    torch.manual_seed(0)
    inputs = _inputs(torch.randn(16, 2), [0, 1] * 8, [k % 4 for k in range(16)])
    objective = _with_group_samples(FISTaskLoss(4, fusion=1.0), torch.randn(16, 2), [0, 1] * 8, [k // 4 for k in range(16)])
    objective(inputs)
    assert torch.isclose(objective._batch_diagnostics["group_weight"].sum(), torch.tensor(1.0))


def test_fis_requires_group_samples() -> None:
    with pytest.raises(ValueError, match="群ごとに別に引いたバッチ"):
        FISTaskLoss(2)(_inputs(torch.randn(4, 2), [0, 1, 0, 1], [0, 0, 1, 1]))


def test_fis_gradient_flows_through_individual_weights_only() -> None:
    """s^I = softmax(ℓ) には勾配が流れる（公式実装は detach しない）。s^G は定数。"""
    logits = torch.randn(6, 2, requires_grad=True)
    inputs = _inputs(logits, [0, 1, 0, 1, 0, 1], [0, 0, 0, 1, 1, 1])
    _with_group_samples(FISTaskLoss(2, fusion=0.0), torch.randn(6, 2), [0, 1] * 3, [0, 0, 0, 1, 1, 1])(inputs).backward()
    reference, logits.grad = logits.grad.clone(), None
    losses = F.cross_entropy(logits, inputs.target, reduction="none")
    (torch.softmax(losses, 0) * losses).mean().backward()
    assert torch.allclose(reference, logits.grad, atol=1e-6)


def test_group_sample_batches_draw_the_same_count_from_every_group() -> None:
    from projects.hypernet_e2e.data.group_datamodule import GroupSampleBatchSampler

    groups = torch.tensor([0] * 5 + [1] * 2 + [2] * 9)
    batches = iter(GroupSampleBatchSampler(groups, 3, per_group=4, seed=0))
    seen = []
    for _ in range(3):
        batch = torch.tensor(next(batches))
        assert torch.equal(torch.bincount(groups[batch], minlength=3), torch.tensor([4, 4, 4]))
        seen.append(batch)
    # 群 0（5 件）は 1 周目の 5 件を、並べ直す前にすべて使う
    first_cycle = torch.cat([b[:4] for b in seen])[:5]
    assert sorted(first_cycle.tolist()) == [0, 1, 2, 3, 4]


def test_polynomial_score_recovers_a_gaussian_score() -> None:
    """N(μ, σ²I) では ∇ ln p(x) = −(x − μ)/σ²（SD 2）。3 次の基底は 1 次を含むので、標本誤差の範囲で推定できる。"""
    torch.manual_seed(0)
    x = 2.0 + 0.5 * torch.randn(4000, 2)
    error = polynomial_log_density_gradient(x).float() + (x - 2.0) / 0.25
    assert error.abs().mean() < 0.1


def test_fami_score_difference_is_zero_when_groups_are_identically_distributed() -> None:
    v = torch.randn(40, 2)
    v = torch.cat([v, v])
    sfd = FaMITaskLoss(2).score_difference(v, torch.tensor([0] * 40 + [1] * 40))
    assert torch.allclose(sfd, torch.zeros_like(sfd), atol=1e-5)


def test_fami_penalty_pulls_separated_groups_together() -> None:
    torch.manual_seed(0)
    low, high = torch.randn(30, 2) * 0.5 - 1.0, torch.randn(30, 2) * 0.5 + 1.0
    v = torch.cat([low, high]).requires_grad_()
    groups = torch.tensor([0] * 30 + [1] * 30)
    loss = FaMITaskLoss(2, mi_weight=1.0)
    (loss.score_difference(v, groups) * v).sum(1).mean().backward()
    # 降下方向（−勾配）が、低い群を上へ、高い群を下へ動かす
    assert (-v.grad[:30].mean(0) > 0).all() and (-v.grad[30:].mean(0) < 0).all()
    assert torch.isfinite(loss(_inputs(v.detach(), [0] * 60, groups.tolist())))


def test_fami_skips_groups_smaller_than_the_basis() -> None:
    v = torch.randn(20, 2)
    sfd = FaMITaskLoss(2).score_difference(v, torch.tensor([0] * 14 + [1] * 6))
    assert torch.equal(sfd[14:], torch.zeros(6, 2))


def _segments(losses: list[float]) -> list[_Segment]:
    return [_Segment(epoch, {"w": torch.tensor([float(epoch)])}, 10, loss) for epoch, loss in enumerate(losses)]


def test_swad_averages_from_the_valley_start_until_it_dies() -> None:
    swad = SWADCallback(n_converge=3, n_tolerance=6, tolerance_ratio=0.3)
    # epoch 2 で最小（以後 2 区間はそれより高い）→ epoch 2 から平均。epoch 10 以降はしきい値を超え続ける
    losses = [1.0, 0.8, 0.5, 0.55, 0.6, 0.52, 0.58, 0.6, 0.62, 0.64] + [2.0] * 8
    for segment in _segments(losses):
        swad.update(segment)
    swad.finalize()
    assert swad.converge_epoch == 2
    assert swad.threshold == pytest.approx((0.5 + 0.55 + 0.6) / 3 * 1.3)
    assert swad.dead
    assert swad.averaged_epochs == list(range(2, 10))
    assert swad.total["w"] / swad.total_steps == pytest.approx(sum(range(2, 10)) / 8)


def test_swad_adds_the_remaining_valley_at_the_end_of_training() -> None:
    swad = SWADCallback(n_converge=3, n_tolerance=6, tolerance_ratio=0.3)
    for segment in _segments([1.0, 0.5, 0.6, 0.6, 0.55, 0.6]):
        swad.update(segment)
    swad.finalize()
    assert swad.converge_epoch == 1
    assert swad.averaged_epochs == [1, 2, 3, 4, 5]


def test_lightning_passes_group_sample_batches_to_fis(tmp_path) -> None:
    """学習 step ごとに DataModule の群別バッチを勾配なしで forward し、FIS の s^G に使う。"""
    import lightning as L
    from torch.utils.data import DataLoader, TensorDataset

    from projects.hypernet_e2e.module import LitModule

    class Data(TensorDataset):
        def __getitem__(self, index):
            x, group, target = super().__getitem__(index)
            return x, {"group_id": group}, int(target)

    groups, targets = torch.tensor([0, 0, 1, 1] * 2), torch.tensor([0, 1, 0, 1] * 2)
    train = Data(torch.randn(8, 2), groups, targets)
    sample_batch = next(iter(DataLoader(Data(torch.randn(6, 2), torch.tensor([0, 0, 0, 1, 1, 1]), torch.tensor([0, 1, 1, 0, 0, 1])), batch_size=6)))
    requested = []

    class DM(L.LightningDataModule):
        def train_dataloader(self):
            return DataLoader(train, batch_size=8)

        def group_sample_batches(self, per_group):
            requested.append(per_group)
            while True:
                yield sample_batch

    net = torch.nn.Linear(2, 2)
    objective = FISTaskLoss(2, group_sample_size=3)
    module = LitModule(net, objective, optimizer=lambda params: torch.optim.SGD(params, lr=0.0), scheduler=None, backbone_checkpoint_path=None)
    trainer = L.Trainer(
        accelerator="cpu", max_epochs=1, logger=False, enable_checkpointing=False, enable_model_summary=False, enable_progress_bar=False, limit_val_batches=0, default_root_dir=tmp_path
    )
    trainer.fit(module, datamodule=DM())

    x, attributes, target = next(iter(DataLoader(train, batch_size=8)))
    with torch.no_grad():
        expected = FISTaskLoss(2)
        sx, sattributes, starget = sample_batch
        expected.set_group_sample_source(lambda: ObjectiveInput(net(sx), starget, sattributes))
        expected(ObjectiveInput(net(x), target, attributes))
    assert requested == [3]
    # 学習が終われば、module を参照する関数は目的関数から外れる。
    assert objective._group_sample_source is None
    assert torch.allclose(objective._batch_diagnostics["group_weight"], expected._batch_diagnostics["group_weight"], atol=1e-6)


def test_worst_cohort_diagnostics_keep_the_group_dimension() -> None:
    logits = torch.tensor([[2.0, 0.0], [0.0, 2.0], [2.0, 0.0], [0.0, 0.0]])
    inputs = _inputs(logits, [0, 1, 1, 1], [0, 0, 1, 1])
    objective = WorstCohortTaskLoss(3)
    objective(inputs)
    diagnostics = objective.batch_diagnostics()
    losses = F.cross_entropy(logits, inputs.target, reduction="none")
    assert torch.allclose(diagnostics["group_loss"], torch.stack([losses[:2].mean(), losses[2:].mean(), torch.tensor(0.0)]))
    assert diagnostics["group_present"].tolist() == [1.0, 1.0, 0.0]
    assert diagnostics["worst_group"].tolist() == [0.0, 1.0, 0.0]


def test_fis_reports_the_share_of_the_group_term() -> None:
    torch.manual_seed(0)
    logits, sample_logits = torch.randn(8, 2), torch.randn(12, 2)
    groups = [0] * 6 + [1] * 2
    inputs = _inputs(logits, [0, 1] * 4, groups)
    objective = _with_group_samples(FISTaskLoss(2, fusion=0.5), sample_logits, [0, 1] * 6, [0] * 6 + [1] * 6)
    objective(inputs)
    diagnostics = objective.batch_diagnostics()
    group_term = 0.5 * diagnostics["group_weight"][torch.tensor(groups)]
    individual = 0.5 * torch.softmax(F.cross_entropy(logits, inputs.target, reduction="none"), 0)
    assert torch.isclose(diagnostics["group_term_share"], group_term.sum() / (group_term + individual).sum())
    # 論文の正規化では、群の項は 1/|A|、個別の項は 1/B の大きさなので、c = 0.5 でも群の項が大半を占める。
    assert diagnostics["group_term_share"] > 0.7


def test_fami_diagnostics_mark_groups_without_an_estimate() -> None:
    torch.manual_seed(0)
    logits = torch.randn(20, 2)
    groups = [0] * 12 + [1] * 8
    objective = FaMITaskLoss(3)
    objective(_inputs(logits, [0, 1] * 10, groups))
    diagnostics = objective.batch_diagnostics()
    # 基底は 1 + 3·2 = 7 個なので、12 件の群 0 と 8 件の群 1 は推定でき、居ない群 2 は推定しない。
    assert diagnostics["lambda_active"].tolist() == [1.0, 1.0, 0.0]
    assert (diagnostics["lambda_norm"][:2] > 0).all() and diagnostics["lambda_norm"][2] == 0
    assert diagnostics["penalty"].ndim == 0


def test_default_recorder_writes_cohort_diagnostics_every_epoch(tmp_path) -> None:
    """既定の callback に入った ObjectiveDiagnosticsRecorder が、LHCF 系の診断値を epoch ごとに書き出す。"""
    import json

    import lightning as L
    from torch.utils.data import DataLoader, TensorDataset

    from projects.hypernet_e2e.callbacks.objective_diagnostics import ObjectiveDiagnosticsRecorder
    from projects.hypernet_e2e.module import LitModule

    class Data(TensorDataset):
        def __getitem__(self, index):
            x, group, target = super().__getitem__(index)
            return x, {"group_id": group}, int(target)

    train = Data(torch.randn(8, 2), torch.tensor([0, 0, 1, 1] * 2), torch.tensor([0, 1, 0, 1] * 2))
    module = LitModule(torch.nn.Linear(2, 2), WorstCohortTaskLoss(3), optimizer=lambda params: torch.optim.SGD(params, lr=0.1), scheduler=None)
    trainer = L.Trainer(
        accelerator="cpu",
        max_epochs=2,
        logger=False,
        enable_checkpointing=False,
        enable_model_summary=False,
        enable_progress_bar=False,
        limit_val_batches=0,
        callbacks=[ObjectiveDiagnosticsRecorder()],
        default_root_dir=tmp_path,
    )
    trainer.fit(module, train_dataloaders=DataLoader(train, batch_size=4))
    rows = [json.loads(line) for line in (tmp_path / "metrics" / "objective_diagnostics.jsonl").read_text().splitlines()]
    assert [row["epoch"] for row in rows] == [0, 1]
    assert all(row["batches"] == 2 and len(row["epoch_mean"]["group_loss"]) == 3 for row in rows)
    assert rows[0]["epoch_mean"]["group_present"] == [1.0, 1.0, 0.0]
