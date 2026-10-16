"""夜間キューの承認範囲、依存解決、二重起動防止を副作用なしで検証する。"""

import copy
import hashlib
import subprocess
from pathlib import Path

import pytest

from projects.hypernet_e2e import overnight as queue
from projects.hypernet_e2e.overnight_probe import _checkpoint


def _parent(remote, status="running"):
    return {
        "status": status, "seed": 42, "study": queue.STUDY,
        "experiment": "resnet-chexpert-race-known" if remote else "resnet-chexpert-age-race-group-dro-race-known",
        "preflight": 0, "wandb": [{"id": "test"}], "checkpoint_error": None,
        "checkpoint": "projects/hypernet_e2e/runs/parent/checkpoints/best_val_auroc_001.ckpt" if status == "succeeded" else None,
        "config": {"data": {"cv_splits_dir": "/data/chexpert/splits_race_known"}, "model": {"freeze_backbone": False}},
    }


def _snapshots(remote_status="running", local_status="running"):
    return {
        "remote": {"runs": {queue.M01: _parent(True, remote_status)}, "gpus": [], "containers": {"e2e_m01_s42_20261002": {"status": "running", "exit_code": 0}}},
        "local": {"runs": {queue.M02: _parent(False, local_status)}, "gpus": [], "containers": {}, "sessions": ["e2e_m02_s42_20261002T1203"]},
    }


def _gpu(index, **overrides):
    return {"index": index, "memory": 14, "utilization": 0, "compute": False, **overrides}


def test_authorized_scope_is_exactly_five_seed42_conditions():
    assert set(queue.TASKS) == {"M11", "M06", "M13", "M10", "M15"}
    assert all("race_known" in preset for _, preset, _ in queue.TASKS.values())
    assert all("seed=42" in queue._fit_args(preset, None) for _, preset, _ in queue.TASKS.values())


def test_free_gpu_rejects_compute_process_utilization_memory_and_reservation():
    snap = {"gpus": [_gpu(0), _gpu(2), _gpu(3, compute=True), _gpu(4, memory=600), _gpu(5, reserved=True), _gpu(6, utilization=10)]}
    assert queue._free_gpus(snap, range(2, 9)) == [2]


def test_running_parents_never_trigger_launch(tmp_path, monkeypatch):
    monkeypatch.setattr(queue, "_snapshots", _snapshots)
    monkeypatch.setattr(queue, "_execute", lambda *_args, **_kwargs: pytest.fail("launch attempted"))
    controller = queue.Controller(tmp_path)
    controller.tick(execute=True)
    assert all(entry["status"] == "pending" for entry in controller.state["tasks"].values())


def test_parent_failure_blocks_only_its_dependents(tmp_path, monkeypatch):
    monkeypatch.setattr(queue, "_snapshots", lambda: _snapshots("failed"))
    controller = queue.Controller(tmp_path)
    controller.tick(execute=True)
    assert controller.state["tasks"]["M11"]["status"] == "pending"
    assert all(controller.state["tasks"][task]["status"] == "blocked" for task in ["M06", "M13", "M10", "M15"])


@pytest.mark.parametrize("field,value", [("seed", 43), ("study", "scratch"), ("experiment", "legacy"), ("checkpoint_error", "bad hash")])
def test_parent_mismatch_is_rejected(field, value):
    run = _parent(True, "succeeded")
    run[field] = value
    with pytest.raises(ValueError):
        queue._validate_parent(run, remote=True)


def test_read_only_tick_does_not_launch_after_success(tmp_path, monkeypatch):
    monkeypatch.setattr(queue, "_snapshots", lambda: _snapshots("succeeded", "succeeded"))
    controller = queue.Controller(tmp_path)
    controller.tick()
    assert all(entry["status"] == "pending" for entry in controller.state["tasks"].values())


def test_missing_process_after_launch_intent_never_retries(tmp_path):
    controller = queue.Controller(tmp_path)
    controller.state["tasks"]["M06"] = {"status": "starting", "submitted_at": 0}
    assert controller._reconcile("M06", _snapshots()["remote"])
    assert controller.state["tasks"]["M06"]["status"] == "failed"
    restored = queue.Controller(tmp_path)
    restored.state = controller.state
    assert restored._reconcile("M06", _snapshots()["remote"])


def test_exited_container_without_run_is_not_relaunched(tmp_path):
    controller = queue.Controller(tmp_path)
    snap = _snapshots()["remote"]
    snap["containers"][queue._name("M06")] = {"status": "exited", "exit_code": 1}
    assert controller._reconcile("M06", snap)
    assert controller.state["tasks"]["M06"]["status"] == "failed"


def test_launch_error_leaves_persistent_no_retry_state(tmp_path, monkeypatch):
    snap = _snapshots("running", "succeeded")
    snap["local"]["gpus"] = [_gpu(0)]
    monkeypatch.setattr(queue, "_snapshots", lambda: copy.deepcopy(snap))
    calls = []

    def execute(args, **kwargs):
        calls.append(args)
        if args[0] == "tmux":
            raise subprocess.TimeoutExpired(args, 60)
        return "dry run succeeded"

    monkeypatch.setattr(queue, "_execute", execute)
    controller = queue.Controller(tmp_path)
    controller.tick(execute=True)
    assert controller.state["tasks"]["M11"]["status"] == "blocked"
    restored = queue.Controller(tmp_path)
    restored.tick(execute=True)
    assert sum(args[0] == "tmux" for args in calls) == 1


def test_checkpoint_uses_best_monitor_and_digest(tmp_path):
    run_id = queue.M01
    relative = Path("projects/hypernet_e2e/runs") / run_id / "checkpoints/best_val_auroc_001.ckpt"
    path = tmp_path / relative
    path.parent.mkdir(parents=True)
    path.write_bytes(b"weights")
    record = {"status": "running", "run_id": run_id, "checkpoints": []}
    assert _checkpoint(record, tmp_path) is None
    best = {"path": str(Path(queue.CONTAINER_ROOT) / relative), "sha256": hashlib.sha256(b"weights").hexdigest()}
    record.update(status="succeeded", checkpoints=[{"monitor": "val/auroc", "mode": "max", "best": best}])
    assert _checkpoint(record, tmp_path) == str(relative)
    path.write_bytes(b"corrupt")
    with pytest.raises(ValueError, match="SHA"):
        _checkpoint(record, tmp_path)


def test_commands_preserve_docker_mounts_gpu_and_checkpoint():
    dry, launch = queue._launch_commands("M06", 7, "projects/hypernet_e2e/runs/parent/checkpoints/best.ckpt")
    assert "--rm" in dry
    assert "device=7" in launch and ":ro" in launch and "--ipc=host" in launch
    assert "backbone_checkpoint_path=projects/" in launch
    assert "logger=none" not in launch


def test_all_completed_runs_stop_controller(tmp_path, monkeypatch):
    snap = _snapshots("succeeded", "succeeded")
    for rid in queue.MONITORED_REMOTE:
        snap["remote"]["runs"][rid] = _parent(True, "succeeded")
    monkeypatch.setattr(queue, "_snapshots", lambda: snap)
    controller = queue.Controller(tmp_path)
    for entry in controller.state["tasks"].values():
        entry["status"] = "blocked"
    assert controller.tick()


def test_interrupted_parent_blocks_dependents_even_if_run_json_is_still_running(tmp_path, monkeypatch):
    snap = _snapshots()
    snap["remote"]["containers"]["e2e_m01_s42_20261002"]["status"] = "exited"
    monkeypatch.setattr(queue, "_snapshots", lambda: snap)
    controller = queue.Controller(tmp_path)
    controller.tick(execute=True)
    assert controller.state["tasks"]["M06"]["status"] == "blocked"
    assert controller.state["tasks"]["M11"]["status"] == "pending"


def test_successful_remote_launches_use_distinct_gpus_and_fixed_parent(tmp_path, monkeypatch):
    snap = _snapshots("succeeded")
    snap["remote"]["gpus"] = [_gpu(index) for index in range(2, 6)]
    monkeypatch.setattr(queue, "_snapshots", lambda: copy.deepcopy(snap))
    launches = []
    controller = queue.Controller(tmp_path)

    def ssh(command, **kwargs):
        if "dry_run=true" in command:
            return "dry-run OK"
        task = next(task for task in queue.TASKS if queue._name(task) in command)
        entry = controller.state["tasks"][task]
        assert queue.Controller(tmp_path).state["tasks"][task]["status"] == "starting"
        launches.append(entry["gpu"])
        for gpu in snap["remote"]["gpus"]:
            if gpu["index"] == entry["gpu"]:
                gpu["reserved"] = True
        snap["remote"]["containers"][queue._name(task)] = {"status": "running", "exit_code": 0}
        return "container-id"

    monkeypatch.setattr(queue, "_ssh", ssh)
    controller.tick(execute=True)
    assert launches == [2, 3, 4, 5]
    controller.tick(execute=True)
    assert launches == [2, 3, 4, 5]
    assert all(controller.state["tasks"][task]["checkpoint"] == snap["remote"]["runs"][queue.M01]["checkpoint"] for task in ["M06", "M13", "M10", "M15"])


def test_success_without_free_gpu_does_not_launch(tmp_path, monkeypatch):
    monkeypatch.setattr(queue, "_snapshots", lambda: _snapshots("succeeded"))
    monkeypatch.setattr(queue, "_ssh", lambda *_args, **_kwargs: "dry-run OK")
    controller = queue.Controller(tmp_path)
    controller.tick(execute=True)
    assert all(entry["status"] == "pending" for entry in controller.state["tasks"].values())
