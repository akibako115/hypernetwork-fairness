"""metadata-KMeans cohort artifact の契約を検証する。"""

from pathlib import Path

import numpy as np
import pandas as pd
import torch

from projects.hypernet_iterative.cohorts.build import SplitEmbeddings, save_artifact


def test_cohort_artifact_is_fit_on_train_and_records_its_reference(tmp_path: Path) -> None:
    checkpoint = tmp_path / "reference.ckpt"
    torch.save({"state_dict": {}}, checkpoint)
    embeddings = {
        split: SplitEmbeddings(
            images=np.asarray([f"{split}_0.png", f"{split}_1.png"], dtype=str),
            targets=np.asarray([0, 1], dtype="int64"),
            values=np.asarray([[0.0, 0.0], [10.0, 10.0]], dtype="float32"),
        )
        for split in ("train", "val", "test")
    }

    assignment_path = save_artifact(
        tmp_path / "cohort",
        embeddings,
        clusters=2,
        n_init=1,
        random_state=0,
        reference_checkpoint=checkpoint,
        reference_id="stage01",
    )

    assignments = pd.read_parquet(assignment_path)
    assert assignments.columns.tolist() == ["split", "image", "group_id"]
    assert set(assignments.loc[assignments["split"] == "train", "group_id"]) == {0, 1}
    metadata = __import__("json").loads((tmp_path / "cohort" / "cohort.json").read_text())
    assert metadata["reference_id"] == "stage01"
    assert metadata["num_groups"] == 2
