"""race 欠損除外と元属性を保持する派生 split の検証。"""

import pandas as pd
import pytest

from data_pipeline.chexpert_race_known_splits import create_splits, race_known_frame


def test_mapping_preserves_original_columns_and_order():
    source = pd.DataFrame({"race": [5, 0, 1, 2, 3, 4, 0], "race_missing": [0] * 6 + [1], "target": [1, 0, 0, 1, 0, 1, 1]})
    result = race_known_frame(source)
    pd.testing.assert_frame_equal(result[source.columns], source.iloc[:6])
    assert result.race_group.tolist() == [3, 0, 3, 1, 2, 3]
    assert not result.race_group_missing.any()


@pytest.mark.parametrize("race,missing", [(float("nan"), 0), (6, 0), (1.5, 0), (0, 2), (0, None)])
def test_invalid_known_values_or_flags_are_rejected(race, missing):
    with pytest.raises(ValueError):
        race_known_frame(pd.DataFrame({"race": [race], "race_missing": [missing]}))


def test_manifest_and_no_overwrite(tmp_path):
    source_dir = tmp_path / "input"
    source_dir.mkdir()
    for split in ("train", "val", "test"):
        pd.DataFrame({"image": ["a", "b"], "target": [0, 1], "race": [0, 0], "race_missing": [0, 1]}).to_csv(source_dir / f"{split}.csv", index=False)
    output_dir = tmp_path / "output"
    manifest = create_splits(source_dir, output_dir)
    assert manifest["splits"]["train"]["removed_rows"] == 1
    assert len(manifest["splits"]["train"]["input_sha256"]) == 64
    assert len(pd.read_csv(output_dir / "test.csv")) == 1
    with pytest.raises(FileExistsError):
        create_splits(source_dir, output_dir)
