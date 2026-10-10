import json

import numpy as np
import pytest

from evaluate_linear import _split_manifest
from summarize_comparison import _aggregate_results


def test_downstream_split_manifest_is_reused_and_bound_to_label_order(tmp_path):
    labels = np.repeat(np.arange(3), 20)
    path = tmp_path / "deepmimo_splits.json"
    first = _split_manifest(path, labels, "classification", (0, 1, 2))
    second = _split_manifest(path, labels.copy(), "classification", (0, 1, 2))

    assert first == second == json.loads(path.read_text(encoding="utf-8"))
    for split in first["splits"].values():
        train = set(split["train"])
        test = set(split["test"])
        assert len(train) == 48
        assert len(test) == 12
        assert train.isdisjoint(test)
        assert train | test == set(range(len(labels)))

    changed_labels = labels.copy()
    changed_labels[0] = 2
    with pytest.raises(ValueError, match="does not match"):
        _split_manifest(path, changed_labels, "classification", (0, 1, 2))


def test_comparison_aggregates_pretraining_and_split_seed_scores():
    results = [
        {
            "task": "rfs",
            "architecture": "mamba2",
            "pretraining_seed": 0,
            "metric": "mean_per_class_accuracy_percent",
            "seeds": {"0": 40.0, "1": 60.0},
            "checkpoint": "mamba0.pt",
            "result_path": "mamba0.json",
        },
        {
            "task": "rfs",
            "architecture": "mamba2",
            "pretraining_seed": 1,
            "metric": "mean_per_class_accuracy_percent",
            "seeds": {"0": 80.0, "1": 100.0},
            "checkpoint": "mamba1.pt",
            "result_path": "mamba1.json",
        },
    ]

    (record,) = _aggregate_results(results)
    assert record["mean"] == pytest.approx(70.0)
    assert record["std"] == pytest.approx(np.std([40, 60, 80, 100], ddof=1))
    assert record["n_scores"] == 4
    assert record["pretraining_seeds"] == "[0, 1]"
    assert record["split_seeds"] == "[0, 1]"
    assert '"pretrain_1/split_1": 100.0' in record["scores"]
