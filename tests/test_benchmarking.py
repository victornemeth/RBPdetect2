import numpy as np
import pandas as pd

from rbpdetect2.benchmarking import run_linear_probe


def test_run_linear_probe_returns_metrics_and_test_predictions() -> None:
    labels = ["TF", "TSP", "nonRBP"]
    centers = {
        "TF": np.array([4.0, 0.0, 0.0]),
        "TSP": np.array([0.0, 4.0, 0.0]),
        "nonRBP": np.array([0.0, 0.0, 4.0]),
    }
    rows = []
    embeddings = []
    for split_name, offset in [("train", 0), ("val", 10), ("test", 20)]:
        for label_index, label in enumerate(labels):
            for replicate in range(3):
                sequence_id = f"{split_name}_{label}_{replicate}"
                rows.append(
                    {
                        "id": sequence_id,
                        "label": label,
                        "cluster": f"cluster_{offset + label_index * 3 + replicate}",
                        "split": split_name,
                    }
                )
                embeddings.append(centers[label] + replicate * 0.01)

    split = pd.DataFrame(rows)
    ids = split[["id", "label"]].copy()

    metrics, predictions = run_linear_probe(
        embeddings=np.asarray(embeddings, dtype=np.float32),
        ids=ids,
        split=split,
        c_values=[0.1, 1.0],
        bootstrap_replicates=20,
    )

    assert metrics["macro_f1"] == 1.0
    assert len(predictions) == 9
    assert set(predictions["prediction"]) == set(labels)

