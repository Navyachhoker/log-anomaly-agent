"""Toy-data tests for the duplicate analysis (they say nothing about the real results)."""
from array import array

import duplicate_analysis as da


def test_identical_sequences_are_found_and_scored_separately():
    blocks = {"ref": array("I", [1, 2, 3]), "dup_normal": array("I", [1, 2, 3]), "dup_anom": array("I", [9, 9]),
              "new_normal": array("I", [4, 5]), "new_anom": array("I", [7, 8]), "ref_anom": array("I", [9, 9])}
    labels = {"ref": 0, "dup_normal": 0, "dup_anom": 1, "new_normal": 0, "new_anom": 1, "ref_anom": 1}
    twins = da.with_identical_sequence(blocks, {"ref", "ref_anom"}, ["dup_normal", "dup_anom", "new_normal", "new_anom"])
    assert twins == {"dup_normal", "dup_anom"}

    def rec(label, generated):
        return {"label": label, "generated": generated}
    preds = {"m": {"dup_normal": rec("Normal", "Normal"), "dup_anom": rec("Anomaly", "Anomaly"),
                   "new_normal": rec("Normal", "Anomaly"), "new_anom": rec("Anomaly", "Normal")}}
    out = da.analyze(blocks, labels, preds, {"ref": {"ref", "ref_anom"}})
    rows = out["references"]["ref"]["models"]["m"]
    assert rows["all"]["n"] == 4 and rows["with_identical_sequence"]["n"] == 2 and rows["without_identical_sequence"]["n"] == 2
    assert rows["with_identical_sequence"]["accuracy"] == 1.0
    assert rows["without_identical_sequence"]["accuracy"] == 0.0 and rows["without_identical_sequence"]["confusion"]["fn"] == 1


def test_trained_block_ids_follow_the_training_cap(tmp_path):
    lines = ['{"block_id": "a1", "label": "Anomaly"}', '{"block_id": "n1", "label": "Normal"}',
             '{"block_id": "a2", "label": "Anomaly"}', '{"block_id": "n2", "label": "Normal"}', '{"block_id": "a3", "label": "Anomaly"}']
    path = tmp_path / "train.jsonl"
    path.write_text("\n".join(lines))
    assert da.trained_block_ids(path, 2) == {"a1", "a2", "n1", "n2"}