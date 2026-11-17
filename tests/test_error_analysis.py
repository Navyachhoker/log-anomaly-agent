import error_analysis as ea


def test_finds_false_alarms_and_misses_and_counts_unparseable_as_errors():
    def rec(block, label, generated):
        return {"block_id": block, "label": label, "generated": generated, "p_anomaly": 0.5}
    preds = {"a": rec("a", "Normal", "Anomaly"), "b": rec("b", "Anomaly", "Normal"), "c": rec("c", "Normal", "Normal"),
             "d": rec("d", "Anomaly", "???")}
    snippets = {"a": {"snippet": "INFO x\nWARN y", "num_lines": 2, "truncated": False}}
    errors = ea.find_errors(preds, snippets)
    kinds = {r["block_id"]: k for k, r, _ in errors}
    assert set(kinds) == {"a", "b", "d"} and kinds["a"].startswith("FALSE ALARM") and kinds["b"] == "MISSED ANOMALY"
    assert "WARN y" in ea.describe(*errors[0])