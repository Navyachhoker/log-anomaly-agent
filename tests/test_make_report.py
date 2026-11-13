"""Metrics and report tests on tiny hand-checkable data (no real results are implied)."""
import json

import numpy as np
import pytest

import make_report as mr


def test_score_matches_hand_computation():
    y_true = np.array([1, 1, 1, 0, 0, 0, 0, 0], dtype=np.int8)
    y_pred = np.array([1, 1, 0, 1, 0, 0, 0, 0], dtype=np.int8)
    s = mr.score(y_true, y_pred)
    assert s["confusion"] == {"tp": 2, "fp": 1, "fn": 1, "tn": 4}
    assert s["accuracy"] == pytest.approx(6 / 8) and s["precision"] == pytest.approx(2 / 3)
    assert s["recall"] == pytest.approx(2 / 3) and s["f1"] == pytest.approx(2 / 3)


def test_score_handles_no_predicted_positives():
    s = mr.score(np.array([1, 0], dtype=np.int8), np.array([0, 0], dtype=np.int8))
    assert s["precision"] == 0.0 and s["recall"] == 0.0 and s["f1"] == 0.0


def test_bootstrap_interval_contains_the_point_estimate():
    rng = np.random.default_rng(0)
    y_true = rng.integers(0, 2, 400).astype(np.int8)
    y_pred = np.where(rng.random(400) < 0.9, y_true, 1 - y_true).astype(np.int8)
    point = mr.score(y_true, y_pred)["f1"]
    low, high = mr.bootstrap_ci(y_true, y_pred, 300)["f1"]
    assert low <= point <= high


def test_threshold_tuning_finds_a_separating_threshold():
    y = np.array([0] * 50 + [1] * 50, dtype=np.int8)
    p = np.array([0.1] * 50 + [0.3] * 50)           # anomalies score low, so 0.5 would miss them all
    t = mr.tune_threshold(y, p)
    assert 0.1 < t <= 0.3 and mr.score(y, (p >= t).astype(np.int8))["f1"] == 1.0


def test_unparseable_answers_count_as_errors():
    records = [{"generated": "Anomaly"}, {"generated": "I think so"}, {"generated": "Normal"}, {"generated": "???"}]
    y_true = np.array([1, 1, 0, 0], dtype=np.int8)
    y_pred, bad = mr.predictions_from_text(records, y_true)
    assert bad == 2 and list(y_pred) == [1, 0, 0, 1]


def _write(path, rows, meta):
    path.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    (path.parent / (path.name + ".meta.json")).write_text(json.dumps(meta))


def test_full_report_from_synthetic_predictions(tmp_path):
    pred, res = tmp_path / "pred", tmp_path / "res"
    pred.mkdir()
    rng = np.random.default_rng(1)
    ids = [f"blk_{i}" for i in range(200)]
    truth = {b: "Anomaly" if i % 10 == 0 else "Normal" for i, b in enumerate(ids)}

    def rows(accuracy, with_p=True, tokens=False):
        out = []
        for i, b in enumerate(ids):
            right = rng.random() < accuracy
            label = truth[b] if right else ("Normal" if truth[b] == "Anomaly" else "Anomaly")
            row = {"block_id": b, "label": truth[b], "generated": label, "latency_s": 0.1 + 0.001 * i, "warmup": i < 3}
            if with_p:
                row["p_anomaly"] = 0.9 if label == "Anomaly" else 0.1
            if tokens:
                row.update(input_tokens=500, output_tokens=2)
            out.append(row)
        return out

    _write(pred / "finetuned_test_eval.jsonl", rows(0.97), {"hardware": "T4 (test)"})
    _write(pred / "untuned_test_eval.jsonl", rows(0.6), {"hardware": "T4 (test)"})
    _write(pred / "zeroshot_test_eval.jsonl", rows(0.8, False, True), {"hardware": "API", "price_in_per_mtok_usd": 1.0, "price_out_per_mtok_usd": 2.0})
    _write(pred / "finetuned_val_eval.jsonl", rows(0.97), {})

    report = mr.build_report(pred, res, gpu_hourly_usd=0.5)
    assert report["test_examples"]["n"] == 200 and report["test_examples"]["anomalies"] == 20
    assert [m["key"] for m in report["models"]] == ["finetuned", "untuned", "zeroshot"]
    finetuned, zeroshot = report["models"][0], report["models"][2]
    assert finetuned["as_generated"]["accuracy"] > report["models"][1]["as_generated"]["accuracy"]
    assert "scored_at_tuned_threshold" in finetuned and (res / "threshold.json").exists()
    assert zeroshot["cost"]["input_tokens"] == 200 * 500 and zeroshot["cost"]["estimated_commercial_usd_total"] == pytest.approx((200 * 500 * 1.0 + 200 * 2 * 2.0) / 1e6)
    assert finetuned["latency"]["n"] == 197      # warm-up calls excluded
    assert json.loads((res / "evaluation.json").read_text())["threshold"]["tuned_on"].startswith("validation")