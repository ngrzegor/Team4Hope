import pytest
from src.cli.main import evaluate_url, validate_ndjson

def test_evaluate_url_structure():
    url = "https://huggingface.co/someuser/somemodel"
    rec = evaluate_url(url)

    # Check top-level keys
    assert "url" in rec
    assert "scores" in rec
    assert "overall" in rec

    # Scores should be a dict with nested metrics
    for metric, value in rec["scores"].items():
        assert isinstance(value, dict)
        assert "score" in value
        assert "latency" in value
        # score must be None or a float between 0 and 1
        if value["score"] is not None:
            assert 0 <= value["score"] <= 1
        # latency must be None or int
        if value["latency"] is not None:
            assert isinstance(value["latency"], int)


def test_validate_ndjson_valid_record():
    rec = evaluate_url("https://huggingface.co/someuser/somemodel")
    assert validate_ndjson(rec)


def test_validate_ndjson_invalid_score_type():
    rec = evaluate_url("https://huggingface.co/someuser/somemodel")
    rec["scores"]["size"]["score"] = "bad"  # invalid type
    assert not validate_ndjson(rec)


def test_validate_ndjson_invalid_latency_type():
    rec = evaluate_url("https://huggingface.co/someuser/somemodel")
    rec["scores"]["size"]["latency"] = "bad"  # invalid type
    assert not validate_ndjson(rec)
