"""Consent attestations and strict, private JSON serialization."""
import json
import pytest
from src.common import consent_subjects, local_path, write_json


@pytest.mark.parametrize("data", [{"attested": False, "subjects": ["a"]},
                                    {"attested": True, "subjects": []},
                                    {"attested": True, "subjects": ["__UNKNOWN__"]},
                                    {"attested": True, "subjects": ["a", "a"]}])
def test_invalid_consent_rejected(tmp_path, data):
    path = tmp_path / "consent.json"
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError):
        consent_subjects(path)


def test_local_json_and_paths(tmp_path):
    path = tmp_path / "result.json"
    write_json(path, {"metric": None})
    assert json.loads(path.read_text()) == {"metric": None}
    assert path.stat().st_mode & 0o777 == 0o600
    with pytest.raises(ValueError):
        local_path("https://example.com/data")
    with pytest.raises(ValueError):
        write_json(tmp_path / "bad.json", {"metric": float("nan")})
