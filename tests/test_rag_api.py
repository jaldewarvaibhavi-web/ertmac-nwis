import json

from tests.helpers import ctx, skip
from src.explainability import assess
from src.rag import Retriever, WELL_RE, ask, grounding_problems


def _state(depth=2690):
    return assess(ctx(), "ACTIVE-01", depth)


def test_offline_answers_only_mention_wells_in_evidence():
    r = Retriever(ctx().documents)
    for q in ["Why is this risk indication being shown?", "Show historical mud-loss cases near the current depth.",
              "Which offset wells are most relevant?", "Summarize the historical mitigation actions."]:
        res = ask(q, _state(), r, use_claude=False)
        allowed = set(WELL_RE.findall(json.dumps(res["evidence"], default=str)))
        allowed |= {e["well_id"] for e in res["excerpts"]}
        assert set(WELL_RE.findall(res["answer"])) <= allowed, q


def test_insufficient_evidence_message():
    res = ask("Why is this risk indication being shown?", _state(2300), None, use_claude=False)
    assert "Insufficient historical evidence" in res["answer"]


def test_grounding_check_catches_invented_well():
    assert grounding_problems("W-102 and W-999 lost mud", {"w": "W-102"}, []) == ["W-999"]


def test_api_services():
    from api import services as s
    assert s.health()["status"] == "ok"
    r = s.risk_predict("ACTIVE-01", 2690)
    assert r["risks"][0]["level"] == "Elevated"
    json.dumps(s.match_offsets("ACTIVE-01", 2690))
    try:
        s.get_well("NOPE")
        assert False
    except KeyError:
        pass


def test_fastapi_endpoints():
    try:
        from fastapi.testclient import TestClient
    except ImportError:
        skip("fastapi not installed")
    from api.main import app
    c = TestClient(app)
    assert c.get("/health").status_code == 200
    assert c.get("/wells/nearby", params={"well_id": "ACTIVE-01", "radius": 5}).status_code == 200
    assert c.get("/wells/NOPE").status_code == 404
    r = c.post("/risk/predict", json={"well_id": "ACTIVE-01", "depth": 2690})
    assert r.status_code == 200 and r.json()["risks"][0]["level"] == "Elevated"
