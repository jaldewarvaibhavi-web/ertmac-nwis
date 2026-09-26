"""
main.py - eRTMAC-NWIS REST API (FastAPI).   Run:  uvicorn api.main:app --reload   (or python run.py api)
Interactive documentation: http://localhost:8000/docs

Endpoints (all JSON):
  GET  /health                         service status + data label
  GET  /wells                          all wells (with data_origin: SYNTHETIC / REAL-PUBLIC)
  GET  /wells/nearby?well_id=&radius=  wells within radius (km) of a well
  GET  /wells/{well_id}                one well + formation tops
  GET  /events                         all historical events (with source document)
  GET  /events/{well_id}               events of one well
  GET  /current/{well_id}?depth=       current stream up to a depth (synthetic eRTMAC-like)
  POST /match-offsets                  ranked offset wells with 5 score components + events near the bit
  POST /risk/predict                   risk indications (Low/Moderate/Elevated) with components
  POST /explain-alert                  grounded explanation (Claude if configured, else deterministic)
  POST /documents/ingest               upload a .txt/.pdf report -> extracted events (for review)
  POST /documents/search               retrieve report excerpts
NOTE: /wells/nearby is declared before /wells/{well_id} so "nearby" is not read as a well id.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from fastapi import FastAPI, File, HTTPException, Query, UploadFile  # noqa: E402
from pydantic import BaseModel, Field  # noqa: E402

from api import services  # noqa: E402

app = FastAPI(title="eRTMAC-NWIS API", version="0.2",
              description="Nearby Wells Intelligence System - decision support prototype. "
                          "SYNTHETIC DATA — FOR PROTOTYPE DEMONSTRATION ONLY.")


class Weights(BaseModel):
    spatial: float = 0.30
    formation: float = 0.20
    depth: float = 0.20
    parameter: float = 0.20
    geology: float = 0.10


class Config(BaseModel):
    radius_km: float = Field(10.0, gt=0, description="search radius")
    max_offsets: int = Field(8, ge=1)
    relevance_threshold: float = Field(0.35, ge=0, le=1)
    depth_tolerance_m: float = Field(25.0, gt=0)
    lookahead_m: float = Field(200.0, gt=0)
    weights: Weights = Weights()


class DepthRequest(BaseModel):
    well_id: str = "ACTIVE-01"
    depth: float = Field(2690, gt=0, description="bit depth, m MD")
    config: Config = Config()


class ExplainRequest(DepthRequest):
    risk_type: str | None = Field(None, description="e.g. 'Mud Loss'; default = highest indication")
    use_claude: bool = True


class SearchRequest(BaseModel):
    query: str
    k: int = Field(5, ge=1, le=20)


def _run(fn, *a, **k):
    try:
        return fn(*a, **k)
    except KeyError as err:
        raise HTTPException(status_code=404, detail=str(err))
    except ValueError as err:
        raise HTTPException(status_code=400, detail=str(err))


def _cfg(c: Config) -> dict:
    d = c.model_dump()
    return d


@app.get("/health", tags=["system"])
def health():
    """Service status and data label."""
    return services.health()


@app.get("/wells", tags=["wells"])
def wells():
    """All wells with coordinates, field, basin, status and data origin."""
    return services.list_wells()


@app.get("/wells/nearby", tags=["wells"])
def wells_nearby(well_id: str = Query(..., examples=["ACTIVE-01"]), radius: float = Query(10.0, gt=0)):
    """Wells within `radius` km of `well_id` (Haversine distance)."""
    return _run(services.nearby, well_id, radius)


@app.get("/wells/{well_id}", tags=["wells"])
def well(well_id: str):
    """One well, its formation tops and number of historical events."""
    return _run(services.get_well, well_id)


@app.get("/events", tags=["events"])
def all_events():
    """All historical events with provenance (source document, extraction method)."""
    return services.events()


@app.get("/events/{well_id}", tags=["events"])
def well_events(well_id: str):
    """Historical events of one well."""
    return _run(services.events, well_id)


@app.get("/current/{well_id}", tags=["stream"])
def current(well_id: str, depth: float | None = None, limit: int = Query(200, ge=1, le=5000)):
    """Current-well stream up to `depth` (synthetic eRTMAC-like, or replay of a public log)."""
    return _run(services.current, well_id, depth, limit)


@app.post("/match-offsets", tags=["analysis"])
def match_offsets(req: DepthRequest):
    """Ranked offset wells (relevance + 5 components) and historical events near the bit."""
    return _run(services.match_offsets, req.well_id, req.depth, _cfg(req.config))


@app.post("/risk/predict", tags=["analysis"])
def risk_predict(req: DepthRequest):
    """Risk indications per event type. Scores are prototype indications, not calibrated probabilities."""
    return _run(services.risk_predict, req.well_id, req.depth, _cfg(req.config))


@app.post("/explain-alert", tags=["analysis"])
def explain_alert(req: ExplainRequest):
    """Grounded explanation of an alert; Claude is used only if ANTHROPIC_API_KEY is set."""
    return _run(services.explain_alert, req.well_id, req.depth, req.risk_type, req.use_claude, _cfg(req.config))


@app.post("/documents/ingest", tags=["documents"])
async def documents_ingest(file: UploadFile = File(...)):
    """Upload a daily drilling report (.txt/.pdf); returns extracted events for review."""
    return _run(services.documents_ingest, file.filename, await file.read())


@app.post("/documents/search", tags=["documents"])
def documents_search(req: SearchRequest):
    """Retrieve the most relevant report excerpts (TF-IDF or ChromaDB)."""
    return services.documents_search(req.query, req.k)
