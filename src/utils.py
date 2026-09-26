"""
utils.py - shared settings, vocabulary and small helpers.

WHAT:  one place for paths, default configuration (radius, weights, tolerances),
       the canonical event / formation vocabulary and tiny maths helpers.
WHY:   every other module must use the SAME names and settings, otherwise the
       dashboard, API and tests would disagree with each other.
"""

from __future__ import annotations

import math
import re
from copy import deepcopy
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
FORGE_DIR = DATA_DIR / "forge"
TRUTH_DIR = DATA_DIR / "_truth"          # answer keys for validation only - never used by the app
DOCS_DIR = ROOT / "documents"
MODELS_DIR = ROOT / "models"
DB_PATH = DATA_DIR / "nwis.db"

SYNTHETIC_BANNER = "SYNTHETIC DATA — FOR PROTOTYPE DEMONSTRATION ONLY"
VALIDATION_DISCLAIMER = "Prototype validation only — not representative of field deployment accuracy."
DECISION_NOTE = "Review historical evidence and use engineering judgment. NWIS is decision support only."
INSUFFICIENT_ML = "Insufficient labeled data for reliable ML training."

# ------------------------------------------------------------------ configuration
DEFAULT_CONFIG: dict = {
    "radius_km": 10.0,
    "max_offsets": 8,
    "relevance_threshold": 0.35,
    # prototype weights only - NOT scientifically validated; editable in the UI
    "weights": {"spatial": 0.30, "formation": 0.20, "depth": 0.20, "parameter": 0.20, "geology": 0.10},
    "depth_tolerance_m": 25.0,       # "comparable depth" window around a formation-aligned event depth
    "lookahead_m": 200.0,            # how far below the bit to look for historical event zones
    "min_events_for_ml": 8,          # train a supervised model only above these counts
    "min_wells_for_ml": 5,
}

CURRENT_PARAMS = ["ROP", "WOB", "RPM", "torque", "SPP", "flow_rate", "mud_weight", "ECD", "gas"]


def get_config(overrides: dict | None = None) -> dict:
    cfg = deepcopy(DEFAULT_CONFIG)
    for k, v in (overrides or {}).items():
        if k == "weights" and isinstance(v, dict):
            cfg["weights"].update(v)
        else:
            cfg[k] = v
    return cfg


# ------------------------------------------------------------------ vocabulary
EVENT_TYPES = ["Mud Loss", "Kick", "Stuck Pipe", "Torque Spike", "Pressure/Gas Anomaly", "NPT",
               "Fishing", "Cementing Issue", "Casing Issue", "Wellbore Instability", "Bit Problem", "Other"]

_EVENT_ALIASES = {
    "mud loss": "Mud Loss", "mud_loss": "Mud Loss", "lost circulation": "Mud Loss", "losses": "Mud Loss",
    "kick": "Kick", "influx": "Kick", "well control": "Kick",
    "stuck pipe": "Stuck Pipe", "stuck_pipe": "Stuck Pipe", "differential sticking": "Stuck Pipe",
    "torque spike": "Torque Spike", "torque_spike": "Torque Spike", "high torque": "Torque Spike",
    "pressure": "Pressure/Gas Anomaly", "gas anomaly": "Pressure/Gas Anomaly", "pressure/gas anomaly": "Pressure/Gas Anomaly",
    "npt": "NPT", "fishing": "Fishing",
    "cementing issue": "Cementing Issue", "cementing_issue": "Cementing Issue", "poor cbl": "Cementing Issue",
    "casing issue": "Casing Issue",
    "wellbore instability": "Wellbore Instability", "wellbore_instability": "Wellbore Instability",
    "tight hole": "Wellbore Instability",
    "bit problem": "Bit Problem", "bit_problem": "Bit Problem", "bit balling": "Bit Problem",
}


def normalize_event_type(raw) -> str:
    """Map any spelling to the canonical list; unknown -> 'Other'. Original is kept by callers."""
    if raw is None or (isinstance(raw, float) and math.isnan(raw)):
        return "Other"
    key = re.sub(r"\s+", " ", str(raw).strip().lower())
    if key in _EVENT_ALIASES:
        return _EVENT_ALIASES[key]
    for canon in EVENT_TYPES:
        if canon.lower() == key:
            return canon
    return "Other"


FORMATION_ORDER = ["Alluvium", "Namsang", "Girujan", "Tipam", "Barail", "Kopili"]


def normalize_formation(raw) -> str | None:
    """'barail fm.' / 'BARAIL' / ' Barail ' -> 'Barail'. Unknown names are title-cased, not dropped."""
    if raw is None or (isinstance(raw, float) and math.isnan(raw)):
        return None
    s = re.sub(r"\b(fm|formation)\.?$", "", str(raw).strip(), flags=re.I).strip()
    for f in FORMATION_ORDER:
        if s.lower() == f.lower():
            return f
    return s.title() if s else None


# ------------------------------------------------------------------ maths helpers
def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Great-circle distance between two points in km."""
    r = 6371.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp, dl = p2 - p1, math.radians(lon2 - lon1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * r * math.asin(math.sqrt(a))


def clip01(x: float) -> float:
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return float("nan")
    return max(0.0, min(1.0, float(x)))


def risk_label(score: float) -> str:
    """Words used everywhere for risk. Never 'will occur'."""
    if score >= 0.6:
        return "Elevated"
    if score >= 0.3:
        return "Moderate"
    return "Low"
