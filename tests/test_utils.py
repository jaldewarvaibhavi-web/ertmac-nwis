from tests.helpers import ROOT  # noqa: F401
from src.utils import get_config, haversine_km, normalize_event_type, normalize_formation, risk_label


def test_haversine_basic():
    assert haversine_km(27.3, 95.3, 27.3, 95.3) == 0
    assert 110 < haversine_km(27.0, 95.0, 28.0, 95.0) < 112          # 1 degree latitude ~111 km


def test_event_type_normalisation_keeps_vocabulary():
    assert normalize_event_type("lost circulation") == "Mud Loss"
    assert normalize_event_type("stuck_pipe") == "Stuck Pipe"
    assert normalize_event_type("something odd") == "Other"
    assert normalize_event_type(None) == "Other"


def test_formation_normalisation():
    assert normalize_formation(" barail fm. ") == "Barail"
    assert normalize_formation("BARAIL") == "Barail"
    assert normalize_formation("formation b") == "Formation B"   # unknown names kept, not dropped


def test_risk_words_never_certain():
    assert [risk_label(x) for x in (0.1, 0.4, 0.8)] == ["Low", "Moderate", "Elevated"]


def test_config_overrides_are_merged():
    cfg = get_config({"radius_km": 3, "weights": {"spatial": 1.0}})
    assert cfg["radius_km"] == 3 and cfg["weights"]["spatial"] == 1.0 and cfg["weights"]["geology"] == 0.10
