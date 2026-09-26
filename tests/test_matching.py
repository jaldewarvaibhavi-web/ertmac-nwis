import math

from tests.helpers import ctx
from src.event_engine import correlate
from src.geo_matching import nearby_wells, relevance
from src.similarity import param_similarity
import pandas as pd


def test_radius_filter():
    n = nearby_wells(ctx().wells, "ACTIVE-01", 3.0)
    assert (n[n.in_radius].distance_km <= 3.0).all() and (n[~n.in_radius].distance_km > 3.0).all()


def test_relevance_scores_bounded_and_sorted():
    rel = relevance(ctx(), "ACTIVE-01", 2660)
    comps = ["relevance_score", "spatial_score", "formation_score", "depth_score", "parameter_score", "geology_score"]
    vals = rel[comps].astype(float).values.ravel()
    assert ((vals >= 0) & (vals <= 1)).all()
    assert rel.relevance_score.is_monotonic_decreasing


def test_weights_are_configurable():
    rel = relevance(ctx(), "ACTIVE-01", 2660, {"weights": {"spatial": 1, "formation": 0, "depth": 0,
                                                            "parameter": 0, "geology": 0}})
    assert rel.well_id.tolist() == rel.sort_values("distance_km").well_id.tolist()


def test_param_similarity_properties():
    a = pd.Series([0.0, 1.0, float("nan")])
    assert param_similarity(a, a)[0] == 1.0
    s, k = param_similarity(a, pd.Series([1.0, 2.0, 5.0]))
    assert 0 < s < 1 and k == 2                                 # NaN ignored, not invented


def test_formation_aligned_depth():
    al = ctx().aligned_events("ACTIVE-01")
    tops = ctx().tops
    e = al[(al.well_id == "W-102") & (al.event_type == "Mud Loss") & (al.formation == "Barail")].iloc[0]
    top_o = tops[(tops.well_id == "W-102") & (tops.formation == "Barail")].top_depth_m.iloc[0]
    top_a = tops[(tops.well_id == "ACTIVE-01") & (tops.formation == "Barail")].top_depth_m.iloc[0]
    assert math.isclose(e.aligned_depth, top_a + (e.depth - top_o))


def test_depth_tolerance_is_configurable():
    rel = relevance(ctx(), "ACTIVE-01", 2660)
    narrow = correlate(ctx().aligned_events("ACTIVE-01"), rel, 2660, {"depth_tolerance_m": 5})
    wide = correlate(ctx().aligned_events("ACTIVE-01"), rel, 2660, {"depth_tolerance_m": 50})
    assert (wide.status == "AT CURRENT DEPTH").sum() > (narrow.status == "AT CURRENT DEPTH").sum()
