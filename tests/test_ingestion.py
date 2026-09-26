import pandas as pd

from tests.helpers import ROOT  # noqa: F401
from src.ingestion import map_columns, validate
from src.preprocessing import model_view


def _params(**over):
    base = dict(well_id=["W1"] * 4, depth=[1, 2, 3, 3], ROP=[10, 900, 12, 12], WOB=[10, 10, 10, 10],
                torque=[5, 5, 5, 5])
    base.update(over)
    return pd.DataFrame(base)


def test_out_of_range_is_flagged_not_dropped():
    df, issues = validate(_params(), "historical_parameters", "t.csv")
    assert len(df) == 4                                    # nothing silently removed
    assert "ROP:out_of_range" in df.qc_flags.iloc[1]
    assert any(i["check_name"].startswith("outside plausible range") for i in issues)


def test_duplicates_flagged():
    df, issues = validate(_params(), "historical_parameters", "t.csv")
    assert "duplicate" in df.qc_flags.iloc[3]
    assert any(i["check_name"] == "duplicate records" for i in issues)


def test_missing_required_column_is_an_error():
    _, issues = validate(pd.DataFrame({"depth": [1]}), "historical_parameters", "t.csv")
    assert issues[0]["severity"] == "error"


def test_model_view_blanks_only_the_bad_value():
    df, _ = validate(_params(), "historical_parameters", "t.csv")
    mv = model_view(df)
    assert pd.isna(mv.loc[mv.depth == 2, "ROP"]).all()          # bad ROP blanked
    assert (mv.loc[mv.depth == 2, "WOB"] == 10).all()           # the rest of the row is kept
    assert len(mv) == 3                                         # flagged duplicate excluded from models


def test_forge_column_mapping_and_feet_conversion():
    df = map_columns(pd.DataFrame({"Depth(ft)": [1000.0], "Surface Torque (psi)": [120], "Flow Out %": [80]}))
    assert abs(df.depth.iloc[0] - 304.8) < 1e-6 and "torque" in df and "flow_out_pct" in df


def test_event_type_normalised_and_raw_kept():
    ev = pd.DataFrame(dict(event_id=["e1"], well_id=["W1"], depth=[100], event_type=["lost circulation"],
                           source_document=["x.pdf"]))
    df, _ = validate(ev, "events", "e.csv")
    assert df.event_type.iloc[0] == "Mud Loss" and df.event_type_raw.iloc[0] == "lost circulation"


def test_real_forge_quality_problems_are_reported():
    from tests.helpers import ctx
    qc = ctx().qc
    forge = qc[qc.source_file.str.contains("58-32")]
    assert {"duplicate records", "depth goes backwards"} <= set(forge.check_name)
