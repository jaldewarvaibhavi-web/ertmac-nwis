"""
ingestion.py - Phase 1 & 2: load, validate and store all data (CSV / Excel / JSON).

WHAT:  file -> column mapping -> cleaning -> unit normalisation -> missing-value handling
       -> schema validation -> SQLite database (data/nwis.db).
WHY:   bad data gives bad alerts. The spec forbids silently discarding values, so every
       problem is FLAGGED (qc_flags column + qc_issues table) and shown in the dashboard.
INPUT: data/*.csv (synthetic field), documents -> data/historical_events.csv,
       data/forge/Well_58-32_processed_pason_log.csv (real public log).
OUTPUT: SQLite tables wells, geology, formation_tops, historical_parameters, events,
        current_stream, documents, qc_issues (+ empty alert_feedback, alert_log).
The table layout mirrors the spec so it can move to PostgreSQL + PostGIS later
(wells.latitude/longitude -> a geography(Point) column).
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd

from src.utils import DATA_DIR, DB_PATH, FORGE_DIR, normalize_event_type, normalize_formation

# ------------------------------------------------------------------ column mapping
COLUMN_ALIASES = {
    # generic snake_case names
    "depth_md_m": "depth", "depth_m": "depth", "md": "depth", "rop_m_hr": "ROP", "rop": "ROP",
    "wob_klbs": "WOB", "wob": "WOB", "rpm": "RPM", "torque_kftlbs": "torque", "spp_psi": "SPP",
    "spp": "SPP", "flow_rate_gpm": "flow_rate", "flow": "flow_rate", "mud_weight_ppg": "mud_weight",
    "mw": "mud_weight", "ecd": "ECD", "gas": "gas", "pit_volume_bbl": "pit_volume",
    # Pason / FORGE names
    "Depth(m)": "depth", "ROP(1 m)": "ROP", "weight on bit (k-lbs)": "WOB", "Rotary Speed (rpm)": "RPM",
    "Surface Torque (psi)": "torque", "Pump Press (psi)": "SPP", "Flow In (gal/min)": "flow_rate",
    "Pit Total (bbls)": "pit_volume", "Flow Out %": "flow_out_pct", "Hookload (k-lbs)": "hookload",
    "Temp Out( degC)": "mud_temp_out_c", "WH Pressure (psi)": "wellhead_pressure",
}

# ------------------------------------------------------------------ schemas
# range = physically possible limits for the prototype (values outside are flagged, not deleted)
PARAM_RANGES = {"depth": (0, 12000), "ROP": (0, 150), "WOB": (0, 100), "RPM": (0, 300),
                "SPP": (0, 7500), "flow_rate": (0, 1500), "mud_weight": (6, 20), "ECD": (6, 22),
                "gas": (0, 100), "pit_volume": (0, 5000), "flow_out_pct": (0, 150),
                "hookload": (0, 1500), "wellhead_pressure": (0, 15000)}
TORQUE_RANGE = {"kft-lbs": (0, 100), "psi": (0, 2000)}

SCHEMAS = {
    "wells": dict(required=["well_id", "latitude", "longitude", "total_depth", "well_status"],
                  key=["well_id"],
                  ranges={"latitude": (-90, 90), "longitude": (-180, 180), "total_depth": (0, 12000),
                          "drilling_year": (1900, 2100), "current_depth": (0, 12000)}),
    "geology": dict(required=["well_id", "depth", "formation"], key=["well_id", "depth"],
                    ranges={"depth": (0, 12000), "porosity": (0, 0.5), "caliper": (2, 40)}),
    "formation_tops": dict(required=["well_id", "formation", "top_depth_m", "bottom_depth_m"],
                           key=["well_id", "formation"], ranges={"top_depth_m": (0, 12000)}),
    "historical_parameters": dict(required=["well_id", "depth"], key=["well_id", "depth"], ranges=PARAM_RANGES),
    "current_stream": dict(required=["well_id", "depth"], key=["well_id", "depth"], ranges=PARAM_RANGES),
    "events": dict(required=["event_id", "well_id", "depth", "event_type", "source_document"],
                   key=["event_id"], ranges={"depth": (0, 12000), "npt_hours": (0, 2000)}),
    "documents": dict(required=["document_id", "filename"], key=["document_id"], ranges={}),
}

DDL = """
CREATE TABLE IF NOT EXISTS wells (well_id TEXT PRIMARY KEY, well_name TEXT, latitude REAL, longitude REAL,
  field TEXT, basin TEXT, total_depth REAL, drilling_year INTEGER, well_status TEXT, formation TEXT,
  current_depth REAL, data_origin TEXT, location_source TEXT, source_file TEXT, qc_flags TEXT);
CREATE TABLE IF NOT EXISTS geology (well_id TEXT, depth REAL, formation TEXT, porosity REAL, caliper REAL,
  caliper_unit TEXT, data_origin TEXT, source_file TEXT, qc_flags TEXT);
CREATE TABLE IF NOT EXISTS formation_tops (well_id TEXT, formation TEXT, top_depth_m REAL, bottom_depth_m REAL,
  data_origin TEXT, source_file TEXT, qc_flags TEXT);
CREATE TABLE IF NOT EXISTS historical_parameters (well_id TEXT, depth REAL, formation TEXT, ROP REAL, WOB REAL,
  RPM REAL, torque REAL, torque_unit TEXT, SPP REAL, flow_rate REAL, mud_weight REAL, ECD REAL, gas REAL,
  gas_unit TEXT, pit_volume REAL, flow_out_pct REAL, hookload REAL, data_origin TEXT, source_file TEXT, qc_flags TEXT);
CREATE TABLE IF NOT EXISTS events (event_id TEXT PRIMARY KEY, well_id TEXT, depth REAL, formation TEXT,
  event_type TEXT, event_type_raw TEXT, severity TEXT, severity_value REAL, severity_unit TEXT, description TEXT,
  mitigation TEXT, outcome TEXT, npt_hours REAL, report_no TEXT, report_date TEXT, source_document TEXT,
  extraction_method TEXT, data_origin TEXT, source_file TEXT, qc_flags TEXT);
CREATE TABLE IF NOT EXISTS current_stream (timestamp TEXT, well_id TEXT, depth REAL, formation TEXT, ROP REAL,
  WOB REAL, RPM REAL, torque REAL, torque_unit TEXT, SPP REAL, flow_rate REAL, mud_weight REAL, ECD REAL,
  gas REAL, gas_unit TEXT, pit_volume REAL, flow_out_pct REAL, hookload REAL, data_origin TEXT,
  source_file TEXT, qc_flags TEXT);
CREATE TABLE IF NOT EXISTS documents (document_id TEXT PRIMARY KEY, well_id TEXT, document_type TEXT,
  filename TEXT, extracted_text TEXT, source TEXT, extraction_method TEXT, n_events INTEGER,
  source_file TEXT, qc_flags TEXT);
CREATE TABLE IF NOT EXISTS qc_issues (table_name TEXT, column_name TEXT, check_name TEXT, severity TEXT,
  n_rows INTEGER, example TEXT, action TEXT, source_file TEXT);
CREATE TABLE IF NOT EXISTS alert_feedback (time TEXT, well_id TEXT, depth REAL, risk_type TEXT, level TEXT,
  useful TEXT, comment TEXT);
CREATE TABLE IF NOT EXISTS alert_log (time TEXT, well_id TEXT, depth REAL, risk_type TEXT, level TEXT,
  score REAL, summary TEXT);
"""


# ------------------------------------------------------------------ readers
def read_any(path: Path) -> pd.DataFrame:
    """CSV (also a CSV saved as .xls), Excel or JSON."""
    suffix = path.suffix.lower()
    if suffix == ".json":
        return pd.read_json(path)
    if suffix in (".xlsx", ".xls"):
        try:
            return pd.read_excel(path)
        except Exception:
            return pd.read_csv(path)       # e.g. the FORGE file is CSV text named .xls
    return pd.read_csv(path)


def map_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.rename(columns={c: COLUMN_ALIASES.get(c, COLUMN_ALIASES.get(str(c).strip(), c)) for c in df.columns})
    if "depth" not in df and "Depth(ft)" in df:              # unit normalisation ft -> m
        df["depth"] = df["Depth(ft)"] * 0.3048
    return df


# ------------------------------------------------------------------ validation
def _issue(table, col, check, severity, n, example, action, src):
    return dict(table_name=table, column_name=col, check_name=check, severity=severity, n_rows=int(n),
                example=str(example)[:200], action=action, source_file=src)


def validate(df: pd.DataFrame, table: str, source_file: str) -> tuple[pd.DataFrame, list[dict]]:
    """Flag - never drop - problems. Returns (dataframe with qc_flags, list of issues)."""
    schema = SCHEMAS[table]
    df = df.copy()
    issues: list[dict] = []
    flags = pd.Series([[] for _ in range(len(df))], index=df.index, dtype=object)

    def flag(mask, name):
        for i in df.index[mask.fillna(False)]:
            flags.at[i].append(name)

    missing_cols = [c for c in schema["required"] if c not in df.columns]
    if missing_cols:
        issues.append(_issue(table, ",".join(missing_cols), "required column missing", "error", len(df),
                             "", "table cannot be loaded until mapped", source_file))
        return df, issues

    # numeric checks
    ranges = dict(schema["ranges"])
    for col, (lo, hi) in ranges.items():
        if col not in df:
            continue
        raw = df[col]
        num = pd.to_numeric(raw, errors="coerce")
        bad_type = raw.notna() & num.isna()
        if bad_type.any():
            issues.append(_issue(table, col, "non-numeric value", "warning", bad_type.sum(),
                                 raw[bad_type].iloc[0], "set to missing, original kept in qc_flags", source_file))
            flag(bad_type, f"{col}:non_numeric")
        df[col] = num
        n_missing = num.isna().sum() - bad_type.sum()
        if n_missing and n_missing == len(df):
            issues.append(_issue(table, col, "column entirely missing", "info", n_missing, "",
                                 "shown as 'Not available'; excluded from models", source_file))
        elif n_missing:
            issues.append(_issue(table, col, "missing values", "info", n_missing, "",
                                 "left empty (no fake values)", source_file))
        out = (num < lo) | (num > hi)
        if out.any():
            issues.append(_issue(table, col, f"outside plausible range [{lo}, {hi}]", "warning", out.sum(),
                                 f"min {num[out].min():.2f} / max {num[out].max():.2f}",
                                 "kept, flagged; only that value is excluded from models", source_file))
            flag(out, f"{col}:out_of_range")

    # torque: range depends on its unit (kft-lbs vs hydraulic psi are NOT convertible)
    if "torque" in df:
        unit = df.get("torque_unit", pd.Series("kft-lbs", index=df.index)).fillna("kft-lbs")
        df["torque_unit"] = unit
        tq = pd.to_numeric(df.torque, errors="coerce")
        df["torque"] = tq
        for u, (lo, hi) in TORQUE_RANGE.items():
            m = (unit == u) & ((tq < lo) | (tq > hi))
            if m.any():
                issues.append(_issue(table, "torque", f"outside range for {u}", "warning", m.sum(),
                                     tq[m].max(), "kept, flagged", source_file))
                flag(m, "torque:out_of_range")
        if unit.nunique() > 1:
            issues.append(_issue(table, "torque", "mixed torque units", "warning", len(df),
                                 ", ".join(unit.unique()), "compared only within the same unit", source_file))

    # duplicates on the key
    key = [k for k in schema["key"] if k in df]
    dup = df.duplicated(subset=key, keep="first")
    if dup.any():
        issues.append(_issue(table, "+".join(key), "duplicate records", "warning", dup.sum(),
                             df.loc[dup, key].iloc[0].to_dict(), "kept first, later copies flagged", source_file))
        flag(dup, "duplicate")

    # depth order within a well (parameter tables)
    if table in ("historical_parameters", "current_stream") and "depth" in df:
        back = df.groupby("well_id").depth.diff() < 0
        if back.any():
            issues.append(_issue(table, "depth", "depth goes backwards", "warning", back.sum(),
                                 df.loc[back, "depth"].iloc[0], "flagged; sorted copy used by models", source_file))
            flag(back, "depth_backwards")

    # categorical normalisation (original values preserved)
    if "formation" in df:
        norm = df.formation.map(normalize_formation)
        changed = df.formation.notna() & (norm != df.formation)
        if changed.any():
            df["formation_raw"] = df.formation
            issues.append(_issue(table, "formation", "formation name normalised", "info", changed.sum(),
                                 f"{df.formation[changed].iloc[0]} -> {norm[changed].iloc[0]}",
                                 "original kept in formation_raw", source_file))
        df["formation"] = norm
    if table == "events":
        if "event_type_raw" not in df:
            df["event_type_raw"] = df.event_type
        norm = df.event_type.map(normalize_event_type)
        unknown = norm.eq("Other") & ~df.event_type.astype(str).str.lower().eq("other")
        if unknown.any():
            issues.append(_issue(table, "event_type", "unknown event type", "warning", unknown.sum(),
                                 df.event_type[unknown].iloc[0], "mapped to 'Other', original kept", source_file))
        df["event_type"] = norm

    df["qc_flags"] = flags.map(lambda x: ";".join(x) if x else "")
    df["source_file"] = source_file
    return df, issues


# ------------------------------------------------------------------ FORGE (real public log)
def load_forge(path: Path | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Real Utah FORGE 58-32 Pason log -> historical_parameters-shaped frame + a wells row."""
    path = path or FORGE_DIR / "Well_58-32_processed_pason_log.csv"
    raw = read_any(path)
    df = map_columns(raw)
    keep = ["depth", "ROP", "WOB", "RPM", "torque", "SPP", "flow_rate", "pit_volume", "flow_out_pct",
            "hookload", "wellhead_pressure", "mud_temp_out_c"]
    df = df[[c for c in keep if c in df]].copy()
    df.insert(0, "well_id", "FORGE-58-32")
    df["formation"] = None                        # not in the log - add from End-of-Well report
    df["torque_unit"] = "psi"                     # hydraulic surface torque
    for c in ["mud_weight", "ECD", "gas"]:
        df[c] = np.nan                            # NOT available - never invented
    df["data_origin"] = "REAL-PUBLIC"
    gps_file = FORGE_DIR / "Utah FORGE Well and Seismic Locations.csv"
    lat, lon, loc_src = 38.50, -112.90, "APPROXIMATE - add official FORGE GPS file"
    if gps_file.exists():
        g = pd.read_csv(gps_file)
        row = g[g.apply(lambda r: r.astype(str).str.contains("58-32", regex=False).any(), axis=1)]
        if len(row):
            latc = [c for c in g.columns if "lat" in c.lower()][0]
            lonc = [c for c in g.columns if "lon" in c.lower()][0]
            lat, lon, loc_src = float(row.iloc[0][latc]), float(row.iloc[0][lonc]), f"official ({gps_file.name})"
    well = pd.DataFrame([dict(well_id="FORGE-58-32", well_name="Utah FORGE 58-32 (MU-ESW1)", latitude=lat,
                              longitude=lon, field="Utah FORGE (geothermal research site)",
                              basin="Not available (geothermal site, not a petroleum basin)",
                              total_depth=float(df.depth.max()), drilling_year=2017, well_status="Completed",
                              formation=None, current_depth=float(df.depth.max()), data_origin="REAL-PUBLIC",
                              location_source=loc_src)])
    return df, well


# ------------------------------------------------------------------ build database
def build_database(db_path: Path = DB_PATH, data_dir: Path = DATA_DIR, include_forge: bool = True,
                   verbose: bool = True) -> pd.DataFrame:
    """Validate every input file and (re)create the SQLite database. Returns the QC issues."""
    if db_path.exists():
        db_path.unlink()
    con = sqlite3.connect(db_path)
    con.executescript(DDL)
    all_issues: list[dict] = []

    plan = [("wells", "well_master.csv"), ("geology", "well_geology.csv"), ("formation_tops", "formation_tops.csv"),
            ("historical_parameters", "historical_parameters.csv"), ("current_stream", "current_well_stream.csv"),
            ("events", "historical_events.csv"), ("documents", "documents.csv")]
    for table, fname in plan:
        path = data_dir / fname
        if not path.exists():
            all_issues.append(_issue(table, "", "file not found", "error", 0, fname,
                                     "run `python run.py setup` first", fname))
            continue
        df = map_columns(read_any(path))
        df, issues = validate(df, table, fname)
        all_issues += issues
        if any(i["severity"] == "error" for i in issues):
            continue
        cols = [r[1] for r in con.execute(f"PRAGMA table_info({table})")]
        df[[c for c in cols if c in df]].to_sql(table, con, if_exists="append", index=False)

    if include_forge and (FORGE_DIR / "Well_58-32_processed_pason_log.csv").exists():
        params, well = load_forge()
        params, issues = validate(params, "historical_parameters", "Well_58-32_processed_pason_log.csv")
        all_issues += issues
        cols = [r[1] for r in con.execute("PRAGMA table_info(historical_parameters)")]
        params[[c for c in cols if c in params]].to_sql("historical_parameters", con, if_exists="append", index=False)
        well["source_file"] = "Well_58-32_processed_pason_log.csv"
        well["qc_flags"] = ""
        well.to_sql("wells", con, if_exists="append", index=False)

    qc = pd.DataFrame(all_issues)
    if len(qc):
        qc.to_sql("qc_issues", con, if_exists="append", index=False)
    con.commit()
    con.close()
    if verbose:
        print(f"Database written: {db_path}")
        if len(qc):
            print(qc[qc.severity != "info"][["table_name", "column_name", "check_name", "n_rows", "action"]]
                  .to_string(index=False))
    return qc


# ------------------------------------------------------------------ read back
def load_tables(db_path: Path = DB_PATH) -> dict[str, pd.DataFrame]:
    con = sqlite3.connect(db_path)
    names = [r[0] for r in con.execute("SELECT name FROM sqlite_master WHERE type='table'")]
    out = {n: pd.read_sql(f"SELECT * FROM {n}", con) for n in names}
    con.close()
    return out


def append_row(table: str, row: dict, db_path: Path = DB_PATH) -> None:
    con = sqlite3.connect(db_path)
    pd.DataFrame([row]).to_sql(table, con, if_exists="append", index=False)
    con.close()


if __name__ == "__main__":
    build_database()
    meta = DATA_DIR / "dataset_metadata.json"
    if meta.exists():
        print(json.loads(meta.read_text())["label"])
