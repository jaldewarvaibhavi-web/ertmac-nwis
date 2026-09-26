"""
build_spec_files.py - turn the raw synthetic field into the input files the NWIS spec expects.

    data/well_master.csv            well metadata (+ field, basin, data_origin)
    data/well_geology.csv           every 2 m: formation, porosity, caliper      (SYNTHETIC)
    data/formation_tops.csv         derived helper: top/bottom of each formation per well
    data/historical_parameters.csv  offset wells, 1 m: ROP WOB RPM torque SPP flow MW ECD gas pit
    data/current_well_stream.csv    active well ACTIVE-01 with timestamps (synthetic eRTMAC-like)
    data/_truth/events_truth.csv    answer key (event intervals) - ONLY for validating NLP/ML
    data/dataset_metadata.json      provenance + synthetic label

ECD, gas, porosity and caliper do not exist in any real file we have. They are generated here
so every NWIS feature can be demonstrated, and are labelled SYNTHETIC everywhere.
"""

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "_truth" / "raw"
DATA = ROOT / "data"
TRUTH = ROOT / "data" / "_truth"
rng = np.random.default_rng(7)

ACTIVE_OLD, ACTIVE_NEW = "W-ACT", "ACTIVE-01"
FAULT_ZONE, FAULT_RADIUS_KM = (27.314, 95.312), 4.0          # same as base_generator
POROSITY = {"Alluvium": 0.32, "Namsang": 0.27, "Girujan": 0.10, "Tipam": 0.24, "Barail": 0.19, "Kopili": 0.07}
GAS = {"Alluvium": 0.05, "Namsang": 0.10, "Girujan": 0.20, "Tipam": 0.40, "Barail": 0.80, "Kopili": 1.20}
ORIGIN = "SYNTHETIC"


def hav(lat1, lon1, lat2, lon2):
    r = 6371.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    a = math.sin((p2 - p1) / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(math.radians(lon2 - lon1) / 2) ** 2
    return 2 * r * math.asin(math.sqrt(a))


def rename_active(df):
    if "well_id" in df:
        df["well_id"] = df.well_id.replace(ACTIVE_OLD, ACTIVE_NEW)
    return df


def event_intervals(drill):
    """Answer key: contiguous runs of the per-metre label -> (well, type, top, bottom)."""
    rows = []
    for wid, g in drill.groupby("well_id"):
        lab = g.event_label.values
        dep = g.depth_md_m.values
        start = None
        for i in range(len(lab)):
            if lab[i] != "normal" and (start is None or lab[i] != lab[i - 1]):
                start = i
            end_run = lab[i] != "normal" and (i == len(lab) - 1 or lab[i + 1] != lab[i])
            if end_run:
                rows.append(dict(well_id=wid, event_type=lab[i], top_m=int(dep[start]), bottom_m=int(dep[i])))
                start = None
    return pd.DataFrame(rows)


def bit_size(depth, barail_top):
    return 17.5 if depth < 800 else (12.25 if depth < barail_top - 20 else 8.5)


def main():
    wells = rename_active(pd.read_csv(RAW / "wells.csv"))
    tops = rename_active(pd.read_csv(RAW / "formation_tops.csv"))
    drill = pd.read_csv(RAW / "drilling_data.csv")
    stream = rename_active(pd.read_csv(RAW / "active_well_stream.csv"))
    ev_truth = pd.read_csv(RAW / "events.csv")

    # ---------------------------------------------------------------- well master
    td_form = tops.sort_values("top_depth_m").groupby("well_id").formation.last()
    wm = pd.DataFrame(dict(
        well_id=wells.well_id, well_name=wells.well_name.replace("NHK-ACTIVE", "ACTIVE-01"),
        latitude=wells.latitude, longitude=wells.longitude,
        field="Synthetic Assam-style field", basin="Assam-Arakan (synthetic analogue)",
        total_depth=wells.total_depth_m, drilling_year=wells.spud_year,
        well_status=wells.status, formation=wells.well_id.map(td_form),
        current_depth=wells.current_depth_m, data_origin=ORIGIN))
    wm.to_csv(DATA / "well_master.csv", index=False)
    tops.assign(data_origin=ORIGIN).to_csv(DATA / "formation_tops.csv", index=False)

    # ---------------------------------------------------------------- truth intervals
    both = pd.concat([drill, stream.drop(columns=["timestamp"])], ignore_index=True)
    intervals = rename_active(event_intervals(both))
    TRUTH.mkdir(parents=True, exist_ok=True)
    intervals.to_csv(TRUTH / "event_intervals_truth.csv", index=False)
    ev_truth.to_csv(TRUTH / "events_truth.csv", index=False)

    # ---------------------------------------------------------------- geology (2 m)
    geo_rows = []
    for w in wm.itertuples():
        wt = tops[tops.well_id == w.well_id].sort_values("top_depth_m")
        b_top = int(wt[wt.formation == "Barail"].top_depth_m.iloc[0])
        near_fault = hav(w.latitude, w.longitude, *FAULT_ZONE) <= FAULT_RADIUS_KM
        iv = intervals[intervals.well_id == w.well_id]
        for d in range(2, int(w.total_depth) + 1, 2):
            form = wt[wt.top_depth_m <= d].formation.iloc[-1]
            por = POROSITY[form] + rng.normal(0, 0.015)
            if near_fault and form in ("Tipam", "Barail"):
                por += 0.05                                    # fractured / high-perm zone
            cal = bit_size(d, b_top) + abs(rng.normal(0, 0.1))
            if form == "Girujan":
                cal += rng.uniform(0.3, 1.2)                   # reactive clay washout
            here = iv[(iv.top_m - 5 <= d) & (iv.bottom_m + 5 >= d)]
            for e in here.itertuples():
                if e.event_type == "mud_loss":
                    por += 0.04
                if e.event_type == "wellbore_instability":
                    cal += rng.uniform(1.5, 3.0)
            geo_rows.append(dict(well_id=w.well_id, depth=d, formation=form,
                                 porosity=round(float(np.clip(por, 0.01, 0.45)), 3),
                                 caliper=round(cal, 2), caliper_unit="in", data_origin=ORIGIN))
    pd.DataFrame(geo_rows).to_csv(DATA / "well_geology.csv", index=False)

    # ---------------------------------------------------------------- parameters (+ECD, gas)
    def add_ecd_gas(df):
        df = df.copy()
        flow_term = (df.flow_rate_gpm / 750.0).clip(lower=0) ** 1.8
        ecd = df.mud_weight_ppg + 0.25 * flow_term * (df.depth_md_m / 3000.0) + rng.normal(0, 0.01, len(df))
        gas = df.formation.map(GAS) * rng.lognormal(0, 0.25, len(df))
        lab = df.get("event_label", pd.Series("normal", index=df.index))
        # kick: gas rises (and 10 m before it), ECD drops slightly
        for wid, g in df.groupby("well_id"):
            kick_idx = g.index[lab.loc[g.index] == "kick"]
            for i in kick_idx:
                gas.loc[i] *= rng.uniform(3, 6)
            if len(kick_idx):
                first = kick_idx.min()
                pre = g.loc[max(g.index.min(), first - 10):first - 1].index
                gas.loc[pre] *= np.linspace(1.2, 2.5, len(pre))
                ecd.loc[kick_idx] -= 0.05
        df["ECD"] = ecd.round(2)
        df["gas"] = gas.round(3)
        return df

    def to_spec(df):
        return pd.DataFrame(dict(
            well_id=df.well_id, depth=df.depth_md_m, formation=df.formation,
            ROP=df.rop_m_hr, WOB=df.wob_klbs, RPM=df.rpm, torque=df.torque_kftlbs, torque_unit="kft-lbs",
            SPP=df.spp_psi, flow_rate=df.flow_rate_gpm, mud_weight=df.mud_weight_ppg,
            ECD=df.ECD, gas=df.gas, gas_unit="% total gas", pit_volume=df.pit_volume_bbl,
            data_origin=ORIGIN))

    hp = to_spec(add_ecd_gas(drill))
    hp.to_csv(DATA / "historical_parameters.csv", index=False)

    s = add_ecd_gas(stream)
    cs = to_spec(s)
    cs.insert(0, "timestamp", s.timestamp)
    cs.to_csv(DATA / "current_well_stream.csv", index=False)

    meta = {
        "label": "SYNTHETIC DATA — FOR PROTOTYPE DEMONSTRATION ONLY",
        "not": "Not Oil India data. Not eRTMAC data.",
        "generator": "data_generation/base_generator.py + build_spec_files.py (seeded, reproducible)",
        "formation_names": "Real Upper Assam stratigraphic names used for realism; all values generated.",
        "synthetic_only_parameters": ["porosity", "caliper", "ECD", "gas"],
        "real_public_data": "data/forge/Well_58-32_processed_pason_log.csv (Utah FORGE 58-32, public)",
        "files": {f.name: len(pd.read_csv(f)) for f in sorted(DATA.glob("*.csv"))},
    }
    (DATA / "dataset_metadata.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False))
    print(json.dumps(meta["files"], indent=1))


if __name__ == "__main__":
    main()
