"""
NWIS - Step 2: Synthetic dataset generator
------------------------------------------
Creates an Assam-style (Upper Assam shelf) synthetic field:
  * 15 historical (offset) wells + 1 active well
  * formation tops that vary from well to well (formations dip to the east)
  * depth-wise drilling data with the 10 chosen parameters
  * historical drilling events injected with realistic signatures

ALL DATA IS SYNTHETIC - generated for prototype/demo purposes only.
Formation names follow the public Upper Assam stratigraphy; nothing here
is real Oil India data.

Run:  python generate_dataset.py      -> writes CSV files into ./data
"""

import math
from pathlib import Path

import numpy as np
import pandas as pd

SEED = 42
rng = np.random.default_rng(SEED)
OUT = Path(__file__).resolve().parents[1] / "data" / "_truth" / "raw"
OUT.mkdir(parents=True, exist_ok=True)

# ----------------------------------------------------------------------------
# 1. Field settings
# ----------------------------------------------------------------------------
FIELD_CENTER = (27.300, 95.300)          # lat, lon (Upper Assam area)
N_HISTORICAL = 15
ACTIVE_CURRENT_DEPTH = 2450              # active well is now drilling here (m)

# Loss-prone fault zone: wells close to it lose mud in Tipam/Barail more often.
# This makes "nearby" wells genuinely more relevant than far ones.
FAULT_ZONE = (27.314, 95.312)
FAULT_RADIUS_KM = 4.0

# Formation order and base top depth (m MD) at the field centre
FORMATIONS = ["Alluvium", "Namsang", "Girujan", "Tipam", "Barail", "Kopili"]
BASE_TOPS = {"Alluvium": 0, "Namsang": 350, "Girujan": 900,
             "Tipam": 1900, "Barail": 2600, "Kopili": 3150}
DIP_M_PER_KM_EAST = 12                   # tops get deeper towards the east

# Normal drilling behaviour per formation
# rop m/hr, wob klbs, rpm, torque kft-lbs
FORMATION_BASE = {
    "Alluvium": dict(rop=28, wob=10, rpm=120, torque=5.0),
    "Namsang":  dict(rop=22, wob=13, rpm=120, torque=6.5),
    "Girujan":  dict(rop=13, wob=18, rpm=100, torque=9.0),   # sticky clay
    "Tipam":    dict(rop=19, wob=16, rpm=110, torque=10.0),  # sandstone
    "Barail":   dict(rop=10, wob=22, rpm=100, torque=12.0),  # sand/shale/coal
    "Kopili":   dict(rop=7,  wob=24, rpm=90,  torque=13.0),  # shale
}

# Event probability per formation for a well (far from fault / near fault)
EVENT_RULES = {
    "Girujan": [("wellbore_instability", 0.35, 0.35), ("bit_problem", 0.40, 0.40)],
    "Tipam":   [("mud_loss", 0.20, 0.70)],
    "Barail":  [("mud_loss", 0.20, 0.85), ("stuck_pipe", 0.35, 0.45),
                ("torque_spike", 0.30, 0.35)],
    "Kopili":  [("kick", 0.25, 0.30), ("wellbore_instability", 0.25, 0.25)],
}

MITIGATIONS = {
    "mud_loss": [("Pumped 50 bbl LCM pill (40 ppb fine+medium), reduced MW by 0.3 ppg", True),
                 ("Reduced flow rate to 550 gpm, pumped 30 bbl LCM pill", True),
                 ("Pumped 2 LCM pills, losses continued; set cement plug", False)],
    "kick": [("Shut in well, circulated out kick with 0.5 ppg heavier kill mud", True),
             ("Shut in, driller's method, raised MW to 11.2 ppg", True)],
    "stuck_pipe": [("Jarred up, spotted pipe-freeing pill, pipe free after 6 hrs", True),
                   ("Worked pipe, circulated 3 bottoms up, freed pipe", True),
                   ("Could not free pipe, backed off and fished; sidetracked", False)],
    "torque_spike": [("Reduced WOB and RPM, circulated hole clean", True),
                     ("Added lubricant (2%) to mud, torque normalised", True)],
    "wellbore_instability": [("Raised MW by 0.2 ppg, added KCl/polymer inhibitor, reamed tight spots", True),
                             ("Back-reamed tight section, pumped hi-vis sweep", True)],
    "bit_problem": [("Pulled out, found balled bit; changed bit, added anti-balling agent", True),
                    ("POOH, bit worn (grade 4-4), ran new PDC bit", True)],
}

SEVERITY_TEXT = {
    "mud_loss": lambda s: f"{int(10 + s*50)} bbl/hr",
    "kick": lambda s: f"{int(5 + s*20)} bbl pit gain",
    "stuck_pipe": lambda s: f"overpull {int(20 + s*40)} klbs",
    "torque_spike": lambda s: f"torque up to {int(16 + s*8)} kft-lbs",
    "wellbore_instability": lambda s: "tight hole, heavy cavings" if s > 0.5 else "tight hole",
    "bit_problem": lambda s: f"ROP dropped {int(30 + s*40)}%",
}


# ----------------------------------------------------------------------------
# 2. Helpers
# ----------------------------------------------------------------------------
def haversine_km(lat1, lon1, lat2, lon2):
    r = 6371.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp, dl = p2 - p1, math.radians(lon2 - lon1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * r * math.asin(math.sqrt(a))


def random_point_near(center, max_km):
    """Random lat/lon within max_km of the centre."""
    d = max_km * math.sqrt(rng.random())
    theta = rng.random() * 2 * math.pi
    dlat = (d * math.cos(theta)) / 111.0
    dlon = (d * math.sin(theta)) / (111.0 * math.cos(math.radians(center[0])))
    return round(center[0] + dlat, 5), round(center[1] + dlon, 5)


def east_km(lon):
    return (lon - FIELD_CENTER[1]) * 111.0 * math.cos(math.radians(FIELD_CENTER[0]))


def hole_section(depth, barail_top):
    """Hole size / flow / mud weight by section (simple casing design)."""
    if depth < 800:
        return '17 1/2"', 900, 9.0
    if depth < barail_top - 20:
        return '12 1/4"', 750, 9.6
    return '8 1/2"', 500, 10.2


# ----------------------------------------------------------------------------
# 3. Wells + formation tops
# ----------------------------------------------------------------------------
wells, tops_rows = [], []
for i in range(N_HISTORICAL + 1):
    active = i == N_HISTORICAL
    if active:
        wid, name = "W-ACT", "NHK-ACTIVE"
        lat, lon = 27.305, 95.305
        status, spud = "Active", 2026
    else:
        wid, name = f"W-{101 + i}", f"NHK-{101 + i}"
        lat, lon = random_point_near(FIELD_CENTER, 6.5)
        status, spud = "Completed", int(rng.integers(2008, 2025))

    shift = DIP_M_PER_KM_EAST * east_km(lon)
    tops = {}
    for f in FORMATIONS:
        tops[f] = 0 if f == "Alluvium" else int(BASE_TOPS[f] + shift + rng.normal(0, 25))
    td = int(tops["Kopili"] + rng.integers(200, 320))

    wells.append(dict(well_id=wid, well_name=name, latitude=lat, longitude=lon,
                      total_depth_m=td, status=status, spud_year=spud,
                      current_depth_m=ACTIVE_CURRENT_DEPTH if active else td,
                      data_type="SYNTHETIC"))
    for j, f in enumerate(FORMATIONS):
        bottom = tops[FORMATIONS[j + 1]] if j + 1 < len(FORMATIONS) else td
        tops_rows.append(dict(well_id=wid, formation=f,
                              top_depth_m=tops[f], bottom_depth_m=bottom))

wells_df = pd.DataFrame(wells)
tops_df = pd.DataFrame(tops_rows)


# ----------------------------------------------------------------------------
# 4. Plan events for each well
# ----------------------------------------------------------------------------
def plan_events(well, well_tops, force_barail_loss=False):
    near_fault = haversine_km(well["latitude"], well["longitude"], *FAULT_ZONE) <= FAULT_RADIUS_KM
    evs = []
    for _, t in well_tops.iterrows():
        rules = EVENT_RULES.get(t.formation, [])
        for etype, p_far, p_near in rules:
            p = p_near if near_fault else p_far
            if force_barail_loss and t.formation == "Barail" and etype == "mud_loss":
                p = 1.0
            if rng.random() < p:
                span = t.bottom_depth_m - t.top_depth_m
                # losses tend to happen near the top of the permeable formation
                if etype == "mud_loss":
                    start = t.top_depth_m + int(rng.integers(5, max(10, min(80, span // 3))))
                else:
                    start = t.top_depth_m + int(rng.integers(20, max(30, span - 40)))
                length = int(rng.integers(6, 20)) if etype != "bit_problem" else int(rng.integers(30, 60))
                evs.append(dict(event_type=etype, start=start, end=start + length,
                                formation=t.formation, severity=float(rng.uniform(0.2, 1.0))))
    # cementing issue (not visible in the 10 parameters; recorded from reports)
    if near_fault and rng.random() < 0.5:
        b_top = int(well_tops.loc[well_tops.formation == "Barail", "top_depth_m"].iloc[0])
        evs.append(dict(event_type="cementing_issue", start=b_top - 20, end=b_top - 20,
                        formation="Tipam", severity=0.5))
    # avoid overlapping events in the same well
    evs.sort(key=lambda e: e["start"])
    clean = []
    for e in evs:
        if not clean or e["start"] > clean[-1]["end"] + 25:
            clean.append(e)
    return clean


# ----------------------------------------------------------------------------
# 5. Generate depth-wise drilling data (1 row per metre)
# ----------------------------------------------------------------------------
def formation_at(depth, well_tops):
    f = well_tops[well_tops.top_depth_m <= depth].iloc[-1]
    return f.formation


def simulate_well(well, well_tops, events, max_depth):
    barail_top = int(well_tops.loc[well_tops.formation == "Barail", "top_depth_m"].iloc[0])
    depths = np.arange(1, max_depth + 1)
    rows = []
    pit = 600.0
    for d in depths:
        form = formation_at(d, well_tops)
        base = FORMATION_BASE[form]
        _, flow, mw = hole_section(d, barail_top)
        if form == "Kopili":
            mw = 10.6
        rop = base["rop"] * rng.normal(1, 0.08)
        wob = base["wob"] * rng.normal(1, 0.05)
        rpm = base["rpm"] * rng.normal(1, 0.03)
        torque = (base["torque"] + d / 1000) * rng.normal(1, 0.05)
        spp = (1400 + 0.55 * d) * (flow / 750) ** 1.8 * rng.normal(1, 0.02)
        pit += rng.normal(0, 0.4)
        pit += (600 - pit) * 0.02            # mud engineers keep pit near 600 bbl
        label = "normal"

        for e in events:
            s, en, sev, et = e["start"], e["end"], e["severity"], e["event_type"]
            if et == "cementing_issue":
                continue
            # precursor window (warning signs before the event)
            pre = 15
            if s - pre <= d < s:
                k = (d - (s - pre)) / pre           # 0 -> 1
                if et == "stuck_pipe":
                    torque *= 1 + 0.25 * k * sev
                elif et == "mud_loss":
                    pit -= 0.4 * k * sev
                elif et == "kick":
                    rop *= 1 + 0.3 * k * sev
                elif et == "bit_problem":
                    pass
            if s <= d <= en:
                label = et
                if et == "mud_loss":
                    pit -= (1.5 + 4 * sev)
                    flow = flow * 0.85                       # crew reduces pump rate
                    spp *= 0.85 ** 1.8 * (1 - 0.10 * sev)    # lower rate + fluid escaping
                elif et == "kick":
                    pit += (1.0 + 3 * sev)
                    rop *= 1.6 + 0.6 * sev
                    spp *= 1 - 0.04 * sev
                elif et == "stuck_pipe":
                    torque *= 1.5 + 0.5 * sev
                    spp *= 1 + rng.uniform(0.02, 0.10) * sev
                    rop *= 0.4
                elif et == "torque_spike":
                    torque *= 1 + rng.choice([0.1, 0.6 + 0.4 * sev])
                    rpm *= rng.normal(1, 0.12)
                elif et == "wellbore_instability":
                    torque *= 1.25 + 0.2 * sev
                    spp *= 1 + rng.uniform(0, 0.05)
                    rop *= 0.7
                elif et == "bit_problem":
                    k = (d - s) / max(1, en - s)
                    rop *= 1 - (0.3 + 0.4 * sev) * k

        rows.append(dict(
            well_id=well["well_id"],
            depth_md_m=int(d),
            formation=form,
            rop_m_hr=round(max(rop, 0.5), 2),
            wob_klbs=round(wob, 2),
            rpm=round(rpm, 1),
            torque_kftlbs=round(torque, 2),
            spp_psi=round(spp, 0),
            flow_rate_gpm=round(flow * rng.normal(1, 0.01), 0),
            mud_weight_ppg=round(mw + rng.normal(0, 0.02), 2),
            pit_volume_bbl=round(pit, 1),
            event_label=label,
        ))
    return rows


all_rows, event_rows, active_stream = [], [], None
for _, w in wells_df.iterrows():
    wt = tops_df[tops_df.well_id == w.well_id].reset_index(drop=True)
    is_active = w.status == "Active"
    # the active well will (hidden) hit Barail losses - used later to test alerts
    evs = plan_events(w, wt, force_barail_loss=is_active)
    rows = simulate_well(w, wt, evs, int(w.total_depth_m))

    if is_active:
        active_stream = pd.DataFrame(rows)
        # timestamps: time per metre from ROP -> realistic "live" stream
        minutes = (60 / active_stream.rop_m_hr).cumsum()
        start = pd.Timestamp("2026-09-01 06:00")
        active_stream.insert(1, "timestamp", start + pd.to_timedelta(minutes, unit="min"))
        active_stream["timestamp"] = active_stream["timestamp"].dt.floor("s")
        continue

    all_rows.extend(rows)
    for n, e in enumerate(evs, start=1):
        mit, worked = MITIGATIONS.get(e["event_type"],
                                      [("Squeeze cement job performed after poor CBL", True)])[
            int(rng.integers(0, len(MITIGATIONS.get(e["event_type"], [1]))))]
        sev_txt = SEVERITY_TEXT.get(e["event_type"], lambda s: "poor CBL across Tipam")(e["severity"])
        event_rows.append(dict(
            event_id=f"{w.well_id}-E{n}",
            well_id=w.well_id,
            depth_m=e["start"],
            formation=e["formation"],
            event_type=e["event_type"],
            severity=sev_txt,
            npt_hours=round(1 + e["severity"] * {"stuck_pipe": 14, "kick": 8, "mud_loss": 6,
                                                   "cementing_issue": 20}.get(e["event_type"], 3), 1),
            mitigation=mit,
            mitigation_worked="Yes" if worked else "No",
            source=f"DDR #{int(e['start'] / 70) + 1} (synthetic)",
        ))

drilling_df = pd.DataFrame(all_rows)
events_df = pd.DataFrame(event_rows)

# ----------------------------------------------------------------------------
# 6. Save
# ----------------------------------------------------------------------------
wells_df.to_csv(OUT / "wells.csv", index=False)
tops_df.to_csv(OUT / "formation_tops.csv", index=False)
events_df.to_csv(OUT / "events.csv", index=False)
drilling_df.to_csv(OUT / "drilling_data.csv", index=False)
active_stream.to_csv(OUT / "active_well_stream.csv", index=False)

print("wells:", len(wells_df), "| tops:", len(tops_df), "| events:", len(events_df),
      "| drilling rows:", len(drilling_df), "| active stream rows:", len(active_stream))
