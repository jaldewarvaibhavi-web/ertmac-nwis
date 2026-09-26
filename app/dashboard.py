"""
dashboard.py - eRTMAC-NWIS Streamlit dashboard (Phase 12 + Demo Mode, Phase 14).

Run:  streamlit run app/dashboard.py        (or: python run.py dashboard)

Layout (spec section 19):
  Header + data-origin banner
  Section 1 Active well            Section 5 Risk intelligence (alert cards with Why / evidence / sources / Ask)
  Tabs:  Map & offset intelligence (S2, S4) | Drilling parameters (S3) | Why & historical evidence (S6, S7)
         | Ask NWIS (S8) | Data quality & validation
Demo Mode: "Start Simulation" drills ACTIVE-01 down through the Barail loss zone.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import folium  # noqa: E402
import pandas as pd  # noqa: E402
import streamlit as st  # noqa: E402
from streamlit_folium import st_folium  # noqa: E402

from app import charts  # noqa: E402
from src.engine import get_context  # noqa: E402
from src.explainability import assess, ml_status  # noqa: E402
from src.ingestion import append_row  # noqa: E402
from src.rag import Retriever, ask, claude_available  # noqa: E402
from src.utils import (DATA_DIR, DECISION_NOTE, DEFAULT_CONFIG, MODELS_DIR, SYNTHETIC_BANNER,  # noqa: E402
                       VALIDATION_DISCLAIMER)

st.set_page_config(page_title="eRTMAC-NWIS", page_icon="🛢️", layout="wide")


@st.cache_resource
def context():
    return get_context()


@st.cache_resource
def retriever():
    return Retriever(context().documents)


ctx = context()
ss = st.session_state
DEMO_START, DEMO_STEP = 2350, 10

# ================================================================ sidebar: source, config, demo
st.sidebar.title("🛢️ eRTMAC-NWIS")
sources = {"ACTIVE-01 · synthetic eRTMAC-like stream": "ACTIVE-01"}
if len(ctx.forge_params):
    sources["FORGE-58-32 · replay of a public drilling log (not eRTMAC)"] = "FORGE-58-32"
src_label = st.sidebar.radio("Current well stream", list(sources))
well = sources[src_label]
stream = ctx.stream(well)
dmin, dmax = int(stream.depth.min()) + 30, int(stream.depth.max())

st.sidebar.subheader("Offset search")
cfg = {"radius_km": st.sidebar.slider("Search radius (km)", 1.0, 20.0, float(DEFAULT_CONFIG["radius_km"]), 0.5),
       "max_offsets": st.sidebar.slider("Number of offset wells", 1, 15, DEFAULT_CONFIG["max_offsets"]),
       "relevance_threshold": st.sidebar.slider("Relevance threshold", 0.0, 1.0,
                                                DEFAULT_CONFIG["relevance_threshold"], 0.05),
       "depth_tolerance_m": st.sidebar.slider("Depth tolerance ± (m)", 5, 100, int(DEFAULT_CONFIG["depth_tolerance_m"]), 5),
       "lookahead_m": st.sidebar.slider("Look-ahead below bit (m)", 50, 400, int(DEFAULT_CONFIG["lookahead_m"]), 25)}
with st.sidebar.expander("Relevance weights (prototype, not validated)"):
    w = {k: st.slider(k.title(), 0.0, 1.0, v, 0.05) for k, v in DEFAULT_CONFIG["weights"].items()}
    total = sum(w.values()) or 1.0
    cfg["weights"] = {k: v / total for k, v in w.items()}
    st.caption("Weights are re-normalised to sum to 1. " +
               ", ".join(f"{k} {v:.0%}" for k, v in cfg["weights"].items()))

# ---- demo / depth state (plain session values, not widget keys -> safe to change anytime)
if ss.get("well") != well:
    ss.well, ss.depth, ss.running, ss.seen = well, (2450 if well == "ACTIVE-01" else 1900), False, set()
ss.setdefault("running", False)
ss.setdefault("seen", set())
ss.setdefault("pause_on_alert", True)


def start():
    if ss.depth >= dmax - DEMO_STEP:
        ss.depth = DEMO_START if well == "ACTIVE-01" else 1900
    ss.running = True


def pause():
    ss.running = False


def reset():
    ss.running, ss.depth, ss.seen = False, (DEMO_START if well == "ACTIVE-01" else 1900), set()


def slider_moved():
    ss.depth = ss.depth_slider
    ss.running = False


st.sidebar.subheader("Demo mode")
b1, b2, b3 = st.sidebar.columns(3)
b1.button("▶ Start", on_click=start, help="Start Simulation")
b2.button("⏸ Pause", on_click=pause)
b3.button("↺ Reset", on_click=reset)
speed = st.sidebar.select_slider("Speed (m per step)", [5, 10, 20, 40], value=DEMO_STEP)
ss.pause_on_alert = st.sidebar.checkbox("Pause when a new elevated indication appears", value=ss.pause_on_alert)
ss.depth = int(min(max(ss.depth, dmin), dmax))
ss.depth_slider = ss.depth
st.sidebar.slider("Bit depth (m)", dmin, dmax, key="depth_slider", on_change=slider_moved)
st.sidebar.caption(("🟢 Simulation running" if ss.running else "⏸ Simulation paused") + f" · {ss.depth:,} m")
depth = ss.depth

# ================================================================ analysis for this bit position
state = assess(ctx, well, depth, cfg)
cur = state["current"]
wrow = ctx.wells[ctx.wells.well_id == well].iloc[0]

# ================================================================ header + banner
st.title("eRTMAC-NWIS · Nearby Wells Intelligence System")
st.caption("AI-powered offset-well knowledge and decision support")
if wrow.data_origin == "SYNTHETIC":
    st.warning(f"**{SYNTHETIC_BANNER}** · Synthetic eRTMAC-like current drilling stream · "
               "not Oil India data · not eRTMAC data")
else:
    st.info(f"**REAL PUBLIC DATA** — Utah FORGE 58-32 drilling log replay (not eRTMAC, not Oil India). "
            f"Location: {wrow.location_source}. No offset wells or labelled events exist for this well.")

# ================================================================ Section 1 - active well
c = st.columns(6)
c[0].metric("Well", well)
c[1].metric("Bit depth", f"{depth:,} m")
c[2].metric("Formation", state["formation"] or "Not available")
c[3].metric("Field", str(wrow.field)[:28])
c[4].metric("Basin", str(wrow.basin)[:28])
c[5].metric("Status", wrow.well_status)

# ================================================================ Section 5 - risk intelligence (always visible)
st.subheader("Risk intelligence")
if not state["alerts"]:
    st.success("No moderate or elevated risk indication at this depth. "
               f"{DECISION_NOTE}")
for i, a in enumerate(state["alerts"][:3]):
    box = st.error if a["level"] == "Elevated" else st.warning
    icon = "🔴" if a["level"] == "Elevated" else "🟠"
    box(f"### {icon} {a['headline']}\n{a['statement']}  \n"
        f"Score {a['score']:.2f} (prototype, not a calibrated probability) · depth {depth:,} m · "
        f"{state['formation'] or 'formation n/a'} · {state['timestamp'] or 'depth-indexed replay'}")
    with st.expander("WHY? — reasons for this indication", expanded=(i == 0)):
        for reason in a["why"]:
            st.markdown(f"- {reason}")
        comp = a["components"]
        st.caption("Components: " + " · ".join(f"{k.replace('_', ' ')}: {v}" for k, v in comp.items()))
    if a["evidence"]:
        with st.expander("View Relevant Wells"):
            ev = pd.DataFrame(a["evidence"])
            st.dataframe(ev.drop_duplicates("well_id")[["well_id", "distance_km", "relevance_score",
                                                        "parameter_similarity", "geology_similarity"]],
                         hide_index=True)
        with st.expander("View Historical Events"):
            st.dataframe(ev[["well_id", "event_type", "event_depth_m", "aligned_depth_m", "delta_to_bit_m", "status",
                             "formation", "severity", "mitigation", "outcome", "source_document",
                             "extraction_method"]], hide_index=True)
            if a["historical_mitigations"]:
                st.markdown("**What was done historically (not an instruction):**")
                for m in a["historical_mitigations"]:
                    st.markdown(f"- {m['mitigation']} → **{m['outcome']}** ({m['times']}×, {m['wells']})")
        with st.expander("View Source Reports"):
            for doc in sorted(set(ev.source_document)):
                d = ctx.documents[ctx.documents.filename == doc]
                st.markdown(f"**{doc}** · {d.extraction_method.iloc[0] if len(d) else ''}")
                if len(d):
                    st.text(d.extracted_text.iloc[0][:2500])
    with st.expander("Ask NWIS about this indication"):
        if st.button("Explain this alert", key=f"explain_{i}"):
            res = ask(f"Why is this {a['risk_type']} risk indication being shown?", state, retriever())
            st.markdown(res["answer"])
            st.caption(f"Answer mode: {res['mode']} · sources: {', '.join(res['sources']) or '-'}")
    f1, f2, _ = st.columns([1, 1, 5])
    if f1.button("👍 Useful", key=f"up_{i}_{depth}"):
        append_row("alert_feedback", dict(time=pd.Timestamp.now().isoformat(timespec="seconds"), well_id=well,
                                          depth=depth, risk_type=a["risk_type"], level=a["level"], useful="yes", comment=""))
        st.toast("Feedback saved")
    if f2.button("👎 Not useful", key=f"down_{i}_{depth}"):
        append_row("alert_feedback", dict(time=pd.Timestamp.now().isoformat(timespec="seconds"), well_id=well,
                                          depth=depth, risk_type=a["risk_type"], level=a["level"], useful="no", comment=""))
        st.toast("Feedback saved")

tab_map, tab_params, tab_why, tab_ask, tab_data = st.tabs(
    ["🗺️ Map & offset intelligence", "📈 Drilling parameters", "🔎 Why & historical evidence",
     "💬 Ask NWIS", "🧪 Data quality & validation"])
rel, corr = state["relevance"], state["correlated"]

# ================================================================ Section 2 + 4 - map & offsets
with tab_map:
    if rel.empty:
        st.info("No offset wells for this well within the radius (the FORGE well has no neighbouring "
                "wells in this dataset). Offset intelligence needs historical wells nearby.")
    else:
        left, right = st.columns([3, 2])
        with left:
            if ss.running:
                st.caption("Map paused during the simulation (press ⏸ to show it).")
            else:
                m = folium.Map(location=[wrow.latitude, wrow.longitude], zoom_start=12, tiles="OpenStreetMap")
                folium.Circle([wrow.latitude, wrow.longitude], radius=cfg["radius_km"] * 1000, color="#2a78d6",
                              fill=True, fill_opacity=0.05, tooltip=f"{cfg['radius_km']} km radius").add_to(m)
                ev_count = ctx.events.groupby("well_id").size()
                others = ctx.wells[(ctx.wells.well_id != well) & (ctx.wells.data_origin == wrow.data_origin)]
                for o in others.itertuples():
                    r = rel[rel.well_id == o.well_id]
                    n = int(ev_count.get(o.well_id, 0))
                    if len(r) and bool(r.selected.iloc[0]):
                        colour, rad = "#2a78d6", 11
                    elif len(r):
                        colour, rad = "#86b6ef", 8
                    else:
                        colour, rad = "#9e9e9e", 5
                    evs = ctx.events[ctx.events.well_id == o.well_id]
                    lines = "".join(f"<li>{e.depth:.0f} m · {e.formation} · {e.event_type} [{e.source_document}]</li>"
                                    for e in evs.itertuples()) or "<li>No recorded events</li>"
                    comp = ("" if r.empty else
                            "<br>".join(f"{k}: {r.iloc[0][f'{k}_score']}" for k in
                                        ["spatial", "formation", "depth", "parameter", "geology"]))
                    popup = (f"<b>{o.well_id}</b> · {o.data_origin}<br>TD {o.total_depth:,.0f} m · {o.drilling_year}"
                             + (f"<br><b>Relevance {r.iloc[0].relevance_score:.2f}</b>"
                                f" ({r.iloc[0].distance_km} km)<br>{comp}" if len(r) else "<br>outside radius")
                             + f"<br><b>{n} historical events</b><ul>{lines}</ul>")
                    folium.CircleMarker([o.latitude, o.longitude], radius=rad, color="white", weight=2,
                                        fill=True, fill_color=colour, fill_opacity=0.95,
                                        tooltip=f"{o.well_id} · {n} events" +
                                                (f" · relevance {r.iloc[0].relevance_score:.2f}" if len(r) else ""),
                                        popup=folium.Popup(popup, max_width=340)).add_to(m)
                    if n:
                        folium.map.Marker([o.latitude, o.longitude], icon=folium.DivIcon(
                            html=f"<div style='font-size:10px;color:#d03b3b;font-weight:700;"
                                 f"transform:translate(8px,-18px)'>{n}⚑</div>")).add_to(m)
                folium.Marker([wrow.latitude, wrow.longitude], tooltip=f"ACTIVE: {well}",
                              icon=folium.Icon(color="red", icon="star")).add_to(m)
                st_folium(m, height=520, use_container_width=True, returned_objects=[])
                st.caption("★ active well · dark blue = selected relevant offsets · light blue = in radius, "
                           "below threshold · grey = outside radius · red n⚑ = number of historical events. "
                           "Click a well for details.")
        with right:
            st.markdown("**Offset relevance ranking** (why each well was selected)")
            st.plotly_chart(charts.relevance_bars(rel, cfg["weights"]))
        st.markdown("**Top relevant offset wells**")
        show = rel.copy()
        show["events"] = show.well_id.map(ctx.events.groupby("well_id").event_type.apply(
            lambda s: ", ".join(f"{k} ×{v}" for k, v in s.value_counts().items()))).fillna("none")
        st.dataframe(show[["selected", "well_id", "distance_km", "relevance_score", "spatial_score",
                           "formation_score", "depth_score", "parameter_score", "geology_score", "events",
                           "missing_components", "why"]], hide_index=True,
                     column_config={"relevance_score": st.column_config.ProgressColumn(
                         "relevance", min_value=0.0, max_value=1.0, format="%.2f")})

# ================================================================ Section 3 - drilling parameters
with tab_params:
    shown = stream[stream.depth <= depth]
    zones = corr[corr.status.isin(["AT CURRENT DEPTH", "AHEAD"])] if len(corr) else None
    _, inc = ctx.anomalies(well)
    inc_now = inc[(inc.top_m <= depth) & (inc.confidence != "Low")] if len(inc) else None
    st.plotly_chart(charts.parameter_tracks(shown, depth, ctx.tops[ctx.tops.well_id == well], zones, inc_now))
    st.caption("Blue = drilling parameters up to the bit · red line = bit · amber = incidents detected in this "
               "well's own data · red bands = offset-well events projected (formation-aligned) onto this well · "
               "'Not available' = parameter missing in this source (never filled with fake values).")
    last = shown.tail(1)
    if len(last):
        vals = {k: (f"{last.iloc[0][k]:.2f}" if k in last and pd.notna(last.iloc[0][k]) else "Not available")
                for k in ["ROP", "WOB", "RPM", "torque", "SPP", "flow_rate", "mud_weight", "ECD", "gas"]}
        st.dataframe(pd.DataFrame([vals]), hide_index=True)

# ================================================================ Section 6 + 7 - why & evidence
with tab_why:
    st.markdown(f"**Current position:** {depth:,} m · formation **{state['formation'] or 'Not available'}**"
                + (f" · {state['m_below_top']:.0f} m below its top" if state["m_below_top"] is not None else ""))
    st.plotly_chart(charts.events_by_depth(corr, depth, ctx.tops[ctx.tops.well_id == well], cfg))
    if len(corr):
        st.markdown("**Historical evidence from relevant offset wells**")
        st.dataframe(corr[["status", "well_id", "relevance_score", "distance_km", "event_type", "depth", "aligned_depth",
                           "delta_to_bit_m", "formation", "severity", "mitigation", "outcome", "npt_hours",
                           "source_document", "extraction_method"]].round(1), hide_index=True)
    if state["risks"]:
        st.markdown("**All risk indications (including Low)**")
        st.dataframe(pd.DataFrame([dict(risk=r["risk_type"], level=r["level"], score=r["score"], **r["components"],
                                        ml=r["ml_note"]) for r in state["risks"]]), hide_index=True)

# ================================================================ Section 8 - Ask NWIS
with tab_ask:
    st.caption(f"Answers are built only from NWIS data and retrieved reports. Mode: "
               f"{'Claude API' if claude_available() else 'offline (add ANTHROPIC_API_KEY to .env for Claude)'} · "
               f"retrieval: {retriever().backend}")
    presets = ["Why is this risk indication being shown?", "Show historical mud-loss cases near the current depth.",
               "Which offset wells are most relevant?", "Summarize the historical mitigation actions."]
    pc = st.columns(4)
    for i, q in enumerate(presets):
        if pc[i].button(q, key=f"preset_{i}"):
            ss.question = q
    q = st.text_input("Ask NWIS", value=ss.get("question", ""), placeholder="e.g. What happened with LCM pills in Barail?")
    if q:
        res = ask(q, state, retriever())
        st.markdown(res["answer"])
        st.caption(f"Mode: {res['mode']} · intent: {res['intent']} · sources: {', '.join(res['sources']) or '-'}")
        if res["grounding_warnings"]:
            st.error("Claude mentioned wells not in the evidence: " + ", ".join(res["grounding_warnings"]))
        if res["excerpts"]:
            with st.expander("Retrieved report excerpts"):
                for e in res["excerpts"]:
                    st.markdown(f"**[{e['filename']}]** (score {e['score']}) — {e['text']}")

# ================================================================ data quality & validation
with tab_data:
    st.markdown("**Data validation issues** (values are flagged, never silently removed)")
    st.dataframe(ctx.qc, hide_index=True)
    st.markdown("**Risk model validation** — leave-one-well-out")
    st.warning(VALIDATION_DISCLAIMER)
    rows = []
    for t, s in ml_status().items():
        m = s.get("metrics", {})
        rows.append(dict(event_type=t, used_in_alerts=s["usable"], status=s["reason"],
                         precision=m.get("precision"), recall=m.get("recall"), f1=m.get("f1"),
                         roc_auc=m.get("roc_auc"), pr_auc=m.get("pr_auc"), fpr=m.get("false_positive_rate"),
                         fnr=m.get("false_negative_rate"), confusion=str(m.get("confusion_matrix", "")),
                         events_warned=(f"{s['event_level']['warned_in_advance']}/{s['event_level']['events']}"
                                        if s.get("event_level") else "")))
    st.dataframe(pd.DataFrame(rows), hide_index=True)
    st.caption(f"Full report: {MODELS_DIR / 'risk_metrics.json'} · dataset metadata: {DATA_DIR / 'dataset_metadata.json'}")
    fb = ctx.tables.get("alert_feedback")
    st.markdown("**Engineer feedback** is stored in the `alert_feedback` table (restart to refresh this view).")
    if fb is not None and len(fb):
        st.dataframe(fb, hide_index=True)

# ================================================================ demo loop
if ss.running:
    new_elevated = {a["risk_type"] for a in state["alerts"] if a["level"] == "Elevated"} - ss.seen
    if new_elevated:
        ss.seen |= new_elevated
        for a in state["alerts"]:
            if a["risk_type"] in new_elevated:
                append_row("alert_log", dict(time=pd.Timestamp.now().isoformat(timespec="seconds"), well_id=well,
                                             depth=depth, risk_type=a["risk_type"], level=a["level"],
                                             score=a["score"], summary=a["statement"]))
        if ss.pause_on_alert:
            ss.running = False
            st.toast(f"Paused: new elevated indication — {', '.join(sorted(new_elevated))}")
    if ss.running:
        time.sleep(0.6)
        ss.depth = min(dmax, depth + speed)
        if ss.depth >= dmax:
            ss.running = False
        st.rerun()
