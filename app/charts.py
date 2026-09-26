"""
charts.py - interactive Plotly figures for the dashboard (hover works on every mark).

Colour rules: one blue for live data, greys for chrome, status colours only for risk
(amber = detected incident, red = offset-event zone), categorical colour + marker shape
per event type so identity never depends on colour alone.
"""

from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

SERIES = "#2a78d6"
MUTED = "#8a8984"
GRID = "#e6e5e1"
STATUS = {"good": "#0ca30c", "warning": "#fab219", "serious": "#ec835a", "critical": "#d03b3b"}
BAND = ["rgba(0,0,0,0.035)", "rgba(0,0,0,0.075)"]
EVENT_STYLE = {
    "Mud Loss": ("#2a78d6", "triangle-down"), "Kick": ("#e34948", "triangle-up"),
    "Stuck Pipe": ("#4a3aa7", "x"), "Torque Spike": ("#eb6834", "diamond"),
    "Wellbore Instability": ("#1baf7a", "square"), "Bit Problem": ("#eda100", "circle"),
    "Cementing Issue": ("#e87ba4", "cross"), "Pressure/Gas Anomaly": ("#008300", "star"),
    "Other": (MUTED, "circle-open"),
}
TRACKS = [("ROP", "ROP (m/hr)"), ("WOB", "WOB (klbs)"), ("RPM", "RPM"), ("torque", "Torque"),
          ("SPP", "SPP (psi)"), ("ECD", "ECD (ppg)"), ("gas", "Gas (%)"), ("pit_volume", "Pit (bbl)"),
          ("flow_out_pct", "Flow out (%)")]


def _layout(fig, height):
    fig.update_layout(height=height, margin=dict(l=10, r=10, t=50, b=10), plot_bgcolor="white",
                      paper_bgcolor="white", font=dict(size=11), hoverlabel=dict(bgcolor="white"))
    fig.update_xaxes(showgrid=True, gridcolor=GRID, zeroline=False)
    fig.update_yaxes(showgrid=True, gridcolor=GRID, zeroline=False)
    return fig


def parameter_tracks(stream: pd.DataFrame, depth: float, tops_w: pd.DataFrame | None = None,
                     zones: pd.DataFrame | None = None, incidents: pd.DataFrame | None = None,
                     up: float = 250, down: float = 200) -> go.Figure:
    """Depth-based log tracks: everything drilled so far, formation bands, bit line,
    detected incidents (amber) and formation-aligned offset-event zones ahead (red)."""
    view = stream[(stream.depth >= depth - up) & (stream.depth <= depth)]
    tracks = [(c, l) for c, l in TRACKS if c in stream]
    fig = make_subplots(rows=1, cols=len(tracks), shared_yaxes=True, horizontal_spacing=0.012,
                        subplot_titles=[l for _, l in tracks])
    for i, (col, label) in enumerate(tracks, start=1):
        if view[col].notna().any():
            fig.add_trace(go.Scatter(x=view[col], y=view.depth, mode="lines", line=dict(color=SERIES, width=1.4),
                                     name=label, showlegend=False,
                                     hovertemplate=f"{label}: %{{x:.2f}}<br>Depth %{{y:.1f}} m<extra></extra>"),
                          row=1, col=i)
        else:
            xref = "x domain" if i == 1 else f"x{i} domain"
            fig.add_annotation(text="Not<br>available", xref=xref, yref="y domain", x=0.5, y=0.5,
                               showarrow=False, font=dict(color=MUTED))
    if tops_w is not None and len(tops_w):
        for j, t in enumerate(tops_w.sort_values("top_depth_m").itertuples()):
            if t.bottom_depth_m >= depth - up and t.top_depth_m <= depth + down:
                fig.add_hrect(y0=t.top_depth_m, y1=t.bottom_depth_m, fillcolor=BAND[j % 2], line_width=0,
                              layer="below", row="all", col="all")
                fig.add_hrect(y0=t.top_depth_m, y1=t.top_depth_m, line_width=0, row=1, col=len(tracks),
                              annotation_text=f"{t.formation} top", annotation_position="bottom right",
                              annotation_font=dict(size=10, color=MUTED))
    if incidents is not None and len(incidents):
        for r in incidents.itertuples():
            fig.add_hrect(y0=r.top_m, y1=min(r.bottom_m, depth), fillcolor=STATUS["warning"], opacity=0.35,
                          line_width=0, row="all", col="all")
    if zones is not None and len(zones):
        for z in zones.itertuples():
            fig.add_hrect(y0=z.aligned_depth - 6, y1=z.aligned_depth + 6, fillcolor=STATUS["critical"],
                          opacity=0.15, line_width=0, row="all", col="all")
            fig.add_hrect(y0=z.aligned_depth, y1=z.aligned_depth, line_width=0, row=1, col=1,
                          annotation_text=f"{z.event_type} ({z.well_id})", annotation_position="top left",
                          annotation_font=dict(size=9, color=STATUS["critical"]))
    fig.add_hline(y=depth, line_color=STATUS["critical"], line_width=2, row="all", col="all")
    fig.update_yaxes(range=[depth + down, depth - up], title_text="Measured depth (m)", row=1, col=1)
    return _layout(fig, 640)


def events_by_depth(corr: pd.DataFrame, depth: float, tops_w: pd.DataFrame | None, cfg: dict) -> go.Figure:
    """Offset-well events placed at their formation-aligned depth in the active well."""
    fig = go.Figure()
    if corr is None or corr.empty:
        fig.add_annotation(text="No events in the selected relevant offset wells", x=0.5, y=0.5,
                           xref="paper", yref="paper", showarrow=False)
        return _layout(fig, 520)
    order = corr.sort_values("relevance_score", ascending=False).well_id.unique().tolist()
    for et, g in corr.groupby("event_type"):
        colour, symbol = EVENT_STYLE.get(et, EVENT_STYLE["Other"])
        fig.add_trace(go.Scatter(
            x=g.well_id, y=g.aligned_depth, mode="markers", name=et,
            marker=dict(color=colour, symbol=symbol, size=13, line=dict(color="white", width=2)),
            customdata=g[["depth", "formation", "severity", "status", "source_document", "relevance_score"]].values,
            hovertemplate=("<b>%{x}</b> · " + et + "<br>Aligned depth %{y:.0f} m (actual %{customdata[0]:.0f} m)"
                           "<br>%{customdata[1]} · %{customdata[2]}<br>%{customdata[3]} · relevance "
                           "%{customdata[5]:.2f}<br>Source: %{customdata[4]}<extra></extra>")))
    if tops_w is not None:
        for t in tops_w.itertuples():
            fig.add_hline(y=t.top_depth_m, line_color=MUTED, line_dash="dot", line_width=1,
                          annotation_text=f"{t.formation} top", annotation_position="right",
                          annotation_font=dict(size=10, color=MUTED))
    fig.add_hrect(y0=depth - cfg["depth_tolerance_m"], y1=depth + cfg["depth_tolerance_m"],
                  fillcolor=STATUS["critical"], opacity=0.08, line_width=0)
    fig.add_hline(y=depth, line_color=STATUS["critical"], line_width=2,
                  annotation_text=f"bit {depth:,.0f} m (±{cfg['depth_tolerance_m']:.0f} m)",
                  annotation_position="left", annotation_font=dict(color=STATUS["critical"]))
    lo = min(corr.aligned_depth.min(), depth) - 60
    hi = max(corr.aligned_depth.max(), depth) + 60
    fig.update_xaxes(categoryorder="array", categoryarray=order, title_text="Offset well (most relevant first)")
    fig.update_yaxes(range=[hi, lo], title_text="Depth in ACTIVE well coordinates (m, formation-aligned)")
    fig.update_layout(legend=dict(orientation="h", y=-0.2))
    return _layout(fig, 560)


def relevance_bars(rel: pd.DataFrame, weights: dict) -> go.Figure:
    """Stacked contribution of each component (weight x score) to the relevance score."""
    comps = ["spatial", "formation", "depth", "parameter", "geology"]
    colours = ["#2a78d6", "#eb6834", "#1baf7a", "#4a3aa7", "#eda100"]
    top = rel.head(10)
    fig = go.Figure()
    for c, col in zip(comps, colours):
        vals = top[f"{c}_score"].astype(float).fillna(0) * weights[c] / sum(weights.values())
        fig.add_trace(go.Bar(y=top.well_id, x=vals, orientation="h", name=f"{c} ({weights[c]:.0%})",
                             marker=dict(color=col, line=dict(color="white", width=2)),
                             hovertemplate=f"%{{y}} · {c}: score %{{customdata:.2f}} → %{{x:.3f}}<extra></extra>",
                             customdata=top[f"{c}_score"].astype(float).fillna(0)))
    fig.update_layout(barmode="stack", legend=dict(orientation="h", y=-0.15))
    fig.update_yaxes(autorange="reversed")
    fig.update_xaxes(range=[0, 1], title_text="Offset relevance score (0-1)")
    return _layout(fig, 380)
