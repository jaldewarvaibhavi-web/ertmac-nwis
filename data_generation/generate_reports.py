"""
NWIS - Step 4a: Generate sample Daily Drilling Reports (DDRs)
-------------------------------------------------------------
Real OIL reports are not available, so we write synthetic DDRs from
data/events.csv - the way a company man would write them:
  * 3 different writers (formal / abbreviated field jargon / narrative)
  * normal days with "no losses", "no tight spots" (tests negation handling)
  * some reports saved as typed PDFs and some as SCANNED image PDFs (tests OCR)

The NLP extractor (nlp_extractor.py) must then recover the events from this
text alone, and we check it against events.csv (the answer key).

Run:  python generate_reports.py   -> reports/*.txt, reports/*.pdf
"""

import random
from pathlib import Path

import pandas as pd

BASE = Path(__file__).resolve().parents[1]
DATA, OUT = BASE / "data" / "_truth" / "raw", BASE / "documents"
OUT.mkdir(parents=True, exist_ok=True)
for old in list(OUT.glob("W-*_DDR*")):          # only our generated reports
    old.unlink()

random.seed(7)
wells = pd.read_csv(DATA / "wells.csv")
tops = pd.read_csv(DATA / "formation_tops.csv")
events = pd.read_csv(DATA / "events.csv")
drill = pd.read_csv(DATA / "drilling_data.csv")

WRITERS = ["R.K.", "P.D.", "S.G."]           # fictional company-man initials

# ---------------------------------------------------------------- phrases
# {d} = depth, {n} = severity number, three styles: 0 formal, 1 jargon, 2 narrative
EVENT_TEXT = {
    "mud_loss": [
        "Drilled to {d} m. Observed partial mud losses at {n} bbl/hr. Stopped drilling.",
        "@ {d}m obs'd losses {n} bph. P/U off btm, red. flow.",
        "While drilling at {d} m the well started losing mud at approx {n} bbl/hr.",
    ],
    "kick": [
        "Drilling break at {d} m. Flow check positive, pit gain {n} bbl. Shut in well.",
        "@ {d}m drlg break, F/C +ve, {n} bbl gain in pits. S/I well on BOP.",
        "At {d} m the ROP suddenly increased and the pits gained {n} bbl, so the well was shut in.",
    ],
    "stuck_pipe": [
        "While making connection at {d} m, pipe stuck. Overpull {n} klbs, unable to rotate.",
        "@ {d}m pipe stuck on conn, o/p {n} klbs, no rotation.",
        "At {d} m the string got stuck during a connection; we pulled {n} klbs over string weight.",
    ],
    "torque_spike": [
        "Drilled to {d} m. Torque increased erratically up to {n} kft-lbs.",
        "@ {d}m erratic torque, spikes to {n} kft-lbs, stick-slip.",
        "Near {d} m the torque started jumping, reaching {n} kft-lbs.",
    ],
    "wellbore_instability": [
        "At {d} m encountered tight hole with {cav} on shakers.",
        "@ {d}m tight hole, {cav} on shaker, had to ream.",
        "Around {d} m the hole became tight and we saw {cav} coming over the shakers.",
    ],
    "bit_problem": [
        "From {d} m ROP dropped by {n}% at constant WOB and RPM. Suspected bit balling.",
        "Fr {d}m ROP dn {n}% same WOB/RPM - bit balled?",
        "Starting at {d} m the ROP fell by about {n}% although WOB and RPM were unchanged.",
    ],
    "cementing_issue": [
        "Ran CBL/VDL, 9 5/8\" shoe at {d} m. Poor cement bond observed above shoe.",
        "CBL run, 9 5/8\" shoe @ {d}m - poor bond abv shoe.",
        "The cement bond log run on the 9 5/8\" casing (shoe at {d} m) showed a poor bond.",
    ],
}

OUTCOME = {
    ("mud_loss", "Yes"): ["Losses cured, well static.", "losses cured, static.", "The losses stopped after this."],
    ("mud_loss", "No"): ["Losses continued.", "losses cont'd.", "The losses did not stop."],
    ("kick", "Yes"): ["Well dead, opened BOP.", "well dead, BOP opened.", "The well was killed successfully."],
    ("stuck_pipe", "Yes"): ["Pipe free.", "pipe free.", "The pipe came free."],
    ("stuck_pipe", "No"): ["Unable to free pipe.", "unable to free pipe.", "We could not free the pipe."],
    ("torque_spike", "Yes"): ["Torque back to normal.", "torque normal.", "Torque came back to normal."],
    ("wellbore_instability", "Yes"): ["Hole in good condition after reaming.", "hole OK after ream.",
                                      "After reaming the hole was fine."],
    ("bit_problem", "Yes"): ["ROP improved with new bit.", "ROP OK w/ new bit.", "ROP recovered with the new bit."],
    ("cementing_issue", "Yes"): ["Re-run CBL shows good bond.", "re-run CBL good.", "The repeat CBL showed a good bond."],
}

NPT_NAME = {
    "mud_loss": ["Lost circulation", "LC", "mud losses"],
    "kick": ["Well control", "WC", "well control"],
    "stuck_pipe": ["Stuck pipe", "stuck pipe", "stuck pipe"],
    "torque_spike": ["High torque", "torque", "high torque"],
    "wellbore_instability": ["Tight hole", "tight hole/reaming", "tight hole"],
    "bit_problem": ["Trip for bit", "bit trip", "bit change"],
    "cementing_issue": ["Remedial cementing", "squeeze", "remedial cementing"],
}

NORMAL_LINES = [
    ["No losses observed. Hole in good condition, no tight spots.",
     "No losses. Hole OK, no tite spots. T&D normal.",
     "There were no losses and no tight spots; torque and drag were normal."],
    ["Background gas 0.3%, no gas shows.", "BG gas 0.3%, nil shows.",
     "Background gas stayed around 0.3% with no shows."],
]


def number_in(text, default=0):
    import re
    m = re.search(r"(\d+)", str(text))
    return int(m.group(1)) if m else default


def formation_at(well_id, depth):
    t = tops[(tops.well_id == well_id) & (tops.top_depth_m <= depth)]
    return t.iloc[-1].formation


def params_at(well_id, depth):
    r = drill[(drill.well_id == well_id) & (drill.depth_md_m == int(depth))]
    return r.iloc[0] if len(r) else None


def hole_size(depth, well_id):
    b_top = tops[(tops.well_id == well_id) & (tops.formation == "Barail")].top_depth_m.iloc[0]
    return '17 1/2"' if depth < 800 else ('12 1/4"' if depth < b_top - 20 else '8 1/2"')


def fmt(d):
    return f"{int(d):,}"


def write_report(well, report_no, day_events, style):
    """Build the text of one DDR (day_events may be empty = normal day)."""
    end_depth = int(min(well.total_depth_m, report_no * 70))
    start_depth = max(1, end_depth - 70)
    spud = pd.Timestamp(f"{well.spud_year}-01-15")
    date = (spud + pd.Timedelta(days=int(report_no))).strftime("%d-%b-%Y")
    p = params_at(well.well_id, start_depth + 5)
    form = formation_at(well.well_id, end_depth)
    hs = hole_size(end_depth, well.well_id)

    lines = [
        "NWIS PROTOTYPE - SYNTHETIC DAILY DRILLING REPORT (not a real OIL India record)",
        "-" * 64,
        f"Well: {well.well_name} ({well.well_id})      Report No: {report_no}",
        f"Date: {date}      Rig: E-2000 HP (synthetic)",
        f"Depth @ 06:00: {fmt(end_depth)} m MD      Progress: {end_depth - start_depth} m",
        f"Formation: {form}      Hole size: {hs}",
        "-" * 64,
        "TIME LOG (06:00 - 06:00)",
    ]
    if style == 1:
        drill_line = (f"Drld {hs} hole fr {start_depth} to {{to}}m. WOB {p.wob_klbs:.0f} klbs, "
                      f"RPM {p.rpm:.0f}, SPP {p.spp_psi:.0f} psi, {p.flow_rate_gpm:.0f} gpm.")
    else:
        drill_line = (f"Drilled {hs} hole from {fmt(start_depth)} m to {{to}} m. WOB {p.wob_klbs:.0f} klbs, "
                      f"RPM {p.rpm:.0f}, SPP {p.spp_psi:.0f} psi, flow {p.flow_rate_gpm:.0f} gpm.")

    hour = 6
    def stamp(h_len):
        nonlocal hour
        a, b = hour, min(hour + h_len, 30)
        hour = b
        return f"{a % 24:02d}:00-{b % 24:02d}:00  "

    npt_parts = []
    if not day_events:
        lines.append(stamp(10) + drill_line.format(to=start_depth + 40 if style == 1 else fmt(start_depth + 40)))
        lines.append(stamp(2) + random.choice(NORMAL_LINES[0][style:style + 1]))
        lines.append(stamp(12) + (f"Drilled ahead to {fmt(end_depth)} m. " if style != 1 else f"Drld ahead to {end_depth}m. ")
                     + NORMAL_LINES[1][style])
    else:
        for e in day_events:
            d = int(e.depth_m)
            if e.event_type != "cementing_issue":
                lines.append(stamp(3) + drill_line.format(to=d if style == 1 else fmt(d)))
            n = number_in(e.severity)
            cav = "heavy cavings" if "heavy" in str(e.severity) else "some cavings"
            lines.append(stamp(1) + EVENT_TEXT[e.event_type][style].format(d=fmt(d) if style != 1 else d, n=n, cav=cav))
            mit = e.mitigation.rstrip(".") + "."
            lines.append(stamp(max(1, int(e.npt_hours))) + (mit if style != 1 else mit.replace("Pumped", "Pmpd")
                                                           .replace("circulated", "circ").replace("Circulated", "Circ")))
            outcome = OUTCOME.get((e.event_type, e.mitigation_worked),
                                  ["Problem persisted.", "prob persisted.", "The problem persisted."])[style]
            lines.append(" " * 14 + outcome)
            lines.append(stamp(2) + ("Resumed drilling." if style != 1 else "Resumed drlg."))
            npt_parts.append(f"{NPT_NAME[e.event_type][style]} {e.npt_hours} hrs")
        lines.append(stamp(24) + (f"Drilled ahead to {fmt(end_depth)} m." if style != 1 else f"Drld ahead to {end_depth}m."))

    mw = p.mud_weight_ppg
    lines += [
        "-" * 64,
        f"MUD: WBM KCl-Polymer   MW {mw:.1f} ppg   Pit volume {p.pit_volume_bbl:.0f} bbl",
        ("NPT: " + (f"{sum(e.npt_hours for e in day_events):.1f} hrs (" + ", ".join(npt_parts) + ")"
                    if npt_parts else "Nil")),
        "-" * 64,
        f"Company Man: {WRITERS[style]}",
    ]
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------- build all reports
events["report_no"] = events.source.str.extract(r"#(\d+)").astype(int)
manifest = []
for _, w in wells[wells.status != "Active"].iterrows():
    style = random.randrange(3)             # each well has one company man
    ev_w = events[events.well_id == w.well_id]
    used = set()
    for rno, grp in ev_w.groupby("report_no"):
        txt = write_report(w, int(rno), list(grp.itertuples()), style)
        name = f"{w.well_id}_DDR{int(rno):03d}"
        (OUT / f"{name}.txt").write_text(txt)
        manifest.append(name)
        used.add(int(rno))
    # two normal days per well
    for rno in random.sample([r for r in range(3, int(w.total_depth_m / 70)) if r not in used], 2):
        txt = write_report(w, rno, [], style)
        name = f"{w.well_id}_DDR{rno:03d}"
        (OUT / f"{name}.txt").write_text(txt)
        manifest.append(name)

# ---------------------------------------------------------------- PDFs (typed + scanned)
from PIL import Image, ImageDraw, ImageFilter, ImageFont
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas


def to_typed_pdf(txt_path):
    pdf = txt_path.with_suffix(".pdf")
    c = canvas.Canvas(str(pdf), pagesize=A4)
    c.setFont("Courier", 8.5)
    y = A4[1] - 50
    for line in txt_path.read_text().splitlines():
        c.drawString(40, y, line)
        y -= 12
    c.save()
    txt_path.unlink()


def to_scanned_pdf(txt_path):
    """Render text as a slightly rotated, noisy page image = looks scanned."""
    font = None
    for f in ["/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf",
              "/System/Library/Fonts/Menlo.ttc", "/Library/Fonts/Courier New.ttf"]:
        if Path(f).exists():
            font = ImageFont.truetype(f, 26)
            break
    font = font or ImageFont.load_default()
    img = Image.new("L", (2480, 3508), 245)
    draw = ImageDraw.Draw(img)
    y = 150
    for line in txt_path.read_text().splitlines():
        draw.text((140, y), line, fill=25, font=font)
        y += 40
    img = img.rotate(random.uniform(-0.8, 0.8), fillcolor=245).filter(ImageFilter.GaussianBlur(0.6))
    pdf = txt_path.with_name(txt_path.stem + "_scanned.pdf")
    img.convert("RGB").save(pdf, "PDF", resolution=300)
    txt_path.unlink()


random.shuffle(manifest)
for name in manifest[:6]:
    to_typed_pdf(OUT / f"{name}.txt")
for name in manifest[6:10]:
    to_scanned_pdf(OUT / f"{name}.txt")

print(f"{len(manifest)} reports written to {OUT} "
      f"({len(list(OUT.glob('*.txt')))} txt, 6 typed PDF, 4 scanned PDF)")
