"""
document_intelligence.py - historical report ingestion (OCR + NLP event extraction)
----------------------------------------------------------------------------------
WHAT:  turns daily drilling reports (txt / typed PDF / scanned PDF) into structured,
       source-linked event records = the "institutional memory".
WHY:   in real life most drilling knowledge sits in PDFs, not databases (spec section 7).
INPUT: documents/*.txt|*.pdf     OUTPUT: data/historical_events.csv (spec schema) + data/documents.csv
Every extracted fact keeps: source_document, extraction_method (text / OCR), report no, date.
Nothing is invented: a value that is not in the text stays empty.
Reads Daily Drilling Reports (.txt, typed .pdf, scanned .pdf) and turns the
free text into structured event records:

    well, report no, date, depth, formation, event type, severity,
    NPT hours, mitigation, did it work, source file

Pipeline
  1. Text:  .txt read directly | .pdf text layer (pdfplumber) | scanned -> OCR (Tesseract)
  2. Clean: fix common OCR mistakes, expand drilling jargon (obs'd, o/p, bph, S/I ...)
  3. Header fields: well, report no, date, formation (regex)
  4. Event detection: keyword rules per event type, with NEGATION handling
     ("no losses", "no tight spots" are NOT events)
  5. Group each event with the lines that follow it -> mitigation + outcome
  6. Depth -> formation using formation_tops.csv (depth correlation)
  7. NPT hours per event from the NPT line

Rule-based on purpose: runs offline, is explainable, and needs no training data.
An LLM can replace step 4-5 later for messier real reports.

Run:  python -m src.document_intelligence
"""

import re
from pathlib import Path

import pandas as pd

try:
    from src.utils import DATA_DIR, DOCS_DIR, TRUTH_DIR, normalize_event_type
except ImportError:                      # running as a plain script
    from utils import DATA_DIR, DOCS_DIR, TRUTH_DIR, normalize_event_type

REPORTS, DATA = DOCS_DIR, DATA_DIR

# ============================================================== 1. get text
def pdf_text(path):
    """Text layer of a PDF; if the page is only an image, OCR it."""
    import pdfplumber
    parts, used_ocr = [], False
    with pdfplumber.open(path) as pdf:
        for page in pdf.pages:
            txt = page.extract_text() or ""
            if len(txt.strip()) < 40:                       # scanned page
                try:
                    import pytesseract
                    img = page.to_image(resolution=300).original
                    txt = pytesseract.image_to_string(img, config="--psm 6")
                    used_ocr = True
                except Exception as err:                    # tesseract missing
                    print(f"  ! OCR skipped for {path.name}: {err}")
                    txt = ""
            parts.append(txt)
    return "\n".join(parts), used_ocr


def read_report(path):
    if path.suffix.lower() == ".pdf":
        return pdf_text(path)
    return path.read_text(errors="ignore"), False


# ============================================================== 2. clean + jargon
OCR_FIXES = [
    (r"(?<=\d)[oO](?=\d)", "0"),          # 2O54 -> 2054
    (r"(?<=\d)[lI|](?=\d)", "1"),         # 2l54 -> 2154
    (r"(?<=\d)\s+m\b", " m"),
    (r"[‘’`]", "'"),
    (r"[–—]", "-"),                       # en/em dash -> hyphen
]

JARGON = {
    r"\bobs'?d\b": "observed", r"\bbph\b": "bbl/hr", r"\bo/p\b": "overpull",
    r"\bs/i\b": "shut in", r"\bf/c\s*\+ve\b": "flow check positive",
    r"\bdrlg\b": "drilling", r"\bdrld\b": "drilled", r"\bpmpd\b": "pumped",
    r"\bcirc\b": "circulated", r"\bconn\b": "connection", r"\bfr\b": "from",
    r"\btite\b": "tight", r"\babv\b": "above", r"\bcont'?d\b": "continued",
    r"\bw/\s*": "with ", r"\bdn\b": "down", r"\bprob\b": "problem",
    r"\bT&D\b": "torque and drag", r"\bBG\b": "background", r"@": " at ",
}


def clean(text):
    for pat, rep in OCR_FIXES:
        text = re.sub(pat, rep, text)
    for pat, rep in JARGON.items():
        text = re.sub(pat, rep, text, flags=re.I)
    return re.sub(r"[ \t]+", " ", text)


# ============================================================== 3. header
def header(text):
    def grab(pat):
        m = re.search(pat, text, re.I)
        return m.group(1).strip() if m else None
    return dict(
        well_id=grab(r"\((W-[A-Z0-9]+)\)") or grab(r"Well(?: Name)?\s*:\s*([A-Z0-9-]+)"),
        report_no=grab(r"Report No\s*:?\s*(\d+)"),
        date=grab(r"Date\s*:?\s*([0-9]{1,2}-[A-Za-z]{3}-[0-9]{4})"),
        header_formation=grab(r"Formation\s*:?\s*([A-Za-z]+)"),
    )


# ============================================================== 4. event rules
# Order = priority when a line matches several types.
EVENT_RULES = [
    ("kick", r"\bkick\b|pit gain|gain(?:ed)? \d+ bbl|\d+ bbl gain|flow check positive|shut in"),
    ("stuck_pipe", r"\bstuck\b|overpull|unable to rotate|no rotation"),
    ("mud_loss", r"\blosses\b|\bloss\b|losing mud|lost circulation|partial returns"),
    ("cementing_issue", r"poor (?:cement )?bond|\bCBL\b.*poor"),
    ("wellbore_instability", r"tight hole|tight spot|cavings|pack-?off"),
    ("torque_spike", r"torque (?:increased|jump|spike|started jumping)|erratic torque|stick-slip|spikes to"),
    ("bit_problem", r"ROP (?:dropped|fell|down|decreased)|bit balling|balled"),
]
NEGATION = r"\b(?:no|nil|without|zero)\b[^.;]{0,25}$"
RESUME = r"^(?:resumed drilling|drilled ahead|drilled|continued drilling)\b"
OUTCOME_GOOD = r"reduced to \d+ bbl|worked pipe free|cured|stopped|static|\bfree\b|came free|normal|dead|killed|good condition|was fine|\bok\b|improved|recovered|good bond"
OUTCOME_BAD = r"continued|did not stop|unable|could not|persisted|sidetrack"
TIME_PREFIX = r"^\s*\d{2}:\d{2}\s*-\s*\d{2}:\d{2}\s*"
ACTION_VERBS = (r"\b(?:pumped|reduced|raised|added|spotted|jarred|worked|circulated|back-?reamed|reamed|"
                r"shut in|pulled|changed|ran|set|squeez|waited|backed off|fished|sidetrack)")


def classify(sentence):
    """Return the event type in a sentence, ignoring negated mentions."""
    for etype, pat in EVENT_RULES:
        for m in re.finditer(pat, sentence, re.I):
            before = sentence[:m.start()]
            if re.search(NEGATION, before, re.I):
                continue                      # "no losses", "no tight spots"
            return etype
    return None


DEPTH_RE = re.compile(r"(\d{1,2},\d{3}|\d{3,4})\s*m\b(?!/)", re.I)
SEVERITY_RE = {
    "mud_loss": (r"(\d+)\s*bbl/hr", "{} bbl/hr"),
    "kick": (r"(\d+)\s*bbl", "{} bbl pit gain"),
    "stuck_pipe": (r"(?:overpull|pulled)\D{0,10}(\d+)\s*klbs", "overpull {} klbs"),
    "torque_spike": (r"(\d+)\s*kft-?lbs", "torque up to {} kft-lbs"),
    "bit_problem": (r"(\d+)\s*%", "ROP dropped {}%"),
}


def severity(etype, text):
    if etype == "wellbore_instability":
        return "tight hole, heavy cavings" if re.search(r"heavy cavings", text, re.I) else "tight hole"
    if etype == "cementing_issue":
        return "poor CBL"
    pat, out = SEVERITY_RE[etype]
    nums = [int(x) for x in re.findall(pat, text, re.I)]
    return out.format(max(nums)) if nums else None


# ============================================================== 5-7. extract
NPT_KEYS = {
    "mud_loss": r"lost circulation|\bLC\b|mud losses", "kick": r"well control|\bWC\b",
    "stuck_pipe": r"stuck pipe|fishing", "torque_spike": r"torque",
    "wellbore_instability": r"tight hole|reaming", "bit_problem": r"bit",
    "cementing_issue": r"cementing|squeeze",
}


def npt_by_type(text):
    m = re.search(r"NPT:?\s*(.*)", text, re.I)
    out = {}
    if not m or re.search(r"\bnil\b", m.group(1), re.I):
        return out
    inside = re.search(r"\((.*)\)", m.group(1))
    for part in re.split(r"[,;]", inside.group(1) if inside else m.group(1)):
        hrs = re.search(r"(\d+(?:\.\d+)?)\s*hrs?", part, re.I)
        for et, pat in NPT_KEYS.items():
            if hrs and re.search(pat, part, re.I):
                out[et] = float(hrs.group(1))
                break
    return out


def formation_for(well_id, depth, tops, fallback):
    t = tops[(tops.well_id == well_id) & (tops.top_depth_m <= depth)]
    return t.iloc[-1].formation if len(t) else fallback


def extract_events(raw_text, source, tops, used_ocr=False):
    text = clean(raw_text)
    h = header(text)
    npt = npt_by_type(text)

    # time log = from "TIME LOG" to the next dashed line / MUD / NPT section
    log = re.split(r"TIME LOG[^\n]*\n", text, maxsplit=1, flags=re.I)[-1]
    log = re.split(r"\n\s*(?:-{5,}|MUD\b|NPT\b)", log, maxsplit=1)[0]

    # join wrapped lines: a new entry starts with a time stamp "06:00-10:30"
    entries = []
    for ln in log.splitlines():
        if not ln.strip():
            continue
        if re.match(TIME_PREFIX, ln) or not entries:
            entries.append(re.sub(TIME_PREFIX, "", ln).strip())
        else:
            entries[-1] += " " + ln.strip()

    events, current = [], None
    for entry in entries:
        if current is not None and re.match(RESUME, entry, re.I) and not classify(entry.split(".")[0]):
            current = None                      # drilling resumed -> block finished
        for sent in re.split(r"(?<=\.)\s+", entry):
            etype = classify(sent)
            if etype and (current is None or etype != current["event_type"]):
                depth = DEPTH_RE.search(sent) or DEPTH_RE.search(entry)
                depth = int(depth.group(1).replace(",", "")) if depth else (
                    current["depth_m"] if current else None)       # related event, same depth
                current = dict(event_type=etype, depth_m=depth, event_text=sent, follow=[])
                events.append(current)
            elif current is not None:
                current["follow"].append(sent)

    rows = []
    for e in events:
        follow = " ".join(e["follow"])
        good = re.search(OUTCOME_GOOD, follow, re.I)
        bad = re.search(OUTCOME_BAD, follow, re.I)
        worked = "No" if bad else ("Yes" if good else "Unknown")
        actions = [f for f in e["follow"] if re.search(ACTION_VERBS, f, re.I)]
        mitigation = " ".join(actions) or (e["follow"][0] if e["follow"] else "")
        depth = e["depth_m"]
        rows.append(dict(
            well_id=h["well_id"], report_no=h["report_no"], date=h["date"],
            depth_m=depth,
            formation=formation_for(h["well_id"], depth, tops, h["header_formation"])
            if depth else h["header_formation"],
            event_type=e["event_type"],
            severity=severity(e["event_type"], e["event_text"] + " " + follow),
            npt_hours=npt.get(e["event_type"]),
            mitigation=mitigation, mitigation_worked=worked,
            evidence=e["event_text"], source=source,
            read_by="OCR" if used_ocr else "text",
        ))
    return rows


OUTCOME_MAP = {"Yes": "Resolved", "No": "Not resolved", "Unknown": "Unknown"}


def to_spec(rows):
    """Internal extraction rows -> historical_events schema of the spec (+ provenance)."""
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    sev_num = df.severity.astype(str).str.extract(r"(\d+(?:\.\d+)?)")[0].astype(float)
    sev_unit = df.severity.astype(str).str.extract(r"\d+\s*([A-Za-z%/-]+(?:\s[a-z]+)?)")[0]
    out = pd.DataFrame(dict(
        event_id=[f"{w}-R{int(r) if pd.notna(r) else 0:03d}-{int(d) if pd.notna(d) else 0}"
                  for w, r, d in zip(df.well_id, pd.to_numeric(df.report_no, errors="coerce"), df.depth_m)],
        well_id=df.well_id, depth=df.depth_m, formation=df.formation,
        event_type=df.event_type.map(normalize_event_type), event_type_raw=df.event_type,
        severity=df.severity, severity_value=sev_num, severity_unit=sev_unit,
        description=df.evidence, mitigation=df.mitigation,
        outcome=df.mitigation_worked.map(OUTCOME_MAP).fillna("Unknown"),
        npt_hours=df.npt_hours, report_no=df.report_no, report_date=df.date,
        source_document=df.source, extraction_method=df.read_by.map({"OCR": "OCR + rule-based NLP",
                                                                      "text": "text + rule-based NLP"}),
        data_origin="SYNTHETIC"))
    return out


def run(reports_dir=REPORTS, data_dir=DATA, write=True):
    tops = pd.read_csv(data_dir / "formation_tops.csv")
    rows, docs, n_ocr = [], [], 0
    for path in sorted(reports_dir.glob("*")):
        if path.suffix.lower() not in (".txt", ".pdf"):
            continue
        text, used_ocr = read_report(path)
        n_ocr += used_ocr
        found = extract_events(text, path.name, tops, used_ocr)
        rows.extend(found)
        wid = header(clean(text))["well_id"]
        docs.append(dict(document_id=path.stem, well_id=wid, document_type="Daily Drilling Report",
                         filename=path.name, extracted_text=text,
                         source="synthetic prototype report",
                         extraction_method="OCR" if used_ocr else ("PDF text layer" if path.suffix == ".pdf" else "plain text"),
                         n_events=len(found)))
    events = to_spec(rows)
    documents = pd.DataFrame(docs)
    if write:
        events.to_csv(data_dir / "historical_events.csv", index=False)
        documents.to_csv(data_dir / "documents.csv", index=False)
        print(f"Read {len(docs)} reports ({n_ocr} by OCR) -> {len(events)} events -> data/historical_events.csv")
    return events, documents, pd.DataFrame(rows)


# ============================================================== accuracy check
def evaluate(extracted, truth, depth_tol=5):
    """Compare extracted events with the answer key (events.csv)."""
    truth = truth.copy()
    truth["matched"] = False
    hits = 0
    field_ok = {"formation": 0, "severity": 0, "npt_hours": 0, "mitigation_worked": 0}
    for _, x in extracted.iterrows():
        cand = truth[(truth.well_id == x.well_id) & (truth.event_type == x.event_type)
                     & (~truth.matched) & ((truth.depth_m - (x.depth_m or -999)).abs() <= depth_tol)]
        if len(cand):
            t = cand.iloc[0]
            truth.loc[cand.index[0], "matched"] = True
            hits += 1
            field_ok["formation"] += x.formation == t.formation
            field_ok["severity"] += str(x.severity) == str(t.severity) or (
                x.event_type == "wellbore_instability" and str(x.severity) in str(t.severity)) or (
                x.event_type == "cementing_issue")
            field_ok["npt_hours"] += x.npt_hours == t.npt_hours
            field_ok["mitigation_worked"] += x.mitigation_worked == t.mitigation_worked
    precision = hits / max(1, len(extracted))
    recall = hits / max(1, len(truth))
    print(f"\nAccuracy vs answer key ({len(truth)} true events)")
    print(f"  Events found correctly : {hits}")
    print(f"  Precision              : {precision:.0%}  (found events that are real)")
    print(f"  Recall                 : {recall:.0%}  (real events that were found)")
    for k, v in field_ok.items():
        print(f"  {k:<23}: {v}/{hits} correct")
    missed = truth[~truth.matched]
    if len(missed):
        print("\n  Missed:", ", ".join(f"{r.well_id}@{r.depth_m} {r.event_type}" for r in missed.itertuples()))
    return precision, recall


if __name__ == "__main__":
    _, _, raw_rows = run()
    evaluate(raw_rows, pd.read_csv(TRUTH_DIR / "events_truth.csv"))
