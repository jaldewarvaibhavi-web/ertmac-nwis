# How NWIS works — module by module (for presenting)

Each module answers one question in the NWIS workflow. For every module: **what it does · why it is
needed · input · processing · output · how it connects · how it keeps NWIS unique.**

---

## 1. `src/ingestion.py` — "Can we trust the data?"  (Phases 1–2)
* **What:** loads CSV / Excel / JSON files, renames columns to one standard (e.g. `Pump Press (psi)` → `SPP`),
  converts feet to metres, checks every value, and stores everything in a SQLite database.
* **Why:** wrong numbers give wrong alerts. The spec forbids silently deleting data.
* **Input:** `data/*.csv`, the FORGE log.  **Output:** `data/nwis.db` + a `qc_issues` table.
* **Processing:** required columns → numeric checks against plausible ranges → duplicates → depth order
  → formation / event-name normalisation (original kept) → a `qc_flags` column on every row.
* **Connects:** every other module reads the database.
* **Uniqueness:** provenance (`source_file`, `data_origin`) is attached here, so every later fact can be traced.
* **Real finding:** on the FORGE log it flagged 48 impossible ROP values, 3 flow spikes, 742 broken
  wellhead-pressure readings, 33 duplicates and 2 depth reversals — all kept and shown, none deleted.

## 2. `src/document_intelligence.py` — "What does the institutional memory say?"
* **What:** reads daily drilling reports (text, PDF, scanned PDF) and turns sentences like
  *"@ 2654m obs'd losses 44 bph"* into a structured event row.
* **Why:** most drilling experience lives in reports, not in databases.
* **Processing:** PDF text or **OCR** (Tesseract) → fix OCR mistakes → expand rig slang (obs'd, o/p, bph, S/I)
  → detect events with keyword rules and **negation** ("no losses" is not an event) → depth → formation from
  formation tops → severity, NPT, what was done, whether it worked.
* **Output:** `historical_events.csv` with `source_document` and `extraction_method` on every row.
* **Uniqueness:** nothing is invented — a value that is not in the text stays empty.

## 3. `src/preprocessing.py` — "Make the numbers comparable"
* Blanks only the flagged *value* (not the whole row), removes duplicates for modelling, sorts by depth.
* Formation helpers: formation at a depth, metres below its top.
* StandardScaler: puts ROP (m/hr), SPP (psi) and gas (%) on the same scale so none dominates.

## 4. `src/geo_matching.py` + `src/similarity.py` — "Which offset wells matter now?"  (Phases 3–4, 7)
* **What:** radius search, then the **Offset Relevance Score** (0–1):
  `30% spatial + 20% formation + 20% depth + 20% parameter + 10% geology` — prototype weights, adjustable.
* **Spatial:** 1 − distance / radius.
* **Formation:** does the offset well contain the current and upcoming formations, and is the current
  formation similarly thick?
* **Depth:** is the current formation top at a similar depth (structure), and did the offset reach the
  equivalent position?
* **Parameter:** standardised average drilling parameters in the shared, already-drilled formations.
* **Geology:** porosity / caliper in shared formations (fallback: formation sequence if missing).
* **Output:** ranked table with every component and a plain-language reason.
* **Uniqueness:** not "nearest well" — a well 4 km away with the same structure can outrank one 2 km away.

## 5. `src/event_engine.py` — "Where are the old problems relative to my bit?"  (Phase 5)
* **What:** moves every offset event into the active well's coordinates:
  `aligned depth = active formation top + (event depth − offset formation top)`.
* Marks each event **AT CURRENT DEPTH** (within ± tolerance), **AHEAD** (within look-ahead) or **PASSED**.
* **Why:** Barail is at 2,620 m in W-102 but 2,610 m in ACTIVE-01; raw depth alone mis-matches events.
* Raw depth difference is shown too, for transparency.

## 6. `src/anomaly.py` — "Is something unusual happening in this well right now?"  (Phase 6)
* Compares each reading with the well's own last ~30 m **in the same formation**: flow out / pit volume
  (losses, kicks), torque ratio (stuck pipe), pump pressure vs pump rate (pack-off / washout), ROP per WOB
  (bit wear), plus an Isolation Forest score. Signals are merged into incidents with High / Medium / Low confidence.
* Uses only readings above the bit (it behaves like a live system).
* It is the only analysis that works on the real FORGE well (no labels, no neighbours).

## 7. `src/risk_model.py` — "Can past wells predict this one?"  (Phases 8–9)
* Random Forest per event type — **only** if ≥ 8 events in ≥ 5 wells; otherwise it says
  *"Insufficient labeled data for reliable ML training."*
* **Leave-one-well-out validation:** the tested well and its events are hidden completely.
* Reports precision, recall, F1, confusion matrix, ROC-AUC, PR-AUC, FPR, FNR and an evidence-only baseline.
* A model is used in alerts only if it passes the gate (PR-AUC ≥ 0.3 and F1 ≥ 0.4). Mud Loss passes;
  Wellbore Instability is trained but fails and is not used.

## 8. `src/explainability.py` — "What is the risk, and WHY?"  (Phase 10)
* Combines: offset evidence (share of relevant wells with this event near this formation position),
  proximity, similarity of current parameters to those wells just before their events, geology similarity,
  live anomaly, and the validated ML score.
* Produces a **Low / Moderate / Elevated risk indication** with a "Why?" list, the supporting events with
  sources, and "what was done historically" (not instructions), plus the decision note.

## 9. `src/rag.py` — "Explain it in plain words, without making things up"  (Phase 11)
* Retrieves report excerpts (TF-IDF, or ChromaDB if enabled).
* Answers the standard questions directly from the analytics; Claude (if a key is set) only rewords the
  supplied evidence and must cite sources. Answers mentioning wells outside the evidence are rejected.

## 10. `src/engine.py` — the conductor
Loads the database once, caches models and analyses, and gives the dashboard, API and demo the same answer.

## 11. `app/dashboard.py`, `app/charts.py` — the engineer's screen  (Phase 12, 14)
Streamlit + Plotly + Folium. Banner showing data origin, active-well panel, alert cards with WHY and buttons
for wells / events / source reports / Ask NWIS, map with relevance colours and event counts, depth-track
charts with offset-event zones ahead of the bit, validation and data-quality tab, and **Demo Mode**.

## 12. `api/main.py`, `api/services.py` — for other systems  (Phase 13)
REST endpoints returning JSON; the logic sits in `services.py` so it is unit-tested without a server.

## 13. `tests/` — proof it behaves  (Phase 15)
29 tests, e.g. values are flagged not dropped; the held-out well never leaks into training; weights and
tolerance are configurable; alerts always carry sources and never say "will occur"; missing FORGE
parameters stay missing; offline answers mention only wells in the evidence.
