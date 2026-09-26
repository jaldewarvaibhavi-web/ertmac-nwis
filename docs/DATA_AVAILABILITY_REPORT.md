# eRTMAC-NWIS — Data Availability Report

*Prepared before implementation, as required by §36 of the master development prompt.*
*Generated from the files actually present on 26-Sep-2026. Nothing below is assumed from the prompt.*

> **SYNTHETIC DATA — FOR PROTOTYPE DEMONSTRATION ONLY.** No Oil India / eRTMAC data is present.

---

## 1. Files inspected

| # | File | Format | Records | Wells | Depth range | Origin |
|---|---|---|---|---|---|---|
| 1 | `data/wells.csv` | CSV | 16 | 15 offset + 1 active (`W-ACT`) | TD 3,292–3,494 m | **Synthetic** |
| 2 | `data/formation_tops.csv` | CSV | 96 | 16 | 0–3,494 m | **Synthetic** (real Upper Assam formation *names*) |
| 3 | `data/drilling_data.csv` | CSV | 51,119 (1 per m) | 15 | 1–3,494 m | **Synthetic** |
| 4 | `data/events.csv` | CSV | 44 | 14 | 1,008–3,419 m | **Synthetic** (answer key) |
| 5 | `data/extracted_events.csv` | CSV | 44 | 14 | 1,008–3,419 m | NLP output from file 7 |
| 6 | `data/active_well_stream.csv` | CSV | 3,382 (1 per m) | 1 (`W-ACT`) | 1–3,382 m | **Synthetic** eRTMAC-like stream |
| 7 | `reports/*` | 64 TXT, 6 PDF, 4 scanned PDF | 74 reports | 15 | — | **Synthetic** DDRs |
| 8 | `Well_58-32_processed_pason_log.csv` | CSV (uploaded as `.xls`) | 7,311 | 1 | 26–2,297 m | **Real, public** (Utah FORGE 58-32) |
| 9 | `data_real/forge_58-32_drilling.csv` | CSV | 7,278 | 1 | 26–2,297 m | Converted from file 8 |
| 10 | `data_real/wells_real.csv` | CSV | 1 | 1 | — | Location **approximate** |
| 11 | `data_real/forge_58-32_suspected_events.csv` | CSV | 67 | 1 | — | Detector output (**not** labelled events) |
| 12 | `SM_GroundwaterWellDrillingReport2008.pdf` | Scanned PDF, 43 pages | 8 shallow borings (B-1…B-8) | — | 4–58 m | Real, public (Colorado environmental site) |

---

## 2. Parameter availability matrix

Legend: ✅ Available · ⚠️ Available with caveat · 🔧 Can be derived · 🧪 Requires synthetic prototype data · 🔒 Requires future OIL integration · ❌ Missing

### 2.1 Well master (`well_master`)

| Field | Synthetic field | FORGE 58-32 | Status / action |
|---|---|---|---|
| well_id | ✅ | ✅ | Available |
| latitude, longitude | ✅ | ⚠️ approximate (38.50, −112.90) | Replace with official FORGE GPS file |
| field | ❌ | ❌ | 🔧 From source metadata: "Synthetic Assam-style field" / "Utah FORGE" |
| basin | ❌ | ❌ | 🔧 From metadata: "Assam-Arakan (synthetic analogue)" / geothermal site — not a petroleum basin |
| total_depth | ✅ | ✅ (2,297 m) | Available |
| drilling_year | ✅ `spud_year` | ⚠️ 2017 from public documentation | Verify from End-of-Well report |
| well_status | ✅ | ✅ | Available |
| formation | ⚠️ per-interval in `formation_tops`, not one value per well | ❌ | Keep as interval table; "formation at TD" 🔧 derivable |

### 2.2 Geology (`well_geology`)

| Field | Synthetic | FORGE 58-32 | Status |
|---|---|---|---|
| depth, formation | ✅ as formation tops (interval) | ❌ | FORGE tops are in the End-of-Well report → 🔧 manual entry |
| porosity | ❌ | ❌ **not in this log** | 🧪 synthetic, **or** download FORGE 58-32 well logs (LAS, GDR submission 1006) |
| caliper | ❌ | ❌ **not in this log** | Same as porosity |

**Consequence:** the 10 % *geological similarity* component cannot use porosity/caliper today.
Fallback (documented): similarity of formation sequence and thickness, derived from formation tops.

### 2.3 Historical drilling parameters (`historical_parameters`)

| Field | Synthetic offsets | FORGE 58-32 | Status |
|---|---|---|---|
| depth | ✅ 1 m step | ✅ ~0.31 m step | Available |
| formation | ✅ | ❌ | FORGE: "Unknown" until tops added |
| ROP | ✅ m/hr | ✅ m/hr | Available |
| WOB | ✅ klbs | ✅ klbs | Available |
| RPM | ✅ | ✅ | Available |
| torque | ✅ kft-lbs | ⚠️ **surface torque in psi (hydraulic)** | **Unit inconsistency** — not convertible; use trend/ratio only |
| SPP | ✅ psi | ✅ psi | Available |
| flow_rate | ✅ gpm | ✅ gpm | Available |
| mud_weight | ✅ ppg | ❌ | FORGE: Not available |
| ECD | ❌ | ❌ | 🔧 estimate from MW + annular losses (needs hole geometry) or 🧪 synthetic |
| gas | ❌ | ❌ (only H₂S safety sensors, max 0.78 ppm) | 🧪 synthetic for demo; 🔒 mud-logging gas from OIL |
| *extra:* pit volume | ✅ | ✅ | Kept (key loss/kick signal) |
| *extra:* flow out % | ❌ | ✅ | Kept (best early loss/kick signal) |
| *extra:* hookload | ❌ | ✅ | Kept (overpull / stuck pipe) |
| *extra:* mud temperature | ❌ | ✅ | Kept, not used for risk |

### 2.4 Historical events (`historical_events`)

| Field | Synthetic | FORGE 58-32 | Status |
|---|---|---|---|
| event_id | ✅ in `events.csv`; ❌ in `extracted_events.csv` | ❌ | 🔧 generate from well + report + depth |
| depth, formation | ✅ | ❌ | — |
| event_type | ✅ 7 types | ❌ **no labels** | Normalise to prompt vocabulary (below) |
| severity | ✅ free text (e.g. "44 bbl/hr") | ❌ | Keep original text + parsed number |
| description | ✅ `evidence` sentence from report | ❌ | Available (NLP output) |
| mitigation | ✅ | ❌ | Available |
| outcome | ✅ `mitigation_worked` Yes/No | ❌ | Map Yes → "Resolved", No → "Not resolved" |
| source_document | ✅ report file name | ❌ | Available |
| NPT hours | ✅ | ❌ | Available |

**Event counts (synthetic, 14 wells):** Mud loss 15 · Wellbore instability 11 · Torque spike 6 · Bit problem 5 · Kick 3 · Stuck pipe 3 · Cementing 1.
**Not present at all:** Fishing (appears only inside mitigation text), Casing issue, Pressure/Gas anomaly (as a labelled type).

**FORGE 58-32 has no event labels.** Its 67 "suspected incidents" were produced by our own detector.
They must **not** be stored as historical events; they go to a separate `suspected_incidents` table
until confirmed from the 58-32 daily drilling reports.

### 2.5 Current well stream (`current_well_stream`)

| Field | Synthetic `W-ACT` | FORGE 58-32 (replay) |
|---|---|---|
| timestamp | ✅ (derived from ROP) | ❌ depth-indexed only → replay by depth |
| well_id, depth, formation | ✅ | ✅ / ✅ / ❌ |
| ROP, WOB, RPM, torque, SPP, flow | ✅ | ✅ (torque in psi) |
| mud_weight | ✅ | ❌ |
| ECD, gas | ❌ → 🧪 | ❌ |

UI label (both): **"Synthetic eRTMAC-like current drilling stream"** for W-ACT; **"Replay of a public
drilling log (not eRTMAC)"** for FORGE.

---

## 3. Data quality findings

### Real FORGE log (file 8)
| Check | Finding | Proposed handling (no silent discard) |
|---|---|---|
| Duplicate depths | 33 rows | Flag `dup_depth`; keep first, log the rest |
| Depth going backwards | 2 steps | Flag; sort, keep original order column |
| ft ↔ m consistency | max 0.12 m difference | Rounding in source; use metres, report |
| ROP impossible | 48 rows > 150 m/hr (max 908) | Flag `rop_spike`; exclude from models, show in UI |
| Flow-in spikes | 3 rows > 1,500 gpm (max 3,318) | Flag `flow_spike` |
| RPM > 250 | 4 rows | Flag |
| Zeros in WOB / RPM / flow | 198 / 380 / 69 rows | Pumps off / off bottom → `valid=False`, not errors |
| WH pressure | 742 rows negative (to −1,232 psi) | Sensor offset / not connected → mark column unusable |
| Torque units | hydraulic psi, not kft-lbs | Store as `torque_surface_psi`; never mix with kft-lbs |

⚠️ **Correction to the current prototype:** `forge_converter.py` currently *clips* ROP at 200 m/hr.
That silently changes data and violates §6. It will be changed to **flag, not modify**.

### Synthetic data
- No missing values or duplicates (by construction).
- All events and signatures were generated by our own rules. Any model trained on them partly
  re-learns the generator → results are **"Prototype validation only — not representative of
  field deployment accuracy."**
- `drilling_data.event_label` is a per-metre label derived from the events → usable for well-level
  validation, never for random row splits.

### Groundwater report PDF (file 12)
Scanned (no text layer), environmental boreholes 4–58 m deep in Colorado, qualitative lithology only,
no porosity/caliper numbers. **Not joinable** with any other file. Use only as an OCR test document.

---

## 4. Can the files be joined?

| Join | Key | Result |
|---|---|---|
| wells ↔ formation_tops ↔ drilling_data ↔ events (synthetic) | `well_id` + depth (interval lookup for formation) | ✅ Fully joinable |
| events ↔ extracted_events | `well_id` + event type + depth (±5 m) | ✅ Used to measure NLP accuracy |
| active stream ↔ offset wells | formation name + depth relative to formation top | ✅ Core NWIS correlation |
| FORGE ↔ synthetic field | none (different site, geology, units, well type) | ❌ **Not joinable** — separate real-data validation track |
| FORGE ↔ other FORGE wells | `well_id` + GPS (official file) | 🔧 Possible once those logs/GPS are downloaded |
| Groundwater PDF ↔ anything | — | ❌ |

---

## 5. Supervised-ML feasibility (§12)

| Target | Labelled events | Wells with it | Decision |
|---|---|---|---|
| Mud loss | 15 | 12 | Train Random Forest, leave-one-well-out; label "prototype validation only" |
| Wellbore instability | 11 | 10 | Train, same caveat |
| Torque spike | 6 | 6 | Borderline → report metrics, flag as unreliable |
| Bit problem | 5 | 5 | **Insufficient labeled data for reliable ML training** → rules/similarity |
| Kick | 3 | 3 | **Insufficient** → rules/similarity |
| Stuck pipe | 3 | 3 | **Insufficient** → rules/similarity |
| Cementing | 1 | 1 | **Insufficient** → evidence retrieval only |
| Any event on FORGE | 0 | — | No supervised training; unsupervised anomaly detection only |

---

## 6. Proposed schema mapping

| Target table.column | Source | Transformation |
|---|---|---|
| wells.well_id | wells.csv `well_id`; wells_real.csv | — |
| wells.latitude/longitude | same | FORGE flagged `location_source=APPROX` |
| wells.field / basin | — | from dataset metadata (labelled) |
| wells.total_depth | `total_depth_m` | metres |
| wells.drilling_year | `spud_year` | FORGE from documentation |
| wells.well_status | `status` | normalise to Active / Completed |
| wells.formation | formation_tops | formation at TD (derived) |
| wells.data_origin *(added)* | `data_type` | SYNTHETIC / REAL-PUBLIC — shown in UI |
| geology.(well_id, depth, formation) | formation_tops | one row per formation top; `depth = top_depth_m`, plus `bottom_depth` |
| geology.porosity / caliper | — | NULL → "Not available" (or 🧪 if chosen) |
| historical_parameters.* | drilling_data.csv; forge_58-32_drilling.csv | rename: `depth_md_m→depth`, `rop_m_hr→ROP`, `wob_klbs→WOB`, `torque_kftlbs→torque` (+`torque_unit`), `spp_psi→SPP`, `flow_rate_gpm→flow_rate`, `mud_weight_ppg→mud_weight`; ECD, gas NULL; keep `pit_volume`, `flow_out_pct`, `hookload`; add `qc_flags` |
| events.event_id | generated `{well}-{report}-{depth}` | — |
| events.event_type | extracted_events `event_type` | `mud_loss→Mud Loss`, `kick→Kick`, `stuck_pipe→Stuck Pipe`, `torque_spike→Torque Spike`, `cementing_issue→Cementing Issue`, `wellbore_instability→Other (Wellbore instability)`, `bit_problem→Other (Bit problem)`; original kept in `event_type_raw` |
| events.severity | `severity` | text kept; number parsed to `severity_value` + `severity_unit` |
| events.description | `evidence` | report sentence |
| events.mitigation | `mitigation` | — |
| events.outcome | `mitigation_worked` | Yes→Resolved, No→Not resolved, Unknown→Unknown |
| events.source_document | `source` | + `extraction_method` (text / OCR) |
| events.npt_hours *(added)* | `npt_hours` | — |
| suspected_incidents *(added)* | forge_58-32_suspected_events.csv | kept separate from events |
| current_stream.* | active_well_stream.csv | same renames; `timestamp` kept; ECD, gas NULL |
| documents.* | reports/* | `document_type=DDR`, `extracted_text`, `source=synthetic` |

---

## 7. Gap analysis against the current prototype

| Requirement | Current prototype | Needed |
|---|---|---|
| Nearby wells in radius | ✅ | — |
| Offset **relevance score** (5 components, configurable weights) | ❌ distance only | New `geo_matching.py` + `similarity.py` |
| Formation-aware depth matching | ✅ (position below formation top) | Add configurable ±depth tolerance |
| Current-vs-historical parameter similarity | ❌ (we compare with the well's own recent normal) | StandardScaler + cosine/Euclidean vs offset wells at same formation position |
| Geological similarity | ❌ | Formation-sequence fallback (porosity/caliper missing) |
| Supervised model with leave-one-well-out + PR-AUC etc. | ❌ (unsupervised + rules only) | `risk_model.py` for mud loss & instability only |
| Risk wording ("indication", never "will/expected") | ❌ says "expected" | Rewrite alert text |
| No operational instructions | ⚠️ some advice ("flow check immediately") | Replace with "historically used: …; review with engineering judgment" |
| Ingestion validation (impossible values, dups, units) | ⚠️ partial; ROP clipping | `ingestion.py` with visible QC report |
| SQLite DB | ❌ CSV files | `nwis.db` with the schema above |
| FastAPI endpoints | ❌ | `api/main.py` |
| RAG + Claude explanation | ❌ | `rag.py` (TF-IDF/Chroma) + Claude API with evidence-only prompt, offline fallback |
| Demo mode "Start Simulation" | ⚠️ auto-play exists | Scripted scenario with the 8 steps |
| Provenance on every fact | ⚠️ partial | `source_file`, `extraction_method` everywhere |
| Synthetic banner everywhere | ⚠️ partial | Banner on every page + README + metadata |
| Tests | ❌ | pytest suite |
