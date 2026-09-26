"""
rag.py - Phase 11: document retrieval + grounded Claude explanations ("Ask NWIS").

WHAT:
  1. RETRIEVAL   historical reports -> chunks -> vector index -> top-k chunks for a question.
                 Backend: TF-IDF (scikit-learn, works offline) by default; set
                 NWIS_VECTOR_BACKEND=chroma to use ChromaDB embeddings if installed.
  2. STRUCTURED  answers the four standard questions straight from the analytics
                 (why this alert / cases near this depth / most relevant wells / mitigations).
  3. CLAUDE      receives ONLY the structured evidence + retrieved chunks and is told to explain
                 nothing else. If no ANTHROPIC_API_KEY is set (or the call fails), an offline
                 answer is written from the same evidence - the app never depends on the LLM.
  4. GROUNDING CHECK  any well ID in Claude's answer that is not in the evidence is reported,
                 and the offline answer is shown instead.
WHY:   Claude is the explanation layer, NOT the risk model (spec section 17). Numbers come from
       the deterministic pipeline; Claude only puts them into words.
"""

from __future__ import annotations

import json
import os
import re

import numpy as np
import pandas as pd

from src.explainability import evidence_payload, explain_text
from src.utils import DECISION_NOTE, ROOT

try:
    from dotenv import load_dotenv
    load_dotenv(ROOT / ".env")
except Exception:  # python-dotenv optional
    pass

SYSTEM_PROMPT = """You are the explanation layer of NWIS, a drilling decision-support prototype.
Rules you must follow:
- Use ONLY the facts in the EVIDENCE and DOCUMENT EXCERPTS provided. Do not add wells, depths,
  events, parameters, mitigation actions or numbers that are not there.
- If the evidence does not answer the question, reply exactly: "Insufficient historical evidence."
- Describe risk as a "risk indication" (Low / Moderate / Elevated). Never say an event WILL occur.
- Do not give operational instructions. You may say what was done historically and its outcome.
- Cite the source document in brackets after each historical fact, e.g. [W-102_DDR038.pdf].
- End with: "Review historical evidence and use engineering judgment."
- The data is synthetic prototype data unless the evidence says otherwise; do not call it Oil India data.
Keep the answer under 180 words, plain language, short bullet points allowed."""

WELL_RE = re.compile(r"\b(?:W-\d{3}|ACTIVE-\d{2}|FORGE-58-32)\b")


# ------------------------------------------------------------------ chunking + index
def chunk_documents(documents: pd.DataFrame, max_chars: int = 450) -> pd.DataFrame:
    rows = []
    for d in documents.itertuples():
        text = str(d.extracted_text or "")
        parts, buf = [], ""
        for line in text.splitlines():
            line = line.strip()
            if not line or set(line) <= set("-"):
                continue
            if len(buf) + len(line) > max_chars and buf:
                parts.append(buf)
                buf = ""
            buf += (" " if buf else "") + line
        if buf:
            parts.append(buf)
        for i, p in enumerate(parts):
            rows.append(dict(chunk_id=f"{d.document_id}#{i}", document_id=d.document_id, well_id=d.well_id,
                             filename=d.filename, extraction_method=d.extraction_method, text=p))
    return pd.DataFrame(rows)


class Retriever:
    def __init__(self, documents: pd.DataFrame):
        self.chunks = chunk_documents(documents)
        self.backend = "tfidf"
        want = os.getenv("NWIS_VECTOR_BACKEND", "tfidf").lower()
        if want == "chroma":
            try:
                import chromadb
                client = chromadb.Client()
                self.col = client.get_or_create_collection("nwis_reports")
                self.col.add(ids=self.chunks.chunk_id.tolist(), documents=self.chunks.text.tolist(),
                             metadatas=self.chunks[["document_id", "well_id", "filename"]].fillna("").to_dict("records"))
                self.backend = "chroma"
            except Exception as err:  # fall back silently to TF-IDF, but record why
                self.backend_note = f"ChromaDB unavailable ({err}); using TF-IDF"
        if self.backend == "tfidf":
            from sklearn.feature_extraction.text import TfidfVectorizer
            self.vec = TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True, stop_words="english")
            self.X = self.vec.fit_transform(self.chunks.text) if len(self.chunks) else None

    def search(self, query: str, k: int = 5, well_ids: list[str] | None = None) -> pd.DataFrame:
        if not len(self.chunks):
            return self.chunks
        if self.backend == "chroma":
            where = {"well_id": {"$in": well_ids}} if well_ids else None
            r = self.col.query(query_texts=[query], n_results=k, where=where)
            hits = self.chunks.set_index("chunk_id").loc[r["ids"][0]].reset_index()
            hits["score"] = [round(1 - d, 3) for d in r["distances"][0]]
            return hits
        q = self.vec.transform([query])
        scores = (self.X @ q.T).toarray().ravel()
        df = self.chunks.assign(score=scores.round(3))
        if well_ids:
            df = df[df.well_id.isin(well_ids)]
        return df[df.score > 0].sort_values("score", ascending=False).head(k)


# ------------------------------------------------------------------ Claude client
def claude_available() -> bool:
    if not os.getenv("ANTHROPIC_API_KEY"):
        return False
    try:
        import anthropic  # noqa: F401
        return True
    except ImportError:
        return False


def call_claude(question: str, evidence: dict, excerpts: list[dict]) -> str:
    import anthropic
    client = anthropic.Anthropic()                       # reads ANTHROPIC_API_KEY
    content = (f"QUESTION:\n{question}\n\nEVIDENCE (structured, from the NWIS analytics):\n"
               f"{json.dumps(evidence, indent=1, default=str)}\n\nDOCUMENT EXCERPTS:\n" +
               "\n".join(f"[{e['filename']}] {e['text']}" for e in excerpts))
    msg = client.messages.create(model=os.getenv("NWIS_CLAUDE_MODEL", "claude-sonnet-5"), max_tokens=600,
                                 system=SYSTEM_PROMPT, messages=[{"role": "user", "content": content}])
    return "".join(getattr(b, "text", "") for b in msg.content).strip()


def grounding_problems(answer: str, evidence: dict, excerpts: list[dict]) -> list[str]:
    allowed = set(WELL_RE.findall(json.dumps(evidence, default=str) + " ".join(e["text"] for e in excerpts)))
    allowed |= {e.get("well_id") for e in excerpts if e.get("well_id")}
    return sorted(set(WELL_RE.findall(answer)) - allowed)


# ------------------------------------------------------------------ Ask NWIS
def _intent(q: str) -> str:
    q = q.lower()
    if "why" in q:
        return "why"
    if "mitigation" in q or "what was done" in q or "summar" in q:
        return "mitigation"
    if "relevant" in q or "which offset" in q or "most similar" in q:
        return "relevance"
    if "case" in q or "near the current depth" in q or "history" in q or "historical" in q:
        return "cases"
    return "search"


def _event_type_in(q: str, default: str | None) -> str | None:
    for t in ["mud loss", "kick", "stuck pipe", "torque spike", "wellbore instability", "bit problem", "cementing"]:
        if t in q.lower() or t.replace(" ", "-") in q.lower():
            return t.title().replace("Cementing", "Cementing Issue")
    return default


def ask(question: str, state: dict, retriever: Retriever | None = None, use_claude: bool = True) -> dict:
    """Answer a question about the current situation, grounded in NWIS data. Returns
    {answer, mode, intent, evidence, sources, grounding_warnings}."""
    intent = _intent(question)
    alert = state["alerts"][0] if state.get("alerts") else None
    etype = _event_type_in(question, alert["risk_type"] if alert else None)
    corr, rel = state.get("correlated", pd.DataFrame()), state.get("relevance", pd.DataFrame())
    evidence: dict = {"current_depth_m": state["depth"], "formation": state["formation"], "well": state["well_id"]}
    offline = ""

    if intent == "why":
        a = next((r for r in state.get("risks", []) if r["risk_type"] == etype), alert)
        if a is None:
            offline = "Insufficient historical evidence. No risk indication is active at this depth."
        else:
            evidence = evidence_payload(a, state)
            offline = explain_text(a)
    elif intent == "cases":
        c = corr[(corr.event_type == etype)] if len(corr) and etype else corr
        c = c[c.status.isin(["AT CURRENT DEPTH", "AHEAD"])] if len(c) else c
        evidence["cases"] = c[["well_id", "event_type", "depth", "aligned_depth", "delta_to_bit_m", "status",
                               "severity", "mitigation", "outcome", "source_document"]].round(1).to_dict("records") if len(c) else []
        offline = ("Insufficient historical evidence." if not evidence["cases"] else
                   f"{etype or 'Event'} cases in relevant offset wells near {state['depth']:.0f} m "
                   f"(formation-aligned):\n" + "\n".join(
                       f"- {x['well_id']}: {x['event_type']} at {x['depth']:.0f} m (aligned {x['aligned_depth']:.0f} m, "
                       f"{x['delta_to_bit_m']:+.0f} m from bit, {x['status']}), {x['severity']}; "
                       f"historically: {x['mitigation']} -> {x['outcome']} [{x['source_document']}]"
                       for x in evidence["cases"]))
    elif intent == "relevance":
        top = rel.head(5) if len(rel) else rel
        evidence["relevant_wells"] = top[["well_id", "distance_km", "relevance_score", "spatial_score",
                                          "formation_score", "depth_score", "parameter_score", "geology_score",
                                          "why"]].to_dict("records") if len(top) else []
        offline = ("Insufficient historical evidence (no offset wells within the radius)." if not len(top) else
                   "Most relevant offset wells (prototype weights, configurable):\n" + "\n".join(
                       f"- {r.well_id}: relevance {r.relevance_score:.2f} (spatial {r.spatial_score}, formation "
                       f"{r.formation_score}, depth {r.depth_score}, parameters {r.parameter_score}, geology "
                       f"{r.geology_score}); {r.distance_km} km" for r in top.itertuples()))
    elif intent == "mitigation":
        a = next((r for r in state.get("risks", []) if r["risk_type"] == etype), alert)
        mit = a["historical_mitigations"] if a else []
        evidence["historical_mitigations"] = mit
        offline = ("Insufficient historical evidence." if not mit else
                   f"Historical actions for {a['risk_type']} in relevant offset wells (what was done, not an instruction):\n" +
                   "\n".join(f"- {m['mitigation']} -> {m['outcome']} ({m['times']}x; {m['wells']}) [{m['sources']}]"
                             for m in mit))

    excerpts = []
    if retriever is not None:
        wells = list(rel.well_id[rel.selected]) if len(rel) else None
        hits = retriever.search(question + (" " + etype if etype else ""), k=4, well_ids=wells)
        excerpts = hits[["filename", "well_id", "text", "score"]].to_dict("records") if len(hits) else []
        if intent == "search":
            evidence["retrieved"] = len(excerpts)
            offline = ("Insufficient historical evidence." if not excerpts else
                       "Most relevant report excerpts:\n" + "\n".join(f"- [{e['filename']}] {e['text'][:220]}…"
                                                                     for e in excerpts))
    offline = offline.rstrip() + ("" if DECISION_NOTE in offline else f"\n\n{DECISION_NOTE}")

    mode, answer, warnings = "offline (no Claude API key)", offline, []
    if use_claude and claude_available():
        try:
            answer = call_claude(question, evidence, excerpts)
            mode = "claude"
            warnings = grounding_problems(answer, evidence, excerpts)
            if warnings:
                answer, mode = offline, "offline (Claude answer rejected: mentioned wells not in evidence)"
        except Exception as err:
            mode = f"offline (Claude call failed: {type(err).__name__})"
    sources = sorted({e["filename"] for e in excerpts} |
                     set(re.findall(r"\b[\w-]+_DDR\d+(?:_scanned)?\.(?:txt|pdf)\b", json.dumps(evidence, default=str))))
    return dict(answer=answer, mode=mode, intent=intent, evidence=evidence, excerpts=excerpts,
                sources=sources, grounding_warnings=warnings)
