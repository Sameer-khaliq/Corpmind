# CorpMind — Scope & Requirements (v1)
### Multi-Supplier Catalog Reconciliation & Enrichment Agent

## 1. Problem Statement

E-commerce sellers sourcing from multiple suppliers or dropship partners end up with catalog data that's structurally inconsistent — every supplier formats their feed differently, uses different attribute names, and has different gaps in what they report. Today this gets fixed by hand in spreadsheets, which doesn't scale past a few hundred SKUs and breaks down completely once a business is juggling three or more supplier feeds.

CorpMind is a multi-agent pipeline that ingests raw supplier feeds and produces a reconciled, de-duplicated, enriched catalog — with every automated decision backed by an evaluation gate, so nothing gets published without a confidence check.

## 2. Real Business Problems This Solves

| # | Pain Point | Why Existing Tools Fail | How CorpMind Addresses It |
|---|---|---|---|
| 1 | Inconsistent attributes across suppliers (`Color: Red` vs `Colour: Crimson` vs no field at all, buried in free text) | Rule-based PIM tools need a hand-built mapping per supplier format and can't parse free text | Extraction Agent normalizes every source into one fixed schema regardless of format |
| 2 | Duplicate/near-duplicate SKUs across supplier feeds | Exact-string or SKU matching misses products with different titles/descriptions for the same physical item | Matching Agent uses hybrid search (semantic + keyword), not exact matching |
| 3 | New supplier onboarding is a recurring, not one-time, manual cost | Every new supplier means redoing the mapping work by hand, again | Pipeline absorbs new feeds without new manual mapping — this is also why it supports a retainer, not just a one-off fee |
| 4 | Manual categorization doesn't scale and isn't consistent between staff | No systematic source of truth for taxonomy assignment | Extraction Agent assigns category as a structured, consistent output |
| 5 | Messy catalog data hurts on-site search relevance and conversion | Search quality is capped by input data quality | Clean, consistent output feeds directly into a search-relevance project (natural upsell) |
| 6 | Naive AI automation risks silently wrong data (wrong material, wrong size) reaching customers | Most "AI enrichment" tools just generate a plausible-sounding fill with no way to catch when it's wrong | Enrichment Agent grounds fills in an actual web search instead of generating from nothing; anything unverified gets flagged, not published |

## 3. Scope

**In scope (v1):**
- Ingest supplier feeds in CSV and XLSX
- Extract & normalize into a fixed schema: title, brand, category, color, material, size, price, SKU, description
- Cross-supplier product matching via hybrid search (BM25 + dense + RRF)
- Web-search-grounded enrichment for missing/low-confidence attributes
- RAGAS-style evaluation gate before any match or enrichment is accepted
- Human review queue for anything below threshold
- Structured output: reconciled catalog export (CSV/JSON) + a change/audit report
- Simple demo UI (Streamlit/Gradio): upload a feed, watch the pipeline run, review flagged items
- Dockerized, single-tenant demo deployment

**Out of scope (v1 — deferred, and good upsell items for a real client later):**
- PDF/scanned catalogs — needs OCR, a materially harder problem than structured data; add only if a real client needs it
- Real-time/streaming feed sync — v1 is batch (upload → result); live sync with a store's inventory is infrastructure work worth billing separately
- Multi-tenant architecture — this is a single-business system, not a SaaS platform
- Direct marketplace push (Shopify/Amazon API) — a strong v2 upsell once someone's paying
- Non-English catalogs — relevant later for Gulf/EU clients specifically, not needed for an English-first portfolio piece

## 4. System Architecture

```
Supplier Feed (CSV/XLSX)
        |
        v
1. Ingestion Node
   parses raw files, handles missing columns/encoding gracefully
        |
        v
2. Extraction & Normalization Agent
   maps messy fields into the fixed schema (LLM-based)
        |
        v
3. Matching / Deduplication Agent
   hybrid search against the existing catalog -> match-confidence score
        |
        v
4. Enrichment Agent
   web-search tool-calling to ground-fill missing/low-confidence fields
        |
        v
5. Evaluation Agent (Gate)
   RAGAS-style faithfulness/confidence scoring against a gold set
        |
   -----+-----
   |         |
Above      Below
threshold  threshold
   |         |
   v         v
6a. Report   6b. Human Review Queue
    Agent    (surfaced in the UI)
```

A supervisor node in LangGraph routes between steps and handles retries when a tool call (web search, embedding lookup) fails.

**Maps to what you've already built — this is integration and extension, not a from-zero build:**

| Node | Tech | Reuses |
|---|---|---|
| Ingestion | pandas/openpyxl | New, but small |
| Extraction & Normalization | Groq/Gemini LLM call | New prompt, same API pattern as your 40-day project |
| Matching/Dedup | ChromaDB + BM25 + RRF | Same hybrid-search pipeline as your 40-day e-commerce RAG project |
| Enrichment | LangGraph ReAct + web search tool | Same pattern as your 40-day ReAct agent |
| Evaluation Gate | RAGAS | Same eval approach as your 40-day project |
| Orchestration | LangGraph | Same framework as your research-agent project |
| Deployment | Docker, HF Spaces | Same as your 40-day deployment |

## 5. Functional Requirements

| ID | Requirement |
|---|---|
| FR-1 | Ingest supplier product feeds in CSV and XLSX format |
| FR-2 | Extract and normalize attributes from inconsistent source fields, including unstructured description text, into a fixed schema |
| FR-3 | Detect candidate duplicate/matching products across supplier feeds via hybrid search, with a numeric match-confidence score per candidate |
| FR-4 | Attempt web-search-grounded enrichment for missing/low-confidence attributes before flagging for review |
| FR-5 | Score every accepted match and enrichment against a RAGAS-style faithfulness/confidence threshold |
| FR-6 | Route anything below threshold to a human review queue instead of auto-publishing |
| FR-7 | Output a reconciled catalog export (CSV/JSON) plus a structured change/audit report |
| FR-8 | Provide a UI to upload a feed, monitor pipeline progress, and review/approve flagged items |
| FR-9 | Log every agent decision (which node acted, what changed, what score it got) for auditability |

## 6. Non-Functional Requirements

| Category | Requirement | Target for v1 |
|---|---|---|
| Throughput | Batch processing time | Reconcile + enrich a 500-SKU batch in under 15 minutes |
| Cost | LLM spend per batch | Define a $/1,000-SKU ceiling; route easy/high-confidence items to a cheaper model (Groq), reserve a stronger model for ambiguous cases only |
| Accuracy — matching | False-positive rate | Weight precision over recall — merging two different products is worse than missing a real duplicate; tune conservatively |
| Accuracy — enrichment | RAGAS faithfulness score | Don't auto-publish below 0.85 faithfulness against the retrieved source |
| Reliability | Malformed input / failed tool calls | Flag for review, never silently drop or silently guess |
| Security | Retrieved web content handling | Treat retrieved web text as untrusted data, not instructions — guard against prompt injection from a scraped page |
| Auditability | Traceability | Every merge/enrichment/flag traceable to which agent made the call and why |
| Scalability | Target catalog size for v1 | Hundreds to a few thousand SKUs — right-sized for portfolio/small-client use, not enterprise scale |

## 7. Definition of Done

- A deliberately messy, overlapping test feed (built from 2-3 combined/altered public catalogs) runs end-to-end with minimal manual intervention
- Planted "trap" cases resolve correctly: a near-duplicate that should match, a genuinely different product that should NOT match, a missing attribute that gets correctly enriched, a missing attribute with no reliable source that gets correctly flagged instead of guessed
- The audit report clearly shows what changed and why, per item
- Runs in Docker, reachable via a public demo link
- You can explain the business problem and show the before/after in under 2 minutes — that's the real test, since that's what a client conversation requires
