"""
CorpMind UI -- Day 19 (Upload + Monitor) + Day 20 (Human Review Queue)
Run: streamlit run ui/app.py
"""

import queue
import time

import streamlit as st

from pipeline_adapter import (
    ingest_uploaded_files,
    run_pipeline_in_background,
    approve_flagged_item,
    reject_flagged_item,
    regenerate_exports,
)

st.set_page_config(page_title="CorpMind", layout="wide")


# ---------------------------------------------------------------------------
# Session state init
# ---------------------------------------------------------------------------

def _init_state():
    defaults = {
        "view": "Upload",
        "progress_q": None,
        "thread": None,
        "stage_log": [],
        "batch_result": None,
        "report_md": None,
        "catalog_csv": None,
        "catalog_json": None,
        "report_dir": None,
        "pipeline_error": None,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


_init_state()


# ---------------------------------------------------------------------------
# Sidebar nav
# ---------------------------------------------------------------------------

with st.sidebar:
    st.title("CorpMind")
    views = ["Upload", "Monitor", "Review Queue"]
    st.session_state.view = st.radio("View", views, index=views.index(st.session_state.view))

    if st.session_state.batch_result:
        st.divider()
        st.metric("Accepted", len(st.session_state.batch_result.get("accepted_items", [])))
        st.metric("Flagged for review", len(st.session_state.batch_result.get("flagged_items", [])))


# ---------------------------------------------------------------------------
# View: Upload
# ---------------------------------------------------------------------------

def view_upload():
    st.header("Upload supplier feeds")
    st.caption("CSV or XLSX. One row per feed with its supplier ID.")

    n_feeds = st.number_input("Number of supplier feeds", min_value=1, max_value=10, value=1)

    uploaded_files, supplier_ids = [], []
    for i in range(int(n_feeds)):
        col1, col2 = st.columns([2, 1])
        with col1:
            f = st.file_uploader(f"Feed {i + 1}", type=["csv", "xlsx"], key=f"file_{i}")
        with col2:
            sid = st.text_input(f"Supplier ID {i + 1}", value=f"supplier_{chr(65 + i)}", key=f"sid_{i}")
        if f is not None:
            uploaded_files.append(f)
            supplier_ids.append(sid)

    run_disabled = len(uploaded_files) == 0 or st.session_state.thread is not None
    if st.button("Run pipeline", type="primary", disabled=run_disabled):
        raw_rows = ingest_uploaded_files(uploaded_files, supplier_ids)
        st.session_state.progress_q = queue.Queue()
        st.session_state.stage_log = []
        st.session_state.batch_result = None
        st.session_state.pipeline_error = None
        st.session_state.thread = run_pipeline_in_background(raw_rows, st.session_state.progress_q)
        st.session_state.view = "Monitor"
        st.rerun()

    if st.session_state.thread is not None:
        st.info("A run is already in progress -- switch to Monitor to watch it.")


# ---------------------------------------------------------------------------
# View: Monitor
# ---------------------------------------------------------------------------

STAGE_ORDER = ["ingestion", "extract_and_match", "review_processing"]


def view_monitor():
    st.header("Pipeline progress")

    if st.session_state.progress_q is None:
        st.info("No run yet -- start one from the Upload view.")
        return

    # Drain everything that's arrived since the last rerun
    q = st.session_state.progress_q
    while True:
        try:
            event = q.get_nowait()
        except queue.Empty:
            break

        if event["type"] == "error":
            st.session_state.pipeline_error = event["error"]
            st.session_state.thread = None
        elif event["type"] == "done":
            st.session_state.batch_result = event["result"]
            st.session_state.report_md = event["report_md"]
            st.session_state.catalog_csv = event["catalog_csv"]
            st.session_state.catalog_json = event["catalog_json"]
            st.session_state.report_dir = event["report_dir"]
            st.session_state.thread = None
        else:
            st.session_state.stage_log.append(event)

    if st.session_state.pipeline_error:
        st.error(f"Pipeline failed: {st.session_state.pipeline_error}")
        return

    done_stages = {e["stage"]: e for e in st.session_state.stage_log if e["type"] == "stage_done"}
    for stage in STAGE_ORDER:
        if stage in done_stages:
            e = done_stages[stage]
            st.success(f"✅ {e['label']} -- {e.get('detail', '')}")
        elif st.session_state.thread is not None:
            label = {"ingestion": "Ingestion", "extract_and_match": "Extraction + Matching",
                      "review_processing": "Enrichment + Evaluation"}[stage]
            st.info(f"⏳ {label} -- waiting")

    if st.session_state.batch_result is not None:
        result = st.session_state.batch_result
        st.divider()
        c1, c2, c3 = st.columns(3)
        c1.metric("Accepted", len(result.get("accepted_items", [])))
        c2.metric("Flagged for review", len(result.get("flagged_items", [])))
        c3.metric("Audit log entries", len(result.get("audit_log", [])))

        dl1, dl2 = st.columns(2)
        with dl1:
            st.download_button("Download catalog (CSV)", st.session_state.catalog_csv,
                                file_name="catalog_export.csv", mime="text/csv")
        with dl2:
            st.download_button("Download catalog (JSON)", st.session_state.catalog_json,
                                file_name="catalog_export.json", mime="application/json")

        with st.expander("Audit report"):
            st.markdown(st.session_state.report_md)

        if result.get("flagged_items"):
            st.warning(f"{len(result['flagged_items'])} items need review -- see the Review Queue view.")
    elif st.session_state.thread is not None:
        time.sleep(1)
        st.rerun()


# ---------------------------------------------------------------------------
# View: Review Queue (Day 20)
# ---------------------------------------------------------------------------

def view_review_queue():
    st.header("Human review queue")

    result = st.session_state.batch_result
    if result is None:
        st.info("No completed run yet -- run a batch first.")
        return

    flagged = result.get("flagged_items", [])
    if not flagged:
        st.success("Nothing flagged -- queue is empty.")
        return

    for item in flagged:
        cid = item.get("audit_catalog_id", "unknown")
        # normalized_product is a NormalizedProduct object (extraction.py),
        # not a flat dict key on item -- mirrors report.py's own
        # getattr(normalized, "title", None) approach.
        normalized = item.get("normalized_product")
        title = getattr(normalized, "title", None) if normalized is not None else None
        title = title or cid

        with st.expander(f"{title}  --  {cid}"):
            # item holds pydantic sub-objects (normalized_product,
            # match_result, enrichment_result, evaluation_record) --
            # st.json needs plain JSON, so model_dump() anything that has it.
            display = {}
            for k, v in item.items():
                if k == "audit_catalog_id":
                    continue
                display[k] = v.model_dump() if hasattr(v, "model_dump") else v
            st.json(display, expanded=False)

            reasons = [
                (e.reason if hasattr(e, "reason") else e.get("reason"))
                for e in result.get("audit_log", [])
                if (e.catalog_id if hasattr(e, "catalog_id") else e.get("catalog_id")) == cid
            ]
            if reasons:
                st.caption("Why it was flagged:")
                for r in reasons:
                    st.write(f"- {r}")

            note = st.text_input("Reviewer note (optional)", key=f"note_{cid}")

            col1, col2 = st.columns(2)
            with col1:
                if st.button("Approve", key=f"approve_{cid}", type="primary"):
                    st.session_state.batch_result = approve_flagged_item(item, result, note)
                    report_md, csv_bytes, json_bytes, report_dir = regenerate_exports(
                        st.session_state.batch_result, st.session_state.report_dir
                    )
                    st.session_state.report_md = report_md
                    st.session_state.catalog_csv = csv_bytes
                    st.session_state.catalog_json = json_bytes
                    st.session_state.report_dir = report_dir
                    st.rerun()
            with col2:
                if st.button("Reject", key=f"reject_{cid}"):
                    st.session_state.batch_result = reject_flagged_item(item, result, note)
                    report_md, csv_bytes, json_bytes, report_dir = regenerate_exports(
                        st.session_state.batch_result, st.session_state.report_dir
                    )
                    st.session_state.report_md = report_md
                    st.session_state.catalog_csv = csv_bytes
                    st.session_state.catalog_json = json_bytes
                    st.session_state.report_dir = report_dir
                    st.rerun()


# ---------------------------------------------------------------------------
# Router
# ---------------------------------------------------------------------------

if st.session_state.view == "Upload":
    view_upload()
elif st.session_state.view == "Monitor":
    view_monitor()
elif st.session_state.view == "Review Queue":
    view_review_queue()