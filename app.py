# ============================================================
# 📄 Contract Clause Deviation Detector — FINAL STABLE APP
# ============================================================

import streamlit as st

# ------------------------------------------------------------
# Page config (MUST be first Streamlit command)
# ------------------------------------------------------------
st.set_page_config(
    page_title="Contract Clause Deviation Detector",
    layout="wide"
)

# ------------------------------------------------------------
# Ensure models are present (Streamlit Cloud safe)
# ------------------------------------------------------------
import os

if not os.path.exists("resources/deberta-clause-final"):
    from download_models import ensure_models
    ensure_models()

import tempfile
from llm.llm_client import ollama_client

from pipeline import (
    analyze_document,
    ask_document,
    build_contract_summary,
    narrate_contract_summary,
)

# ------------------------------------------------------------
# Session State Init
# ------------------------------------------------------------
if "analyzed" not in st.session_state:
    st.session_state.analyzed = False

if "last_answer" not in st.session_state:
    st.session_state.last_answer = None

if "active_tab" not in st.session_state:
    st.session_state.active_tab = "overview"

# ------------------------------------------------------------
# Title
# ------------------------------------------------------------
st.title("📄 Contract Clause Deviation Detector")
st.caption(
    "AI-assisted contract analysis using clause detection, deviation analysis, "
    "and explainable reasoning (not legal advice)."
)

# ------------------------------------------------------------
# Sidebar — Upload & Analyze
# ------------------------------------------------------------
st.sidebar.header("📂 Upload Contract")

uploaded_pdf = st.sidebar.file_uploader(
    "Upload a contract PDF",
    type=["pdf"]
)

if st.sidebar.button("▶ Analyze Contract"):
    if uploaded_pdf is None:
        st.sidebar.error("Please upload a PDF file first.")
    else:
        with st.spinner("Analyzing contract… this may take a minute"):
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                tmp.write(uploaded_pdf.read())
                pdf_path = tmp.name

            clause_df, spans, embeddings, embedder = analyze_document(pdf_path)

            # Known clauses first, Unknown last
            clause_df["known"] = clause_df["final_clause"] != "Unknown"
            clause_df = clause_df.sort_values(
                by=["known", "span_id"],
                ascending=[False, True]
            ).drop(columns="known")

            # Deterministic summary (NO LLM)
            contract_summary = build_contract_summary(clause_df, spans)

            # Optional guarded narration
            summary_narration = narrate_contract_summary(
                contract_summary,
                llm_client=ollama_client
            )

            st.session_state.clause_df = clause_df
            st.session_state.spans = spans
            st.session_state.embeddings = embeddings
            st.session_state.embedder = embedder
            st.session_state.contract_summary = contract_summary
            st.session_state.summary_narration = summary_narration
            st.session_state.last_answer = None
            st.session_state.analyzed = True
            st.session_state.active_tab = "overview"

        st.sidebar.success("Analysis complete ✅")

# ------------------------------------------------------------
# Main UI
# ------------------------------------------------------------
if st.session_state.analyzed:
    clause_df = st.session_state.clause_df
    spans = st.session_state.spans
    summary = st.session_state.contract_summary

    # Tab persistence
    tab_labels = ["📘 Overview", "⚠️ Deviating Clauses", "❓ Ask the Contract"]
    tab1, tab2, tab3 = st.tabs(tab_labels)

    # ========================================================
    # TAB 1 — OVERVIEW
    # ========================================================
    with tab1:
        st.subheader("🧾 Contract Summary")
        st.caption("Auto-generated overview based on detected clauses and deviations")

        st.markdown("### 📝 Executive Summary")
        st.write(st.session_state.summary_narration)

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Total Spans", summary["overview"]["total_spans"])
        col2.metric("Recognized Clauses", summary["overview"]["recognized_clauses"])
        col3.metric("Unknown Sections", summary["overview"]["unknown_spans"])
        col4.metric("Deviating Clauses", summary["overview"]["deviating_spans"])

        if summary["deviations"]:
            st.markdown("### ⚠️ Risk Snapshot")
            for d in summary["deviations"]:
                st.write(f"**{d['clause']}** — {', '.join(d['reasons'])}")
        else:
            st.success("✅ No non-standard clause patterns detected.")

        st.markdown("### 📌 Clause Coverage")
        st.write(", ".join(summary["coverage"]["detected_clauses"]))
        st.caption(summary["coverage"]["undetected_note"])

        # ------------------------------
        # ✅ FIX 1 — Clause → Span Mapping
        # ------------------------------
        st.markdown("---")
        st.subheader("📄 Clause to Text Mapping")

        for clause_name in sorted(clause_df["final_clause"].unique()):
            if clause_name == "Unknown":
                continue

            with st.expander(f"{clause_name}"):
                rows = clause_df[clause_df["final_clause"] == clause_name]

                for _, row in rows.iterrows():
                    sid = row["span_id"]
                    conf = row["confidence"]
                    flag = " ⚠️" if row["final_deviation"] else ""

                    st.markdown(
                        f"**Span {sid} | confidence={conf:.2f}{flag}**"
                    )
                    st.write(spans[sid])
                    st.markdown("---")

        st.markdown("---")
        for note in summary["confidence_notes"]:
            st.caption(note)

    # ========================================================
    # TAB 2 — DEVIATING CLAUSES
    # ========================================================
    with tab2:
        deviating = clause_df[clause_df["final_deviation"]]

        if deviating.empty:
            st.success("No non-standard / deviating clauses detected.")
        else:
            st.warning(f"{len(deviating)} deviating clause(s) detected.")

            for _, row in deviating.iterrows():
                sid = row["span_id"]
                with st.expander(f"{row['final_clause']} | span {sid}"):
                    st.markdown("**Deviation reasons:**")
                    for r in row["deviation_reasons"]:
                        st.write(f"- {r}")

                    st.markdown("**Clause text:**")
                    st.write(spans[sid])

    # ========================================================
    # TAB 3 — ASK THE CONTRACT (SYNC RAG, NO ASYNC)
    # ========================================================
    with tab3:
        st.subheader("Ask a question about the contract")

        question = st.text_input(
            "Example: What does the cap on liability mean for me?"
        )

        if st.button("Ask"):
            if not question.strip():
                st.warning("Please enter a question.")
            else:
                st.session_state.active_tab = "ask"

                # Step 1 — ML-grounded retrieval
                retrieval = ask_document(
                    question,
                    clause_df,
                    spans,
                    st.session_state.embeddings,
                    st.session_state.embedder
                )

                # Step 2 — SAFE RAG context
                rag_context = {
                    "intent": retrieval["intent"],
                    "detected_clauses": list(
                        set(e["clause"] for e in retrieval.get("evidence", []))
                    ),
                    "deviations": [
                        {
                            "clause": e["clause"],
                            "reasons": e["reasons"],
                            "explanations": e["explanations"]
                        }
                        for e in retrieval.get("evidence", [])
                        if e["deviating"]
                    ]
                }

                # Step 3 — Synchronous LLM reasoning (NO LangChain)
                prompt = f"""
You are explaining a contract to a non-legal audience.

ONLY use the information below. Do NOT invent anything.

CONTEXT:
{rag_context}

QUESTION:
{question}

RULES:
- Be neutral and simple
- No legal advice
- No speculation
"""

                explanation = ollama_client(prompt)

                st.session_state.last_answer = {
                    "explanation": explanation,
                    "evidence": retrieval.get("evidence", []),
                    "confidence_notes": retrieval.get("confidence_notes", [])
                }

        # ----------------------------
        # Render Answer (persistent)
        # ----------------------------
        if st.session_state.last_answer:
            answer = st.session_state.last_answer

            st.markdown("### 📌 Answer")
            st.write(answer["explanation"])

            if answer["evidence"]:
                st.markdown("---")
                st.caption("Supporting sections from the contract:")

                for ev in answer["evidence"]:
                    with st.expander(f"{ev['clause']} | span {ev['span_id']}"):
                        st.write(ev["text"])

            st.markdown("---")
            for note in answer["confidence_notes"]:
                st.caption(note)

else:
    st.info("⬅️ Upload a PDF and click **Analyze Contract** to begin.")