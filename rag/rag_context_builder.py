# rag/rag_context_builder.py

def build_rag_context(clause_df, contract_summary):
    """
    Builds a SAFE, ABSTRACTED context for RAG-based reasoning.
    No raw contract text is included.
    """

    # -----------------------------
    # Clause overview
    # -----------------------------
    clauses = []

    for clause in sorted(clause_df["final_clause"].unique()):
        if clause == "Unknown":
            continue

        subset = clause_df[clause_df["final_clause"] == clause]
        deviating = subset["final_deviation"].any()

        clauses.append({
            "name": clause,
            "present": True,
            "deviating": bool(deviating),
            "count": int(len(subset))
        })

    # -----------------------------
    # Deviations (high-level only)
    # -----------------------------
    deviations = []

    for d in contract_summary["deviations"]:
        deviations.append({
            "clause": d["clause"],
            "severity": d["severity_hint"],
            "reasons": d["reasons"]
        })

    # -----------------------------
    # Final RAG context
    # -----------------------------
    rag_context = {
        "overview": {
            "total_sections": contract_summary["overview"]["total_spans"],
            "recognized_clauses": contract_summary["overview"]["recognized_clauses"],
            "deviating_clauses": contract_summary["overview"]["deviating_spans"]
        },
        "clauses": clauses,
        "deviations": deviations,
        "confidence_notes": contract_summary["confidence_notes"]
    }

    return rag_context
