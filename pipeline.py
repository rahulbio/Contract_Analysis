# # # ============================================================
# # # CONTRACT CLAUSE DEVIATION — PIPELINE
# # # ============================================================

# # # ---------------- IMPORTS ----------------
# # import pdfplumber
# # import re
# # import ast
# # import torch
# # import numpy as np
# # import pandas as pd

# # from collections import defaultdict
# # from sklearn.metrics.pairwise import cosine_distances, cosine_similarity
# # from transformers import AutoTokenizer, AutoModelForSequenceClassification
# # from sentence_transformers import SentenceTransformer

# # # ---------------- CONFIG ----------------
# # MAX_SEQ_LENGTH = 128
# # CONFIDENCE_THRESHOLD = 0.5

# # device = torch.device("cpu")  # enable GPU later if needed

# # # ---------------- PATHS ----------------
# # CLAUSE_MODEL_PATH = r"C:\Users\Rahul K\OneDrive\Desktop\contract_deviation_app\resources\deberta-clause-final"
# # TRAIN_CSV_PATH = r"C:\Users\Rahul K\OneDrive\Desktop\contract_deviation_app\resources\final_cleaned_version (2).csv"
# # EMBEDDING_MODEL_NAME = "sentence-transformers/all-mpnet-base-v2"

# # # ============================================================
# # # SPAN GENERATION
# # # ============================================================
# # def generate_paragraph_spans(text, min_len=80):
# #     paragraphs = [p.strip() for p in text.split("\n\n")]
# #     return [p for p in paragraphs if len(p) >= min_len]

# # def split_on_subsections(text, min_len=80):
# #     parts = re.split(r'(?=\n?\d+\.\d+\s)', text)
# #     return [p.strip() for p in parts if len(p.strip()) >= min_len]

# # # ============================================================
# # # TEXT NORMALIZATION
# # # ============================================================
# # def clean_text(text):
# #     text = text.lower()
# #     text = re.sub(r"\s+", " ", text)
# #     text = re.sub(r"[^a-z0-9\s]", "", text)
# #     return text.strip()

# # def normalize_span(text):
# #     text = re.sub(r"\b(company|licensor|licensee|producer|ma|ent)\b", "party", text)
# #     text = re.sub(r"\b\d+(\.\d+)?\b", "num", text)
# #     text = re.sub(r"\b(day|days|month|months|year|years)\b", "time", text)
# #     return text

# # # ============================================================
# # # INVARIANTS
# # # ============================================================
# # LICENSE_OWNERSHIP_TERMS = [
# #     "ownership", "owned", "own", "title",
# #     "sell", "sale", "transfer ownership",
# #     "full ownership", "full rights"
# # ]

# # UNCAPPED_LIABILITY_TERMS = [
# #     "unlimited liability",
# #     "without limitation",
# #     "no limitation",
# #     "liable for all damages"
# # ]

# # def license_grant_invariant(text):
# #     return any(t in text for t in LICENSE_OWNERSHIP_TERMS)

# # def cap_liability_invariant(text):
# #     return any(t in text for t in UNCAPPED_LIABILITY_TERMS)

# # # ============================================================
# # # LOAD CLAUSE CLASSIFICATION MODEL
# # # ============================================================
# # clause_tokenizer = AutoTokenizer.from_pretrained(CLAUSE_MODEL_PATH)
# # clause_model = AutoModelForSequenceClassification.from_pretrained(CLAUSE_MODEL_PATH)
# # clause_model.to(device).eval()

# # ID_TO_CLAUSE = {
# #     int(k): v for k, v in clause_model.config.id2label.items()
# # }

# # # ============================================================
# # # LOAD TRAINING DATA & BASELINES
# # # ============================================================
# # def extract_span_text(span):
# #     try:
# #         lst = ast.literal_eval(span)
# #         if isinstance(lst, list) and len(lst) > 0:
# #             return lst[0]
# #     except:
# #         return None

# # df_train = pd.read_csv(TRAIN_CSV_PATH)
# # df_train["span_text"] = df_train["Span"].apply(extract_span_text)
# # df_train = df_train.dropna(subset=["span_text"]).reset_index(drop=True)

# # df_train["norm_span"] = df_train["span_text"].apply(
# #     lambda x: normalize_span(clean_text(x))
# # )

# # embedder = SentenceTransformer(EMBEDDING_MODEL_NAME)

# # train_embeddings = embedder.encode(
# #     df_train["norm_span"].tolist(),
# #     batch_size=32,
# #     show_progress_bar=False
# # )

# # clause_embeddings = defaultdict(list)
# # for emb, clause in zip(train_embeddings, df_train["Clause"]):
# #     clause_embeddings[clause].append(emb)

# # clause_centroids = {}
# # clause_thresholds = {}
# # clause_applicability_thresholds = {}

# # for clause, embs in clause_embeddings.items():
# #     embs = np.vstack(embs)
# #     centroid = embs.mean(axis=0)
# #     distances = cosine_distances(embs, centroid.reshape(1, -1)).flatten()

# #     clause_centroids[clause] = centroid
# #     clause_thresholds[clause] = np.percentile(distances, 95)
# #     clause_applicability_thresholds[clause] = np.percentile(distances, 99)

# # # ============================================================
# # # POLARITY PROFILES
# # # ============================================================
# # def learn_clause_polarity_profile(df, clause):
# #     texts = df[df["Clause"] == clause]["norm_span"]
# #     total = len(texts)
# #     signals = ["shall", "may", "must", "not", "without", "freely"]
# #     counts = {s: 0 for s in signals}
# #     for t in texts:
# #         for s in signals:
# #             if s in t:
# #                 counts[s] += 1
# #     return {k: v / total for k, v in counts.items()}

# # clause_polarity_profiles = {
# #     c: learn_clause_polarity_profile(df_train, c)
# #     for c in df_train["Clause"].unique()
# # }

# # def generic_polarity_violation(text, profile):
# #     permission_terms = [
# #         "freely",
# #         "at its discretion",
# #         "without limitation",
# #         "without approval",
# #         "without restriction"
# #     ]
# #     if profile.get("not", 0) > 0.6:
# #         for p in permission_terms:
# #             if p in text:
# #                 return True
# #     return False

# # # ============================================================
# # # MAIN ANALYSIS FUNCTION
# # # ============================================================
# # def analyze_document(pdf_path):
# #     # ---- PDF TO TEXT ----
# #     all_text = []
# #     with pdfplumber.open(pdf_path) as pdf:
# #         for page in pdf.pages:
# #             text = page.extract_text()
# #             if text:
# #                 all_text.append(text)

# #     document_text = "\n\n".join(all_text)

# #     # ---- SPANS ----
# #     spans = generate_paragraph_spans(document_text)
# #     refined_spans = []
# #     for span in spans:
# #         if len(span) > 1000:
# #             refined_spans.extend(split_on_subsections(span))
# #         else:
# #             refined_spans.append(span)

# #     # ---- CLAUSE CLASSIFICATION ----
# #     results = []
# #     for i, span in enumerate(refined_spans):
# #         encoded = clause_tokenizer(
# #             span,
# #             return_tensors="pt",
# #             truncation=True,
# #             padding=True,
# #             max_length=MAX_SEQ_LENGTH
# #         )

# #         with torch.no_grad():
# #             logits = clause_model(
# #                 input_ids=encoded["input_ids"].to(device),
# #                 attention_mask=encoded["attention_mask"].to(device)
# #             ).logits

# #         probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
# #         pred_id = int(np.argmax(probs))
# #         pred_clause = ID_TO_CLAUSE.get(pred_id, "Unknown")
# #         confidence = float(probs[pred_id])

# #         final_clause = pred_clause if confidence >= CONFIDENCE_THRESHOLD else "Unknown"

# #         results.append({
# #             "span_id": i,
# #             "predicted_clause": pred_clause,
# #             "final_clause": final_clause,
# #             "confidence": confidence
# #         })

# #     clause_df = pd.DataFrame(results)

# #     # ---- OCR EMBEDDINGS ----
# #     ocr_clean = [normalize_span(clean_text(s)) for s in refined_spans]
# #     ocr_embeddings = embedder.encode(ocr_clean, batch_size=16, show_progress_bar=False)

# #     # ---- DEVIATION DETECTION ----
# #     deviation_rows = []

# #     for _, row in clause_df.iterrows():
# #         sid = row["span_id"]
# #         clause = row["final_clause"]
# #         raw_text = refined_spans[sid].lower()

# #         if clause == "Unknown" or clause not in clause_centroids:
# #             deviation_rows.append({
# #                 "semantic_distance": None,
# #                 "clause_applicable": False,
# #                 "semantic_deviation": False,
# #                 "polarity_violation": False,
# #                 "invariant_violation": False,
# #                 "final_deviation": False
# #             })
# #             continue

# #         dist = cosine_distances(
# #             ocr_embeddings[sid].reshape(1, -1),
# #             clause_centroids[clause].reshape(1, -1)
# #         )[0][0]

# #         clause_applicable = dist <= clause_applicability_thresholds[clause]
# #         sem_dev = dist > clause_thresholds[clause] if clause_applicable else False
# #         pol_dev = generic_polarity_violation(
# #             raw_text, clause_polarity_profiles[clause]
# #         ) if clause_applicable else False

# #         inv_dev = False
# #         if clause_applicable:
# #             if clause == "License Grant":
# #                 inv_dev = license_grant_invariant(raw_text)
# #             elif clause == "Cap On Liability":
# #                 inv_dev = cap_liability_invariant(raw_text)

# #         deviation_rows.append({
# #             "semantic_distance": float(dist),
# #             "clause_applicable": clause_applicable,
# #             "semantic_deviation": sem_dev,
# #             "polarity_violation": pol_dev,
# #             "invariant_violation": inv_dev,
# #             "final_deviation": sem_dev or pol_dev or inv_dev
# #         })

# #     deviation_df = pd.DataFrame(deviation_rows)

# #     clause_df = pd.concat(
# #         [clause_df.reset_index(drop=True), deviation_df.reset_index(drop=True)],
# #         axis=1
# #     )

# #     clause_df["final_deviation"] = clause_df["final_deviation"].astype(bool)

# #     return clause_df, refined_spans, ocr_embeddings, embedder

# # # ============================================================
# # # QUESTION ANSWERING (API-SAFE)
# # # ============================================================
# # def ask_document(question, clause_df, refined_spans, ocr_embeddings, embedder):
# #     q_emb = embedder.encode([question])[0]
# #     sims = cosine_similarity(q_emb.reshape(1, -1), ocr_embeddings)[0]
# #     idx = int(np.argmax(sims))

# #     return {
# #         "clause": clause_df.iloc[idx]["final_clause"],
# #         "confidence": clause_df.iloc[idx]["confidence"],
# #         "deviating": clause_df.iloc[idx]["final_deviation"],
# #         "text": refined_spans[idx]
# #     }

# # # ============================================================
# # # SAFETY CHECK
# # # ============================================================
# # if __name__ == "__main__":
# #     print("pipeline.py loaded successfully")
# # ============================================================
# # CONTRACT CLAUSE DEVIATION — PIPELINE (FINAL OPTIMIZED)
# # ============================================================

# import pdfplumber
# import re
# import torch
# import numpy as np
# import pandas as pd
# import streamlit as st

# # ============================================================
# # EXPLANATION ENGINE (WHY LAYER)
# # ============================================================

# EXPLANATION_TEMPLATES = {
#     "Uncapped liability detected": (
#         "This clause is considered risky because it does not place a clear limit on liability. "
#         "In standard contracts, liability is typically capped to control financial exposure. "
#         "Without a cap, potential losses may be unlimited."
#     ),

#     "Permission / obligation polarity mismatch": (
#         "This clause alters the typical balance between obligations and permissions. "
#         "Such deviations may create ambiguity around responsibilities and enforcement."
#     ),

#     "Semantic deviation from standard clause language": (
#         "This clause uses language that differs significantly from commonly observed contract patterns. "
#         "Unusual phrasing may lead to interpretation issues or unintended obligations."
#     ),

#     "Violation of non-negotiable license ownership invariant": (
#         "This clause appears to affect ownership or transfer of rights in a way that is generally "
#         "treated as non-negotiable in standard agreements. This may have long-term implications."
#     ),

#     "Uncapped liability detected": (
#         "The absence of a liability cap may expose a party to unlimited financial responsibility, "
#         "which is uncommon in standard contractual practice."
#     )
# }


# CLAUSE_ALIASES = {
#     "Warranty Duration": ["warranty", "warranty period"],
#     "Cap On Liability": ["liability", "liability cap", "cap on liability"],
#     "License Grant": ["license", "licensing"],
#     "Termination": ["termination", "terminate"],
# }

# def explain_deviation_reasons(reasons):
#     """
#     Converts deviation reasons into conservative, factual explanations.
#     """
#     explanations = []

#     for r in reasons:
#         explanation = EXPLANATION_TEMPLATES.get(
#             r,
#             "This clause deviates from standard contractual patterns and may warrant closer review."
#         )
#         explanations.append(explanation)

#     return explanations


# from sklearn.metrics.pairwise import cosine_distances, cosine_similarity
# from transformers import AutoTokenizer, AutoModelForSequenceClassification
# from sentence_transformers import SentenceTransformer

# # ---------------- CONFIG ----------------
# MAX_SEQ_LENGTH = 128
# CONFIDENCE_THRESHOLD = 0.5
# device = torch.device("cpu")

# CLAUSE_MODEL_PATH = r"C:\Users\Rahul K\OneDrive\Desktop\contract_deviation_app\resources\deberta-clause-final"
# EMBEDDING_MODEL_NAME = "sentence-transformers/all-mpnet-base-v2"

# # ---------------- CACHED LOADERS ----------------
# @st.cache_resource
# def load_models():
#     tokenizer = AutoTokenizer.from_pretrained(CLAUSE_MODEL_PATH)
#     model = AutoModelForSequenceClassification.from_pretrained(CLAUSE_MODEL_PATH)
#     model.eval()
#     embedder = SentenceTransformer(EMBEDDING_MODEL_NAME)
#     return tokenizer, model, embedder

# @st.cache_resource
# def load_baselines():
#     return (
#         np.load("clause_centroids.npy", allow_pickle=True).item(),
#         np.load("clause_thresholds.npy", allow_pickle=True).item(),
#         np.load("clause_applicability.npy", allow_pickle=True).item(),
#         np.load("clause_polarity.npy", allow_pickle=True).item()
#     )

# clause_tokenizer, clause_model, embedder = load_models()
# clause_centroids, clause_thresholds, clause_applicability_thresholds, clause_polarity_profiles = load_baselines()

# ID_TO_CLAUSE = {int(k): v for k, v in clause_model.config.id2label.items()}

# # ---------------- TEXT HELPERS ----------------
# def clean_text(text):
#     text = text.lower()
#     text = re.sub(r"\s+", " ", text)
#     text = re.sub(r"[^a-z0-9\s]", "", text)
#     return text.strip()

# def normalize_span(text):
#     text = re.sub(r"\b(company|licensor|licensee|producer|ma|ent)\b", "party", text)
#     text = re.sub(r"\b\d+(\.\d+)?\b", "num", text)
#     text = re.sub(r"\b(day|days|month|months|year|years)\b", "time", text)
#     return text

# def generate_spans(text, min_len=80):
#     return [p.strip() for p in text.split("\n\n") if len(p.strip()) >= min_len]

# # ---------------- INVARIANTS ----------------
# def license_grant_invariant(text):
#     return any(t in text for t in [
#         "ownership", "transfer ownership", "full ownership", "full rights"
#     ])

# def cap_liability_invariant(text):
#     return any(t in text for t in [
#         "unlimited liability", "without limitation", "no limitation"
#     ])

# def polarity_violation(text, profile):
#     if profile.get("not", 0) > 0.6:
#         return any(p in text for p in [
#             "freely", "without restriction", "without approval"
#         ])
#     return False

# # ---------------- MAIN ANALYSIS ----------------
# def analyze_document(pdf_path):
#     # ---- PDF TO TEXT ----
#     pages = []
#     with pdfplumber.open(pdf_path) as pdf:
#         for p in pdf.pages:
#             if p.extract_text():
#                 pages.append(p.extract_text())

#     spans = generate_spans("\n\n".join(pages))

#     # ---- CLAUSE CLASSIFICATION ----
#     records = []
#     for i, span in enumerate(spans):
#         encoded = clause_tokenizer(
#             span,
#             return_tensors="pt",
#             truncation=True,
#             max_length=MAX_SEQ_LENGTH
#         )

#         with torch.no_grad():
#             logits = clause_model(**encoded).logits

#         probs = torch.softmax(logits, dim=1)[0].cpu().numpy()
#         pred_id = int(np.argmax(probs))
#         pred_clause = ID_TO_CLAUSE[pred_id]
#         confidence = float(probs[pred_id])

#         final_clause = pred_clause if confidence >= CONFIDENCE_THRESHOLD else "Unknown"

#         records.append({
#             "span_id": i,
#             "final_clause": final_clause,
#             "confidence": confidence
#         })

#     clause_df = pd.DataFrame(records)

#     # ---- EMBEDDINGS ----
#     norm_spans = [normalize_span(clean_text(s)) for s in spans]
#     embeddings = embedder.encode(norm_spans, batch_size=16)

#     # ---- DEVIATION DETECTION ----
#     deviation_rows = []
#     for idx, row in clause_df.iterrows():
#         clause = row["final_clause"]
#         raw_text = spans[idx].lower()
#         reasons = []

#         if clause == "Unknown" or clause not in clause_centroids:
#             deviation_rows.append({
#                 "final_deviation": False,
#                 "deviation_reasons": []
#             })
#             continue

#         dist = cosine_distances(
#             embeddings[idx].reshape(1, -1),
#             clause_centroids[clause].reshape(1, -1)
#         )[0][0]

#         if dist > clause_thresholds[clause]:
#             reasons.append("Semantic deviation from standard clause language")

#         if polarity_violation(raw_text, clause_polarity_profiles[clause]):
#             reasons.append("Permission / obligation polarity mismatch")

#         if clause == "License Grant" and license_grant_invariant(raw_text):
#             reasons.append("Violation of non-negotiable license ownership invariant")

#         if clause == "Cap On Liability" and cap_liability_invariant(raw_text):
#             reasons.append("Uncapped liability detected")

#         deviation_rows.append({
#             "final_deviation": len(reasons) > 0,
#             "deviation_reasons": reasons
#         })

#     deviation_df = pd.DataFrame(deviation_rows)

#     clause_df = pd.concat(
#         [clause_df.reset_index(drop=True), deviation_df.reset_index(drop=True)],
#         axis=1
#     )

#     return clause_df, spans, embeddings, embedder

# # ---------------- QUESTION ANSWERING ----------------
# def detect_intent(question: str) -> str:
#     q = question.lower()
#     INTENTS = {
#         "RISK_OVERVIEW": ["careful", "risk", "concern", "problem", "danger"],
#         "DEVIATION_EXPLANATION": ["why", "deviation", "non-standard"],
#         "SPECIFIC_CLAUSE": ["clause", "liability", "termination", "warranty", "license"],
#         "OBLIGATION": ["shall", "must", "obligation", "responsible"],
#         "RIGHT": ["right", "may", "allowed", "permission"]
#     }
#     for intent, keywords in INTENTS.items():
#         if any(k in q for k in keywords):
#             return intent
#     return "GENERAL"


# def ask_document(
#     question,
#     clause_df,
#     spans,
#     embeddings,
#     embedder,
#     top_k=5,
#     similarity_threshold=0.35
# ):
#     q = question.lower()

#     # ========================================================
#     # SYSTEM-ANSWERABLE QUESTIONS (NO RETRIEVAL)
#     # ========================================================
#     if any(k in q for k in ["deviat", "risk", "non-standard", "red flag"]):
#         deviating = clause_df[clause_df["final_deviation"]]

#         evidence = []
#         for _, row in deviating.iterrows():
#             reasons = row.get("deviation_reasons", [])
#             evidence.append({
#                 "span_id": row["span_id"],
#                 "clause": row["final_clause"],
#                 "deviating": True,
#                 "reasons": reasons,
#                 "explanations": explain_deviation_reasons(reasons),
#                 "text": spans[row["span_id"]]
#             })

#         return {
#             "intent": "RISK_EXPLANATION",
#             "evidence": evidence,
#             "confidence_notes": [
#                 "Explanations are based on detected deviations and reference contract patterns.",
#                 "This analysis is informational and not legal advice."
#             ]
#         }
#     for clause_name in clause_df["final_clause"].unique():
#         aliases = CLAUSE_ALIASES.get(clause_name, [])
#         if clause_name != "Unknown" and (
#             clause_name.lower() in q or any(a in q for a in aliases)
#         ):
#             matched_rows = clause_df[clause_df["final_clause"] == clause_name]

#             evidence = []
#             for _, row in matched_rows.iterrows():
#                 reasons = row.get("deviation_reasons", [])
#                 evidence.append({
#                     "span_id": row["span_id"],
#                     "clause": clause_name,
#                     "deviating": row["final_deviation"],
#                     "reasons": reasons,
#                     "explanations": explain_deviation_reasons(reasons),
#                     "text": spans[row["span_id"]]
#                 })

#             return {
#                 "intent": "CLAUSE_EXPLANATION",
#                 "evidence": evidence,
#                 "confidence_notes": [
#                     f"Answer based on direct lookup of the '{clause_name}' clause.",
#                     "This is not legal advice."
#                 ]
#             }
#     # ========================================================
#     # RETRIEVAL-BASED QUESTIONS
#     # ========================================================
#     q_emb = embedder.encode([question])[0]
#     sims = cosine_similarity(q_emb.reshape(1, -1), embeddings)[0]

#     candidates = [
#         (i, sims[i])
#         for i in range(len(sims))
#         if sims[i] >= similarity_threshold
#     ]

#     candidates = sorted(candidates, key=lambda x: x[1], reverse=True)[:top_k]

#     evidence = []
#     for idx, _ in candidates:
#         row = clause_df.iloc[idx]
#         reasons = row.get("deviation_reasons", [])

#         evidence.append({
#             "span_id": idx,
#             "clause": row["final_clause"],
#             "deviating": row["final_deviation"],
#             "reasons": reasons,
#             "explanations": explain_deviation_reasons(reasons),
#             "text": spans[idx]
#         })

#     return {
#         "intent": "EVIDENCE_LOOKUP",
#         "evidence": evidence,
#         "confidence_notes": [
#             "Answer derived from semantic similarity and clause analysis.",
#             "This is not legal advice."
#         ]
#     }


# # ============================================================
# # CONTRACT SUMMARY BUILDER (DETERMINISTIC, LLM-SAFE)
# # ============================================================

# def build_contract_summary(clause_df, spans):
#     """
#     Build a deterministic contract summary object from analysis results.
#     This function does NOT use any LLM and has no side effects.
#     """

#     # ---------------- OVERVIEW ----------------
#     total_spans = len(clause_df)
#     recognized_df = clause_df[clause_df["final_clause"] != "Unknown"]
#     unknown_df = clause_df[clause_df["final_clause"] == "Unknown"]
#     deviating_df = clause_df[clause_df["final_deviation"] == True]

#     overview = {
#         "total_spans": int(total_spans),
#         "recognized_clauses": int(len(recognized_df)),
#         "unknown_spans": int(len(unknown_df)),
#         "deviating_spans": int(len(deviating_df))
#     }

#     # ---------------- COVERAGE ----------------
#     detected_clauses = sorted(
#         recognized_df["final_clause"].unique().tolist()
#     )

#     coverage = {
#         "detected_clauses": detected_clauses,
#         "undetected_note": (
#             "Some sections could not be confidently mapped "
#             "to known clause categories."
#         )
#     }

#     # ---------------- DEVIATIONS ----------------
#     deviations = []

#     for _, row in deviating_df.iterrows():
#         sid = int(row["span_id"])
#         reasons = row.get("deviation_reasons", [])

#         # Simple, explainable severity hint
#         severity = "High" if any(
#             "invariant" in r.lower() or "uncapped" in r.lower()
#             for r in reasons
#         ) else "Medium"

#         deviations.append({
#             "clause": row["final_clause"],
#             "span_id": sid,
#             "reasons": reasons,
#             "severity_hint": severity,
#             "excerpt": spans[sid][:300] + ("..." if len(spans[sid]) > 300 else "")
#         })

#     # ---------------- UNKNOWN SECTIONS ----------------
#     unknown_sections = []

#     for _, row in unknown_df.iterrows():
#         sid = int(row["span_id"])
#         unknown_sections.append({
#             "span_id": sid,
#             "excerpt": spans[sid][:300] + ("..." if len(spans[sid]) > 300 else "")
#         })

#     # ---------------- CONFIDENCE NOTES ----------------
#     confidence_notes = [
#         "This summary highlights deviations based on learned reference patterns.",
#         "The absence of a deviation does not guarantee standard or low-risk language.",
#         "This system supports contract review and does not provide legal advice."
#     ]

#     # ---------------- FINAL SUMMARY ----------------
#     contract_summary = {
#         "overview": overview,
#         "coverage": coverage,
#         "deviations": deviations,
#         "unknown_sections": unknown_sections,
#         "confidence_notes": confidence_notes
#     }

#     return contract_summary


# def narrate_contract_summary(summary, llm_client=None):
#     """
#     Guarded LLM narration for contract summary.
#     The LLM NEVER sees the PDF or raw spans — only this summary object.
#     """

#     # ---------------- SAFE FALLBACK (NO LLM) ----------------
#     if llm_client is None:
#         if summary["deviations"]:
#             return (
#                 "This contract contains one or more clauses that deviate from "
#                 "reference standards. Review of the highlighted sections is recommended."
#             )
#         else:
#             return (
#                 "No non-standard clause patterns were detected based on "
#                 "reference baselines."
#             )

#     # ---------------- GUARDED PROMPT ----------------
#     prompt = f"""
# You are summarizing contract analysis findings.

# STRICT RULES:
# - Use ONLY the information provided.
# - Do NOT add legal advice.
# - Do NOT infer or speculate.
# - Do NOT introduce new risks.

# CONTRACT ANALYSIS FINDINGS:
# Overview:
# - Total sections analyzed: {summary['overview']['total_spans']}
# - Deviating sections: {summary['overview']['deviating_spans']}

# Detected Clauses:
# {", ".join(summary['coverage']['detected_clauses'])}

# Deviations:
# {summary['deviations']}

# TASK:
# Write a short, neutral, professional executive summary (3–5 sentences)
# describing what was detected and where attention may be needed.
# """

#     try:
#         response = llm_client(prompt)
#         return response.strip()
#     except Exception:
#         return (
#             "An executive summary could not be generated at this time. "
#             "Please review the detected clauses and deviations manually."
#         )

# ============================================================
# 📄 CONTRACT CLAUSE DEVIATION — PIPELINE (FINAL)
# ============================================================

import pdfplumber
import re
import torch
import numpy as np
import pandas as pd
import streamlit as st

from sklearn.metrics.pairwise import cosine_distances, cosine_similarity
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sentence_transformers import SentenceTransformer


# ============================================================
# 🔎 EXPLANATION ENGINE (WHY LAYER)
# ============================================================

EXPLANATION_TEMPLATES = {
    "Uncapped liability detected": (
        "This clause is considered risky because it does not place a clear limit on liability. "
        "In standard contracts, liability is typically capped to control financial exposure. "
        "Without a cap, potential losses may be unlimited."
    ),
    "Permission / obligation polarity mismatch": (
        "This clause alters the usual balance between obligations and permissions. "
        "Such deviations can create ambiguity in enforcement or responsibility."
    ),
    "Semantic deviation from standard clause language": (
        "The language in this clause differs from commonly observed contract patterns. "
        "Unusual phrasing may increase interpretation risk."
    ),
    "Violation of non-negotiable license ownership invariant": (
        "This clause appears to affect ownership or transfer of rights, which is usually treated "
        "as non-negotiable in standard agreements."
    ),
}
from download_models import ensure_models
ensure_models() 
CLAUSE_ALIASES = {
    "Warranty Duration": ["warranty", "warranty period"],
    "Cap On Liability": ["liability", "liability cap", "cap on liability"],
    "License Grant": ["license", "licensing"],
    "Termination": ["termination", "terminate"],
}

def explain_deviation_reasons(reasons):
    explanations = []
    for r in reasons:
        explanations.append(
            EXPLANATION_TEMPLATES.get(
                r,
                "This clause deviates from standard contractual patterns and may warrant review."
            )
        )
    return explanations


# ============================================================
# ⚙️ CONFIG
# ============================================================

MAX_SEQ_LENGTH = 128
CONFIDENCE_THRESHOLD = 0.5
device = torch.device("cpu")

CLAUSE_MODEL_PATH = "resources/deberta-clause-final"
EMBEDDING_MODEL_NAME = "sentence-transformers/all-mpnet-base-v2"

from download_models import ensure_models
ensure_models()

# ============================================================
# ♻️ CACHED LOADERS
# ============================================================

@st.cache_resource
def load_models():
    tokenizer = AutoTokenizer.from_pretrained(
    CLAUSE_MODEL_PATH,
    use_fast=False,          # 🔑 critical fix
    local_files_only=False   # allow download
)
    model = AutoModelForSequenceClassification.from_pretrained(CLAUSE_MODEL_PATH)
    model.eval()
    embedder = SentenceTransformer(EMBEDDING_MODEL_NAME)
    return tokenizer, model, embedder

@st.cache_resource
def load_baselines():
    return (
        np.load("resources/clause_centroids.npy", allow_pickle=True).item(),
        np.load("resources/clause_thresholds.npy", allow_pickle=True).item(),
        np.load("resources/clause_applicability.npy", allow_pickle=True).item(),
        np.load("resources/clause_polarity.npy", allow_pickle=True).item()
    )

clause_tokenizer, clause_model, embedder = load_models()
clause_centroids, clause_thresholds, clause_applicability_thresholds, clause_polarity_profiles = load_baselines()

ID_TO_CLAUSE = {int(k): v for k, v in clause_model.config.id2label.items()}


# ============================================================
# 🧹 TEXT HELPERS
# ============================================================

def clean_text(text):
    text = text.lower()
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"[^a-z0-9\s]", "", text)
    return text.strip()

def normalize_span(text):
    text = re.sub(r"\b(company|licensor|licensee|producer|party)\b", "party", text)
    text = re.sub(r"\b\d+(\.\d+)?\b", "num", text)
    text = re.sub(r"\b(day|days|month|months|year|years)\b", "time", text)
    return text

def generate_spans(text, min_len=80):
    return [p.strip() for p in text.split("\n\n") if len(p.strip()) >= min_len]


# ============================================================
# 🚨 INVARIANTS
# ============================================================

def license_grant_invariant(text):
    return any(t in text for t in [
        "ownership", "transfer ownership", "full ownership", "full rights"
    ])

def cap_liability_invariant(text):
    return any(t in text for t in [
        "unlimited liability", "without limitation", "no limitation"
    ])

def polarity_violation(text, profile):
    if profile.get("not", 0) > 0.6:
        return any(p in text for p in [
            "freely", "without restriction", "without approval"
        ])
    return False


# ============================================================
# 🧠 MAIN ANALYSIS PIPELINE
# ============================================================

def analyze_document(pdf_path):
    # ---- PDF TO TEXT ----
    pages = []
    with pdfplumber.open(pdf_path) as pdf:
        for p in pdf.pages:
            if p.extract_text():
                pages.append(p.extract_text())

    spans = generate_spans("\n\n".join(pages))

    # ---- CLAUSE CLASSIFICATION ----
    records = []
    for i, span in enumerate(spans):
        encoded = clause_tokenizer(
            span,
            return_tensors="pt",
            truncation=True,
            max_length=MAX_SEQ_LENGTH
        )

        with torch.no_grad():
            logits = clause_model(**encoded).logits

        probs = torch.softmax(logits, dim=1)[0].cpu().numpy()
        pred_id = int(np.argmax(probs))
        pred_clause = ID_TO_CLAUSE[pred_id]
        confidence = float(probs[pred_id])

        final_clause = pred_clause if confidence >= CONFIDENCE_THRESHOLD else "Unknown"

        records.append({
            "span_id": i,
            "final_clause": final_clause,
            "confidence": confidence
        })

    clause_df = pd.DataFrame(records)

    # ---- EMBEDDINGS ----
    norm_spans = [normalize_span(clean_text(s)) for s in spans]
    embeddings = embedder.encode(norm_spans, batch_size=16)

    # ---- DEVIATION DETECTION ----
    deviation_rows = []
    for idx, row in clause_df.iterrows():
        clause = row["final_clause"]
        raw_text = spans[idx].lower()
        reasons = []

        if clause == "Unknown" or clause not in clause_centroids:
            deviation_rows.append({
                "final_deviation": False,
                "deviation_reasons": []
            })
            continue

        dist = cosine_distances(
            embeddings[idx].reshape(1, -1),
            clause_centroids[clause].reshape(1, -1)
        )[0][0]

        if dist > clause_thresholds[clause]:
            reasons.append("Semantic deviation from standard clause language")

        if polarity_violation(raw_text, clause_polarity_profiles[clause]):
            reasons.append("Permission / obligation polarity mismatch")

        if clause == "License Grant" and license_grant_invariant(raw_text):
            reasons.append("Violation of non-negotiable license ownership invariant")

        if clause == "Cap On Liability" and cap_liability_invariant(raw_text):
            reasons.append("Uncapped liability detected")

        deviation_rows.append({
            "final_deviation": len(reasons) > 0,
            "deviation_reasons": reasons
        })

    deviation_df = pd.DataFrame(deviation_rows)

    clause_df = pd.concat(
        [clause_df.reset_index(drop=True), deviation_df.reset_index(drop=True)],
        axis=1
    )

    return clause_df, spans, embeddings, embedder


# ============================================================
# ❓ QUESTION ANSWERING (ML + RAG READY)
# ============================================================

def ask_document(
    question,
    clause_df,
    spans,
    embeddings,
    embedder,
    top_k=5,
    similarity_threshold=0.35
):
    q = question.lower()

    # --- Risk overview ---
    if any(k in q for k in ["risk", "deviation", "non-standard", "red flag"]):
        deviating = clause_df[clause_df["final_deviation"]]

        evidence = []
        for _, row in deviating.iterrows():
            reasons = row.get("deviation_reasons", [])
            evidence.append({
                "span_id": row["span_id"],
                "clause": row["final_clause"],
                "deviating": True,
                "reasons": reasons,
                "explanations": explain_deviation_reasons(reasons),
                "text": spans[row["span_id"]]
            })

        return {
            "intent": "RISK_EXPLANATION",
            "evidence": evidence,
            "confidence_notes": [
                "Explanations are derived from detected deviations.",
                "This analysis is informational and not legal advice."
            ]
        }

    # --- Clause lookup ---
    for clause_name in clause_df["final_clause"].unique():
        aliases = CLAUSE_ALIASES.get(clause_name, [])
        if clause_name != "Unknown" and (
            clause_name.lower() in q or any(a in q for a in aliases)
        ):
            matched = clause_df[clause_df["final_clause"] == clause_name]

            evidence = []
            for _, row in matched.iterrows():
                reasons = row.get("deviation_reasons", [])
                evidence.append({
                    "span_id": row["span_id"],
                    "clause": clause_name,
                    "deviating": row["final_deviation"],
                    "reasons": reasons,
                    "explanations": explain_deviation_reasons(reasons),
                    "text": spans[row["span_id"]]
                })

            return {
                "intent": "CLAUSE_EXPLANATION",
                "evidence": evidence,
                "confidence_notes": [
                    f"Answer based on detected '{clause_name}' clauses.",
                    "This is not legal advice."
                ]
            }

    # --- Semantic retrieval ---
    q_emb = embedder.encode([question])[0]
    sims = cosine_similarity(q_emb.reshape(1, -1), embeddings)[0]

    candidates = sorted(
        [(i, sims[i]) for i in range(len(sims)) if sims[i] >= similarity_threshold],
        key=lambda x: x[1],
        reverse=True
    )[:top_k]

    evidence = []
    for idx, _ in candidates:
        row = clause_df.iloc[idx]
        reasons = row.get("deviation_reasons", [])

        evidence.append({
            "span_id": idx,
            "clause": row["final_clause"],
            "deviating": row["final_deviation"],
            "reasons": reasons,
            "explanations": explain_deviation_reasons(reasons),
            "text": spans[idx]
        })

    return {
        "intent": "EVIDENCE_LOOKUP",
        "evidence": evidence,
        "confidence_notes": [
            "Answer derived from semantic similarity and clause analysis.",
            "This is not legal advice."
        ]
    }


# ============================================================
# 📊 DETERMINISTIC SUMMARY BUILDER (NO LLM)
# ============================================================

def build_contract_summary(clause_df, spans):
    recognized = clause_df[clause_df["final_clause"] != "Unknown"]
    unknown = clause_df[clause_df["final_clause"] == "Unknown"]
    deviating = clause_df[clause_df["final_deviation"]]

    return {
        "overview": {
            "total_spans": int(len(clause_df)),
            "recognized_clauses": int(len(recognized)),
            "unknown_spans": int(len(unknown)),
            "deviating_spans": int(len(deviating)),
        },
        "coverage": {
            "detected_clauses": sorted(recognized["final_clause"].unique().tolist()),
            "undetected_note": "Some sections could not be confidently mapped."
        },
        "deviations": [
            {
                "clause": row["final_clause"],
                "span_id": row["span_id"],
                "reasons": row["deviation_reasons"],
            }
            for _, row in deviating.iterrows()
        ],
        "confidence_notes": [
            "Deviation detection is based on learned reference patterns.",
            "Absence of deviation does not imply low legal risk.",
            "This system does not provide legal advice."
        ]
    }


# ============================================================
# 🧾 GUARDED NARRATION (OPTIONAL LLM)
# ============================================================

def narrate_contract_summary(summary, llm_client=None):
    if llm_client is None:
        return (
            "This contract has been analyzed for standard clause structure and deviations. "
            "Highlighted sections may require closer review."
        )

    prompt = f"""
You are summarizing contract analysis findings.

Rules:
- Use only the provided information
- Do not add legal advice
- Be neutral and concise

Detected clauses:
{", ".join(summary["coverage"]["detected_clauses"])}

Number of deviating sections:
{summary["overview"]["deviating_spans"]}

Write a short executive summary (3–5 sentences).
"""

    try:
        return llm_client(prompt).strip()
    except Exception:
        return "Executive summary could not be generated at this time."
