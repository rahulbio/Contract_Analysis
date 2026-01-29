import numpy as np
import pandas as pd
import ast
import re
from collections import defaultdict
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_distances

TRAIN_CSV_PATH = r"C:\Users\Rahul K\OneDrive\Desktop\contract_deviation_app\resources\final_cleaned_version (2).csv"
EMBEDDING_MODEL_NAME = "sentence-transformers/all-mpnet-base-v2"

def clean_text(text):
    text = text.lower()
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"[^a-z0-9\s]", "", text)
    return text.strip()

def normalize_span(text):
    text = re.sub(r"\b(company|licensor|licensee|producer|ma|ent)\b", "party", text)
    text = re.sub(r"\b\d+(\.\d+)?\b", "num", text)
    text = re.sub(r"\b(day|days|month|months|year|years)\b", "time", text)
    return text

def extract_span_text(span):
    try:
        lst = ast.literal_eval(span)
        if isinstance(lst, list) and len(lst) > 0:
            return lst[0]
    except:
        return None

df = pd.read_csv(TRAIN_CSV_PATH)
df["span_text"] = df["Span"].apply(extract_span_text)
df = df.dropna(subset=["span_text"])
df["norm_span"] = df["span_text"].apply(lambda x: normalize_span(clean_text(x)))

embedder = SentenceTransformer(EMBEDDING_MODEL_NAME)
embeddings = embedder.encode(df["norm_span"].tolist(), batch_size=32)

clause_embeddings = defaultdict(list)
for emb, clause in zip(embeddings, df["Clause"]):
    clause_embeddings[clause].append(emb)

centroids, thresholds, applicability = {}, {}, {}

for clause, embs in clause_embeddings.items():
    embs = np.vstack(embs)
    centroid = embs.mean(axis=0)
    dists = cosine_distances(embs, centroid.reshape(1, -1)).flatten()
    centroids[clause] = centroid
    thresholds[clause] = np.percentile(dists, 95)
    applicability[clause] = np.percentile(dists, 99)

def polarity_profile(df, clause):
    texts = df[df["Clause"] == clause]["norm_span"]
    signals = ["shall", "may", "must", "not", "without", "freely"]
    return {s: sum(s in t for t in texts) / len(texts) for s in signals}

polarity_profiles = {
    c: polarity_profile(df, c) for c in df["Clause"].unique()
}

np.save("clause_centroids.npy", centroids)
np.save("clause_thresholds.npy", thresholds)
np.save("clause_applicability.npy", applicability)
np.save("clause_polarity.npy", polarity_profiles)

print("✅ Baselines saved")
