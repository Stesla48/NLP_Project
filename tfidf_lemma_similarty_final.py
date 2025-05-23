import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# --- TF-IDF matrisini yükle ---
df = pd.read_csv("tfidf_lemmatized.csv")
vector_cols = [col for col in df.columns if col not in ['subject_id', 'hadm_id']]

# --- Yeni query tokenlar ---
query_tokens = ['emergency', 'transfer', 'hospextram', 'immune', 'thrombocyt', 'purpra']

# --- Query vektörü oluştur ---
query_vec = np.zeros(len(vector_cols))
for idx, token in enumerate(vector_cols):
    query_vec[idx] = query_tokens.count(token)

query_vec = query_vec.reshape(1, -1)
X = df[vector_cols].values

# --- Cosine similarity hesapla ---
similarities = cosine_similarity(query_vec, X).flatten()
top5_idx = similarities.argsort()[-5:][::-1]

# --- Sonuç tablosu ---
result_lemma = df.iloc[top5_idx][['subject_id', 'hadm_id']].copy()
result_lemma['similarity_score'] = similarities[top5_idx]

# --- Zenginleştirme ---
df_full = pd.read_csv("preprocessed_data_lemma_and_stem.csv")
complaints = []
diagnoses = []

for sid, hid in zip(result_lemma['subject_id'], result_lemma['hadm_id']):
    row = df_full[(df_full['subject_id'] == sid) & (df_full['hadm_id'] == hid)].iloc[0]
    complaints.append(row['complaint_lemma'])
    diagnoses.append(row['diagnosis_lemma'])

result_lemma['complaint_lemma'] = complaints
result_lemma['diagnosis_lemma'] = diagnoses

print(result_lemma)
# --- Sonuçları CSV dosyasına kaydet ---
result_lemma.to_csv("tfidf_lemma_similarity_final.csv", index=False)