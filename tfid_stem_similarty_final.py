import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# --- Ana veri setini yükle ---
df_full = pd.read_csv("preprocessed_data_lemma_and_stem.csv")

# -- 'purpra' içeren satırı bul --
row = df_full[df_full['diagnosis_stem'].str.contains('purpra')].iloc[0]
complaint_stem = eval(row['complaint_stem'])
diagnosis_stem = eval(row['diagnosis_stem'])
query_tokens = complaint_stem + diagnosis_stem

# --- TF-IDF matrisini yükle ---
df = pd.read_csv("tfidf_stemmed.csv")
vector_cols = [col for col in df.columns if col not in ['subject_id', 'hadm_id']]

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
result_stem = df.iloc[top5_idx][['subject_id', 'hadm_id']].copy()
result_stem['similarity_score'] = similarities[top5_idx]

# --- Zenginleştirme ---
complaints = []
diagnoses = []

for sid, hid in zip(result_stem['subject_id'], result_stem['hadm_id']):
    row2 = df_full[(df_full['subject_id'] == sid) & (df_full['hadm_id'] == hid)].iloc[0]
    complaints.append(row2['complaint_stem'])
    diagnoses.append(row2['diagnosis_stem'])

result_stem['complaint_stem'] = complaints
result_stem['diagnosis_stem'] = diagnoses

print(result_stem)
# --- Sonucu CSV'ye kaydetmek istersen ---
result_stem.to_csv("tfidf_stem_similarity_final.csv", index=False)
