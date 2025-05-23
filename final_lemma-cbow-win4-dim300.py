import pandas as pd
import numpy as np
from gensim.models import Word2Vec
from sklearn.metrics.pairwise import cosine_similarity

# --- Modeli yükle ---
model = Word2Vec.load("model_lemma_cbow_win4_dim300.model")
vector_size = model.vector_size

# --- Ana veri setini yükle ---
df_full = pd.read_csv("preprocessed_data_lemma_and_stem.csv")

# --- Query tokenlarını seç (lemma için) ---
query_tokens = ['emergency', 'transfer', 'hospextram', 'immune', 'thrombocyt', 'purpra']

# --- Ortalama vektör hesaplama fonksiyonu ---
def get_mean_vector(tokens):
    vectors = [model.wv[word] for word in tokens if word in model.wv]
    if not vectors:
        return np.zeros(vector_size)
    return np.mean(vectors, axis=0)

query_vec = get_mean_vector(query_tokens).reshape(1, -1)

# --- Tüm corpus için vektörleri hesapla (lemma tokenları ile) ---
all_vectors = []
for idx, row in df_full.iterrows():
    tokens = eval(row['complaint_lemma']) + eval(row['diagnosis_lemma'])
    all_vectors.append(get_mean_vector(tokens))
X = np.vstack(all_vectors)

# --- Cosine similarity ---
similarities = cosine_similarity(query_vec, X).flatten()
top5_idx = similarities.argsort()[-5:][::-1]

# --- Sonuç tablosu ---
result = df_full.iloc[top5_idx][['subject_id', 'hadm_id', 'complaint_lemma', 'diagnosis_lemma']].copy()
result['similarity_score'] = similarities[top5_idx]
print(result)
# ---  CSV'ye kaydet ---
result.to_csv("final_lemma_cbow_win4_dim300_similarity.csv", index=False)
