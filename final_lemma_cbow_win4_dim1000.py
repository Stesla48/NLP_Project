import pandas as pd
import numpy as np
from gensim.models import Word2Vec
from sklearn.metrics.pairwise import cosine_similarity

model_path = "model_lemma_cbow_win4_dim1000.model"
model = Word2Vec.load(model_path)
vector_size = model.vector_size

df_full = pd.read_csv("preprocessed_data_lemma_and_stem.csv")
query_tokens = ['emergency', 'transfer', 'hospextram', 'immune', 'thrombocyt', 'purpra']

def get_mean_vector(tokens):
    vectors = [model.wv[word] for word in tokens if word in model.wv]
    if not vectors:
        return np.zeros(vector_size)
    return np.mean(vectors, axis=0)

query_vec = get_mean_vector(query_tokens).reshape(1, -1)

all_vectors = []
for idx, row in df_full.iterrows():
    tokens = eval(row['complaint_lemma']) + eval(row['diagnosis_lemma'])
    all_vectors.append(get_mean_vector(tokens))
X = np.vstack(all_vectors)

similarities = cosine_similarity(query_vec, X).flatten()
top5_idx = similarities.argsort()[-5:][::-1]

result = df_full.iloc[top5_idx][['subject_id', 'hadm_id', 'complaint_lemma', 'diagnosis_lemma']].copy()
result['similarity_score'] = similarities[top5_idx]
print(result)
result.to_csv("final_lemma_cbow_win4_dim1000_similarity.csv", index=False)
