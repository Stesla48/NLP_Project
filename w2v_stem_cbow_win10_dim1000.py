import pandas as pd
import numpy as np
from gensim.models import Word2Vec
from sklearn.metrics.pairwise import cosine_similarity
import ast

# === Parametreler ===
window_size = 10
vector_size = 1000
sg = 0  # CBOW
output_csv = f"similarity_stem_cbow_win{window_size}_dim{vector_size}.csv"
model_file = f"model_stem_cbow_win{window_size}_dim{vector_size}.model"

# === Veriyi Yükle ===
df = pd.read_csv("preprocessed_data_lemma_and_stem.csv", converters={
    'complaint_stem': ast.literal_eval,
    'diagnosis_stem': ast.literal_eval
})

# === Eğitim Verisi: Şikayet + Teşhis
sentences = df['complaint_stem'].tolist() + df['diagnosis_stem'].tolist()

# === Word2Vec Eğitimi
model = Word2Vec(sentences, vector_size=vector_size, window=window_size, sg=sg, min_count=1, workers=4)

# === Ortalama Vektör Hesapla
def get_mean_vector(tokens):
    vectors = [model.wv[word] for word in tokens if word in model.wv]
    return np.mean(vectors, axis=0) if vectors else np.zeros(vector_size)

# === Vektörleri Hesapla
df['complaint_vec'] = df['complaint_stem'].apply(get_mean_vector)
df['diagnosis_vec'] = df['diagnosis_stem'].apply(get_mean_vector)

# === Cosine Similarity Hesapla
df['similarity'] = df.apply(lambda row: cosine_similarity(
    [row['complaint_vec']], [row['diagnosis_vec']])[0][0], axis=1)

# === Kaydet
df[['subject_id', 'hadm_id', 'similarity']].to_csv(output_csv, index=False)
model.save(model_file)

print(f"Similarity CSV kaydedildi: {output_csv}")
print(f"Word2Vec modeli kaydedildi: {model_file}")
