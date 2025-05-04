import pandas as pd
import numpy as np
import string
from nltk.corpus import stopwords
from gensim.models import Word2Vec
from sklearn.metrics.pairwise import cosine_similarity
import nltk
nltk.download('stopwords')

# === Temizleme Fonksiyonu ===
def preprocess_text(text):
    text = str(text).lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    tokens = text.split()  # NLTK yerine split() kullanılıyor
    stop_words = set(stopwords.words('english'))
    return [w for w in tokens if w not in stop_words]

# === Vektör Hesabı ===
def get_vec(tokens, model, vector_size=100):
    vecs = [model.wv[word] for word in tokens if word in model.wv]
    return np.mean(vecs, axis=0) if vecs else np.zeros(vector_size)

# === Veriyi Yükle ===
data_dir = "mimic/"
admissions = pd.read_csv(data_dir + "ADMISSIONS.csv")
diagnoses = pd.read_csv(data_dir + "DIAGNOSES_ICD.csv")
diag_desc = pd.read_csv(data_dir + "D_ICD_DIAGNOSES.csv")

# Küçük harf yap
admissions.columns = admissions.columns.str.lower()
diagnoses.columns = diagnoses.columns.str.lower()
diag_desc.columns = diag_desc.columns.str.lower()

# Eşleştir
merged = pd.merge(admissions, diagnoses, on=["subject_id", "hadm_id"])
merged = pd.merge(merged, diag_desc, how="left", on="icd9_code")
merged = merged.dropna(subset=["short_title"])

# Şikayet ve teşhis
merged['complaint'] = merged['admission_type'] + " " + merged['admission_location']
merged['diagnosis'] = merged['short_title']

# Temizleme
merged['complaint_tokens'] = merged['complaint'].apply(preprocess_text)
merged['diagnosis_tokens'] = merged['diagnosis'].apply(preprocess_text)

# Word2Vec eğitimi
sentences = merged['complaint_tokens'].tolist() + merged['diagnosis_tokens'].tolist()
w2v = Word2Vec(sentences, vector_size=100, window=5, min_count=1, workers=4)

# Vektörleştir
merged['complaint_vec'] = merged['complaint_tokens'].apply(lambda x: get_vec(x, w2v))
merged['diagnosis_vec'] = merged['diagnosis_tokens'].apply(lambda x: get_vec(x, w2v))

# Benzerlik
merged['similarity'] = merged.apply(lambda row: cosine_similarity([row['complaint_vec']], [row['diagnosis_vec']])[0][0], axis=1)

# En iyi eşleşmeleri yazdır
top_matches = merged.sort_values(by='similarity', ascending=False).head(10)
print("\n--- En Benzer Şikayet - Teşhis Eşleşmeleri ---\n")
print(top_matches[['complaint', 'diagnosis', 'similarity']])
