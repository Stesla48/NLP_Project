import pandas as pd
import numpy as np
import ast
from sklearn.feature_extraction.text import TfidfVectorizer

# === Veriyi Yükle ===
df = pd.read_csv("preprocessed_data_lemma_and_stem.csv", converters={
    'complaint_lemma': ast.literal_eval,
    'diagnosis_lemma': ast.literal_eval
})

# === Belgeleri oluştur ===
df['combined_text'] = df['complaint_lemma'].apply(lambda x: ' '.join(x)) + ' ' + df['diagnosis_lemma'].apply(lambda x: ' '.join(x))

# === TF-IDF Vektörleştirme ===
vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform(df['combined_text'])

# === DataFrame'e çevir ===
tfidf_df = pd.DataFrame(tfidf_matrix.toarray(), columns=vectorizer.get_feature_names_out())
tfidf_df.insert(0, "hadm_id", df['hadm_id'])
tfidf_df.insert(0, "subject_id", df['subject_id'])

# === CSV olarak kaydet ===
tfidf_df.to_csv("tfidf_lemmatized.csv", index=False)
print("Tam TF-IDF matrisi kaydedildi: tfidf_lemmatized.csv")
