import pandas as pd
import numpy as np
import re
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer, PorterStemmer

# Gerekli verileri indir (ilk çalıştırmada)
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')

# === Lemmatization fonksiyonu ===
def preprocess_lemma(text):
    if pd.isnull(text):
        return []
    text = text.lower()
    text = text.encode('ascii', 'ignore').decode()
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = re.sub(r'\d+', '', text)
    tokens = text.split()
    stop_words = set(stopwords.words('english'))
    tokens = [w for w in tokens if w not in stop_words and len(w) > 2]
    lemmatizer = WordNetLemmatizer()
    return [lemmatizer.lemmatize(w) for w in tokens]

# === Stemming fonksiyonu ===
def apply_stemming(tokens):
    stemmer = PorterStemmer()
    return [stemmer.stem(w) for w in tokens]

# === Veriyi Yükle ===
data_dir = "mimic/"
admissions = pd.read_csv(data_dir + "ADMISSIONS.csv")
diagnoses = pd.read_csv(data_dir + "DIAGNOSES_ICD.csv")
diag_desc = pd.read_csv(data_dir + "D_ICD_DIAGNOSES.csv")

# Küçük harfe çevir
admissions.columns = admissions.columns.str.lower()
diagnoses.columns = diagnoses.columns.str.lower()
diag_desc.columns = diag_desc.columns.str.lower()

# Tabloları birleştir
merged = pd.merge(admissions, diagnoses, on=["subject_id", "hadm_id"])
merged = pd.merge(merged, diag_desc, how="left", on="icd9_code")
merged = merged.dropna(subset=["short_title"])

# Şikayet ve teşhis metinlerini oluştur
merged['complaint'] = merged['admission_type'] + " " + merged['admission_location']
merged['diagnosis'] = merged['short_title']

# Lemmatize et
merged['complaint_lemma'] = merged['complaint'].apply(preprocess_lemma)
merged['diagnosis_lemma'] = merged['diagnosis'].apply(preprocess_lemma)

# Stem uygula (lemmatized tokenlar üzerine)
merged['complaint_stem'] = merged['complaint_lemma'].apply(apply_stemming)
merged['diagnosis_stem'] = merged['diagnosis_lemma'].apply(apply_stemming)

# === Kaydet ===
output_path = "preprocessed_data_lemma_and_stem.csv"
merged[['subject_id', 'hadm_id', 'complaint_lemma', 'complaint_stem', 'diagnosis_lemma', 'diagnosis_stem']].to_csv(output_path, index=False)

print(f"\nLemma ve Stem ayrı ayrı olacak şekilde '{output_path}' dosyasına kaydedildi!")
