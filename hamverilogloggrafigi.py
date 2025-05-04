import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from collections import Counter
import re

# === Veriyi Yükle ===
df = pd.read_csv("mimic/ADMISSIONS.csv")
diag = pd.read_csv("mimic/D_ICD_DIAGNOSES.csv")
diag_icd = pd.read_csv("mimic/DIAGNOSES_ICD.csv")

# Sütun isimlerini küçült
df.columns = df.columns.str.lower()
diag.columns = diag.columns.str.lower()
diag_icd.columns = diag_icd.columns.str.lower()

# ICD-9 kod açıklamaları ile teşhisleri birleştir
merged = pd.merge(df, diag_icd, on=["subject_id", "hadm_id"])
merged = pd.merge(merged, diag, how="left", on="icd9_code")

# Ham şikayet ve teşhis metinleri
merged['complaint'] = merged['admission_type'].fillna('') + " " + merged['admission_location'].fillna('')
merged['diagnosis'] = merged['short_title'].fillna('')

# Temizleme ve tokenizasyon
def tokenize(text):
    text = str(text).lower()
    text = re.sub(r'[^a-z\s]', '', text)  # sadece harf ve boşluk
    return text.split()

all_tokens = []
for col in ['complaint', 'diagnosis']:
    merged[col + '_tokens'] = merged[col].apply(tokenize)
    all_tokens.extend(merged[col + '_tokens'].sum())

# Frekansları say
word_counts = Counter(all_tokens)
most_common = word_counts.most_common()
words, counts = zip(*most_common)

# Sıralama
ranks = np.arange(1, len(counts) + 1)

# === Zipf Grafiği ===
plt.figure(figsize=(10, 6))
plt.plot(np.log(ranks), np.log(counts))
plt.title("Zipf's Law (Ham Veri Üzerinde)", fontsize=14)
plt.xlabel("log(Sıra)", fontsize=12)
plt.ylabel("log(Frekans)", fontsize=12)
plt.grid(True)

# İlk 10 kelimeyi etiketle
for i in range(min(10, len(words))):
    plt.annotate(words[i], (np.log(ranks[i]), np.log(counts[i])), fontsize=9)

plt.tight_layout()
plt.savefig("zipf_ham_grafik_detayli.png")
plt.show()
