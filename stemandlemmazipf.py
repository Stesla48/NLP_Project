import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from collections import Counter
import ast

# === Veriyi Yükle ===
df = pd.read_csv("preprocessed_data_lemma_and_stem.csv", converters={
    'complaint_lemma': ast.literal_eval,
    'diagnosis_lemma': ast.literal_eval,
    'complaint_stem': ast.literal_eval,
    'diagnosis_stem': ast.literal_eval
})

# === Tüm kelimeleri topla ===
def get_all_tokens(df, col1, col2):
    all_tokens = df[col1].explode().tolist() + df[col2].explode().tolist()
    return [token for token in all_tokens if isinstance(token, str)]

lemma_tokens = get_all_tokens(df, 'complaint_lemma', 'diagnosis_lemma')
stem_tokens = get_all_tokens(df, 'complaint_stem', 'diagnosis_stem')

# === Frekansları say ===
lemma_counts = Counter(lemma_tokens)
stem_counts = Counter(stem_tokens)

# === Sırala ve log-log grafiğe hazırla ===
lemma_freqs = sorted(lemma_counts.values(), reverse=True)
stem_freqs = sorted(stem_counts.values(), reverse=True)

ranks_lemma = range(1, len(lemma_freqs) + 1)
ranks_stem = range(1, len(stem_freqs) + 1)

# === Grafik ===
plt.figure(figsize=(12,5))

plt.subplot(1, 2, 1)
plt.plot(np.log(ranks_lemma), np.log(lemma_freqs))
plt.title("Zipf's Law - Lemmatized")
plt.xlabel("log(Sıra)")
plt.ylabel("log(Frekans)")
plt.grid(True)

plt.subplot(1, 2, 2)
plt.plot(np.log(ranks_stem), np.log(stem_freqs))
plt.title("Zipf's Law - Stemmed")
plt.xlabel("log(Sıra)")
plt.ylabel("log(Frekans)")
plt.grid(True)

plt.tight_layout()
plt.show()
