import pandas as pd
import ast

# Veri dosyasını yükle
df = pd.read_csv("preprocessed_data_lemma_and_stem.csv", converters={
    'complaint_lemma': ast.literal_eval,
    'diagnosis_lemma': ast.literal_eval,
    'complaint_stem': ast.literal_eval,
    'diagnosis_stem': ast.literal_eval,
})

# Tüm lemma kelimeleri birleştir
all_lemma = df['complaint_lemma'].explode().tolist() + df['diagnosis_lemma'].explode().tolist()
unique_lemma = set(all_lemma)

# Tüm stem kelimeleri birleştir
all_stem = df['complaint_stem'].explode().tolist() + df['diagnosis_stem'].explode().tolist()
unique_stem = set(all_stem)

# Sonuçları yazdır
print(f"Toplam benzersiz lemma kelime sayısı: {len(unique_lemma)}")
print(f"Toplam benzersiz stem kelime sayısı: {len(unique_stem)}")
