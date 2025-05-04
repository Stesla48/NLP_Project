# 🧠 MIMIC-III Tabanlı Şikayet-Teşhis Benzerlik Modeli

Bu projede MIMIC-III klinik veri setinden elde edilen hasta şikayetleri ve teşhisleri arasında anlamsal benzerlik hesaplanmıştır. Doğal dil işleme (NLP) teknikleriyle TF-IDF ve Word2Vec modelleri kullanılarak semantik benzerlik skorları üretilmiştir.

---

## 📚 İçindekiler | Table of Contents

- [🔍 Proje Amacı](#-proje-amacı)
- [📦 Kullanılan Veri Setleri](#-kullanılan-veri-setleri)
- [🧹 Veri Önişleme Adımları](#-veri-önişleme-adımları)
- [🧮 Vektörleştirme Yöntemleri](#-vektörleştirme-yöntemleri)
  - [TF-IDF](#tf-idf)
  - [Word2Vec](#word2vec)
- [📈 Modellerin Eğitimi](#-modellerin-eğitimi)
- [📝 Modelden Örnek Çıktılar](#-modelden-örnek-çıktılar)
- [⚙️ Kurulum Talimatları](#️-kurulum-talimatları)
- [🚀 Çalıştırma Adımları](#-çalıştırma-adımları)
- [📌 Sonuç ve Değerlendirme](#-sonuç-ve-değerlendirme)
- [👤 Geliştirici Bilgisi](#-geliştirici-bilgisi)

---

## 🔍 Proje Amacı

MIMIC-III veritabanındaki hasta kayıtları kullanılarak, şikayet (admission_type + admission_location) ve teşhis (ICD açıklamaları) cümleleri arasında vektör temelli benzerlik hesaplamaları yapılmıştır. Bu yapı, ileri düzey öneri sistemlerinin ve otomatik teşhis eşleştirmelerinin temelini oluşturabilir.

---

## 📦 Kullanılan Veri Setleri

| Dosya Adı             | Açıklama                                                |
|------------------------|----------------------------------------------------------|
| `ADMISSIONS.csv`       | Hastanın kabul tipi ve kabul lokasyonu (şikayet gibi)    |
| `DIAGNOSES_ICD.csv`    | Hasta kabulüne bağlı ICD kodları                         |
| `D_ICD_DIAGNOSES.csv`  | ICD kodlarının kısa açıklamaları                         |

Veriler `subject_id` ve `hadm_id` üzerinden birleştirilerek analiz yapılmıştır.

---

## 🧹 Veri Önişleme Adımları

Kütüphaneler: `pandas`, `re`, `nltk`, `string`

1. Tüm metinler küçük harfe çevrildi. (Lowercasing)
2. ASCII dışı karakterler temizlendi.
3. Noktalama işaretleri ve sayılar kaldırıldı.
4. Tokenization uygulandı.
5. İngilizce stopword’ler çıkarıldı. (NLTK stopwords)
6. 3 harften kısa kelimeler çıkarıldı.
7. Lemmatization: `WordNetLemmatizer()`
8. Stemming: `PorterStemmer()`

📁 Önişleme sonrası: `preprocessed_data_lemma_and_stem.csv`

---

## 🧮 Vektörleştirme Yöntemleri

### TF-IDF

- `TfidfVectorizer` ile lemma ve stem verileri ayrı ayrı vektörleştirildi.
- Ardından cosine similarity hesaplandı.
- Teşhis verisinin dahil edilmesiyle similarity skorları anlamlı hale geldi.
- Çıktılar: `tfidf_matrix_lemma.csv`, `tfidf_matrix_stem.csv`

### Word2Vec

Gensim kütüphanesi ile 16 farklı model eğitildi.

- 2 mimari: CBOW ve SkipGram
- 2 pencere boyutu: 4 ve 10
- 2 boyut sayısı: 300 ve 1000
- 2 veri tipi: Lemmatized & Stemmed

📁 Örnek model dosyası: `model_lemma_cbow_win4_dim300.model`

---

## 📈 Modellerin Eğitimi

```python
model = Word2Vec(sentences, vector_size=300, window=4, sg=0)
cosine_similarity([row['complaint_vec']], [row['diagnosis_vec']])
```

Her model için benzerlik hesaplaması yapılmış ve `.csv` dosyası olarak kaydedilmiştir.

---

## 📝 Modelden Örnek Çıktılar

📌 *Örnek model*: `model_lemma_cbow_win4_dim300.model`

```python
model.wv.most_similar("sepsis", topn=5)
```

📤 Çıktı:

```
('chr', 0.9968)
('oth', 0.9968)
('nec', 0.9968)
('pressure', 0.9967)
('mal', 0.9965)
```

---

## ⚙️ Kurulum Talimatları

```bash
pip install pandas numpy scikit-learn nltk gensim matplotlib
```

İlk çalıştırmada NLTK için ek veri setlerini indirin:

```python
import nltk
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')
```

---

## 🚀 Çalıştırma Adımları

### 1. Veriyi Ön İşleyin:

```bash
python preprocessing.py
```

### 2. TF-IDF için çalıştırın:

```bash
python tfidf_lemma.py
python tfidf_stem.py
```

### 3. Word2Vec modellerini çalıştırın:

```bash
python w2v_lemma_cbow_win4_dim300.py
python w2v_stem_skipgram_win10_dim1000.py
...
```

Her model için `.model` ve `similarity_...csv` dosyaları oluşur.

---

## 📌 Sonuç ve Değerlendirme

- TF-IDF modeli bağlam bilgisi taşımadığı için sınırlı performans gösterdi.
- Teşhislerin eklenmesiyle skorlar anlamlı hale geldi.
- Word2Vec modelleri, kelimeler arası anlamsal benzerlikleri daha iyi yakaladı.
- CBOW + lemma + window 4 + dim 300 kombinasyonu en başarılı sonuçları verdi.

---

## 👤 Geliştirici Bilgisi

- İsim: Selim Yağlıoğlu
- Ders: Doğal Dil İşleme (2025)
- Proje: MIMIC-III Şikayet-Teşhis Benzerlik Analizi

