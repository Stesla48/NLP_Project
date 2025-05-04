from gensim.models import Word2Vec

# === Modeli yükle ===
model_path = "model_lemma_cbow_win4_dim300.model"
model = Word2Vec.load(model_path)

# === Örnek bir kelime için benzer terimler ===
print("En benzer kelimeler (örnek: 'sepsis'):\n")
similar = model.wv.most_similar('sepsis', topn=10)
for word, score in similar:
    print(f"{word} -> {score:.4f}")
