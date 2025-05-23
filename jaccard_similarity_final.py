import pandas as pd
import numpy as np

# Tabloyu burada oluştur
data = [
    # ('Model Adı', [ilk5 subject-hadm id])
    ('tfidf_stem_similarity_final.csv', ['44212-163189', '10088-149044', '10059-142582', '10019-177759', '41914-101361']),
    ('tfidf_lemma_similarity_final.csv', ['44212-163189', '10088-149044', '42075-151323', '43827-149950', '10059-142582']),
    ('final_lemma_cbow_win4_dim300_similarity.csv', ['44212-163189', '40304-174997', '10011-105331', '44212-163189', '42321-114648']),
    ('final_lemma_cbow_win4_dim1000_similarity.csv', ['44212-163189', '40304-174997', '44212-163189', '42321-114648', '10011-105331']),
    ('final_lemma_cbow_win10_dim300_similarity.csv', ['44212-163189', '43870-142633', '40304-174997', '44212-163189', '10011-105331']),
    ('final_lemma_cbow_win10_dim1000_similarity.csv', ['44212-163189', '42321-114648', '44212-163189', '40304-174997', '10011-105331']),
    ('final_lemma_skipgram_win4_dim300_similarity.csv', ['44212-163189', '43870-142633', '42321-114648', '10088-149044', '10088-149044']),
    ('final_lemma_skipgram_win4_dim1000_similarity.csv', ['44212-163189', '40595-116518', '10088-149044', '43870-142633', '10042-148562']),
    ('final_lemma_skipgram_win10_dim300_similarity.csv', ['44212-163189', '43870-142633', '42321-114648', '41914-101361', '10088-149044']),
    ('final_lemma_skipgram_win10_dim1000_similarity.csv', ['44212-163189', '10042-148562', '41914-101361', '43870-142633', '10088-149044']),
    ('final_stem_cbow_win4_dim300_similarity.csv', ['44212-163189', '40177-198480', '42321-114648', '44212-163189', '42321-114648']),
    ('final_stem_cbow_win4_dim1000_similarity.csv', ['44212-163189', '42321-114648', '40304-174997', '44212-163189', '10011-105331']),
    ('final_stem_cbow_win10_dim300_similarity.csv', ['44212-163189', '40177-198480', '42321-114648', '40310-186361', '44212-163189']),
    ('final_stem_cbow_win10_dim1000_similarity.csv', ['44212-163189', '42321-114648', '40304-174997', '44212-163189', '10011-105331']),
    ('final_stem_skipgram_win4_dim300_similarity.csv', ['44212-163189', '42321-114648', '40310-186361', '43870-142633', '41914-101361']),
    ('final_stem_skipgram_win4_dim1000_similarity.csv', ['44212-163189', '40595-116518', '10088-149044', '42321-114648', '43870-142633']),
    ('final_stem_skipgram_win10_dim300_similarity.csv', ['44212-163189', '40310-186361', '42321-114648', '10042-148562', '40503-168803']),
    ('final_stem_skipgram_win10_dim1000_similarity.csv', ['44212-163189', '40595-116518', '43870-142633', '10120-193924', '10088-149044']),
]

models = [x[0] for x in data]
top5_lists = [set(x[1]) for x in data]

n = len(models)
jaccard_matrix = np.zeros((n, n))

for i in range(n):
    for j in range(n):
        intersect = len(top5_lists[i] & top5_lists[j])
        union = len(top5_lists[i] | top5_lists[j])
        jaccard = intersect / union if union != 0 else 1.0
        jaccard_matrix[i, j] = round(jaccard, 2)

# Sonuçları DataFrame olarak göster ve kaydet
df_jaccard = pd.DataFrame(jaccard_matrix, columns=models, index=models)
print(df_jaccard)
df_jaccard.to_csv("jaccard_similarity_matrix.csv")
