from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from rouge_score import rouge_scorer

# === Données à évaluer ===
question = "Qu’est-ce que la myopie axile ?"

retrieved_context = """
La myopie axile est la forme la plus fréquente de myopie. Elle est causée par un allongement de l’axe antéro-postérieur de l’œil, ce qui provoque un flou de la vision de loin.
"""

generated_answer = """
La myopie axile est causée par un œil plus long que la normale, ce qui déplace le point focal en avant de la rétine.
"""

# === Score de similarité (fidélité) ===
vectorizer = TfidfVectorizer().fit([retrieved_context, generated_answer])
vecs = vectorizer.transform([retrieved_context, generated_answer])
cos_sim = cosine_similarity(vecs[0], vecs[1])[0][0]

# === Score de ROUGE (pertinence lexicale) ===
scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
rouge = scorer.score(retrieved_context, generated_answer)

# === Résultats ===
print("🔎 Similarité Cosine (Fidélité) :", round(cos_sim, 3))
print("📚 ROUGE-L F1 (Pertinence lexicale) :", round(rouge["rougeL"].fmeasure, 3))
