import faiss
import os
import pickle
from sentence_transformers import SentenceTransformer

# 📍 Chemins des fichiers d'index
INDEX_PATH = "C:/Users/fred_/OneDrive/ML/RAGonPDF/Index_faiss"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# 🔄 Charger l'index FAISS et les métadonnées
index = faiss.read_index(os.path.join(INDEX_PATH, "faiss.index"))

with open(os.path.join(INDEX_PATH, "metadata.pkl"), "rb") as f:
    metadata = pickle.load(f)

# 🔹 Charger le modèle d'embeddings
model = SentenceTransformer(EMBEDDING_MODEL)

def search(query, top_k=5):
    """ Recherche les passages les plus pertinents en fonction d'une requête. """
    query_embedding = model.encode([query], convert_to_numpy=True)
    distances, indices = index.search(query_embedding, top_k)

    results = []
    for i in range(top_k):
        idx = indices[0][i]
        filename, segment_id, text = metadata[idx]
        results.append((filename, segment_id, text, distances[0][i]))

    return results

# 🧐 Exemple d'utilisation
query = "promitente compradora"
results = search(query)

for filename, segment_id, text, score in results:
    print(f"\n📄 {filename} - Segment {segment_id+1} (Score: {score:.4f})\n{text}")

