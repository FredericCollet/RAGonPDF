import faiss
import pickle
from fastapi import FastAPI
from pydantic import BaseModel
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

# 🚀 Initialisation de FastAPI
app = FastAPI(title="RAG API", description="API pour rechercher des passages pertinents dans les PDF indexés.")

class QueryRequest(BaseModel):
    query: str
    top_k: int = 5

@app.get("/")
def home():
    return {"message": "Bienvenue sur l'API RAG 🔍"}

@app.post("/search/")
def search_documents(request: QueryRequest):
    """ Recherche les passages les plus pertinents en fonction d'une requête. """
    query_embedding = model.encode([request.query], convert_to_numpy=True)
    distances, indices = index.search(query_embedding, request.top_k)

    results = []
    for i in range(request.top_k):
        idx = indices[0][i]
        filename, segment_id, text = metadata[idx]
        results.append({
            "filename": filename,
            "segment_id": segment_id,
            "text": text,
            "score": float(distances[0][i])
        })

    return {"query": request.query, "results": results}
