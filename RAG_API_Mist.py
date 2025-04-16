import os
import faiss
import pickle
import httpx
from fastapi import FastAPI
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer

# 📍 Chemins des fichiers d'index
INDEX_PATH = "C:/Users/fred_/OneDrive/ML/RAGonPDF/Index_faiss"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
OLLAMA_URL = "http://localhost:11434"

def check_ollama():
    try:
        response = httpx.get(f"{OLLAMA_URL}/", timeout=10.0)
        return response.status_code == 200
    except httpx.RequestError:
        return False


# 🔄 Charger l'index FAISS et les métadonnées
index = faiss.read_index(os.path.join(INDEX_PATH, "faiss.index"))

with open(os.path.join(INDEX_PATH, "metadata.pkl"), "rb") as f:
    metadata = pickle.load(f)

# 🔹 Charger le modèle d'embeddings
model = SentenceTransformer(EMBEDDING_MODEL)

# 🚀 Initialisation de FastAPI
app = FastAPI(title="RAG API avec Mistral", description="API pour rechercher et générer des réponses enrichies.")

class QueryRequest(BaseModel):
    query: str
    top_k: int = 3

@app.get("/")
def home():
    return {"message": "Bienvenue sur l'API RAG avec Mistral 🔍 mod 01"}

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

    print(f"Fin de la Search ....")

    return {"query": request.query, "results": results}

@app.post("/rag/")
def rag_generate(request: QueryRequest):
    """ Recherche les passages pertinents et génère une réponse avec Mistral via Ollama. """
    
    # ⚠️ Vérifie que Ollama est actif
    if not check_ollama():
        return {
            "query": request.query,
            "documents": [],
            "generated_response": "Erreur : le serveur Ollama ne répond pas. Veuillez lancer 'ollama run mistral'."
        }
    
    print(f"Ollama up and running")
        
    
    # 🔍 Recherche des documents pertinents
    search_results = search_documents(request)["results"]
    
    # 📜 Construction du prompt avec les passages récupérés
    context = "\n\n".join([f"Document: {res['filename']}\nTexte: {res['text']}" for res in search_results])
    
    prompt = f"""
    Tu es un assistant intelligent. Réponds à la question suivante en utilisant uniquement les informations suivantes :
    
    {context}
    
    Question : {request.query}
    Réponse :
    """
    
    # 🚀 Appel à l'API locale Ollama pour générer une réponse
    print(f"Appel API locale Ollama ")
    
    """
    response = httpx.post(
        "http://localhost:11434/api/generate",
         json={"model": "mistral", "prompt": prompt},
         timeout=10.0  # ← max 10 secondes d'attente
    )


    if response.status_code == 200:
        generated_text = response.json()["response"]
    else:
        generated_text = "Erreur lors de la génération avec Ollama."

    """
    try:
        response = httpx.post(
            "http://localhost:11434/api/generate",
            json={"model": "mistral", "prompt": prompt},
            timeout=10.0  # ← max 10 secondes d'attente
        )
        response.raise_for_status()
        generated_text = response.json()["response"]
    
    except Exception as e:
        generated_text = f"Erreur lors de la génération avec Ollama : {e}"
        
    
    return {
        "query": request.query,
        "documents": search_results,
        "generated_response": generated_text
    }
