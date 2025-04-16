import os
import fitz  # PyMuPDF
import re
import nltk
import faiss
import pickle
from nltk.tokenize import sent_tokenize
from sentence_transformers import SentenceTransformer
from datetime import datetime

nltk.download('punkt_tab')

# 📥 Dossier contenant les PDF
INPUT_FOLDER = "C:/Users/fred_/OneDrive/ML/ML-LLM/CONTRATOS/JEUX_REDUIT"
OUTPUT_FOLDER = "C:/users/fred_/OneDrive/ML/RAGonPDF/Text"
INDEX_PATH = "C:/Users/fred_/OneDrive/ML/RAGonPDF/Index_faiss"
LOG_FILE = "log.txt"

# ⚙️ Paramètres
CHUNK_SIZE = 500  # Taille des segments en caractères
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"  # Modèle pour les embeddings

# 🔄 Initialisation
nltk.download("punkt")
os.makedirs(OUTPUT_FOLDER, exist_ok=True)
model = SentenceTransformer(EMBEDDING_MODEL)  # Charge le modèle d'embedding

# 🔹 Création de la base vectorielle FAISS
embedding_dim = model.get_sentence_embedding_dimension()
index = faiss.IndexFlatL2(embedding_dim)  # Index basé sur la distance L2
metadata = []  # Stocke les segments associés aux embeddings

def clean_text(text):
    """ Nettoie le texte : suppression des espaces et des lignes inutiles. """
    text = re.sub(r"\s+", " ", text).strip()
    return text

def segment_text(text, chunk_size):
    """ Segmente le texte en morceaux de taille définie. """
    sentences = sent_tokenize(text)
    chunks = []
    current_chunk = ""

    for sentence in sentences:
        if len(current_chunk) + len(sentence) <= chunk_size:
            current_chunk += " " + sentence
        else:
            chunks.append(current_chunk.strip())
            current_chunk = sentence

    if current_chunk:
        chunks.append(current_chunk.strip())

    return chunks

# 📘 Ouverture du fichier de log
log_file = open(LOG_FILE, "w", encoding="utf-8")

def log(msg):
    timestamp = datetime.now().strftime("[%Y-%m-%d %H:%M:%S]")
    full_msg = f"{timestamp} {msg}"
    print(full_msg)
    log_file.write(full_msg + "\n")

# 🔍 Parcours des PDF
for root, dirs, files in os.walk(INPUT_FOLDER):
    for filename in files:
        if filename.endswith(".pdf"):
            pdf_path = os.path.join(root, filename)
            relative_path = os.path.relpath(root, INPUT_FOLDER)
            output_dir = os.path.join(OUTPUT_FOLDER, relative_path)
            os.makedirs(output_dir, exist_ok=True)
            text_output_path = os.path.join(output_dir, filename.replace(".pdf", "_segments.txt"))


            # 🚫 Vérifie que le fichier n'est pas vide
            if os.path.getsize(pdf_path) == 0:
                log(f"❌ Fichier vide : {filename}, ignoré.")
                continue

            try:
                doc = fitz.open(pdf_path)
            except Exception as e:
                log(f"❌ Erreur à l'ouverture de {filename} : {e}")
                continue
            
            # 📝 Extraction du texte
            doc = fitz.open(pdf_path)
            text = " ".join([page.get_text("text") for page in doc])

            # 🔍 Ignore les PDF vides
            if not text.strip():
                log(f"⚠️ {filename} est vide ou illisible.")
                continue

            # ✨ Nettoyage
            text = clean_text(text)

            # ✂️ Segmentation
            chunks = segment_text(text, CHUNK_SIZE)

            if not chunks:
                log(f"⚠️ Aucun chunk trouvé dans {filename}.")
                continue

            # 🧠 Conversion en embeddings et ajout à FAISS
            chunk_embeddings = model.encode(chunks, convert_to_numpy=True)
            
            
             # 🛠️ Forcer la forme correcte si 1 seul chunk
            if len(chunk_embeddings.shape) == 1:
                chunk_embeddings = chunk_embeddings.reshape(1, -1)

            # 🔍 Debug
            log(f"📐 Embeddings shape pour {filename} : {chunk_embeddings.shape}")

            # 🛠️ Correction forme 1D
            if chunk_embeddings.ndim == 1:
                log(f"ℹ️ 1 seul embedding pour {filename}, reshape vers (1, -1).")
                chunk_embeddings = chunk_embeddings.reshape(1, -1)

            # 🧱 Vérifie que c’est bien une matrice 2D
            if chunk_embeddings.ndim != 2:
                log(f"❌ Erreur : embeddings avec shape inattendue pour {filename}, ignoré.")
                continue

            # 🚫 Vérifie que ce n’est pas vide
            if chunk_embeddings.shape[0] == 0:
                log(f"⚠️ Embeddings vides pour {filename}, fichier ignoré.")
                continue
                      
            # ➕ Ajout à FAISS
            index.add(chunk_embeddings)
            

            # 📌 Stockage des métadonnées
            metadata.extend([(filename, i, chunk) for i, chunk in enumerate(chunks)])

            # 📂 Sauvegarde des segments
            with open(text_output_path, "w", encoding="utf-8") as text_file:
                for i, chunk in enumerate(chunks):
                    text_file.write(f"### Segment {i+1} ###\n{chunk}\n\n")

            log(f"📄 {filename} : texte segmenté et indexé.")


# 💾 Sauvegarde de l'index FAISS et des métadonnées
faiss.write_index(index, os.path.join(INDEX_PATH, "faiss.index"))

with open(os.path.join(INDEX_PATH, "metadata.pkl"), "wb") as f:
    pickle.dump(metadata, f)

log("✅ Indexation terminée et sauvegardée !")


# ✅ Fermeture du fichier de log
log_file.close()
