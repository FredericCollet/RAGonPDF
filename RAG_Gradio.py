import gradio as gr
import requests

# 📌 URL de ton API FastAPI (modifie si nécessaire)
API_URL = "http://127.0.0.1:8000/rag/"

def query_rag(question, top_k):
    """ Envoie une requête à l'API FastAPI pour obtenir une réponse. """
    response = requests.post(API_URL, json={"query": question, "top_k": int(top_k)})

    if response.status_code == 200:
        data = response.json()
        answer = data["generated_response"]
        documents = "\n\n".join([f"📄 **{doc['filename']}**\n{doc['text']}" for doc in data["documents"]])
        return answer, documents
    else:
        return "Erreur : Impossible de contacter l'API.", ""

# 🎨 Interface Gradio
with gr.Blocks() as demo:
    gr.Markdown("## 🔍 RAG avec Mistral et Ollama")
    gr.Markdown("Pose une question et laisse l'IA répondre en s'appuyant sur les documents indexés.")

    with gr.Row():
        question = gr.Textbox(label="Question", placeholder="Tape ta question ici...")
        top_k = gr.Slider(1, 10, value=5, step=1, label="Nombre de documents à récupérer")

    search_btn = gr.Button("Rechercher")

    with gr.Row():
        answer_box = gr.Textbox(label="Réponse générée", interactive=False)
        docs_box = gr.Textbox(label="Documents pertinents", interactive=False, lines=10)

    search_btn.click(query_rag, inputs=[question, top_k], outputs=[answer_box, docs_box])

# 🚀 Lancer l'interface
demo.launch()
