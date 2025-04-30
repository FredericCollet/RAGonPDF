import gradio as gr
import httpx
import json

API_URL = "http://127.0.0.1:8000"  # Ton API FastAPI RAG_API_stream
OLLAMA_URL = "http://192.168.1.123:11434" # PC_MAX
#OLLAMA_URL = "http://192.168.1.73:11434" # PC_HENRI


def stream_ollama(message, history):
    try:
        # 🔍 Étape 1 : Appel à l'API /search
        search_payload = {"query": message, "top_k": 3}
        search_response = httpx.post(f"{API_URL}/search/", json=search_payload)
        search_response.raise_for_status()
        search_results = search_response.json()["results"]

        #Debug
        print(f"Resultal de la rechecheds faiss :  {search_results}" )

        # 📄 Concatène les résultats pour créer le contexte du prompt
        context = "\n\n".join([f"Document: {res['filename']}\nTexte: {res['text']}" for res in search_results])
        prompt = f"""
        Eres un asistente inteligente. Responde a la siguiente pregunta utilizando únicamente la siguiente información:
        {context}

        Question : {message}
        Réponse :
        """

        # 🔁 Étape 2 : Streaming depuis Ollama
        stream_text = ""
        with httpx.stream(
            "POST",
            f"{OLLAMA_URL}/api/generate",
            #json={"model": "mistral", "prompt": prompt},
            json={"model": "gemma3", "prompt": prompt},
            timeout=60.0
        ) as response:
            response.raise_for_status()
            for line in response.iter_lines():
                if line:
                    print(f"🔹 {line}")  # ← utile pour debug
                    data = json.loads(line)
                    token = data.get("response", "")
                    stream_text += token
                    yield stream_text
                    if data.get("done", False):
                        break
    except Exception as e:
        yield f"❌ Erreur : {str(e)}"

chat = gr.ChatInterface(stream_ollama, title="La App Streaming Ollama")

#chat.launch(share=False)
chat.launch(server_name="127.0.0.1", server_port=7860)