import gradio as gr
import httpx
import json

API_URL = "http://localhost:8000/rag/"  # ← Assure-toi que ton API tourne sur ce port

def rag_chat(message, history):
    headers = {"Content-Type": "application/json"}
    json_data = {"query": message, "top_k": 3}
    
    response_text = ""
    chat_stream = ""

    try:
        with httpx.stream("POST", API_URL, json=json_data, timeout=None) as response:
            response.raise_for_status()
            for line in response.iter_lines():
                if line:
                    data = json.loads(line)
                    token = data.get("response", "")
                    response_text += token
                    chat_stream += token
                    yield chat_stream  # ← renvoie le texte généré en continu
                    if data.get("done", False):
                        break
    except Exception as e:
        yield f"❌ Erreur : {str(e)}"


chat_interface = gr.ChatInterface(
    fn=rag_chat,
    title="🧠 Assistant PDF avec Mistral",
    chatbot=gr.Chatbot(label="Chat RAG"),
    textbox=gr.Textbox(placeholder="Pose une question...", label="Votre question", lines=2),
    theme="soft",
    fill_height=True,
    retry_btn="↻ Réessayer",
    stop_btn="🛑 Stop",
    clear_btn="🧹 Effacer"
)

if __name__ == "__main__":
    chat_interface.queue().launch(server_name="127.1.1.0", server_port=7860)
