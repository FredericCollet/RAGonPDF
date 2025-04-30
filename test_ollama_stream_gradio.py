import gradio as gr
import httpx
import json

OLLAMA_URL = "http://192.168.1.123:11434"  # ← ajuste si besoin
#OLLAMA_URL = "http://192.168.1.73:11434" # PC_HENRI

def stream_ollama(prompt,histo):
    stream_text = ""

    try:
        with httpx.stream(
            "POST",
            f"{OLLAMA_URL}/api/generate",
            #json={"model": "mistral", "prompt": prompt},
            json={"model": "gemma3", "prompt": prompt},
            #json={"model": "llama2", "prompt": prompt},
            timeout=60.0
        ) as response:
            response.raise_for_status()
            yield "✅ Connexion établie. Génération en cours...\n"

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

demo = gr.ChatInterface(stream_ollama, title="Test Streaming Ollama")

demo.launch(share=False)
